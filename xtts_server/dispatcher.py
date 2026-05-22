"""
dispatcher.py — worker process pool management and request routing.

Responsibility boundary:
  - Dispatcher owns worker lifecycle and routing.
  - QueueManager owns the asyncio waiting queue and overflow policy.

Locking model
-------------
A single asyncio.Condition (_cond) protects active_requests on every
WorkerHandle.  Using the Condition's built-in lock for *all* mutations and
reads of active_requests ensures consistency: dispatch(), release(), and
wait_for_free_worker() all hold the same underlying lock, so the notify from
release() is always visible to a waiting wait_for_free_worker().

multiprocessing context
-----------------------
All Queue objects are created via the same "spawn" context used for worker
Process objects.  Mixing contexts (e.g. bare multiprocessing.Queue() on Linux
which defaults to "fork") can produce incompatible pipe handles.  Use
Dispatcher.make_queue() everywhere a result queue is needed.
"""

import asyncio
from collections.abc import Callable
import contextlib
from dataclasses import dataclass, field
import functools
import multiprocessing
import multiprocessing.managers
import queue
import time

from logging_config import get_logger
from worker import (
    ComputeLatentsRequest,
    LatentsResult,
    StreamSynthesisRequest,
    SynthesisRequest,
    worker_main,
)

logger = get_logger(__name__)

# Shared spawn context — used for both Process and Queue creation so that
# the communication pipes are always compatible.
_MP_CTX = multiprocessing.get_context("spawn")


# ---------------------------------------------------------------------------
# Per-worker state
# ---------------------------------------------------------------------------


@dataclass
class WorkerHandle:
    worker_id: str
    gpu_index: int
    process: multiprocessing.Process
    request_queue: multiprocessing.Queue
    active_requests: int = 0
    total_requests: int = 0
    total_synthesis_ms: float = 0.0
    # job_id → monotonic start time; populated by dispatch(), evicted by release()
    active_jobs: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class Dispatcher:
    def __init__(
        self,
        model_path: str,
        workers_per_gpu_list: list[int],
        use_fp16: bool = False,
        worker_timeout_seconds: int = 300,
    ) -> None:
        self._model_path = model_path
        self._workers_per_gpu = workers_per_gpu_list
        self._use_fp16 = use_fp16
        self._worker_timeout = worker_timeout_seconds
        self._workers: list[WorkerHandle] = []
        # Single Condition for all active_requests mutations.  Its internal
        # lock is held by dispatch(), release(), and wait_for_free_worker().
        # Initialized lazily on first use so it binds to the running event loop.
        self._cond: asyncio.Condition | None = None
        self._status_task: asyncio.Task | None = None
        # Optional callable that returns the current queue depth — injected at
        # start_background_tasks() time to avoid a circular import with QueueManager.
        self._queue_depth_fn: Callable[[], int] | None = None
        # Manager is used to create result queues that can be passed to worker
        # processes at runtime via the request queue.  Spawn-context Queues
        # cannot be pickled after process creation (Python 3.11 strictly enforces
        # this), but Manager proxy queues can be shared across processes freely.
        self._manager: multiprocessing.managers.SyncManager | None = None

    def _get_cond(self) -> asyncio.Condition:
        """Return the Condition, creating it lazily inside a running loop."""
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn all worker processes. Call before the event loop starts."""
        # Start the Manager server process first so make_queue() is available.
        self._manager = multiprocessing.Manager()
        total = sum(self._workers_per_gpu)
        logger.info(
            "Dispatcher — spawning %d worker(s) across %d GPU slot(s)",
            total,
            len(self._workers_per_gpu),
        )

        # Import torch only for the device-name lookup — keep CUDA init out of
        # the main process as much as possible to avoid re-init errors in workers.
        try:
            import torch

            cuda_ok = torch.cuda.is_available()
        except ImportError:
            cuda_ok = False

        for gpu_index, count in enumerate(self._workers_per_gpu):
            device_name = "CPU"
            if cuda_ok:
                with contextlib.suppress(Exception):
                    device_name = torch.cuda.get_device_name(gpu_index)

            for slot in range(count):
                worker_id = f"cpu-w{slot}" if not cuda_ok else f"gpu{gpu_index}-w{slot}"
                q = _MP_CTX.Queue()
                p = _MP_CTX.Process(
                    target=worker_main,
                    args=(worker_id, gpu_index, self._model_path, q, self._use_fp16),
                    name=f"xtts-{worker_id}",
                    daemon=True,
                )
                p.start()
                self._workers.append(
                    WorkerHandle(
                        worker_id=worker_id,
                        gpu_index=gpu_index,
                        process=p,
                        request_queue=q,
                    )
                )
                logger.info(
                    "Spawned worker %s on GPU %d (%s) — PID %d",
                    worker_id,
                    gpu_index,
                    device_name,
                    p.pid,
                )

        logger.info("Dispatcher — all workers spawned")

    async def start_background_tasks(self, queue_depth_fn: Callable[[], int] | None = None) -> None:
        """Start periodic status logging. Call after the event loop is running."""
        self._queue_depth_fn = queue_depth_fn
        self._status_task = asyncio.create_task(self._periodic_status())

    async def shutdown(self) -> None:
        """Send shutdown sentinel to every worker and wait for them to exit."""
        if self._status_task:
            self._status_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._status_task

        loop = asyncio.get_running_loop()
        logger.info("Dispatcher — shutting down %d worker(s)", len(self._workers))

        for w in self._workers:
            await loop.run_in_executor(None, w.request_queue.put, None)

        for w in self._workers:
            await loop.run_in_executor(None, w.process.join, 15)
            if w.process.is_alive():
                logger.warning("Worker %s did not exit — terminating", w.worker_id)
                w.process.terminate()

        if self._manager:
            self._manager.shutdown()

        logger.info("Dispatcher — shutdown complete")

    # ------------------------------------------------------------------
    # Queue factory (M-10)
    # ------------------------------------------------------------------

    def make_queue(self) -> multiprocessing.Queue:
        """
        Create a result queue that can be sent to a worker at runtime.

        Spawn-context Queue objects cannot be pickled after process creation —
        Python 3.11 enforces this strictly (assert_spawning raises if you try
        to put a Queue into another Queue's message).  Manager proxy queues are
        exempt from this restriction because they are proxied through a separate
        server process, so they serialize as a thin connection object.
        """
        if self._manager is None:
            raise RuntimeError("Call start() before make_queue()")
        return self._manager.Queue()  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        request: SynthesisRequest | StreamSynthesisRequest | ComputeLatentsRequest,
    ) -> "WorkerHandle":
        """
        Route request to the least-loaded worker and increment its counter.
        Holds the Condition lock during the counter mutation so release()
        and wait_for_free_worker() always see a consistent view.
        """
        if not self._workers:
            raise RuntimeError("No workers available — did you call dispatcher.start()?")

        cond = self._get_cond()
        async with cond:
            worker = self._pick_worker()
            worker.active_requests += 1
            worker.active_jobs[request.job_id] = time.monotonic()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, worker.request_queue.put, request)

        logger.info(
            "Dispatched job=%s → worker=%s (gpu=%d, active=%d)",
            request.job_id,
            worker.worker_id,
            worker.gpu_index,
            worker.active_requests,
        )
        return worker

    async def release(self, worker_id: str, elapsed_ms: float = 0.0, job_id: str = "") -> None:
        """
        Decrement active_requests for the given worker and notify all waiters.
        Must be called every time a dispatched request completes (or errors).
        """
        if not worker_id:
            return

        cond = self._get_cond()
        async with cond:
            for w in self._workers:
                if w.worker_id == worker_id:
                    w.active_requests = max(0, w.active_requests - 1)
                    w.total_synthesis_ms += elapsed_ms
                    w.total_requests += 1
                    if job_id:
                        w.active_jobs.pop(job_id, None)
                    break
            cond.notify_all()

    # ------------------------------------------------------------------
    # Latent computation helper (speaker registration path)
    # ------------------------------------------------------------------

    async def compute_latents(self, job_id: str, wav_path: str) -> LatentsResult:
        """
        Send a ComputeLatentsRequest to any worker and await the result.
        This bypasses the QueueManager waiting queue — latent computation
        is fast and called only from the clone endpoint (not high-frequency).
        """
        loop = asyncio.get_running_loop()
        result_queue = self.make_queue()

        request = ComputeLatentsRequest(
            job_id=job_id,
            wav_path=wav_path,
            result_queue=result_queue,
        )
        worker = await self.dispatch(request)

        try:
            result: LatentsResult = await loop.run_in_executor(
                None, functools.partial(result_queue.get, timeout=self._worker_timeout)
            )
        except queue.Empty:
            logger.error(
                "compute_latents timed out | job=%s | worker=%s | timeout=%ds",
                job_id,
                worker.worker_id,
                self._worker_timeout,
            )
            result = LatentsResult(
                job_id=job_id,
                error=f"Worker did not respond within {self._worker_timeout}s — it may have crashed.",
            )
        finally:
            await self.release(worker.worker_id, elapsed_ms=0.0, job_id=job_id)

        return result

    # ------------------------------------------------------------------
    # Capacity queries (called by QueueManager)
    # ------------------------------------------------------------------

    def min_active_requests(self) -> int:
        if not self._workers:
            return 0
        return min(w.active_requests for w in self._workers)

    def all_busy(self) -> bool:
        """True when every worker has at least one active request."""
        return bool(self._workers) and all(w.active_requests >= 1 for w in self._workers)

    async def wait_for_free_worker(self) -> None:
        """
        Suspend until at least one worker drops to active_requests == 0.
        Uses the same Condition lock as dispatch() and release() so the
        notify from release() is guaranteed to wake this coroutine.
        """
        cond = self._get_cond()
        async with cond:
            while self.all_busy():
                await cond.wait()

    def worker_stats(self) -> list[dict]:
        """Snapshot of per-worker stats. Used for /v1/system/info and logging."""
        now = time.monotonic()
        return [
            {
                "worker_id": w.worker_id,
                "gpu_index": w.gpu_index,
                "active_requests": w.active_requests,
                "total_requests": w.total_requests,
                "avg_synthesis_ms": (
                    w.total_synthesis_ms / w.total_requests if w.total_requests else 0.0
                ),
                "alive": w.process.is_alive(),
                "active_jobs": [
                    {"job_id": jid, "elapsed_s": round(now - started_at, 2)}
                    for jid, started_at in w.active_jobs.items()
                ],
            }
            for w in self._workers
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _pick_worker(self) -> WorkerHandle:
        """Return worker with the lowest active_requests. Call while holding _cond."""
        return min(self._workers, key=lambda w: w.active_requests)

    async def _periodic_status(self) -> None:
        while True:
            await asyncio.sleep(60)
            depth = self._queue_depth_fn() if self._queue_depth_fn else None
            self._log_status(queue_depth=depth)

    def _log_status(self, queue_depth: int | None = None) -> None:
        now = time.monotonic()
        queue_info = f" | queue={queue_depth} waiting" if queue_depth is not None else ""
        logger.info("--- Worker status%s ---", queue_info)

        for w in self._workers:
            avg_ms = w.total_synthesis_ms / w.total_requests if w.total_requests else 0.0
            logger.info(
                "  worker=%s  gpu=%d  alive=%s  active=%d  total=%d  avg_ms=%.1f",
                w.worker_id,
                w.gpu_index,
                w.process.is_alive(),
                w.active_requests,
                w.total_requests,
                avg_ms,
            )
            if w.active_jobs:
                for job_id, started_at in w.active_jobs.items():
                    elapsed = now - started_at
                    logger.info("    job=%s  elapsed=%.1fs", job_id, elapsed)
            else:
                logger.info("    (idle)")

        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    used_mb = torch.cuda.memory_allocated(i) / (1024**2)
                    total_mb = torch.cuda.get_device_properties(i).total_memory / (1024**2)
                    logger.info(
                        "  GPU %d VRAM: %.1f / %.1f MB (%.1f%% used)",
                        i,
                        used_mb,
                        total_mb,
                        100 * used_mb / total_mb,
                    )
        except ImportError:
            pass

        logger.info("--- End worker status ---")
