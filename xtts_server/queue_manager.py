"""
QueueManager — asyncio waiting queue that sits in front of the Dispatcher.

Why this layer exists
---------------------
The Dispatcher routes to workers but never waits. When all workers are busy,
requests would pile up inside the workers' multiprocessing.Queues with no
fairness or backpressure. QueueManager fixes this:

  1. Holds an asyncio.Queue (bounded by MAX_QUEUE_SIZE).
  2. A single drain-loop task pops one item at a time, waits until a worker
     slot is free, then dispatches.  This keeps the backpressure in Python
     (asyncio) rather than inside OS pipe buffers.
  3. Returns 503 only when the asyncio queue itself is full (all workers busy
     AND more than MAX_QUEUE_SIZE requests already waiting).

Two submission paths
--------------------
* submit_job(request, on_complete)
    For async fire-and-poll jobs.  After the worker finishes, `on_complete`
    is called with (WorkerHandle, SynthesisResult) so the caller can update
    the job store.

* submit_stream(request)
    For WebSocket streaming.  Queues the request the same way, but there is
    no callback — the WebSocket handler owns the result_queue and reads chunks
    directly.  The drain loop simply dispatches and moves on.
"""

import asyncio
from collections.abc import Awaitable, Callable
import contextlib
from dataclasses import dataclass, field
import functools
import queue
import time
from typing import Any

from dispatcher import Dispatcher, WorkerHandle
from logging_config import get_logger
from worker import StreamSynthesisRequest, SynthesisRequest, SynthesisResult

logger = get_logger(__name__)


class QueueFullError(Exception):
    """Raised when the waiting queue has hit MAX_QUEUE_SIZE."""


# ---------------------------------------------------------------------------
# Internal envelope — wraps a request with metadata for the drain loop
# ---------------------------------------------------------------------------


@dataclass
class _QueuedItem:
    request: Any  # SynthesisRequest | StreamSynthesisRequest
    enqueued_at: float = field(default_factory=time.monotonic)
    # on_complete is None for streaming jobs; the WS handler owns the result.
    on_complete: Callable[[WorkerHandle, SynthesisResult, float], Awaitable[None]] | None = None
    # Resolved by the drain loop once dispatched; lets the WS handler learn
    # which worker was assigned so it can call dispatcher.release() correctly.
    worker_future: asyncio.Future | None = None


# ---------------------------------------------------------------------------
# QueueManager
# ---------------------------------------------------------------------------


class QueueManager:
    def __init__(self, dispatcher: Dispatcher, max_queue_size: int, worker_timeout_seconds: int = 300) -> None:
        self._dispatcher = dispatcher
        self._max_size = max_queue_size
        self._worker_timeout = worker_timeout_seconds
        # asyncio.Queue with a hard size cap — put_nowait raises QueueFull when full.
        self._queue: asyncio.Queue[_QueuedItem] = asyncio.Queue(maxsize=max_queue_size)
        self._drain_task: asyncio.Task | None = None
        logger.info("QueueManager initialised — max_queue_size=%d", max_queue_size)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background drain loop. Call after the event loop is running."""
        self._drain_task = asyncio.create_task(self._drain_loop(), name="queue-drain")
        logger.info("QueueManager drain loop started")

    async def shutdown(self) -> None:
        if self._drain_task:
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._drain_task
        logger.info("QueueManager shut down")

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    async def submit_job(
        self,
        request: SynthesisRequest,
        on_complete: Callable[[WorkerHandle, SynthesisResult, float], Awaitable[None]],
    ) -> None:
        """
        Enqueue an async job request.

        on_complete(worker, result) is awaited after synthesis finishes.
        Raises QueueFullError if the queue is at capacity.
        """
        item = _QueuedItem(request=request, on_complete=on_complete)
        self._enqueue(item, request.job_id)

    async def submit_stream(
        self, request: StreamSynthesisRequest
    ) -> "asyncio.Future[WorkerHandle]":
        """
        Enqueue a streaming request.

        Returns a Future that resolves to the WorkerHandle once the drain loop
        dispatches the request.  The WebSocket handler awaits this to learn the
        worker_id so it can call dispatcher.release() when the stream ends.
        Raises QueueFullError if the queue is at capacity.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        item = _QueuedItem(request=request, on_complete=None, worker_future=future)
        self._enqueue(item, request.job_id)
        return future

    def _enqueue(self, item: _QueuedItem, job_id: str) -> None:
        depth = self._queue.qsize()
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            logger.error(
                "Queue full — rejecting job=%s | depth=%d | max=%d",
                job_id,
                depth,
                self._max_size,
            )
            raise QueueFullError(
                f"Request queue is full ({depth}/{self._max_size}). Try again later."
            ) from None

        # +1 because we just added to the queue
        position = self._queue.qsize()
        logger.info(
            "Enqueued job=%s | position≈%d | depth=%d/%d",
            job_id,
            position,
            position,
            self._max_size,
        )

    # ------------------------------------------------------------------
    # Drain loop
    # ------------------------------------------------------------------

    async def _drain_loop(self) -> None:
        """
        Continuously:
          1. Pop the next item from the asyncio queue.
          2. Wait until a worker slot is free (active_requests == 0 on any worker).
          3. Dispatch to the least-loaded worker.
          4. For job requests: spawn a result-collector task.
          5. For stream requests: just dispatch — WS handler handles the rest.
        """
        logger.info("Drain loop running")
        while True:
            item = await self._queue.get()

            wait_ms = (time.monotonic() - item.enqueued_at) * 1000
            logger.info(
                "Dequeued job=%s | queue_wait_ms=%.1f | remaining_queue=%d",
                item.request.job_id,
                wait_ms,
                self._queue.qsize(),
            )

            # Wait for a worker that has no active requests before dispatching.
            # This prevents unbounded buildup inside the workers' OS queues.
            await self._dispatcher.wait_for_free_worker()

            worker = await self._dispatcher.dispatch(item.request)

            if item.on_complete is not None:
                # Async job — spawn a task to collect the result without blocking
                # the drain loop (we want to keep draining for other requests).
                # Store the reference to avoid it being garbage-collected (RUF006).
                _task = asyncio.create_task(
                    self._collect_result(
                        worker, item.request, item.on_complete, item.enqueued_at
                    ),
                    name=f"collect-{item.request.job_id}",
                )
                del _task  # intentionally fire-and-forget; name kept for debuggability
            # For streaming requests on_complete is None; the WS handler owns
            # result_queue and will call dispatcher.release() when done.
            # Resolve the future so the WS handler unblocks and learns worker_id.
            if item.worker_future is not None and not item.worker_future.done():
                item.worker_future.set_result(worker)

    async def _collect_result(
        self,
        worker: WorkerHandle,
        request: SynthesisRequest,
        on_complete: Callable[[WorkerHandle, SynthesisResult, float], Awaitable[None]],
        enqueued_at: float,
    ) -> None:
        """
        Await the worker's result (blocking get in a thread-pool executor),
        release the worker slot, then fire the caller's on_complete callback.
        """
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()

        try:
            result: SynthesisResult = await loop.run_in_executor(
                None, functools.partial(request.result_queue.get, timeout=self._worker_timeout)
            )
            elapsed_ms = (time.monotonic() - t0) * 1000
        except queue.Empty:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.error(
                "Worker timed out | job=%s | worker=%s | timeout=%ds",
                request.job_id,
                worker.worker_id,
                self._worker_timeout,
            )
            result = SynthesisResult(
                job_id=request.job_id,
                audio=None,
                error=f"Worker did not respond within {self._worker_timeout}s — it may have crashed.",
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.error(
                "Result collection error | job=%s | worker=%s | %s",
                request.job_id,
                worker.worker_id,
                exc,
            )
            result = SynthesisResult(job_id=request.job_id, audio=None, error=str(exc))

        # Release the worker slot so the drain loop can dispatch the next request.
        await self._dispatcher.release(
            worker.worker_id, elapsed_ms=elapsed_ms, job_id=request.job_id
        )

        # Notify the job store (or whatever the caller registered).
        await on_complete(worker, result, elapsed_ms)

        # Total lifecycle: from the moment the request entered the asyncio queue
        # until on_complete finishes (includes queue wait + synthesis + audio save).
        total_ms = (time.monotonic() - enqueued_at) * 1000
        logger.info(
            "Job lifecycle complete | job=%s | worker=%s | synthesis_ms=%.1f | total_ms=%.1f",
            request.job_id,
            worker.worker_id,
            elapsed_ms,
            total_ms,
        )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def depth(self) -> int:
        """Current number of requests waiting in the asyncio queue."""
        return self._queue.qsize()

    def stats(self) -> dict:
        return {"depth": self._queue.qsize(), "max_size": self._max_size}
