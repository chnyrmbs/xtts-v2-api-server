"""
JobStore — in-memory store for async synthesis jobs.

Lifecycle of a job
------------------
  PENDING  → job created, waiting in the asyncio queue
  RUNNING  → dispatched to a worker, synthesis in progress
  DONE     → synthesis succeeded, audio file written to disk
  FAILED   → synthesis raised an exception, error message stored

Storage model
-------------
Jobs are kept in a plain dict guarded by asyncio.Lock.  There is no Redis or
database dependency.  Completed audio is written to OUTPUTS_DIR as a WAV file
named {job_id}.wav; the job record stores the path so the /jobs/{id}/audio
endpoint can stream it back.

TTL / cleanup
-------------
A background task runs every 60 s and removes jobs older than JOB_TTL_SECONDS.
Only DONE and FAILED jobs are pruned — PENDING / RUNNING jobs are never evicted.
"""

import asyncio
import contextlib
from dataclasses import dataclass, field
import enum
import os
import time
import uuid

from logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Job model
# ---------------------------------------------------------------------------


class JobStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    finished_at: float | None = None
    # Path to the output audio file on disk (set when status → DONE).
    audio_path: str | None = None
    # Human-readable error message (set when status → FAILED).
    error: str | None = None
    # Request metadata stored for observability / polling responses.
    text_preview: str = ""
    language: str = ""
    speaker_id: str = ""
    worker_id: str = ""
    gpu_index: int = -1

    # ------------------------------------------------------------------
    # Derived timing fields (computed on read, not stored separately)
    # ------------------------------------------------------------------

    def queue_wait_ms(self) -> float | None:
        if self.started_at is None:
            return None
        return (self.started_at - self.created_at) * 1000

    def synthesis_ms(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at) * 1000

    def total_ms(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.created_at) * 1000


# ---------------------------------------------------------------------------
# JobStore
# ---------------------------------------------------------------------------


class JobStore:
    def __init__(self, outputs_dir: str, ttl_seconds: int = 300) -> None:
        self._outputs_dir = outputs_dir
        self._ttl = ttl_seconds
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

        os.makedirs(outputs_dir, exist_ok=True)
        logger.info(
            "JobStore initialised — outputs_dir=%s, ttl=%ds",
            outputs_dir,
            ttl_seconds,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the TTL cleanup background task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop(), name="job-cleanup")
        logger.info("JobStore cleanup loop started (interval=60s, ttl=%ds)", self._ttl)

    async def shutdown(self) -> None:
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    async def create(
        self,
        text: str,
        language: str,
        speaker_id: str,
    ) -> Job:
        """Create a new PENDING job and return it."""
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            text_preview=text[:60].replace("\n", " "),
            language=language,
            speaker_id=speaker_id,
        )
        async with self._lock:
            self._jobs[job_id] = job
        logger.info(
            "Job created | job_id=%s | lang=%s | speaker=%s | preview='%s'",
            job_id,
            language,
            speaker_id,
            job.text_preview,
        )
        return job

    async def mark_running(self, job_id: str, worker_id: str, gpu_index: int, elapsed_ms: float = 0.0) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                logger.warning("mark_running — job not found: %s", job_id)
                return
            job.status = JobStatus.RUNNING
            # Back-calculate started_at so synthesis_ms() reflects actual synthesis
            # duration. elapsed_ms is the time the worker spent on synthesis,
            # measured by _collect_result before on_complete is called.
            job.started_at = time.monotonic() - elapsed_ms / 1000
            job.worker_id = worker_id
            job.gpu_index = gpu_index
        logger.info(
            "Job running | job_id=%s | worker=%s | gpu=%d",
            job_id,
            worker_id,
            gpu_index,
        )

    async def mark_done(self, job_id: str, audio_path: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                logger.warning("mark_done — job not found: %s", job_id)
                return
            job.status = JobStatus.DONE
            job.finished_at = time.monotonic()
            job.audio_path = audio_path
            # Capture synthesis_ms while still holding the lock so the TTL
            # eviction loop cannot delete the job between the lock release and
            # the log call below.
            synthesis_ms = job.synthesis_ms() or 0.0

        logger.info(
            "Job done | job_id=%s | synthesis_ms=%.1f | audio_path=%s",
            job_id,
            synthesis_ms,
            audio_path,
        )

    async def mark_failed(self, job_id: str, error: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                logger.warning("mark_failed — job not found: %s", job_id)
                return
            job.status = JobStatus.FAILED
            job.finished_at = time.monotonic()
            job.error = error
        logger.error("Job failed | job_id=%s | error=%s", job_id, error[:200])

    async def clear_audio_path(self, job_id: str) -> None:
        """Clear audio_path after the file has been served and deleted from disk."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.audio_path = None

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    async def get(self, job_id: str) -> Job | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def list_all(self) -> list:
        """Return a snapshot of all jobs. Used by GET /v1/jobs."""
        async with self._lock:
            return list(self._jobs.values())

    async def stats(self) -> dict:
        async with self._lock:
            counts: dict[str, int] = {s.value: 0 for s in JobStatus}
            for job in self._jobs.values():
                counts[job.status.value] += 1
        return {"total": sum(counts.values()), **counts}

    # ------------------------------------------------------------------
    # TTL cleanup
    # ------------------------------------------------------------------

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            await self._evict_expired()

    async def _evict_expired(self) -> None:
        now = time.monotonic()
        cutoff = now - self._ttl
        evicted = []

        async with self._lock:
            for job_id, job in list(self._jobs.items()):
                # Only evict terminal jobs — never touch PENDING or RUNNING.
                if job.status not in (JobStatus.DONE, JobStatus.FAILED):
                    continue
                if job.finished_at is not None and job.finished_at < cutoff:
                    # Remove the audio file from disk to reclaim space.
                    if job.audio_path and os.path.isfile(job.audio_path):
                        try:
                            os.remove(job.audio_path)
                        except OSError as e:
                            logger.warning(
                                "Could not delete audio file %s: %s",
                                job.audio_path,
                                e,
                            )
                    del self._jobs[job_id]
                    evicted.append(job_id)

        if evicted:
            logger.info(
                "JobStore TTL eviction — removed %d job(s): %s",
                len(evicted),
                evicted[:10],
            )

        # Periodic summary even when nothing was evicted.
        s = await self.stats()
        logger.info(
            "JobStore status — pending=%d running=%d done=%d failed=%d total=%d",
            s["pending"],
            s["running"],
            s["done"],
            s["failed"],
            s["total"],
        )
