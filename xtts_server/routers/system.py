"""
routers/system.py — health check and system status endpoints.

Endpoints
---------
  GET /health
      Minimal liveness probe.  Returns 200 as long as the server is running.
      Suitable for Kubernetes/Docker health checks — no heavy computation.

  GET /v1/system/info
      Full observability snapshot: GPU inventory, per-worker stats, queue
      depth, and job store counts.  Intended for dashboards and debugging.

All shared state (dispatcher, queue_manager, job_store) is read from
`request.app.state`, which is populated during the FastAPI startup event
in main.py.
"""

import platform
import time

from fastapi import APIRouter, Request
from pydantic import BaseModel
import torch

from logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["system"])

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str  # always "ok" when the server is reachable
    uptime_s: float  # seconds since the server started


class GpuInfo(BaseModel):
    index: int
    name: str
    total_vram_mb: float
    used_vram_mb: float
    free_vram_mb: float
    utilisation_pct: float


class ActiveJobInfo(BaseModel):
    job_id: str
    elapsed_s: float


class WorkerInfo(BaseModel):
    worker_id: str
    gpu_index: int
    active_requests: int
    total_requests: int
    avg_synthesis_ms: float
    alive: bool
    active_jobs: list[ActiveJobInfo]


class QueueInfo(BaseModel):
    depth: int
    max_size: int


class JobCounts(BaseModel):
    total: int
    pending: int
    running: int
    done: int
    failed: int


class SystemInfoResponse(BaseModel):
    server_start_time: float  # unix timestamp
    uptime_s: float
    python_version: str
    torch_version: str
    cuda_available: bool
    gpus: list[GpuInfo]
    workers: list[WorkerInfo]
    queue: QueueInfo
    jobs: JobCounts


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns 200 immediately. Use this for load-balancer / container health checks.",
)
async def health(request: Request) -> HealthResponse:
    uptime = time.monotonic() - request.app.state.start_time
    return HealthResponse(status="ok", uptime_s=round(uptime, 2))


@router.get(
    "/v1/system/info",
    response_model=SystemInfoResponse,
    summary="Full system status",
    description=(
        "Returns GPU inventory, per-worker stats, asyncio queue depth, "
        "and job store counts.  Aggregates all runtime state in one call."
    ),
)
async def system_info(request: Request) -> SystemInfoResponse:
    state = request.app.state
    uptime = time.monotonic() - state.start_time

    # ---- GPU inventory ------------------------------------------------
    gpus: list[GpuInfo] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mb = props.total_memory / (1024**2)
            used_mb = torch.cuda.memory_allocated(i) / (1024**2)
            free_mb = total_mb - used_mb
            util_pct = (used_mb / total_mb * 100) if total_mb > 0 else 0.0
            gpus.append(
                GpuInfo(
                    index=i,
                    name=props.name,
                    total_vram_mb=round(total_mb, 1),
                    used_vram_mb=round(used_mb, 1),
                    free_vram_mb=round(free_mb, 1),
                    utilisation_pct=round(util_pct, 1),
                )
            )

    # ---- Worker stats -------------------------------------------------
    raw_workers = state.dispatcher.worker_stats()
    workers = [WorkerInfo(**w) for w in raw_workers]

    # ---- Queue --------------------------------------------------------
    q_stats = state.queue_manager.stats()
    queue = QueueInfo(**q_stats)

    # ---- Job store ----------------------------------------------------
    j_stats = await state.job_store.stats()
    jobs = JobCounts(**j_stats)

    logger.debug(
        "system_info — uptime=%.1fs workers=%d queue=%d jobs_total=%d",
        uptime,
        len(workers),
        queue.depth,
        jobs.total,
    )

    return SystemInfoResponse(
        server_start_time=state.start_timestamp,
        uptime_s=round(uptime, 2),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        gpus=gpus,
        workers=workers,
        queue=queue,
        jobs=jobs,
    )
