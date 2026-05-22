"""
main.py — FastAPI application factory and startup/shutdown lifecycle.

Startup sequence (order matters)
---------------------------------
  1. Load and validate Settings (MODEL_PATH check, WORKERS_PER_GPU parse).
  2. Spawn worker processes (Dispatcher.start) — blocking, runs before the
     event loop so multiprocessing.Process objects are created in the main
     process context.
  3. Initialise JobStore, SpeakerStore, QueueManager.
  4. Start async background tasks: QueueManager drain loop, Dispatcher
     periodic status logger, JobStore TTL cleanup.
  5. Preload all speakers from disk into the in-memory cache.
  6. Attach all shared objects to app.state so routers can reach them via
     request.app.state.

Shutdown sequence
-----------------
  1. QueueManager shut down (drain loop cancelled).
  2. Dispatcher shut down (sends None sentinels to every worker, joins procs).
  3. JobStore shut down (cleanup task cancelled).

All routers are registered under the same FastAPI app instance.
OpenAPI docs are available at /docs (Swagger UI) and /redoc.
"""

from contextlib import asynccontextmanager
import os
import time
import uuid

from fastapi import FastAPI, Request

from config import Settings, load_settings
from dispatcher import Dispatcher
from job_store import JobStore
from logging_config import get_logger
from queue_manager import QueueManager
from routers import batch, clone, jobs, system, tts
from routers import speakers as speakers_router
from speakers import SpeakerStore
from ws.stream import router as ws_router

logger = get_logger(__name__)
_access_log = get_logger("access")

_VERSION = "1.0.0"


def _log_banner(settings: Settings) -> None:
    """Log a Spring Boot / vLLM style ASCII art banner after settings are loaded."""
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        cuda_ok = False

    total_workers = sum(settings.workers_per_gpu_list)
    device_label = f"{settings.NUM_GPUS} GPU(s)" if cuda_ok else "CPU"
    worker_label = f"{total_workers} × {device_label}"
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", "8000")

    banner = f"""
  ██████╗ ██╗   ██╗ █████╗  ██╗  ██╗███████╗██████╗       ██╗  ██╗████████╗████████╗███████╗
  ██╔═══██╗██║   ██║██╔══██╗ ██║ ██╔╝██╔════╝██╔══██╗      ╚██╗██╔╝╚══██╔══╝╚══██╔══╝██╔════╝
  ██║   ██║██║   ██║███████║ █████╔╝ █████╗  ██████╔╝  ═══  ╚███╔╝    ██║      ██║   ███████╗
  ██║   ██║██║   ██║██╔══██║ ██╔═██╗ ██╔══╝  ██╔═██╗        ██╔██╗    ██║      ██║   ╚════██║
  ╚██████╔╝╚██████╔╝██║  ██║ ██║  ██╗███████╗██║  ╚██╗     ██╔╝ ██╗   ██║      ██║   ███████║
   ╚═════╝  ╚═════╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝╚══════╝╚═╝   ╚═╝     ╚═╝  ╚═╝   ╚═╝      ╚═╝   ╚══════╝
  ═════════════════════════════════════════════════════════════════════════════════════════════════
  :: QUAKER-XTTS Inference Server ::                                               (v{_VERSION})

  Model    : {settings.MODEL_PATH}
  Workers  : {worker_label}
  Language : {settings.DEFAULT_LANGUAGE}
  Queue    : max {settings.MAX_QUEUE_SIZE} concurrent requests
  Endpoint : http://{host}:{port}
  Docs     : http://{host}:{port}/docs
  ═════════════════════════════════════════════════════════════════════════════════════════════════"""

    logger.info(banner)


# ---------------------------------------------------------------------------
# Application lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ------------------------------------------------------------------ UP
    start_time = time.monotonic()
    app.state.start_time = start_time
    app.state.start_timestamp = time.time()

    logger.info("=== XTTS-v2 Server starting up ===")

    # 1. Settings — exits immediately if MODEL_PATH or required files are missing.
    settings = load_settings()
    app.state.settings = settings
    _log_banner(settings)

    # 2. Spawn worker processes synchronously before entering async code.
    #    multiprocessing.Process.start() is not async-safe inside a running
    #    event loop, so we do it here while still in the sync portion of the
    #    lifespan generator (before the first `await`).
    dispatcher = Dispatcher(
        model_path=settings.MODEL_PATH,
        workers_per_gpu_list=settings.workers_per_gpu_list,
        use_fp16=settings.USE_FP16,
        worker_timeout_seconds=settings.WORKER_TIMEOUT_SECONDS,
    )
    dispatcher.start()
    app.state.dispatcher = dispatcher

    # 3. Initialise stores.
    job_store = JobStore(
        outputs_dir=settings.OUTPUTS_DIR,
        ttl_seconds=settings.JOB_TTL_SECONDS,
    )
    app.state.job_store = job_store

    speaker_store = SpeakerStore(speakers_dir=settings.SPEAKERS_DIR)
    app.state.speaker_store = speaker_store

    queue_manager = QueueManager(
        dispatcher=dispatcher,
        max_queue_size=settings.MAX_QUEUE_SIZE,
        worker_timeout_seconds=settings.WORKER_TIMEOUT_SECONDS,
    )
    app.state.queue_manager = queue_manager

    # 4. Start async background tasks.
    await job_store.start()
    await queue_manager.start()
    await dispatcher.start_background_tasks(queue_depth_fn=queue_manager.depth)

    # 5. Warm the speaker cache from disk.
    loaded = speaker_store.preload_all()
    logger.info("Speaker cache warm — %d speaker(s) loaded", loaded)

    logger.info("=== XTTS-v2 Server ready — startup took %.2f s ===", time.monotonic() - start_time)

    yield  # Application runs here.

    # ------------------------------------------------------------------ DOWN
    logger.info("=== XTTS-v2 Server shutting down ===")
    await queue_manager.shutdown()
    await dispatcher.shutdown()
    await job_store.shutdown()
    logger.info("=== Shutdown complete ===")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(
        title="XTTS-v2 Inference Server",
        description=(
            "Production-grade multi-GPU text-to-speech API powered by Coqui XTTS-v2. "
            "Supports async fire-and-poll jobs, WebSocket streaming, voice cloning, "
            "and batch synthesis."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # ---- Access log middleware ----------------------------------------
    # Logs one line per HTTP request: method, path, status, duration, client.
    # /health is logged at DEBUG to avoid noise from frequent liveness probes.
    # A unique X-Request-ID is generated (or forwarded from the client) and
    # attached to every response for client-side correlation.
    @app.middleware("http")
    async def _access_log_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:8]
        client_ip = request.client.host if request.client else "-"
        t0 = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.monotonic() - t0) * 1000
            _access_log.error(
                "%s %s → 500 | %.1f ms | client=%s | req=%s",
                request.method, request.url.path, elapsed_ms, client_ip, req_id,
            )
            raise
        elapsed_ms = (time.monotonic() - t0) * 1000
        log_fn = _access_log.debug if request.url.path == "/health" else _access_log.info
        log_fn(
            "%s %s → %d | %.1f ms | client=%s | req=%s",
            request.method, request.url.path, response.status_code,
            elapsed_ms, client_ip, req_id,
        )
        response.headers["X-Request-ID"] = req_id
        return response

    # ---- Routers -----------------------------------------------------
    app.include_router(system.router)
    app.include_router(tts.router)
    app.include_router(tts.audio_router)
    app.include_router(clone.router)
    app.include_router(batch.router)
    app.include_router(jobs.router)
    app.include_router(speakers_router.router)
    app.include_router(ws_router)

    return app


app = create_app()


# ---------------------------------------------------------------------------
# Entry point — run directly with:  python main.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    # Re-read settings just for the uvicorn bind config.
    # The full settings load (with model validation) happens inside lifespan.
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    log_level = os.environ.get("LOG_LEVEL", "info").lower()

    # Use workers=1 — horizontal scaling is handled by our own worker pool,
    # not by uvicorn's multi-process mode (which would spawn multiple model
    # instances without coordination).
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level=log_level,
        workers=1,
    )
