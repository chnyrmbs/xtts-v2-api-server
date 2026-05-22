"""
Worker process — one XTTS-v2 model instance per process, pinned to a single GPU.

Request types (put onto request_queue by the dispatcher):
  SynthesisRequest       → full audio at once via model.inference()
  StreamSynthesisRequest → chunked audio via model.inference_stream()
  ComputeLatentsRequest  → extract gpt_cond_latent + speaker_embedding from a wav

All latents travel as CPU numpy arrays so they are picklable through
multiprocessing.Queue without GPU-to-GPU transfer issues.

Shutdown: put None on request_queue.
"""

from dataclasses import dataclass, field
from multiprocessing import Queue
import os
import sys
import time
import traceback

import numpy as np

# ---------------------------------------------------------------------------
# Request / result data classes  (imported by dispatcher and routers too)
# ---------------------------------------------------------------------------


@dataclass
class SynthesisRequest:
    job_id: str
    text: str
    language: str
    gpt_cond_latent: np.ndarray  # float32, CPU numpy — shape (1, T, 1024)
    speaker_embedding: np.ndarray  # float32, CPU numpy — shape (1, 512, 1)
    result_queue: object = field(repr=False)  # Queue[SynthesisResult]
    temperature: float = 0.75
    length_penalty: float = 1.0
    repetition_penalty: float = 10.0
    top_k: int = 50
    top_p: float = 0.85
    do_sample: bool = True
    num_beams: int = 1
    speed: float = 1.0
    enable_text_splitting: bool = False


@dataclass
class StreamSynthesisRequest:
    job_id: str
    text: str
    language: str
    gpt_cond_latent: np.ndarray
    speaker_embedding: np.ndarray
    result_queue: object = field(repr=False)  # Queue[SynthesisChunk | SynthesisStreamEnd]
    temperature: float = 0.75
    length_penalty: float = 1.0
    repetition_penalty: float = 10.0
    top_k: int = 50
    top_p: float = 0.85
    do_sample: bool = True
    num_beams: int = 1
    speed: float = 1.0
    enable_text_splitting: bool = False


@dataclass
class ComputeLatentsRequest:
    job_id: str
    wav_path: str  # absolute path to reference wav
    result_queue: object = field(repr=False)  # Queue[LatentsResult]


@dataclass
class SynthesisResult:
    job_id: str
    audio: np.ndarray | None  # float32, shape (N,), 24 kHz
    sample_rate: int = 24000
    error: str | None = None


@dataclass
class SynthesisChunk:
    job_id: str
    chunk: np.ndarray  # float32 PCM chunk


@dataclass
class SynthesisStreamEnd:
    job_id: str
    error: str | None = None


@dataclass
class LatentsResult:
    job_id: str
    gpt_cond_latent: np.ndarray | None = None  # float32 CPU numpy
    speaker_embedding: np.ndarray | None = None  # float32 CPU numpy
    error: str | None = None


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------


def worker_main(
    worker_id: str,
    gpu_index: int,
    model_path: str,
    request_queue: Queue,
    use_fp16: bool = False,
) -> None:
    from logging_config import get_logger

    logger = get_logger(f"worker.{worker_id}")

    logger.info(
        "Worker %s starting — gpu_index=%d, model_path=%s",
        worker_id,
        gpu_index,
        model_path,
    )

    # ---- Device -------------------------------------------------------
    import torch

    if torch.cuda.is_available() and gpu_index >= 0:
        device = f"cuda:{gpu_index}"
        props = torch.cuda.get_device_properties(gpu_index)
        device_name = props.name
        total_vram_gb = props.total_memory / (1024**3)
        logger.info(
            "Worker %s — device=%s name='%s' total_vram=%.2f GB",
            worker_id,
            device,
            device_name,
            total_vram_gb,
        )

        # Apply VRAM cap before any GPU memory is allocated.
        # GPU_MEMORY_FRACTION is exported by start-server.sh; 1.0 means no cap.
        _apply_memory_fraction(gpu_index, worker_id, logger)
    else:
        device = "cpu"
        device_name = "CPU"
        logger.warning("Worker %s — CUDA not available, using CPU", worker_id)

    # ---- Validate model path ------------------------------------------
    _validate_model_path(model_path, worker_id, logger)

    # ---- Load model ---------------------------------------------------
    logger.info("Worker %s — loading model …", worker_id)
    t0 = time.monotonic()

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.load_json(os.path.join(model_path, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
    model.to(device)

    fp16 = use_fp16 and device.startswith("cuda")
    if fp16:
        logger.info("Worker %s — fp16 autocast enabled", worker_id)

    elapsed = time.monotonic() - t0
    logger.info("Worker %s — model loaded in %.2f s", worker_id, elapsed)

    if device.startswith("cuda"):
        vram_mb = torch.cuda.memory_allocated(gpu_index) / (1024**2)
        logger.info("Worker %s — VRAM after load: %.1f MB", worker_id, vram_mb)

    # ---- Stats -----------------------------------------------------------
    total_requests = 0
    total_synthesis_ms = 0.0

    # ---- Request loop ----------------------------------------------------
    logger.info("Worker %s — ready", worker_id)

    while True:
        request = request_queue.get()

        if request is None:
            logger.info("Worker %s — shutdown sentinel received", worker_id)
            break

        if isinstance(request, SynthesisRequest):
            ms = _handle_synthesis(request, model, device, gpu_index, worker_id, logger, fp16)
            total_requests += 1
            total_synthesis_ms += ms

        elif isinstance(request, StreamSynthesisRequest):
            ms = _handle_stream(request, model, device, gpu_index, worker_id, logger, fp16)
            total_requests += 1
            total_synthesis_ms += ms

        elif isinstance(request, ComputeLatentsRequest):
            _handle_compute_latents(request, model, worker_id, logger)

        else:
            logger.warning("Worker %s — unknown request type: %s", worker_id, type(request))

    avg = total_synthesis_ms / total_requests if total_requests else 0.0
    logger.info(
        "Worker %s — shutdown complete | total=%d | avg_ms=%.1f",
        worker_id,
        total_requests,
        avg,
    )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _to_tensor(arr: np.ndarray, device: str):
    import torch

    return torch.from_numpy(arr).to(device)


def _handle_synthesis(
    request: SynthesisRequest,
    model,
    device: str,
    gpu_index: int,
    worker_id: str,
    logger,
    fp16: bool = False,
) -> float:
    elapsed_ms = 0.0  # guard: ensure always defined even if put() raises
    text_preview = request.text[:60].replace("\n", " ")
    logger.info(
        "Worker %s — inference start | job=%s | lang=%s | text='%s…'",
        worker_id,
        request.job_id,
        request.language,
        text_preview,
    )

    import torch

    t0 = time.monotonic()
    try:
        gpt_cond_latent = _to_tensor(request.gpt_cond_latent, device)
        speaker_embedding = _to_tensor(request.speaker_embedding, device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
            outputs = model.inference(
                request.text,
                request.language,
                gpt_cond_latent,
                speaker_embedding,
                temperature=request.temperature,
                length_penalty=request.length_penalty,
                repetition_penalty=request.repetition_penalty,
                top_k=request.top_k,
                top_p=request.top_p,
                do_sample=request.do_sample,
                num_beams=request.num_beams,
                speed=request.speed,
                enable_text_splitting=request.enable_text_splitting,
            )
        audio = np.array(outputs["wav"], dtype=np.float32)

        elapsed_ms = (time.monotonic() - t0) * 1000
        audio_s = len(audio) / 24000
        rtf = audio_s / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0

        logger.info(
            "Worker %s — inference done | job=%s | ms=%.1f | audio_s=%.2f | RTF=%.2f",
            worker_id,
            request.job_id,
            elapsed_ms,
            audio_s,
            rtf,
        )

        if device.startswith("cuda"):
            vram_mb = torch.cuda.memory_allocated(gpu_index) / (1024**2)
            logger.debug("Worker %s — VRAM after inference: %.1f MB", worker_id, vram_mb)

        request.result_queue.put(SynthesisResult(job_id=request.job_id, audio=audio))

    except Exception:
        elapsed_ms = (time.monotonic() - t0) * 1000
        tb = traceback.format_exc()
        logger.error(
            "Worker %s — inference exception | job=%s | text='%s…'\n%s",
            worker_id,
            request.job_id,
            text_preview,
            tb,
        )
        request.result_queue.put(SynthesisResult(job_id=request.job_id, audio=None, error=tb))

    return elapsed_ms


def _handle_stream(
    request: StreamSynthesisRequest,
    model,
    device: str,
    gpu_index: int,
    worker_id: str,
    logger,
    fp16: bool = False,
) -> float:
    elapsed_ms = 0.0  # guard: ensure always defined even if put() raises
    text_preview = request.text[:60].replace("\n", " ")
    logger.info(
        "Worker %s — stream start | job=%s | lang=%s | text='%s…'",
        worker_id,
        request.job_id,
        request.language,
        text_preview,
    )

    import torch

    t0 = time.monotonic()
    elapsed_ms = 0.0
    chunk_count = 0

    try:
        gpt_cond_latent = _to_tensor(request.gpt_cond_latent, device)
        speaker_embedding = _to_tensor(request.speaker_embedding, device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
            for chunk in model.inference_stream(
                request.text,
                request.language,
                gpt_cond_latent,
                speaker_embedding,
                temperature=request.temperature,
                length_penalty=request.length_penalty,
                repetition_penalty=request.repetition_penalty,
                top_k=request.top_k,
                top_p=request.top_p,
                do_sample=request.do_sample,
                num_beams=request.num_beams,
                speed=request.speed,
                enable_text_splitting=request.enable_text_splitting,
            ):
                chunk_np = np.array(chunk, dtype=np.float32)
                request.result_queue.put(SynthesisChunk(job_id=request.job_id, chunk=chunk_np))
                chunk_count += 1

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "Worker %s — stream done | job=%s | ms=%.1f | chunks=%d",
            worker_id,
            request.job_id,
            elapsed_ms,
            chunk_count,
        )
        request.result_queue.put(SynthesisStreamEnd(job_id=request.job_id))

    except Exception:
        elapsed_ms = (time.monotonic() - t0) * 1000
        tb = traceback.format_exc()
        logger.error(
            "Worker %s — stream exception | job=%s | text='%s…'\n%s",
            worker_id,
            request.job_id,
            text_preview,
            tb,
        )
        request.result_queue.put(SynthesisStreamEnd(job_id=request.job_id, error=tb))

    return elapsed_ms


def _handle_compute_latents(
    request: ComputeLatentsRequest,
    model,
    worker_id: str,
    logger,
) -> None:
    logger.info(
        "Worker %s — compute_latents start | job=%s | wav=%s",
        worker_id,
        request.job_id,
        request.wav_path,
    )
    t0 = time.monotonic()
    try:
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[request.wav_path]
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        gpt_np = gpt_cond_latent.cpu().numpy()
        emb_np = speaker_embedding.cpu().numpy()
        result = LatentsResult(
            job_id=request.job_id,
            gpt_cond_latent=gpt_np,
            speaker_embedding=emb_np,
        )
        logger.info(
            "Worker %s — compute_latents done | job=%s | elapsed_ms=%.1f"
            " | gpt_shape=%s | emb_shape=%s",
            worker_id,
            request.job_id,
            elapsed_ms,
            gpt_np.shape,
            emb_np.shape,
        )
    except Exception:
        elapsed_ms = (time.monotonic() - t0) * 1000
        tb = traceback.format_exc()
        logger.error(
            "Worker %s — compute_latents failed | job=%s | elapsed_ms=%.1f\n%s",
            worker_id,
            request.job_id,
            elapsed_ms,
            tb,
        )
        result = LatentsResult(job_id=request.job_id, error=tb)

    request.result_queue.put(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_memory_fraction(gpu_index: int, worker_id: str, logger) -> None:
    """Cap per-process VRAM usage via GPU_MEMORY_FRACTION env var (set by start-server.sh)."""
    import torch

    raw = os.environ.get("GPU_MEMORY_FRACTION", "1.0")
    try:
        fraction = float(raw)
    except ValueError:
        logger.warning(
            "Worker %s — GPU_MEMORY_FRACTION='%s' is not a valid float — ignoring",
            worker_id,
            raw,
        )
        return

    if not (0.0 < fraction <= 1.0):
        logger.warning(
            "Worker %s — GPU_MEMORY_FRACTION=%.3f is out of range (0, 1] — ignoring",
            worker_id,
            fraction,
        )
        return

    if fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(fraction, gpu_index)
        props = torch.cuda.get_device_properties(gpu_index)
        cap_gb = props.total_memory * fraction / (1024**3)
        logger.info(
            "Worker %s — VRAM capped at %.0f%% of GPU %d (≈ %.2f GB / %.2f GB total)",
            worker_id,
            fraction * 100,
            gpu_index,
            cap_gb,
            props.total_memory / (1024**3),
        )


def _validate_model_path(model_path: str, worker_id: str, logger) -> None:
    required = ["config.json", "model.pth", "vocab.json"]

    if not os.path.isdir(model_path):
        logger.error("Worker %s — MODEL_PATH not a directory: %s", worker_id, model_path)
        sys.exit(1)

    missing = []
    for fname in required:
        fpath = os.path.join(model_path, fname)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024**2)
            logger.info("Worker %s — [FOUND] %s (%.2f MB)", worker_id, fname, size_mb)
        else:
            logger.error("Worker %s — [MISSING] %s", worker_id, fname)
            missing.append(fname)

    if missing:
        logger.error(
            "Worker %s — missing required files: %s — aborting",
            worker_id,
            ", ".join(missing),
        )
        sys.exit(1)
