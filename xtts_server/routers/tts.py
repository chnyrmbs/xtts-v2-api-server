"""
routers/tts.py — TTS synthesis endpoints.

POST /v1/tts (async fire-and-poll)
----
  1. Validate request (text length, language, resolve speaker).
  2. Create a PENDING job in the job store.
  3. Build a SynthesisRequest with a fresh multiprocessing result queue.
  4. Register an on_complete callback that writes the audio file and
     transitions the job to DONE or FAILED.
  5. Submit to QueueManager — returns immediately with job_id.

POST /audio/speech-file (synchronous — waits and returns audio directly)
----
  Same validation and queue path as the async endpoint (backpressure preserved),
  but bridges on_complete to an asyncio.Future so the handler can await the
  result and return the encoded audio in the HTTP response body.  No job record
  is written; the client simply waits until synthesis completes.

GET /v1/tts/{job_id}/audio
    Stream the finished audio file back to the client.
    Returns 404 if the job does not exist, 409 if it is not yet DONE.

Speaker resolution
------------------
  The `speaker` field is a discriminated union keyed on `type`:
    - type "SpeakerName"      → speaker_name looked up in SpeakerStore
    - type "SpeakerEmbedding" → gpt_cond_latent (T×1024) + speaker_embedding (512,)
                                 reshaped to (1,T,1024) and (1,512,1) before dispatch

Language handling
-----------------
  Defaults to DEFAULT_LANGUAGE when omitted.
  A WARNING is logged when the request language differs from DEFAULT_LANGUAGE,
  but the request is never rejected on that basis.
"""

import asyncio
import base64
import contextlib
import multiprocessing
import os
import time
import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse
import numpy as np
from pydantic import BaseModel, Field, field_validator
from starlette.background import BackgroundTask

from audio import (
    SUPPORTED_FORMATS,
    AudioFormat,
    audio_to_bytes,
    mime_type,
    output_filename,
    save_audio,
)
from config import SUPPORTED_LANGUAGES
from dispatcher import WorkerHandle
from job_store import JobStatus, JobStore
from logging_config import get_logger
from queue_manager import QueueFullError
from speakers import SpeakerNotFoundError
from worker import SynthesisRequest, SynthesisResult

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/tts", tags=["tts"])
audio_router = APIRouter(prefix="/audio", tags=["tts"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SpeakerName(BaseModel):
    type: Literal["SpeakerName"]
    name: str = Field(..., description="Name of a pre-registered speaker in the speaker store.")


class SpeakerEmbedding(BaseModel):
    type: Literal["SpeakerEmbedding"]
    gpt_cond_latent: list[list[float]] = Field(..., description="Pre-computed GPT conditioning latent as a 2-D array (shape T×1024).")
    speaker_embedding: list[float] = Field(..., description="Pre-computed speaker embedding as a flat array (512 floats).")


class TtsRequest(BaseModel):
    text: str = Field(..., description="Text to synthesise.")
    language: str | None = Field(
        default=None,
        description="BCP-47 language code. Defaults to server DEFAULT_LANGUAGE.",
    )
    speaker: Annotated[SpeakerName | SpeakerEmbedding, Field(discriminator="type")]
    format: AudioFormat = Field(
        default="wav",
        description="Output audio format: wav, mp3, ogg, or flac.",
    )

    # GPT autoregressive inference parameters
    temperature: float = Field(default=0.75, description="Softmax temperature of the autoregressive model.")
    length_penalty: float = Field(default=1.0, description="Length penalty for the autoregressive decoder. Higher values produce more terse outputs.")
    repetition_penalty: float = Field(default=10.0, description="Penalty to prevent the decoder from repeating itself. Reduces long silences.")
    top_k: int = Field(default=50, description="Top-k sampling cutoff. Lower values produce more likely (conservative) outputs.")
    top_p: float = Field(default=0.85, description="Nucleus sampling probability. Lower values produce more likely outputs.")
    do_sample: bool = Field(default=True, description="Whether to sample from the autoregressive decoder.")
    num_beams: int = Field(default=1, description="Number of beams for beam search.")
    speed: float = Field(default=1.0, description="Speaking speed multiplier.")
    enable_text_splitting: bool = Field(default=False, description="Split long texts into sentences before synthesis.")

    @field_validator("format")
    @classmethod
    def _check_format(cls, v: str) -> str:
        if v not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '{v}'. Choose from {SUPPORTED_FORMATS}.")
        return v

    @field_validator("language")
    @classmethod
    def _check_language(cls, v: str | None) -> str | None:
        if v is not None and v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{v}'. Supported: {SUPPORTED_LANGUAGES}")
        return v


class TtsJobResponse(BaseModel):
    job_id: str
    status: str  # always "pending" on creation
    poll_url: str  # convenience URL for the client to poll


class TtsResponse(BaseModel):
    audio_b64: str
    elapsed_time: float
    # Echo of request params
    language: str
    temperature: float
    length_penalty: float
    repetition_penalty: float
    top_k: int
    top_p: float
    do_sample: bool
    num_beams: int
    speed: float
    enable_text_splitting: bool


# ---------------------------------------------------------------------------
# POST /v1/tts — submit job
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=TtsJobResponse,
    status_code=202,
    summary="Submit a TTS synthesis job",
    description=(
        "Enqueues a synthesis request and returns a job_id immediately. "
        "Poll GET /v1/jobs/{job_id} for status, then fetch audio from "
        "GET /v1/tts/{job_id}/audio when status is 'done'."
    ),
)
async def submit_tts(body: TtsRequest, request: Request) -> TtsJobResponse:
    state = request.app.state
    settings = state.settings
    job_store: JobStore = state.job_store

    # ---- Text length guard -------------------------------------------
    if len(body.text) > settings.MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Text length {len(body.text)} exceeds MAX_TEXT_LENGTH "
                f"({settings.MAX_TEXT_LENGTH})."
            ),
        )

    # ---- Language resolution -----------------------------------------
    language = body.language or settings.DEFAULT_LANGUAGE
    if language != settings.DEFAULT_LANGUAGE:
        logger.warning(
            "Non-default language requested | lang=%s | default=%s",
            language,
            settings.DEFAULT_LANGUAGE,
        )

    # ---- Speaker resolution ------------------------------------------
    gpt_cond_latent, speaker_embedding, speaker_id = _resolve_speaker(body, state)

    logger.info(
        "TTS request | lang=%s | speaker=%s | text_len=%d | format=%s",
        language,
        speaker_id,
        len(body.text),
        body.format,
    )

    # ---- Create job --------------------------------------------------
    job = await job_store.create(
        text=body.text,
        language=language,
        speaker_id=speaker_id,
    )

    # ---- Build worker request ----------------------------------------
    # Use the dispatcher's queue factory so all queues share the same
    # spawn context as the worker processes (avoids incompatible pipe handles).
    result_queue: multiprocessing.Queue = state.dispatcher.make_queue()
    synth_request = SynthesisRequest(
        job_id=job.job_id,
        text=body.text,
        language=language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        result_queue=result_queue,
        temperature=body.temperature,
        length_penalty=body.length_penalty,
        repetition_penalty=body.repetition_penalty,
        top_k=body.top_k,
        top_p=body.top_p,
        do_sample=body.do_sample,
        num_beams=body.num_beams,
        speed=body.speed,
        enable_text_splitting=body.enable_text_splitting,
    )

    # ---- Register on_complete callback -------------------------------
    # This closure is called by QueueManager._collect_result once the
    # worker puts a SynthesisResult on result_queue.
    fmt = body.format
    outputs_dir = settings.OUTPUTS_DIR

    async def on_complete(worker: WorkerHandle, result: SynthesisResult, elapsed_ms: float) -> None:
        await job_store.mark_running(job.job_id, worker.worker_id, worker.gpu_index, elapsed_ms)
        if result.error:
            await job_store.mark_failed(job.job_id, result.error)
            return

        # save_audio is synchronous blocking I/O — run in thread pool so we
        # don't stall the event loop while encoding/writing the audio file.
        filename = output_filename(job.job_id, fmt)
        audio_path = os.path.join(outputs_dir, filename)
        loop = asyncio.get_running_loop()
        size_bytes = await loop.run_in_executor(
            None, save_audio, result.audio, audio_path, fmt, result.sample_rate
        )

        await job_store.mark_done(job.job_id, audio_path)
        logger.info(
            "Response ready | job_id=%s | format=%s | size=%d bytes",
            job.job_id,
            fmt,
            size_bytes,
        )

    # ---- Enqueue -------------------------------------------------
    try:
        await state.queue_manager.submit_job(synth_request, on_complete)
    except QueueFullError as exc:
        # Job was created but never dispatched — mark it failed and surface 503.
        await job_store.mark_failed(job.job_id, str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return TtsJobResponse(
        job_id=job.job_id,
        status="pending",
        poll_url=f"/v1/jobs/{job.job_id}",
    )


# ---------------------------------------------------------------------------
# POST /audio/speech-file — synthesise and return audio immediately
# ---------------------------------------------------------------------------


@audio_router.post(
    "/speech-file",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}, "description": "WAV audio file"}},
    summary="Synthesise audio and return raw WAV bytes",
    description=(
        "Runs synthesis through the same worker pool as POST /v1/tts and returns "
        "raw audio bytes (audio/wav). No job record is created."
    ),
)
async def synthesise_speech_file(body: TtsRequest, request: Request) -> Response:
    audio_data, _, _, _ = await _run_synthesis(body, request)
    return Response(content=audio_data, media_type="audio/wav")


@audio_router.post(
    "/speech",
    response_model=TtsResponse,
    summary="Synthesise audio and return JSON with base64-encoded audio",
    description=(
        "Runs synthesis through the same worker pool as POST /v1/tts and returns "
        "a JSON response with base64-encoded audio and echoed parameters. "
        "No job record is created."
    ),
)
async def synthesise_speech(body: TtsRequest, request: Request) -> TtsResponse:
    audio_data, synthesis_ms, language, fmt = await _run_synthesis(body, request)
    return TtsResponse(
        audio_b64=base64.b64encode(audio_data).decode("utf-8"),
        elapsed_time=synthesis_ms / 1000,
        language=language,
        temperature=body.temperature,
        length_penalty=body.length_penalty,
        repetition_penalty=body.repetition_penalty,
        top_k=body.top_k,
        top_p=body.top_p,
        do_sample=body.do_sample,
        num_beams=body.num_beams,
        speed=body.speed,
        enable_text_splitting=body.enable_text_splitting,
    )


async def _run_synthesis(
    body: TtsRequest, request: Request
) -> tuple[bytes, float, str, str]:
    """Shared synthesis path for /audio/* endpoints. Returns (audio_bytes, synthesis_ms, language, fmt)."""
    state = request.app.state
    settings = state.settings

    if len(body.text) > settings.MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Text length {len(body.text)} exceeds MAX_TEXT_LENGTH ({settings.MAX_TEXT_LENGTH}).",
        )

    language = body.language or settings.DEFAULT_LANGUAGE
    if language != settings.DEFAULT_LANGUAGE:
        logger.warning(
            "Non-default language requested | lang=%s | default=%s",
            language,
            settings.DEFAULT_LANGUAGE,
        )

    gpt_cond_latent, speaker_embedding, speaker_id = _resolve_speaker(body, state)

    job_id = str(uuid.uuid4())
    logger.info(
        "TTS sync request | job_id=%s | lang=%s | speaker=%s | text_len=%d | format=%s",
        job_id, language, speaker_id, len(body.text), body.format,
    )

    loop = asyncio.get_running_loop()
    result_future: asyncio.Future[SynthesisResult] = loop.create_future()

    result_queue: multiprocessing.Queue = state.dispatcher.make_queue()
    synth_request = SynthesisRequest(
        job_id=job_id,
        text=body.text,
        language=language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        result_queue=result_queue,
        temperature=body.temperature,
        length_penalty=body.length_penalty,
        repetition_penalty=body.repetition_penalty,
        top_k=body.top_k,
        top_p=body.top_p,
        do_sample=body.do_sample,
        num_beams=body.num_beams,
        speed=body.speed,
        enable_text_splitting=body.enable_text_splitting,
    )

    async def on_complete(worker: WorkerHandle, result: SynthesisResult, elapsed_ms: float) -> None:
        if not result_future.done():
            result_future.set_result(result)

    try:
        t_submit = time.monotonic()
        await state.queue_manager.submit_job(synth_request, on_complete)
    except QueueFullError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    result: SynthesisResult = await result_future
    synthesis_ms = (time.monotonic() - t_submit) * 1000

    if result.error:
        logger.error("TTS sync failed | job_id=%s | error=%s", job_id, result.error[:300])
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {result.error[:300]}")

    fmt = body.format
    audio_data: bytes = await loop.run_in_executor(
        None, audio_to_bytes, result.audio, fmt, result.sample_rate
    )

    audio_s = len(result.audio) / result.sample_rate if result.audio is not None else 0.0
    rtf = audio_s / (synthesis_ms / 1000) if synthesis_ms > 0 else 0.0
    logger.info(
        "TTS sync done | job_id=%s | format=%s | size=%d bytes | audio_s=%.2f | synthesis_ms=%.1f | RTF=%.2f",
        job_id, fmt, len(audio_data), audio_s, synthesis_ms, rtf,
    )

    return audio_data, synthesis_ms, language, fmt


# ---------------------------------------------------------------------------
# GET /v1/tts/{job_id}/audio — download finished audio
# ---------------------------------------------------------------------------


@router.get(
    "/{job_id}/audio",
    summary="Download synthesised audio",
    description=(
        "Returns the audio file for a completed job. "
        "Returns 404 if the job does not exist, 409 if synthesis is not yet done."
    ),
)
async def get_audio(job_id: str, request: Request):
    job = await request.app.state.job_store.get(job_id)

    if job is None:
        logger.warning("Audio download — job not found: %s", job_id)
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job.status == JobStatus.FAILED:
        raise HTTPException(
            status_code=500,
            detail=f"Job '{job_id}' failed: {job.error}",
        )

    if job.status != JobStatus.DONE:
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' is not done yet (status: {job.status.value}).",
        )

    if not job.audio_path:
        raise HTTPException(
            status_code=410,
            detail=f"Audio for job '{job_id}' has already been downloaded and deleted.",
        )

    if not os.path.isfile(job.audio_path):
        logger.error("Audio file missing on disk for job %s: %s", job_id, job.audio_path)
        raise HTTPException(status_code=500, detail="Audio file not found on disk.")

    # Derive the format from the file extension so we set the correct MIME type.
    ext = os.path.splitext(job.audio_path)[1].lstrip(".")
    media_type = mime_type(ext) if ext in SUPPORTED_FORMATS else "application/octet-stream"

    audio_path = job.audio_path
    state = request.app.state

    async def _delete_after_send() -> None:
        with contextlib.suppress(OSError):
            os.remove(audio_path)
        await state.job_store.clear_audio_path(job_id)
        logger.info("Audio deleted after download | job_id=%s", job_id)

    logger.info("Serving audio | job_id=%s | path=%s", job_id, audio_path)
    return FileResponse(
        path=audio_path,
        media_type=media_type,
        filename=os.path.basename(audio_path),
        background=BackgroundTask(_delete_after_send),
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _resolve_speaker(
    body,
    state,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (gpt_cond_latent, speaker_embedding, speaker_id_label)."""
    if isinstance(body.speaker, SpeakerEmbedding):
        # Input: gpt (T, 1024) → model expects (1, T, 1024)
        #        emb (512,)    → model expects (1, 512, 1)
        gpt = np.array(body.speaker.gpt_cond_latent, dtype=np.float32)[np.newaxis]
        emb = np.array(body.speaker.speaker_embedding, dtype=np.float32)[np.newaxis, :, np.newaxis]
        return gpt, emb, "inline"

    # SpeakerName — look up in the store.
    try:
        record = state.speaker_store.get(body.speaker.name)
    except SpeakerNotFoundError as exc:
        logger.warning("Speaker not found: %s", body.speaker.name)
        raise HTTPException(
            status_code=404,
            detail=f"Speaker '{body.speaker.name}' not found.",
        ) from exc
    return record.gpt_cond_latent, record.speaker_embedding, body.speaker.name
