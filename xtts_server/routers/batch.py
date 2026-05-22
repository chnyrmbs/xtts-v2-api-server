"""
routers/batch.py — submit multiple TTS jobs in a single HTTP request.

Design
------
A batch request is simply a list of TTS items.  Each item is validated and
enqueued independently using the same logic as POST /v1/tts.  The response
returns one job_id per item so the client can poll them individually.

This is a thin convenience wrapper — there is no special batch processing
path.  Items enter the same QueueManager as single requests and are served
in arrival order by the same worker pool.

Limits
------
  MAX_BATCH_SIZE = 50 items per request.
  Each item is still subject to MAX_TEXT_LENGTH from settings.
  If any item fails validation the entire batch is rejected (no partial
  acceptance) — this makes failure handling straightforward for the client.

If the queue fills up partway through a batch, already-created jobs are
marked FAILED and the client receives a 503.  Partial success within a
batch is explicitly avoided.
"""

import asyncio
import multiprocessing
import os
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from audio import SUPPORTED_FORMATS, AudioFormat, output_filename, save_audio
from config import SUPPORTED_LANGUAGES
from dispatcher import WorkerHandle
from job_store import JobStore
from logging_config import get_logger
from queue_manager import QueueFullError
from routers.tts import SpeakerEmbedding, SpeakerName, _resolve_speaker
from worker import SynthesisRequest, SynthesisResult

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/batch", tags=["batch"])

MAX_BATCH_SIZE = 50


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class BatchItem(BaseModel):
    text: str = Field(..., description="Text to synthesise for this item.")
    language: str | None = Field(default=None)
    speaker: Annotated[SpeakerName | SpeakerEmbedding, Field(discriminator="type")]
    format: AudioFormat = Field(default="wav")
    temperature: float = Field(default=0.75)
    length_penalty: float = Field(default=1.0)
    repetition_penalty: float = Field(default=10.0)
    top_k: int = Field(default=50)
    top_p: float = Field(default=0.85)
    do_sample: bool = Field(default=True)
    num_beams: int = Field(default=1)
    speed: float = Field(default=1.0)
    enable_text_splitting: bool = Field(default=False)


class BatchRequest(BaseModel):
    items: Annotated[
        list[BatchItem],
        Field(min_length=1, max_length=MAX_BATCH_SIZE, description="TTS items to process."),
    ]


class BatchJobEntry(BaseModel):
    index: int  # 0-based position in the submitted batch
    job_id: str
    status: str  # always "pending" on creation
    poll_url: str


class BatchResponse(BaseModel):
    total: int
    jobs: list[BatchJobEntry]


# ---------------------------------------------------------------------------
# POST /v1/batch
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=BatchResponse,
    status_code=202,
    summary="Submit a batch of TTS jobs",
    description=(
        f"Enqueue up to {MAX_BATCH_SIZE} TTS items in one call. "
        "Each item gets its own job_id. Poll /v1/jobs/{{job_id}} per item."
    ),
)
async def submit_batch(body: BatchRequest, request: Request) -> BatchResponse:
    state = request.app.state
    settings = state.settings
    job_store: JobStore = state.job_store

    # ---- Validate all items before creating any jobs -----------------
    # Fail fast before touching the queue so the client gets a clean error.
    resolved: list[tuple] = []  # (item, language, gpt_latent, speaker_emb, speaker_id)

    for idx, item in enumerate(body.items):
        if len(item.text) > settings.MAX_TEXT_LENGTH:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Item {idx}: text length {len(item.text)} exceeds "
                    f"MAX_TEXT_LENGTH ({settings.MAX_TEXT_LENGTH})."
                ),
            )

        language = item.language or settings.DEFAULT_LANGUAGE
        if language not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=422,
                detail=f"Item {idx}: unsupported language '{language}'.",
            )
        if language != settings.DEFAULT_LANGUAGE:
            logger.warning("Batch item %d — non-default language: %s", idx, language)

        if item.format not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=422,
                detail=f"Item {idx}: unsupported format '{item.format}'.",
            )

        try:
            gpt_lat, spk_emb, spk_id = _resolve_speaker(item, state)
        except HTTPException as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail=f"Item {idx}: {exc.detail}",
            ) from exc

        resolved.append((item, language, gpt_lat, spk_emb, spk_id))

    logger.info(
        "Batch request | items=%d | speakers=%s | formats=%s",
        len(resolved),
        sorted({r[4] for r in resolved}),
        sorted({r[0].format for r in resolved}),
    )

    # ---- Create jobs and enqueue -------------------------------------
    job_entries: list[BatchJobEntry] = []
    created_job_ids: list[str] = []

    for idx, (item, language, gpt_lat, spk_emb, spk_id) in enumerate(resolved):
        job = await job_store.create(
            text=item.text,
            language=language,
            speaker_id=spk_id,
        )
        created_job_ids.append(job.job_id)

        result_queue: multiprocessing.Queue = state.dispatcher.make_queue()
        synth_request = SynthesisRequest(
            job_id=job.job_id,
            text=item.text,
            language=language,
            gpt_cond_latent=gpt_lat,
            speaker_embedding=spk_emb,
            result_queue=result_queue,
            temperature=item.temperature,
            length_penalty=item.length_penalty,
            repetition_penalty=item.repetition_penalty,
            top_k=item.top_k,
            top_p=item.top_p,
            do_sample=item.do_sample,
            num_beams=item.num_beams,
            speed=item.speed,
            enable_text_splitting=item.enable_text_splitting,
        )

        fmt = item.format
        outputs_dir = settings.OUTPUTS_DIR

        # Capture ALL loop variables as default args so each closure is
        # independent — a common Python gotcha with closures inside loops.
        async def on_complete(
            worker: WorkerHandle,
            result: SynthesisResult,
            elapsed_ms: float,
            _job_id: str = job.job_id,
            _fmt: str = fmt,
            _outputs_dir: str = outputs_dir,
        ) -> None:
            await job_store.mark_running(_job_id, worker.worker_id, worker.gpu_index, elapsed_ms)
            if result.error:
                await job_store.mark_failed(_job_id, result.error)
                return
            # Run blocking file I/O in thread pool so event loop is not stalled.
            audio_path = os.path.join(_outputs_dir, output_filename(_job_id, _fmt))
            loop = asyncio.get_running_loop()
            size_bytes = await loop.run_in_executor(
                None, save_audio, result.audio, audio_path, _fmt, result.sample_rate
            )
            await job_store.mark_done(_job_id, audio_path)
            logger.info(
                "Batch item done | job_id=%s | format=%s | size=%d bytes",
                _job_id,
                _fmt,
                size_bytes,
            )

        try:
            await state.queue_manager.submit_job(synth_request, on_complete)
        except QueueFullError as exc:
            # Mark all jobs created so far (including this one) as failed
            # so the store doesn't accumulate ghost PENDING entries.
            for jid in created_job_ids:
                await job_store.mark_failed(jid, str(exc))
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        job_entries.append(
            BatchJobEntry(
                index=idx,
                job_id=job.job_id,
                status="pending",
                poll_url=f"/v1/jobs/{job.job_id}",
            )
        )

    job_ids_preview = [e.job_id for e in job_entries[:3]]
    logger.info(
        "Batch accepted | jobs=%d | job_ids=%s%s",
        len(job_entries),
        job_ids_preview,
        " …" if len(job_entries) > 3 else "",
    )
    return BatchResponse(total=len(job_entries), jobs=job_entries)
