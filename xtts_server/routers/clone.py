"""
routers/clone.py — voice cloning / speaker registration endpoint.

Flow
----
  POST /v1/clone
    1. Accept a multipart upload: speaker name + reference audio file.
    2. Validate the audio file (readable, non-empty).
    3. Save the raw audio to a temp path in SPEAKERS_DIR.
    4. Dispatch a ComputeLatentsRequest to any free worker via
       dispatcher.compute_latents() — this is the only step that touches
       the model and therefore runs in a worker process.
    5. On success: call SpeakerStore.register() which copies the audio and
       saves latents.npz + meta.json to SPEAKERS_DIR/{name}/.
    6. Return the speaker name and metadata.

The latent computation is synchronous from the client's perspective
(the HTTP response is not sent until latents are ready) but it runs in
a worker process so the event loop is never blocked.

Accepted audio formats
----------------------
  Any format soundfile can decode — WAV, FLAC, OGG, MP3 (via libsndfile).
  The file is saved as-is and passed to model.get_conditioning_latents(),
  which handles its own decoding internally.
"""

import os
import time
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from logging_config import get_logger
from speakers import SpeakerStore

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/clone", tags=["clone"])

# Maximum reference audio size accepted (10 MB).
# XTTS-v2 uses at most ~30 s of reference audio; larger files waste memory.
_MAX_AUDIO_BYTES = 10 * 1024 * 1024


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class CloneResponse(BaseModel):
    speaker_name: str
    created_at: str
    message: str


# ---------------------------------------------------------------------------
# POST /v1/clone
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=CloneResponse,
    status_code=201,
    summary="Register a new speaker voice",
    description=(
        "Upload a reference audio clip and a speaker name. "
        "The server computes XTTS-v2 conditioning latents and saves them for "
        "reuse. After registration, pass 'speaker_name' to POST /v1/tts "
        "instead of uploading raw embeddings every time."
    ),
)
async def clone_speaker(
    request: Request,
    speaker_name: str = Form(
        ...,
        description="Unique name for this speaker. Alphanumeric + hyphens/underscores.",
    ),
    audio: UploadFile = File(
        ...,
        description="Reference audio file (WAV, FLAC, OGG, or MP3). Keep it under 30 s.",
    ),
) -> CloneResponse:
    state = request.app.state
    speaker_store: SpeakerStore = state.speaker_store
    settings = state.settings

    # ---- Name validation ---------------------------------------------
    _validate_speaker_name(speaker_name)

    # ---- Duplicate guard — return 409 if already registered ----------
    # Without this check two concurrent requests for the same name would
    # both compute latents and the second would silently overwrite the first.
    if state.speaker_store.exists(speaker_name):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Speaker '{speaker_name}' already exists. "
                "Delete it first with DELETE /v1/speakers/{name}."
            ),
        )

    # ---- Read and size-check the upload ------------------------------
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="Uploaded audio file is empty.")
    if len(audio_bytes) > _MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Audio file too large ({len(audio_bytes) / 1024:.0f} KB). "
                f"Maximum is {_MAX_AUDIO_BYTES // 1024} KB."
            ),
        )

    logger.info(
        "Clone request | speaker_name=%s | filename=%s | content_type=%s | size=%d bytes",
        speaker_name,
        audio.filename,
        audio.content_type,
        len(audio_bytes),
    )

    # ---- Write to a temp file so we can pass a path to the worker ----
    # We keep it inside SPEAKERS_DIR so the final copy stays on the same
    # filesystem and shutil.copy2 is an O(1) rename rather than a full copy.
    tmp_dir = os.path.join(settings.SPEAKERS_DIR, ".tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Preserve original extension for libsndfile format detection.
    original_ext = _safe_extension(audio.filename)
    tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4()}{original_ext}")

    registered = False
    t_clone_start = time.monotonic()
    try:
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        logger.info(
            "Reference audio saved to temp | path=%s | size=%d bytes",
            tmp_path,
            len(audio_bytes),
        )

        # ---- Dispatch latent computation to a worker ----------------
        job_id = str(uuid.uuid4())
        t_latents_start = time.monotonic()
        latents_result = await state.dispatcher.compute_latents(job_id, tmp_path)
        latents_ms = (time.monotonic() - t_latents_start) * 1000

        if latents_result.error:
            logger.error(
                "Latent computation failed | speaker=%s | error=%s",
                speaker_name,
                latents_result.error[:300],
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to compute speaker latents: {latents_result.error[:300]}",
            )

        logger.info(
            "Latents computed | speaker=%s | latents_ms=%.1f"
            " | gpt_shape=%s | emb_shape=%s",
            speaker_name,
            latents_ms,
            latents_result.gpt_cond_latent.shape,
            latents_result.speaker_embedding.shape,
        )

        # ---- Persist speaker to store --------------------------------
        # Re-check existence after the awaited latent computation — a
        # concurrent clone for the same name could have succeeded while we
        # were computing latents (there are no more await points between
        # this check and register(), so this closes the TOCTOU window).
        if state.speaker_store.exists(speaker_name):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Speaker '{speaker_name}' already exists. "
                    "Delete it first with DELETE /v1/speakers/{name}."
                ),
            )
        # register() calls shutil.copy2(tmp_path → wav_dest).  We only
        # delete tmp_path after this completes successfully so a partial
        # copy cannot leave the speaker in a corrupt state.
        record = speaker_store.register(
            name=speaker_name,
            wav_source_path=tmp_path,
            gpt_cond_latent=latents_result.gpt_cond_latent,
            speaker_embedding=latents_result.speaker_embedding,
        )
        registered = True

    finally:
        # Only clean up the temp file after a successful register().
        # On failure, leave it for inspection / debugging.
        if registered and os.path.isfile(tmp_path):
            os.remove(tmp_path)

    total_clone_ms = (time.monotonic() - t_clone_start) * 1000
    logger.info(
        "Speaker clone complete | name=%s | total_ms=%.1f | latents_ms=%.1f",
        speaker_name,
        total_clone_ms,
        latents_ms,
    )

    return CloneResponse(
        speaker_name=record.name,
        created_at=record.created_at,
        message=f"Speaker '{speaker_name}' registered. Use it with POST /v1/tts.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_speaker_name(name: str) -> None:
    """Allow alphanumeric characters, hyphens, and underscores only."""
    import re

    if not name:
        raise HTTPException(status_code=422, detail="speaker_name must not be empty.")
    if len(name) > 64:
        raise HTTPException(
            status_code=422,
            detail="speaker_name must be 64 characters or fewer.",
        )
    if not re.match(r"^[a-zA-Z0-9_\-]+$", name):
        raise HTTPException(
            status_code=422,
            detail=(
                "speaker_name may only contain letters, digits, hyphens (-), and underscores (_)."
            ),
        )


def _safe_extension(filename: str | None) -> str:
    """Return the file extension (e.g. '.wav') or '.bin' as a fallback."""
    if not filename:
        return ".bin"
    _, ext = os.path.splitext(filename)
    # Whitelist known audio extensions to avoid path traversal via the filename.
    return ext.lower() if ext.lower() in {".wav", ".flac", ".ogg", ".mp3", ".m4a"} else ".bin"
