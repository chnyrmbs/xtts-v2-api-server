"""
Speaker store — persists reference audio and pre-computed XTTS-v2 conditioning
latents to disk, with an in-memory cache for fast lookup.

On-disk layout per speaker:
  SPEAKERS_DIR/
    {name}/
      ref.wav        — original reference audio (copied on registration)
      latents.npz    — numpy arrays: gpt_cond_latent, speaker_embedding
      meta.json      — name, created_at (ISO-8601)

Latent computation is intentionally NOT done here — it requires a live model
and must be dispatched to a worker process via ComputeLatentsRequest.
The dispatcher calls SpeakerStore.register() after receiving the LatentsResult.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
import shutil
import threading

import numpy as np

from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SpeakerRecord:
    name: str
    wav_path: str
    gpt_cond_latent: np.ndarray  # float32, shape (1, T, 1024)
    speaker_embedding: np.ndarray  # float32, shape (1, 512, 1) — matches conv1d input (N, C, L)
    created_at: str = ""


class SpeakerNotFoundError(KeyError):
    pass


class SpeakerStore:
    def __init__(self, speakers_dir: str) -> None:
        self._dir = speakers_dir
        self._cache: dict[str, SpeakerRecord] = {}
        # Threading lock so that concurrent delete + get cannot race between
        # the shutil.rmtree call and the cache eviction (or a concurrent
        # _load_from_disk trying to read a half-deleted directory).
        self._cache_lock = threading.Lock()
        os.makedirs(speakers_dir, exist_ok=True)
        logger.info("SpeakerStore ready — dir=%s", speakers_dir)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        wav_source_path: str,
        gpt_cond_latent: np.ndarray,
        speaker_embedding: np.ndarray,
    ) -> SpeakerRecord:
        """Save speaker to disk and cache in RAM. Overwrites if name exists."""
        speaker_dir = os.path.join(self._dir, name)
        os.makedirs(speaker_dir, exist_ok=True)

        # Reference audio
        wav_dest = os.path.join(speaker_dir, "ref.wav")
        shutil.copy2(wav_source_path, wav_dest)

        # Latents
        latents_path = os.path.join(speaker_dir, "latents.npz")
        np.savez(
            latents_path,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
        )

        # Metadata
        created_at = datetime.now(UTC).isoformat()
        meta = {"name": name, "created_at": created_at}
        with open(os.path.join(speaker_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        record = SpeakerRecord(
            name=name,
            wav_path=wav_dest,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            created_at=created_at,
        )
        with self._cache_lock:
            self._cache[name] = record

        logger.info(
            "Speaker registered | name=%s | dir=%s | "
            "gpt_cond_latent.shape=%s | speaker_embedding.shape=%s",
            name,
            speaker_dir,
            gpt_cond_latent.shape,
            speaker_embedding.shape,
        )
        return record

    def delete(self, name: str) -> None:
        """Remove speaker from disk and cache. Raises SpeakerNotFoundError if absent."""
        speaker_dir = os.path.join(self._dir, name)
        with self._cache_lock:
            if not os.path.isdir(speaker_dir):
                logger.warning("Speaker delete — not found: %s", name)
                raise SpeakerNotFoundError(name)
            # Evict from cache before rmtree so no concurrent get() can return
            # a record whose files are already deleted.
            self._cache.pop(name, None)
            shutil.rmtree(speaker_dir)
        logger.info("Speaker deleted: %s", name)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get(self, name: str) -> SpeakerRecord:
        """Return SpeakerRecord, loading from disk if not cached. Raises SpeakerNotFoundError."""
        with self._cache_lock:
            if name in self._cache:
                return self._cache[name]

            record = self._load_from_disk(name)
            if record is None:
                logger.warning("Speaker not found: %s", name)
                raise SpeakerNotFoundError(name)

            self._cache[name] = record
        logger.info("Speaker cache miss — loaded from disk: %s", name)
        return record

    def list_speakers(self) -> list[dict]:
        """Return metadata for every speaker that exists on disk."""
        results = []
        if not os.path.isdir(self._dir):
            return results

        for entry in sorted(os.scandir(self._dir), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            meta_path = os.path.join(entry.path, "meta.json")
            if not os.path.isfile(meta_path):
                continue
            with open(meta_path, encoding="utf-8") as f:
                results.append(json.load(f))

        return results

    def exists(self, name: str) -> bool:
        if name in self._cache:
            return True
        speaker_dir = os.path.join(self._dir, name)
        return os.path.isfile(os.path.join(speaker_dir, "latents.npz"))

    # ------------------------------------------------------------------
    # Startup preload
    # ------------------------------------------------------------------

    def preload_all(self) -> int:
        """Load every speaker on disk into RAM. Called once at server startup."""
        if not os.path.isdir(self._dir):
            return 0

        loaded = 0
        for entry in os.scandir(self._dir):
            if not entry.is_dir():
                continue
            name = entry.name
            if name in self._cache:
                continue
            record = self._load_from_disk(name)
            if record is not None:
                self._cache[name] = record
                loaded += 1

        logger.info("SpeakerStore preload complete — %d speaker(s) loaded", loaded)
        return loaded

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_from_disk(self, name: str) -> SpeakerRecord | None:
        speaker_dir = os.path.join(self._dir, name)
        latents_path = os.path.join(speaker_dir, "latents.npz")
        wav_path = os.path.join(speaker_dir, "ref.wav")
        meta_path = os.path.join(speaker_dir, "meta.json")

        if not os.path.isfile(latents_path):
            return None
        # ref.wav is absent for studio speakers (latents pre-computed from model checkpoint)
        if not os.path.isfile(wav_path):
            wav_path = ""

        data = np.load(latents_path)
        created_at = ""
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                created_at = json.load(f).get("created_at", "")

        return SpeakerRecord(
            name=name,
            wav_path=wav_path,
            gpt_cond_latent=data["gpt_cond_latent"],
            speaker_embedding=data["speaker_embedding"],
            created_at=created_at,
        )
