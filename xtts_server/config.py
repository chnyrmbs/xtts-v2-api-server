import os
import sys

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_LANGUAGES = [
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "pl",
    "tr",
    "ru",
    "nl",
    "cs",
    "ar",
    "zh-cn",
    "ja",
    "hu",
    "ko",
    "hi",
]

_REQUIRED_MODEL_FILES = ["config.json", "model.pth", "vocab.json"]
_LOG_SIZE_FILES = ["config.json", "model.pth"]


class Settings(BaseSettings):
    # Model
    MODEL_PATH: str  # required — no default; startup aborts if unset

    # GPU / worker topology
    NUM_GPUS: int | None = None  # None → auto-detect at runtime
    WORKERS_PER_GPU: str = "1"  # plain int or comma-separated list

    # Language
    DEFAULT_LANGUAGE: str = "tr"

    # Queue
    MAX_QUEUE_SIZE: int = 100

    # Job store (in-memory)
    JOB_TTL_SECONDS: int = 300

    # File system
    SPEAKERS_DIR: str = "./speakers"
    OUTPUTS_DIR: str = "./outputs"

    # Audio
    MAX_TEXT_LENGTH: int = 5000
    SAMPLE_RATE: int = 24000

    # Inference precision
    USE_FP16: bool = False  # cast model to float16 on GPU for faster inference

    # Worker result timeout
    WORKER_TIMEOUT_SECONDS: int = 300  # max seconds to wait for a worker result before giving up

    # Logging
    LOG_LEVEL: str = "INFO"

    # ---- Derived fields (populated by validators) ----
    workers_per_gpu_list: list[int] = []  # parsed from WORKERS_PER_GPU

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # WORKERS_PER_GPU parser
    # ------------------------------------------------------------------
    @field_validator("WORKERS_PER_GPU")
    @classmethod
    def _validate_workers_str(cls, v: str) -> str:
        parts = [p.strip() for p in v.split(",")]
        for p in parts:
            if not p.isdigit() or int(p) < 1:
                raise ValueError(f"WORKERS_PER_GPU must be positive integers, got: '{p}'")
        return v

    @model_validator(mode="after")
    def _parse_and_validate_worker_list(self) -> "Settings":
        import torch  # deferred — not needed at import time

        num_gpus = self.NUM_GPUS
        if num_gpus is None:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            self.NUM_GPUS = num_gpus

        parts = [int(p.strip()) for p in self.WORKERS_PER_GPU.split(",")]

        if len(parts) == 1:
            # Replicate single value across all GPUs (or 1 CPU worker when 0 GPUs)
            count = max(num_gpus, 1)
            self.workers_per_gpu_list = parts * count
        else:
            if len(parts) != num_gpus:
                raise ValueError(
                    f"WORKERS_PER_GPU has {len(parts)} entries but NUM_GPUS={num_gpus}. "
                    "They must match."
                )
            self.workers_per_gpu_list = parts

        return self


def _validate_model_path(model_path: str) -> None:
    """Check MODEL_PATH and required files; exit(1) on any problem."""
    if not os.path.isdir(model_path):
        logger.error("MODEL_PATH does not exist or is not a directory: %s", model_path)
        sys.exit(1)

    missing: list[str] = []
    for fname in _REQUIRED_MODEL_FILES:
        fpath = os.path.join(model_path, fname)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            if fname in _LOG_SIZE_FILES:
                logger.info("  [FOUND] %s — %.2f MB", fname, size_mb)
            else:
                logger.info("  [FOUND] %s", fname)
        else:
            logger.error("  [MISSING] %s", fname)
            missing.append(fname)

    if missing:
        logger.error(
            "MODEL_PATH validation failed. Missing required files: %s — aborting.",
            ", ".join(missing),
        )
        sys.exit(1)

    logger.info("MODEL_PATH validation passed: %s", model_path)


def _log_config(settings: Settings) -> None:
    logger.info("=== XTTS-v2 Server Configuration ===")
    for field_name, value in settings.model_dump().items():
        logger.info("  %s = %s", field_name, value)
    logger.info("=====================================")


def load_settings() -> Settings:
    settings = Settings()
    _log_config(settings)
    _validate_model_path(settings.MODEL_PATH)
    return settings
