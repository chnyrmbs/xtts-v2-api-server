# ---------------------------------------------------------------------------
# XTTS-v2 Inference Server — Dockerfile
#
# Build stages
# ------------
#   deps   — installs Python dependencies via pip-tools (reproducible lockfile)
#   runtime — copies application code on top of the deps image
#
# The two-stage approach means dependency installation is cached separately
# from code changes, so iterative code edits don't re-install all packages.
#
# Usage
# -----
#   docker build -t xtts-server .
#   docker run --gpus all \
#     -e MODEL_PATH=/models/xtts-v2 \
#     -v /host/path/to/xtts-v2:/models/xtts-v2:ro \
#     -v /host/path/to/speakers:/app/speakers \
#     -v /host/path/to/outputs:/app/outputs \
#     -p 8000:8000 \
#     xtts-server
#
# Required environment variables
# --------------------------------
#   MODEL_PATH — absolute path inside the container to the XTTS-v2 model dir.
#                Mount your local model directory at this path.
#
# Optional environment variables (all have defaults — see config.py)
# -------------------------------------------------------------------
#   NUM_GPUS, WORKERS_PER_GPU, DEFAULT_LANGUAGE, MAX_QUEUE_SIZE,
#   JOB_TTL_SECONDS, SPEAKERS_DIR, OUTPUTS_DIR, MAX_TEXT_LENGTH,
#   SAMPLE_RATE, LOG_LEVEL, HOST, PORT
# ---------------------------------------------------------------------------

# ---- Stage 1: dependency installation ---------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS deps

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Use the repository's pinned dependency lock file.
COPY requirements.txt .

# Keep all Python 3.11 packages inside an isolated virtual environment.
RUN python3.11 -m venv /opt/venv \
 && /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel \
 && /opt/venv/bin/python -m pip install --no-cache-dir -r requirements.txt


# ---- Stage 2: runtime image -------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the complete Python 3.11 virtual environment from build stage.
COPY --from=deps /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY xtts_server/ .

RUN mkdir -p speakers outputs logs \
 && useradd -m -u 1000 xtts \
 && chown -R xtts:xtts /app

USER xtts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

ENV MODEL_PATH=""

CMD ["python", "main.py"]
