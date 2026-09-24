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

# System packages needed at runtime:
#   ffmpeg  — required by pydub for MP3 encoding/decoding
#   libsndfile1 — required by soundfile (WAV/FLAC/OGG)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /build

# Copy only the dependency manifests first so Docker caches this layer
# independently of application code changes.
COPY requirements.in .

# Install pip-tools, compile the lockfile, then install from it.
# pip-compile regenerates requirements.txt inside the container so the
# pins are always resolved fresh against the current PyPI index.
RUN pip install --no-cache-dir pip-tools \
 && pip-compile requirements.in -o requirements.txt --no-header \
 && pip install --no-cache-dir -r requirements.txt


# ---- Stage 2: runtime image -------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

# Repeat system packages — we don't copy /usr from the deps stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3      1

# Copy installed Python packages from the deps stage.
COPY --from=deps /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=deps /usr/local/bin            /usr/local/bin

WORKDIR /app

# Copy application source.
COPY xtts_server/ .

# Create directories that the server writes to at runtime.
# These should normally be mounted as volumes so data persists across
# container restarts, but having them here avoids startup errors.
RUN mkdir -p speakers outputs logs

# Non-root user for security — XTTS-v2 doesn't require root.
RUN useradd -m -u 1000 xtts \
 && chown -R xtts:xtts /app
USER xtts

# Expose the default API port.
EXPOSE 8000

# Health check — hits the lightweight /health endpoint.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# MODEL_PATH has no default — the container will exit at startup with a
# clear error message if this is not set.
ENV MODEL_PATH=""

CMD ["python", "main.py"]
