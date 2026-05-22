# CLAUDE.md — Developer Cheatsheet

This file is a checkpoint for Claude Code (and human developers). It captures architecture decisions, non-obvious invariants, known gotchas, and the reasoning behind key design choices so you can continue development without re-deriving context.

---

## Project Goal

A **vllm-style FastAPI inference server** for [Coqui XTTS-v2](https://github.com/coqui-ai/TTS) — multi-GPU, multi-worker, async-first, production-grade. The design borrows vllm's ideas: a central request queue, least-loaded worker routing, async fire-and-poll jobs, and WebSocket streaming for low-latency output. The model itself (XTTS-v2) is a GPT-based TTS model that conditions on speaker latents.

---

## Project Layout

```
xtts-v2-api-server/
├── install.sh            # Linux-only setup script (CUDA/Python/torch checks, .venv, pip install)
├── start-server.sh       # GPU-aware launch script (pre-flight checks, CUDA_VISIBLE_DEVICES, GPU_MEMORY_FRACTION)
├── requirements.in       # hand-edited direct deps
├── requirements.txt      # generated lockfile (pip-compile)
├── pyproject.toml        # ruff config
├── Dockerfile
├── .env.example
└── xtts_server/          # ALL Python source lives here
    ├── main.py
    ├── config.py
    ├── dispatcher.py
    ├── queue_manager.py
    ├── job_store.py
    ├── worker.py
    ├── speakers.py
    ├── audio.py
    ├── logging_config.py
    ├── routers/
    │   ├── system.py
    │   ├── tts.py
    │   ├── clone.py
    │   ├── batch.py
    │   ├── jobs.py
    │   └── speakers.py
    └── ws/
        └── stream.py
```

Run the server from `xtts_server/`:
```bash
cd xtts_server && python main.py
```

---

## API Surface

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — returns `{"status":"ok","uptime_s":…}` |
| `GET` | `/v1/system/info` | GPU/worker stats, queue depth, job counts |
| `POST` | `/v1/tts` | Submit async TTS job → `202 {job_id, poll_url}` |
| `GET` | `/v1/tts/{job_id}/audio` | Download finished audio (`409` if not done yet) |
| `GET` | `/v1/jobs` | List all jobs (debug) |
| `GET` | `/v1/jobs/{job_id}` | Poll job status + timing metadata |
| `POST` | `/v1/batch` | Submit up to 50 TTS items in one request |
| `POST` | `/audio/speech-file` | Synthesise and return raw WAV bytes immediately (client blocks) |
| `POST` | `/audio/speech` | Synthesise and return JSON `TtsResponse` with base64 audio (client blocks) |
| `POST` | `/v1/clone` | Upload reference audio → register speaker name |
| `GET` | `/v1/speakers` | List all registered speakers |
| `GET` | `/v1/speakers/{name}` | Metadata for one speaker |
| `DELETE` | `/v1/speakers/{name}` | Delete speaker (evicts cache + removes disk files) |
| `WS` | `/v1/stream` | WebSocket streaming — raw float32 PCM chunks |

---

## Core Data Flow

```
POST /v1/tts
  → TtsRequest validated (Pydantic)
  → _resolve_speaker() → (gpt_cond_latent: np.ndarray, speaker_embedding: np.ndarray, speaker_id: str)
    (discriminates on body.speaker.type: "SpeakerName" → store lookup, "SpeakerEmbedding" → reshape + use inline)
  → job_store.create() → Job (PENDING)
  → dispatcher.make_queue() → Manager proxy Queue  ← NOT a bare mp.Queue
  → SynthesisRequest(job_id, text, lang, latents, result_queue)
  → queue_manager.submit_job(request, on_complete)
      → asyncio.Queue.put_nowait (raises QueueFullError if full)
  → return 202 {job_id, poll_url}

[drain loop]
  → asyncio.Queue.get()
  → dispatcher.wait_for_free_worker()   # asyncio.Condition wait
  → dispatcher.dispatch(request)         # put into worker's mp.Queue in executor
  → asyncio.create_task(_collect_result)

[worker process]
  → receives SynthesisRequest from its request_queue (spawn-ctx mp.Queue)
  → model.inference(text, lang, gpt_latent_tensor, spk_emb_tensor)
  → puts SynthesisResult(job_id, audio_np, sample_rate) on result_queue (Manager proxy)

[_collect_result task]
  → run_in_executor(result_queue.get)   # blocking get, off event loop
  → dispatcher.release(worker_id)       # notifies Condition → drain loop unblocks
  → on_complete(worker, result)
      → job_store.mark_running → mark_done
      → run_in_executor(save_audio, ...) # blocking I/O off event loop
```

---

## Critical Design Decisions

### 1. `inference()` not `synthesize()`
`model.synthesize()` re-encodes the reference audio WAV on every call — adds 100-500 ms overhead.
This server calls `model.inference(text, lang, gpt_cond_latent, speaker_embedding)` directly with **pre-computed** latents. Latents are computed once at clone time via `model.get_conditioning_latents()` and stored as `.npz` on disk.

For streaming: `model.inference_stream()` — yields chunks so the first audio arrives ~200-400 ms into synthesis.

### 2. Two-tier queue system: spawn-ctx vs Manager proxy
There are **two distinct queue types** in the codebase, and they serve different purposes:

- **Worker request queue** (`_MP_CTX.Queue()`, created at spawn time, passed as a constructor arg to the worker `Process`): used to send `SynthesisRequest` / `StreamSynthesisRequest` / `ComputeLatentsRequest` to the worker. This is fine because it's established before the process starts.

- **Per-job result queue** (`self._manager.Queue()`, Manager proxy, created at request time via `dispatcher.make_queue()`): used to receive `SynthesisResult` / `SynthesisChunk` etc. back from the worker. This **must** be a Manager proxy queue — Python 3.11 strictly enforces that spawn-context `Queue` objects cannot be pickled after process creation (trying to put a Queue into another Queue's message raises `assert_spawning`). Manager proxy queues are serialized as thin connection objects and are exempt from this restriction.

**Rule:** always use `dispatcher.make_queue()` for result queues. Never use `multiprocessing.Queue()` or `_MP_CTX.Queue()` directly for result queues.

### 3. `multiprocessing.get_context("spawn")`
CUDA does not survive `fork()`. All worker `Process` objects are created via `_MP_CTX = multiprocessing.get_context("spawn")` in `dispatcher.py`. The Manager itself is also started before workers via `multiprocessing.Manager()` (which uses its own server process).

### 4. Lazy `asyncio.Condition` in Dispatcher
`asyncio.Condition` (and `asyncio.Lock`) must be created **inside a running event loop** — creating them in `__init__` before the loop exists causes "attached to a different loop" errors. `dispatcher.py` stores `self._cond: asyncio.Condition | None = None` and creates it lazily on first use via `_get_cond()`.

### 5. No Redis
Job state is a plain `dict[str, Job]` guarded by `asyncio.Lock`. TTL cleanup runs every 60 s in a background task. This is sufficient for single-process deployments. If you ever need multi-process uvicorn workers or cross-node sharing, add Redis then.

### 6. Blocking I/O always in `run_in_executor`
The event loop must never block. Five places where this matters:
- `result_queue.get()` — in `queue_manager._collect_result` and `ws/stream.py`
- `save_audio()` (soundfile write) — in the `on_complete` callback in `tts.py` and `batch.py`
- `audio_to_bytes()` (soundfile encode) — in `_run_synthesis` in `tts.py` (sync endpoints must not block either)
- `process.join()` — in `dispatcher.shutdown`
- `worker.request_queue.put(request)` — in `dispatcher.dispatch()`

### 7. Workers are released before `on_complete` is awaited
In `_collect_result`: `dispatcher.release()` is called **before** `on_complete()`. This is intentional — the worker slot is freed as soon as synthesis is done so the drain loop can dispatch the next request. Audio encoding (`save_audio`) happens after release and does not hold up the next job.

### 8. `asyncio.Future[WorkerHandle]` for WebSocket streams
`submit_stream()` returns a `Future` that is resolved by the drain loop once the request is dispatched. The WebSocket handler `await`s this future to learn which `WorkerHandle` was assigned, so it can call `dispatcher.release(worker.worker_id)` in the `finally` block. Without this, the WS handler would not know which worker to release.

### 9. Worker slot release on WebSocket disconnect
`ws/stream.py` wraps the chunk-reading loop in `try/finally`:
```python
try:
    while True:
        item = await loop.run_in_executor(None, result_queue.get)
        ...
finally:
    await state.dispatcher.release(worker.worker_id, elapsed_ms=0.0)
```
This guarantees the slot is freed even if the client disconnects mid-stream or an unexpected error occurs.

### 10. Least-loaded dispatch
`dispatcher._pick_worker()` returns `min(workers, key=lambda w: w.active_requests)`. The `active_requests` counter is incremented in `dispatch()` and decremented in `release()`, both under the `asyncio.Condition` lock.

### 11. Per-worker active job tracking
`WorkerHandle.active_jobs: dict[str, float]` maps job_id → monotonic start time. Populated in `dispatch()`, evicted in `release(job_id=...)`. All three `release()` callers (queue_manager, ws/stream, compute_latents) pass `job_id`. This drives the periodic status log (shows each job and its elapsed time per worker) and the `/v1/system/info` response (`worker_stats()` includes `active_jobs`).

`release()` also increments `WorkerHandle.total_requests` (fixing the long-standing bug where it always showed 0).

### 16. Speaker specification — discriminated union
`TtsRequest`, `BatchItem`, and `StreamRequest` all carry a `speaker` field typed as a Pydantic v2 discriminated union:

```python
speaker: Annotated[SpeakerName | SpeakerEmbedding, Field(discriminator="type")]
```

- **`SpeakerName`** (`type: "SpeakerName"`): `name` — looked up in `SpeakerStore`. Returns latents already in correct numpy shape from disk.
- **`SpeakerEmbedding`** (`type: "SpeakerEmbedding"`): `gpt_cond_latent` (2-D, shape T×1024) + `speaker_embedding` (1-D, 512 floats). `_resolve_speaker` reshapes them to `(1, T, 1024)` and `(1, 512, 1)` before passing to the worker. Sending the full 3-D shapes in JSON is verbose and error-prone; reshaping at the boundary keeps the API ergonomic.

`SpeakerName` and `SpeakerEmbedding` are defined in `routers/tts.py` and imported by `batch.py` and `ws/stream.py`. `_resolve_speaker` is the single resolution point — all three callers go through it.

### 17. FP16 inference — `torch.autocast` not `model.half()`
`USE_FP16=true` enables mixed-precision inference on GPU. The implementation uses `torch.autocast` rather than `model.half()`:

```python
with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
    outputs = model.inference(...)
```

`model.half()` permanently casts all weights to float16, which breaks XTTS-v2's VITS decoder and some normalisation layers on GPU. `torch.autocast` lets PyTorch decide per-operation whether float16 is numerically safe, falling back to float32 where needed. On CPU `fp16` is always `False` (`fp16 = use_fp16 and device.startswith("cuda")`), so autocast is a no-op.

`torch` must be imported at the top of `_handle_synthesis` and `_handle_stream` (before the `with torch.autocast` call) — these functions use lazy imports and `torch` is not available at module level in the worker process.

### 13. GPU memory fraction (`GPU_MEMORY_FRACTION`)
Workers read `os.environ.get("GPU_MEMORY_FRACTION", "1.0")` and call `torch.cuda.set_per_process_memory_fraction(fraction, gpu_index)` **before** the model is loaded. This caps how much VRAM each worker process can allocate. `start-server.sh` validates and exports this value; `worker.py` applies it in `_apply_memory_fraction()` called at device setup time.

A value of `1.0` (default) disables the cap. Only values in `(0.0, 1.0)` apply the limit — `1.0` is passed through silently (PyTorch treats it as no-op).

### 14. Startup banner
`_log_banner(settings)` in `main.py` renders a QUAKER-XTTS ASCII art banner immediately after `load_settings()`. It is intentionally printed as a single `logger.info()` call so the entire block appears atomically in logs. The banner includes model path, worker count, language, queue size, and endpoint URLs.

### 15. HTTP access log middleware
Registered in `create_app()` via `@app.middleware("http")`. Logs one line per request at INFO (DEBUG for `/health`). Generates or forwards `X-Request-ID` (8-char hex) and echoes it as a response header — clients can pass `X-Request-ID` to correlate logs with their own traces. WebSocket upgrades appear as `101`.

### 12. Audio file deleted after first download
`GET /v1/tts/{job_id}/audio` uses a `starlette.background.BackgroundTask` to delete the audio file from disk and clear `job.audio_path` immediately after the response is sent. Subsequent downloads return `410 Gone`. The polling response (`GET /v1/jobs/{id}`) hides `audio_url` once `audio_path` is cleared. The TTL cleanup loop is unaffected (it skips files that are already gone).

---

## WebSocket Protocol (`/v1/stream`)

1. Client connects, sends one JSON text frame:
   ```json
   {"text": "...", "language": "tr", "speaker": {"type": "SpeakerName", "name": "alice"}}
   ```
   Or with raw conditioning arrays:
   ```json
   {"text": "...", "speaker": {"type": "SpeakerEmbedding", "gpt_cond_latent": [[...]], "speaker_embedding": [...]}}
   ```
2. Server replies with **binary frames** — raw `float32` little-endian PCM, 24 000 Hz mono. Each frame is one chunk from `model.inference_stream()`.
3. Server sends a final **text frame**: `{"status": "done"}` or `{"status": "error", "detail": "..."}`.
4. Connection closes. Client reassembles chunks and knows the sample rate is 24 000 Hz.

Close code `1013` (Try Again Later) is used when the queue is full.

---

## Speaker Store

On-disk layout for each speaker (`SPEAKERS_DIR/{name}/`):
```
ref.wav        # copy of the original reference audio (absent for studio speakers)
latents.npz    # gpt_cond_latent (shape 1×T×1024) + speaker_embedding (shape 1×512×1)
meta.json      # {"name": "...", "created_at": "ISO-8601", ["source": "studio", "original_name": "..."]}
```

`SpeakerStore` keeps a `dict[str, SpeakerRecord]` RAM cache. `preload_all()` is called at startup. `get()` falls back to disk on cache miss. `delete()` evicts cache **before** `shutil.rmtree` (under `threading.Lock`) — this order is important; reversing it creates a window where the cache holds a record pointing to deleted files.

Worker processes receive latents as **CPU numpy arrays** (picklable). Each worker converts them to device tensors inside `_handle_synthesis()`. Never pass GPU tensors across process boundaries.

`speaker_embedding` shape is `(1, 512, 1)` — this matches the model's conv1d layer `(N, C_in=512, L=1)`. Storing it as `(1, 512)` causes a channel-mismatch error at inference. The seed script enforces this with `arr.reshape(-1)[None, :, None]`.

**Clone limits:**
- Max upload size: 10 MB (`_MAX_AUDIO_BYTES` in `clone.py`)
- Speaker name: alphanumeric + hyphens/underscores, max 64 chars, must be unique (409 on duplicate)
- Temp files written to `SPEAKERS_DIR/.tmp/` during latent computation, deleted on success

**Studio speakers (pre-seeded):**
58 built-in XTTS-v2 voices from `speakers_xtts.pth` are registered at setup time by running:
```bash
python seed_studio_speakers.py --model-path ./model --speakers-dir ./xtts_server/speakers
```
The script reads `speakers_xtts.pth` directly (no full model load — runs in seconds).
Display names are slugified: spaces → underscores, accents stripped ("Alma María" → "Alma_Maria").
Studio speakers have no `ref.wav`; `_load_from_disk()` treats its absence as allowed and sets `wav_path=""`.
Their `meta.json` includes `"source": "studio"` and `"original_name"` for traceability.
Use `--force` to overwrite existing registrations, `--dry-run` to preview without writing.

---

## Job Lifecycle

```
PENDING  → created, waiting in asyncio.Queue
RUNNING  → dispatched to a worker (mark_running called by on_complete)
DONE     → audio written to disk (audio_path set)
FAILED   → synthesis or I/O error (error message stored)
```

Only DONE and FAILED jobs are evicted by the TTL cleanup. PENDING/RUNNING jobs are never touched.

TTL default: 300 s. Audio files are deleted from disk at eviction time.

---

## Batch Endpoint

`POST /v1/batch` accepts a list of TTS items (same schema as `TtsRequest`).
- Max 50 items per request (`MAX_BATCH_SIZE` in `batch.py`)
- All-or-nothing validation — if any item fails, the entire batch is rejected before any jobs are created
- Each item enters the shared QueueManager and gets its own `job_id`; poll them individually
- If the queue fills up mid-batch, already-created jobs are marked FAILED and a 503 is returned (no partial success)

---

## Error Handling Patterns

- **HTTP exceptions:** always `raise HTTPException(...) from exc` (ruff B904)
- **Silent swallowing:** always `contextlib.suppress(SomeException)` not `try/except: pass` (ruff SIM105)
- **Shutdown cancellation:** `contextlib.suppress(asyncio.CancelledError)` after `task.cancel()`
- **Fire-and-forget tasks:** `_task = asyncio.create_task(...); del _task` — the `del` is intentional, keeps a reference long enough to avoid GC but signals it's fire-and-forget (ruff RUF006)

---

## Config Reference

Key settings in `xtts_server/config.py` → `Settings(BaseSettings)`:

| Field | Type | Default | Notes |
|---|---|---|---|
| `MODEL_PATH` | `str` | — | Required. Startup aborts if dir or files missing |
| `WORKERS_PER_GPU` | `str` | `"1"` | `"1"` or `"2,3,2"`. Parsed into `workers_per_gpu_list` |
| `NUM_GPUS` | `int\|None` | `None` | Auto-detected via `torch.cuda.device_count()` if None |
| `DEFAULT_LANGUAGE` | `str` | `"tr"` | Fallback when request omits language |
| `MAX_QUEUE_SIZE` | `int` | `100` | asyncio.Queue maxsize — 503 only when this is full |
| `JOB_TTL_SECONDS` | `int` | `300` | Eviction age for DONE/FAILED jobs |
| `SPEAKERS_DIR` | `str` | `"./speakers"` | Root for speaker subdirectories |
| `OUTPUTS_DIR` | `str` | `"./outputs"` | Where finished audio files are written |
| `MAX_TEXT_LENGTH` | `int` | `5000` | Hard cap on input text length |
| `SAMPLE_RATE` | `int` | `24000` | XTTS-v2 native sample rate — do not change |
| `LOG_LEVEL` | `str` | `"INFO"` | Passed to logging |
| `CUDA_VISIBLE_DEVICES` | `str` | all GPUs | GPU indices to expose (e.g. `"0,1"`); standard PyTorch env var |
| `GPU_MEMORY_FRACTION` | `float` | `"1.0"` | Per-worker VRAM cap `(0.0, 1.0]`; exported by `start-server.sh`, read by `worker.py` |
| `USE_FP16` | `bool` | `False` | Enable mixed-precision inference on GPU via `torch.autocast`. No-op on CPU. See decision #17. |

Required model files in `MODEL_PATH`: `config.json`, `model.pth`, `vocab.json`.

Supported languages (17): `en es fr de it pt pl tr ru nl cs ar zh-cn ja hu ko hi`

---

## Router Layout

`routers/tts.py` exports **two** `APIRouter` instances:
- `router` (prefix `/v1/tts`) — async fire-and-poll job submission and audio download
- `audio_router` (prefix `/audio`) — synchronous endpoints: `/audio/speech-file` (raw WAV bytes) and `/audio/speech` (JSON `TtsResponse`)

Both are registered in `main.py`. The sync endpoints share `_run_synthesis()` to avoid duplication. Use `response_class=Response` on binary endpoints so Swagger shows a download link instead of trying to render bytes as JSON.

---

## Startup / Shutdown Order

**Startup** (in `main.py` lifespan, order is load-bearing):
1. `load_settings()` — validates config and model files; `_log_banner()` immediately after prints the QUAKER-XTTS ASCII art
2. `Dispatcher.start()` — starts Manager process, then spawns worker processes synchronously (before the event loop enters async code — `multiprocessing.Process.start()` is not async-safe)
3. `JobStore`, `SpeakerStore`, `QueueManager` — plain construction
4. `job_store.start()`, `queue_manager.start()`, `dispatcher.start_background_tasks()` — start async background tasks
5. `speaker_store.preload_all()` — warm the RAM cache

**Shutdown** (reverse dependency order):
1. `queue_manager.shutdown()` — stop accepting new work
2. `dispatcher.shutdown()` — send `None` sentinels to workers, join processes, shut down Manager
3. `job_store.shutdown()` — cancel TTL cleanup task

---

## app.state Attributes

Everything shared across routers lives on `request.app.state`:

| Attribute | Type | Set in |
|---|---|---|
| `settings` | `Settings` | lifespan |
| `dispatcher` | `Dispatcher` | lifespan |
| `job_store` | `JobStore` | lifespan |
| `speaker_store` | `SpeakerStore` | lifespan |
| `queue_manager` | `QueueManager` | lifespan |
| `start_time` | `float` | lifespan (monotonic) |
| `start_timestamp` | `float` | lifespan (wall clock) |

In WebSocket handlers use `websocket.app.state` (not `request.app.state`).

---

## Ruff

Config in `pyproject.toml`. Target: Python 3.11, line length 100.

```bash
ruff format xtts_server/   # format
ruff check xtts_server/    # lint
ruff check --fix xtts_server/  # lint + auto-fix
```

Active rule sets: `E`, `W`, `F`, `I`, `B`, `UP`, `C4`, `SIM`, `RUF`.
Ignored: `E501` (line length, handled by formatter), `B008` (FastAPI Depends pattern), `SIM108` (ternary), `UP007` (X | Y union syntax in older annotations).

---

## Audio Formats

`audio.py` supports `wav`, `mp3`, `ogg`, `flac`.

- `wav` / `ogg` / `flac` — via `soundfile` (libsndfile), PCM-16/PCM-24 depending on format
- `mp3` — via `pydub` + `ffmpeg` (ffmpeg must be on PATH; pydub is only imported when MP3 is requested)
- WebSocket streaming always sends raw `float32` PCM at 24 000 Hz mono (no container)

---

## Known Limitations / Future Work

- **Authentication:** no API key or auth middleware yet. Add as a FastAPI dependency on router level.
- **Persistence:** job store is in-memory. Server restart loses all pending/running jobs. Add SQLite or Redis for durability.
- **Speaker update:** no `PUT /v1/speakers/{name}` endpoint. Delete + re-register to update a voice.
- **Long texts:** XTTS-v2 degrades on very long inputs. Consider splitting at sentence boundaries before `model.inference()` and concatenating audio chunks.
- **CPU fallback:** config supports 0 GPUs (single CPU worker), but latency will be high (~10-30× real time). Only suitable for dev/testing.
- **Dockerfile requirements path:** the Dockerfile copies `xtts_server/requirements.in` — now that deps live at project root, update the `COPY` instruction when rebuilding.

---

## Linux Setup Scripts

### `install.sh`
Linux-only. Checks: OS, `nvidia-smi` presence, CUDA ≥ 12.1, Python 3.11+, `pip`. Creates `.venv/`, runs `pip install -r requirements.txt`, then verifies `torch.__version__`, `torchaudio.__version__`, `torch.cuda.is_available()`, and `torch.cuda.device_count()`. Copies `.env.example → xtts_server/.env` if absent. Creates `xtts_server/speakers/` and `xtts_server/outputs/`.

### `start-server.sh`
Linux-only. CONFIGURATION block at the top — edit before running. Key vars:

| Variable | Description |
|---|---|
| `MODEL_PATH` | Path to XTTS-v2 model directory |
| `CUDA_VISIBLE_DEVICES` | GPU indices to expose (`"0"`, `"0,1"`, etc.) |
| `GPU_MEMORY_FRACTION` | Per-worker VRAM cap as float in `(0, 1]` |
| `WORKERS_PER_GPU` | Workers per visible GPU or comma-separated list |
| `SEED_SPEAKERS` | `true/false` — run `seed_studio_speakers.py` before launch |

Pre-flight checks: OS, MODEL_PATH (catches placeholder), `nvidia-smi`, CUDA version, GPU inventory with utilisation + temperature, `CUDA_VISIBLE_DEVICES` index validation, `GPU_MEMORY_FRACTION` range, `.venv` presence, torch/torchaudio/CUDA install, VRAM summary. Ends with `exec "$PYTHON" main.py` from `xtts_server/`.

`GPU_MEMORY_FRACTION` is exported so worker processes inherit it (they read it via `os.environ.get("GPU_MEMORY_FRACTION", "1.0")`).

---

## macOS Dev Setup

`llvmlite` (via numba → librosa → TTS) does not build from source on macOS without LLVM. Use conda:

```bash
brew install miniconda
conda create -n xtts python=3.11
conda activate xtts
conda install -c conda-forge llvmlite numba librosa
pip install -r requirements.txt
```

For GPU inference you need a Linux machine or Docker with `--gpus all`.

---

## Git / GitHub

- Repo: https://github.com/EngincanVaran/xtts-v2-api-server
- Branch: `main`
- Remote name: `origin`
- `.gitignore` excludes: `__pycache__/`, `.venv/`, `outputs/`, `speakers/`, `logs/`, `*.pth`, `.env`, `.idea/`, `model/`
