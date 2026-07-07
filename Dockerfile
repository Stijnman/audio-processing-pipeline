# ══════════════════════════════════════════════════════════════════════════════
# Audio Processing Pipeline  —  Docker Image
# ══════════════════════════════════════════════════════════════════════════════
#
# Builds a portable, self-contained image with:
#   - ffmpeg (studio audio enhancement + splitting)
#   - faster-whisper (int8 CPU transcription, or float16 GPU)
#   - onnxruntime (Silero VAD)
#   - scipy / numpy (MFCC voice profiling)
#   - openai (LLM post-correction + diarization)
#   - watchdog (efficient file system events)
#   - celery + redis client (distributed task queue)
#
# Usage:
#   # Build:
#   docker build -t audio-pipeline .
#
#   # Run watcher (CPU):
#   docker run -d \
#     --name pipeline \
#     -v /your/inbox:/inbox \
#     -v /your/output:/output \
#     -v /your/processed:/processed \
#     -e OPENAI_API_KEY=sk-... \
#     audio-pipeline
#
#   # Run watcher (GPU — requires NVIDIA Container Toolkit):
#   docker run -d --gpus all \
#     --name pipeline-gpu \
#     -v /your/inbox:/inbox \
#     -v /your/output:/output \
#     -v /your/processed:/processed \
#     -e OPENAI_API_KEY=sk-... \
#     audio-pipeline \
#     python watcher.py --device cuda --compute float16 --model large-v3
#
#   # Run Celery worker:
#   docker run -d \
#     --name celery-worker \
#     -e OPENAI_API_KEY=sk-... \
#     -e CELERY_BROKER_URL=redis://redis:6379/0 \
#     audio-pipeline \
#     celery -A tasks worker --loglevel=info --concurrency=4
#
#   # Run Celery GPU worker:
#   docker run -d --gpus all \
#     --name celery-gpu-worker \
#     -e OPENAI_API_KEY=sk-... \
#     -e CELERY_BROKER_URL=redis://redis:6379/0 \
#     audio-pipeline \
#     celery -A tasks worker --loglevel=info --queues=gpu --concurrency=1
#
# ══════════════════════════════════════════════════════════════════════════════

# ── Base image ────────────────────────────────────────────────────────────────
# Use slim Python 3.11 for a small footprint.
# For GPU support, swap to: nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
FROM python:3.11-slim

# ── Build arguments ───────────────────────────────────────────────────────────
ARG DEBIAN_FRONTEND=noninteractive

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        curl \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Install in a single layer to keep image size minimal.
# Pin versions for reproducibility.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
WORKDIR /app
COPY advanced_pipeline.py .
COPY watcher.py           .
COPY tasks.py             .

# ── Volume mount points ───────────────────────────────────────────────────────
# /inbox     — drop audio files here for automatic processing
# /output    — pipeline results (transcripts, speaker MP3s, JSON)
# /processed — processed input files (moved here after pipeline completes)
# /profiles  — voice profile database (persisted across container restarts)
VOLUME ["/inbox", "/output", "/processed", "/profiles"]

# ── Environment defaults ──────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENAI_API_KEY="" \
    HF_TOKEN="" \
    CELERY_BROKER_URL="redis://redis:6379/0" \
    CELERY_RESULT_URL="redis://redis:6379/1"

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import faster_whisper; print('ok')" || exit 1

# ── Default command: run the directory watcher ────────────────────────────────
# Override with `docker run ... celery -A tasks worker` for Celery mode.
CMD ["python", "watcher.py", \
     "--inbox",     "/inbox", \
     "--output",    "/output", \
     "--processed", "/processed", \
     "--studio", \
     "--post-correct", \
     "--name-speakers", \
     "--profile-db", "/profiles/profiles.json"]
