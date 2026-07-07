#!/usr/bin/env python3
"""
tasks.py
========
Celery task definitions for distributed audio pipeline processing.

Enables parallel processing of high call volumes across multiple machines.
Uses Redis as the message broker (lightweight, single Docker container).

Architecture:
  ┌─────────────┐      ┌───────────────┐      ┌──────────────────┐
  │  Producer   │─────▶│  Redis Broker │─────▶│  Celery Workers  │
  │ (watcher /  │      │  (task queue) │      │  (1–N machines)  │
  │   API call) │      └───────────────┘      └──────────────────┘
  └─────────────┘

Usage:
  # Start Redis (Docker):
  docker run -d -p 6379:6379 redis:alpine

  # Start workers (run on each processing machine):
  celery -A tasks worker --loglevel=info --concurrency=4

  # Start GPU workers (separate queue for GPU-heavy jobs):
  celery -A tasks worker --loglevel=info --queues=gpu --concurrency=1

  # Monitor tasks:
  celery -A tasks flower   # web UI at http://localhost:5555

  # Submit a job programmatically:
  from tasks import process_audio
  result = process_audio.delay("/inbox/call.amr", studio=True, post_correct=True)
  print(result.get(timeout=3600))  # blocks until done

Install:
  pip install celery redis flower

Environment variables:
  CELERY_BROKER_URL   — default: redis://localhost:6379/0
  CELERY_RESULT_URL   — default: redis://localhost:6379/1
  OPENAI_API_KEY      — required for LLM features
  HF_TOKEN            — required for PyAnnote diarization
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

from celery import Celery
from celery.utils.log import get_task_logger

# ── Celery app configuration ──────────────────────────────────────────────────
BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_URL = os.environ.get("CELERY_RESULT_URL", "redis://localhost:6379/1")

app = Celery(
    "audio_pipeline",
    broker=BROKER_URL,
    backend=RESULT_URL,
)

app.conf.update(
    # Serialization
    task_serializer   = "json",
    result_serializer = "json",
    accept_content    = ["json"],

    # Reliability
    task_acks_late              = True,   # ack only after task completes
    task_reject_on_worker_lost  = True,   # requeue if worker dies mid-task
    worker_prefetch_multiplier  = 1,      # one task at a time per worker (fair dispatch)

    # Retries
    task_max_retries = 3,
    task_default_retry_delay = 60,        # seconds between retries

    # Routing: GPU-intensive jobs go to the 'gpu' queue
    task_routes = {
        "tasks.process_audio_gpu": {"queue": "gpu"},
        "tasks.process_audio":     {"queue": "celery"},
    },

    # Result expiry
    result_expires = 86400,               # 24 hours

    # Timezone
    timezone = "UTC",
    enable_utc = True,
)

log = get_task_logger(__name__)
PIPELINE_SCRIPT = str(Path(__file__).parent / "advanced_pipeline.py")


# ══════════════════════════════════════════════════════════════════════════════
# Shared task logic
# ══════════════════════════════════════════════════════════════════════════════

def _build_cmd(audio_path: str, output_dir: str, **kwargs) -> list[str]:
    """Build the advanced_pipeline.py command from keyword arguments."""
    cmd = [
        sys.executable, PIPELINE_SCRIPT,
        audio_path,
        "--output-dir", output_dir,
    ]

    bool_flags = [
        "studio", "vad", "post_correct", "name_speakers",
        "keep_timing", "enroll_unknown", "ecapa",
    ]
    for flag in bool_flags:
        if kwargs.get(flag, False):
            cmd.append(f"--{flag.replace('_', '-')}")

    str_flags = [
        "model", "device", "compute", "diarizer", "format",
        "language", "hf_token", "profile_db", "post_correct_model",
    ]
    defaults = {
        "model":               "base",
        "device":              "auto",
        "compute":             "auto",
        "diarizer":            "llm",
        "format":              "mp3",
        "language":            "auto",
        "profile_db":          "profiles.json",
        "post_correct_model":  "gpt-4o-mini",
    }
    for flag in str_flags:
        val = kwargs.get(flag, defaults.get(flag))
        if val:
            cmd += [f"--{flag.replace('_', '-')}", str(val)]

    float_flags = {"profile_threshold": 0.75, "vad_noise_db": -30.0}
    for flag, default in float_flags.items():
        cmd += [f"--{flag.replace('_', '-')}", str(kwargs.get(flag, default))]

    return cmd


def _run_pipeline(audio_path: str, output_dir: str, timeout: int,
                  **kwargs) -> dict:
    """Execute the pipeline subprocess and return a result dict."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd(audio_path, output_dir, **kwargs)

    log.info("Running pipeline: %s → %s", Path(audio_path).name, output_dir)
    try:
        result = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
        if result.returncode == 0:
            return {
                "status":     "success",
                "input":      audio_path,
                "output_dir": output_dir,
                "stdout":     result.stdout[-2000:],
            }
        else:
            return {
                "status":     "failed",
                "input":      audio_path,
                "output_dir": output_dir,
                "error":      result.stderr[-2000:],
            }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "input":  audio_path,
            "error":  f"Exceeded {timeout}s timeout",
        }


# ══════════════════════════════════════════════════════════════════════════════
# Celery Tasks
# ══════════════════════════════════════════════════════════════════════════════

@app.task(
    bind=True,
    name="tasks.process_audio",
    max_retries=3,
    default_retry_delay=60,
    queue="celery",
)
def process_audio(self, audio_path: str, output_dir: str = None,
                  timeout: int = 3600, **kwargs) -> dict:
    """
    CPU pipeline task.

    Args:
        audio_path:  Absolute path to the input audio file.
        output_dir:  Output directory (defaults to <audio_stem> in /output).
        timeout:     Per-task timeout in seconds.
        **kwargs:    Any advanced_pipeline.py flags (studio=True, model='base', etc.)

    Returns:
        dict with status, input, output_dir, and stdout/error.

    Example:
        from tasks import process_audio
        result = process_audio.delay(
            "/inbox/call.amr",
            studio=True,
            post_correct=True,
            model="base",
        )
        print(result.get(timeout=3600))
    """
    if output_dir is None:
        stem       = Path(audio_path).stem
        output_dir = str(Path("/output") / stem)

    try:
        return _run_pipeline(audio_path, output_dir, timeout, **kwargs)
    except Exception as exc:
        log.error("Task failed: %s — retrying (%d/%d)",
                  exc, self.request.retries, self.max_retries)
        raise self.retry(exc=exc)


@app.task(
    bind=True,
    name="tasks.process_audio_gpu",
    max_retries=3,
    default_retry_delay=60,
    queue="gpu",
)
def process_audio_gpu(self, audio_path: str, output_dir: str = None,
                      timeout: int = 3600, **kwargs) -> dict:
    """
    GPU-optimised pipeline task (routes to the 'gpu' queue).
    Automatically sets device=cuda and compute=float16.

    Example:
        from tasks import process_audio_gpu
        result = process_audio_gpu.delay(
            "/inbox/call.amr",
            studio=True,
            post_correct=True,
            model="large-v3",
        )
    """
    kwargs.setdefault("device",  "cuda")
    kwargs.setdefault("compute", "float16")
    kwargs.setdefault("model",   "large-v3")

    if output_dir is None:
        stem       = Path(audio_path).stem
        output_dir = str(Path("/output") / stem)

    try:
        return _run_pipeline(audio_path, output_dir, timeout, **kwargs)
    except Exception as exc:
        log.error("GPU task failed: %s — retrying (%d/%d)",
                  exc, self.request.retries, self.max_retries)
        raise self.retry(exc=exc)


@app.task(name="tasks.batch_process")
def batch_process(audio_paths: list[str], output_base: str = "/output",
                  use_gpu: bool = False, **kwargs) -> list[str]:
    """
    Submit a batch of audio files as individual tasks.
    Returns a list of task IDs for monitoring.

    Example:
        from tasks import batch_process
        task_ids = batch_process.delay(
            ["/inbox/call1.amr", "/inbox/call2.amr"],
            studio=True,
        ).get()
    """
    task_ids = []
    for path in audio_paths:
        stem       = Path(path).stem
        output_dir = str(Path(output_base) / stem)
        if use_gpu:
            task = process_audio_gpu.delay(path, output_dir, **kwargs)
        else:
            task = process_audio.delay(path, output_dir, **kwargs)
        task_ids.append(task.id)
        log.info("Submitted task %s for %s", task.id, Path(path).name)
    return task_ids


# ══════════════════════════════════════════════════════════════════════════════
# CLI helper: submit a single file from the command line
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Submit an audio file to the Celery pipeline")
    p.add_argument("input",          help="Audio file path")
    p.add_argument("--output-dir",   default=None)
    p.add_argument("--gpu",          action="store_true", help="Use GPU worker queue")
    p.add_argument("--studio",       action="store_true")
    p.add_argument("--post-correct", action="store_true")
    p.add_argument("--name-speakers",action="store_true")
    p.add_argument("--model",        default="base")
    p.add_argument("--wait",         action="store_true", help="Block until task completes")
    args = p.parse_args()

    kwargs = {
        "studio":        args.studio,
        "post_correct":  args.post_correct,
        "name_speakers": args.name_speakers,
        "model":         args.model,
    }

    if args.gpu:
        task = process_audio_gpu.delay(args.input, args.output_dir, **kwargs)
    else:
        task = process_audio.delay(args.input, args.output_dir, **kwargs)

    print(f"Submitted task: {task.id}")

    if args.wait:
        print("Waiting for result...")
        result = task.get(timeout=3600)
        print(f"Status: {result['status']}")
        if result["status"] == "success":
            print(f"Output: {result['output_dir']}")
        else:
            print(f"Error: {result.get('error', 'unknown')}")
