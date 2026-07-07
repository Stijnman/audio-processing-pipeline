#!/usr/bin/env python3
"""
watcher.py
==========
Zero-touch directory watcher for the audio processing pipeline.

Monitors an /inbox directory for new audio files. The moment a file
lands, it is automatically processed by advanced_pipeline.py and the
results are written to /output. Processed files are moved to /processed.

Supports both polling mode (no extra deps) and inotify/FSEvents mode
(via watchdog, much more efficient on large directories).

Usage:
  python watcher.py                          # default dirs
  python watcher.py --inbox /data/calls --output /data/results
  python watcher.py --studio --post-correct --model large-v3
  python watcher.py --workers 4             # parallel processing

Environment variables:
  OPENAI_API_KEY   — required for LLM features
  HF_TOKEN         — required for PyAnnote diarization

Install:
  pip install watchdog   # optional but recommended
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [watcher]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("watcher")

# ── Supported audio extensions ────────────────────────────────────────────────
AUDIO_EXTENSIONS = {
    ".amr", ".mp3", ".wav", ".m4a", ".ogg", ".flac",
    ".aac", ".wma", ".opus", ".webm", ".mp4",
}


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Runner
# ══════════════════════════════════════════════════════════════════════════════

def process_file(audio_path: Path, args: argparse.Namespace) -> bool:
    """
    Run the full pipeline on a single audio file.
    Returns True on success, False on failure.
    """
    stem       = audio_path.stem
    output_dir = Path(args.output) / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Processing: %s → %s", audio_path.name, output_dir)

    # Build the pipeline command
    cmd = [
        sys.executable, str(Path(__file__).parent / "advanced_pipeline.py"),
        str(audio_path),
        "--output-dir", str(output_dir),
    ]

    # Forward relevant flags
    if args.studio:
        cmd.append("--studio")
    if args.vad:
        cmd.append("--vad")
    if args.post_correct:
        cmd.append("--post-correct")
    if args.name_speakers:
        cmd.append("--name-speakers")
    if args.keep_timing:
        cmd.append("--keep-timing")
    if args.enroll_unknown:
        cmd.append("--enroll-unknown")

    cmd += ["--model",             args.model]
    cmd += ["--device",            args.device]
    cmd += ["--compute",           args.compute]
    cmd += ["--diarizer",          args.diarizer]
    cmd += ["--format",            args.format]
    cmd += ["--profile-db",        str(Path(args.output) / "profiles.json")]
    cmd += ["--profile-threshold", str(args.profile_threshold)]
    cmd += ["--post-correct-model", args.post_correct_model]

    if args.hf_token:
        cmd += ["--hf-token", args.hf_token]
    if args.ecapa:
        cmd.append("--ecapa")

    try:
        result = subprocess.run(cmd, timeout=args.timeout)
        if result.returncode == 0:
            log.info("Success: %s", audio_path.name)
            return True
        else:
            log.error("Pipeline failed (exit %d): %s", result.returncode, audio_path.name)
            return False
    except subprocess.TimeoutExpired:
        log.error("Timeout after %ds: %s", args.timeout, audio_path.name)
        return False
    except Exception as e:
        log.error("Unexpected error processing %s: %s", audio_path.name, e)
        return False


def move_file(src: Path, dest_dir: Path, success: bool) -> None:
    """Move a processed file to the processed or failed directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    # Avoid overwriting if a file with the same name already exists
    if dest.exists():
        stem   = src.stem
        suffix = src.suffix
        i = 1
        while dest.exists():
            dest = dest_dir / f"{stem}_{i}{suffix}"
            i += 1
    shutil.move(str(src), str(dest))
    status = "processed" if success else "failed"
    log.info("Moved to %s: %s", status, dest.name)


def scan_and_process(inbox: Path, args: argparse.Namespace,
                     executor: ThreadPoolExecutor) -> None:
    """Scan the inbox for unprocessed audio files and submit them."""
    futures = {}
    for f in sorted(inbox.iterdir()):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
            # Skip files that are still being written (size stable check)
            if not is_file_stable(f):
                log.debug("Skipping unstable file: %s", f.name)
                continue
            future = executor.submit(process_file, f, args)
            futures[future] = f

    for future in as_completed(futures):
        f       = futures[future]
        success = future.result()
        dest_dir = Path(args.processed) if success else Path(args.processed) / "failed"
        move_file(f, dest_dir, success)


def is_file_stable(path: Path, wait: float = 1.0) -> bool:
    """
    Check if a file has finished being written by comparing size twice.
    Returns True if the file size is stable.
    """
    try:
        size1 = path.stat().st_size
        time.sleep(wait)
        size2 = path.stat().st_size
        return size1 == size2 and size1 > 0
    except OSError:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Watchdog Integration (inotify / FSEvents)
# ══════════════════════════════════════════════════════════════════════════════

try:
    from watchdog.observers import Observer
    from watchdog.events    import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


class AudioFileHandler(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    """Watchdog event handler: triggers pipeline on new audio files."""

    def __init__(self, args: argparse.Namespace, executor: ThreadPoolExecutor):
        self.args     = args
        self.executor = executor
        self._pending: set[str] = set()

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            return
        if str(path) in self._pending:
            return
        self._pending.add(str(path))
        log.info("New file detected: %s", path.name)

        # Wait for the file to finish writing, then process
        def delayed_process():
            time.sleep(2.0)  # brief delay for write completion
            if not is_file_stable(path, wait=1.0):
                log.warning("File still unstable, skipping: %s", path.name)
                self._pending.discard(str(path))
                return
            success  = process_file(path, self.args)
            dest_dir = Path(self.args.processed) if success else Path(self.args.processed) / "failed"
            move_file(path, dest_dir, success)
            self._pending.discard(str(path))

        self.executor.submit(delayed_process)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Zero-touch audio pipeline directory watcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Directories
    p.add_argument("--inbox",     default="/inbox",     help="Directory to watch for new audio files")
    p.add_argument("--output",    default="/output",    help="Directory for pipeline output")
    p.add_argument("--processed", default="/processed", help="Directory for processed input files")

    # Processing
    p.add_argument("--workers",  type=int, default=2,   help="Number of parallel pipeline workers")
    p.add_argument("--poll",     action="store_true",   help="Force polling mode (ignore watchdog)")
    p.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds")
    p.add_argument("--timeout",  type=int, default=3600, help="Per-file processing timeout (seconds)")

    # Pipeline flags (forwarded to advanced_pipeline.py)
    p.add_argument("--studio",           action="store_true")
    p.add_argument("--vad",              action="store_true")
    p.add_argument("--post-correct",     action="store_true")
    p.add_argument("--name-speakers",    action="store_true")
    p.add_argument("--keep-timing",      action="store_true")
    p.add_argument("--enroll-unknown",   action="store_true")
    p.add_argument("--ecapa",            action="store_true")
    p.add_argument("--model",            default="base")
    p.add_argument("--device",           default="auto")
    p.add_argument("--compute",          default="auto")
    p.add_argument("--diarizer",         default="llm", choices=["llm", "pyannote"])
    p.add_argument("--hf-token",         default=os.environ.get("HF_TOKEN"))
    p.add_argument("--format",           default="mp3")
    p.add_argument("--profile-threshold", type=float, default=0.75)
    p.add_argument("--post-correct-model", default="gpt-4o-mini")

    args = argparse.parse_args()

    inbox     = Path(args.inbox)
    output    = Path(args.output)
    processed = Path(args.processed)

    for d in [inbox, output, processed]:
        d.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Audio Pipeline Watcher  v2.0")
    log.info("  Inbox:     %s", inbox)
    log.info("  Output:    %s", output)
    log.info("  Processed: %s", processed)
    log.info("  Workers:   %d", args.workers)
    log.info("  Mode:      %s", "polling" if (args.poll or not WATCHDOG_AVAILABLE) else "watchdog")
    log.info("=" * 60)

    executor = ThreadPoolExecutor(max_workers=args.workers)

    # Process any files already in the inbox at startup
    log.info("Scanning inbox for existing files...")
    scan_and_process(inbox, args, executor)

    if not args.poll and WATCHDOG_AVAILABLE:
        # Efficient event-driven mode (inotify on Linux, FSEvents on macOS)
        log.info("Starting watchdog observer...")
        handler  = AudioFileHandler(args, executor)
        observer = Observer()
        observer.schedule(handler, str(inbox), recursive=False)
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("Shutting down watchdog observer...")
            observer.stop()
        observer.join()
    else:
        # Polling fallback
        if not WATCHDOG_AVAILABLE:
            log.info("watchdog not installed — using polling mode. "
                     "Install with: pip install watchdog")
        log.info("Polling inbox every %.1fs...", args.interval)
        try:
            while True:
                scan_and_process(inbox, args, executor)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            log.info("Watcher stopped.")

    executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
