# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.0.0] — 2026-07-07

### Added
- `AudioPipeline.py` — single-file interactive launcher with auto-install and numbered menu
- `--diarizer pyannote` — PyAnnote 3.1 neural diarization for multi-speaker and overlapping speech
- `--ecapa` — ECAPA-TDNN 192-dim voice embeddings via SpeechBrain (replaces MFCC for phone audio)
- `--device auto` — automatic GPU/CPU detection with optimal quantization selection
- `--post-correct` — LLM post-correction for homophones, punctuation, and ASR artefacts
- `--name-speakers` — LLM-based speaker name extraction from transcript context
- `--keep-timing` — full-length sync'd per-speaker tracks with silence where other speaker talks
- `--enroll-unknown` — automatic enrollment of unmatched speakers into the profile database
- Dynamic chunking with SequenceMatcher overlap stitching for files of any length
- `watcher.py` — directory watcher with file stability check and parallel processing
- `tasks.py` — Celery task queue with separate CPU and GPU queues
- `docker-compose.yml` — full stack with Redis, watcher, workers, and Flower monitoring UI
- `termux_install.sh` — one-command installer for Android/Termux
- `docs/ARCHITECTURE.md` — full pipeline architecture and design decisions
- `docs/TERMUX.md` — Android/Termux setup guide

### Changed
- Replaced standard Whisper with **faster-whisper** (CTranslate2) — 4–18× faster depending on hardware
- Replaced per-segment MFCC matching with **per-speaker mean embedding** — eliminates profile DB explosion bug
- Rewrote `stitch_chunks` using token-slice SequenceMatcher — more precise than time-based midpoint fallback
- Studio enhancement chain updated with dB-based thresholds for all filters
- LLM diarization now batches segments in groups of 30 — reduces API calls and latency
- Profile database now uses incremental running average — profiles improve with each new sample

### Fixed
- `--enroll-unknown` no longer creates a new profile for every segment (was creating 50+ profiles per call)
- `apply_profiling` now aggregates all segments per speaker before matching (was matching per-segment, causing instability)
- `--json-only` flag AttributeError on startup
- Timestamp format crash when `start`/`end` were floats formatted with `:.2s`
- argparse inconsistency between `--enroll-unknown` and `--enroll_unknown`

---

## [1.0.0] — 2026-07-05

### Added
- Initial release
- AMR to MP3 conversion via ffmpeg
- 8-stage studio audio enhancement chain
- faster-whisper transcription with Silero VAD
- LLM turn-taking diarization
- MFCC voice profiling with ProfileDB
- Per-speaker audio splitting (concatenated mode)
- `--vad` silence removal
