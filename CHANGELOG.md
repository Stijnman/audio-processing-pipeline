# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- Added pytest test suite for `advanced_pipeline.py` utilities and `watcher.py`
- Added Tests badge to README.md
- Added GitHub Actions CI workflow (`ci.yml`): lint (ruff), type-check (mypy), pytest on Python 3.9â3.12
- Added GitHub Actions Docker workflow (`docker.yml`): build smoke-test and `docker-compose config` validation
- Added `pyproject.toml` with `[tool.ruff]` and `[tool.mypy]` configuration and optional dependency extras
- Added `requirements-dev.txt` for local development setup

### Fixed
- Fixed silent transcript data loss in `stitch_chunks`: overlap windows are now
  selected by timestamp (duplicated audio zone) instead of segment counts, and an
  ambiguous mid-window match keeps both sides instead of dropping real segments
- Fixed `diarize_llm` silently labelling every segment `SPEAKER_A` when the LLM
  response was malformed: responses are validated and require majority coverage
  before being trusted, otherwise the alternating fallback is used
- Fixed `diarize_llm` ignoring `--post-correct-model`; it now honours the model flag
- Fixed watcher polling mode resubmitting in-flight files on every scan: files are
  now tracked in an in-flight set until they are moved out of the inbox
- Fixed watcher missing files moved (not created) into the inbox: added `on_moved`
  handling for watchdog mode
- Fixed `vad_trim` keeping a pad-length sliver of pure silence when the recording
  starts with silence; interval math extracted into testable
  `speech_intervals_from_silence` helper
- Fixed output filenames not matching the README contract: transcript, diarization
  and per-speaker files are now prefixed with the input stem
  (`call_transcript.txt`, `call_speaker_alice.mp3`, ...)
- Fixed Celery tasks never retrying on pipeline failure: failed/timeout results now
  trigger `self.retry` with exponential backoff
- Fixed slow chunk extraction in `transcribe`: `-ss` now uses input seeking instead
  of re-decoding from the start of the file for every chunk
- Fixed `watcher.py` ignoring the documented `INBOX_DIR` / `OUTPUT_DIR` /
  `PROCESSED_DIR` environment variables; they now provide CLI defaults
- Fixed two `test_watcher.py` CLI tests that constructed their own argparse parser
  instead of exercising `watcher.py`'s real parser (now extracted as `build_parser`)
- Fixed bug in `advanced_pipeline.py`'s `stitch_chunks` where the overlapping match was excluded instead of deduplicated.
- Fixed `watcher.py` CLI parser to correctly parse arguments using `p.parse_args()` instead of the module-level function.
- Fixed `termux_install.sh` generating `watch.sh` with an invalid `--output` flag (updated to `--output-dir`).

---

## [2.0.0] â 2026-07-07

### Added
- `AudioPipeline.py` â single-file interactive launcher with auto-install and numbered menu
- `--diarizer pyannote` â PyAnnote 3.1 neural diarization for multi-speaker and overlapping speech
- `--ecapa` â ECAPA-TDNN 192-dim voice embeddings via SpeechBrain (replaces MFCC for phone audio)
- `--device auto` â automatic GPU/CPU detection with optimal quantization selection
- `--post-correct` â LLM post-correction for homophones, punctuation, and ASR artefacts
- `--name-speakers` â LLM-based speaker name extraction from transcript context
- `--keep-timing` â full-length sync'd per-speaker tracks with silence where other speaker talks
- `--enroll-unknown` â automatic enrollment of unmatched speakers into the profile database
- Dynamic chunking with SequenceMatcher overlap stitching for files of any length
- `watcher.py` â directory watcher with file stability check and parallel processing
- `tasks.py` â Celery task queue with separate CPU and GPU queues
- `docker-compose.yml` â full stack with Redis, watcher, workers, and Flower monitoring UI
- `termux_install.sh` â one-command installer for Android/Termux
- `docs/ARCHITECTURE.md` â full pipeline architecture and design decisions
- `docs/TERMUX.md` â Android/Termux setup guide

### Changed
- Replaced standard Whisper with **faster-whisper** (CTranslate2) â 4â18Ã faster depending on hardware
- Replaced per-segment MFCC matching with **per-speaker mean embedding** â eliminates profile DB explosion bug
- Rewrote `stitch_chunks` using token-slice SequenceMatcher â more precise than time-based midpoint fallback
- Studio enhancement chain updated with dB-based thresholds for all filters
- LLM diarization now batches segments in groups of 30 â reduces API calls and latency
- Profile database now uses incremental running average â profiles improve with each new sample

### Fixed
