<div align="center">

# 🎙️ Audio Processing Pipeline

**Production-grade audio pipeline for phone calls and recordings.**  
Convert · Clean · Transcribe · Diarize · Split by Speaker — automatically.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![faster-whisper](https://img.shields.io/badge/ASR-faster--whisper-orange)](https://github.com/SYSTRAN/faster-whisper)
[![ffmpeg](https://img.shields.io/badge/Audio-ffmpeg-darkgreen?logo=ffmpeg)](https://ffmpeg.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](Dockerfile)
[![Android](https://img.shields.io/badge/Android-Termux-brightgreen?logo=android)](docs/TERMUX.md)

</div>

---

## Overview

The Audio Processing Pipeline takes raw phone call recordings in any format (AMR, MP3, WAV, M4A, OGG, FLAC) and produces:

- A **studio-quality cleaned MP3** of the full call
- **Per-speaker audio tracks** — one file per person, either concatenated or sync'd with the original timeline
- A **speaker-labelled transcript** with timestamps
- A **diarization JSON** for downstream processing

All processing runs locally. The only optional external dependency is an OpenAI API key for LLM-powered features (post-correction, name recognition, diarization). Everything else — transcription, audio enhancement, speaker splitting — runs fully offline.

---

## Features

| Feature | Description | Flag |
|---|---|---|
| **Studio enhancement** | 8-stage ffmpeg chain: highpass, lowpass, anlmdn, afftdn, agate, acompressor, speechnorm, alimiter | `--studio` |
| **Silero VAD** | Silence removal before transcription — prevents hallucinations, saves compute | `--vad-threshold` |
| **faster-whisper** | CTranslate2-optimised Whisper with int8/float16 quantization | `--model`, `--compute` |
| **Dynamic chunking** | Rolling chunks with SequenceMatcher overlap stitching for files of any length | `--chunk-sec`, `--overlap-sec` |
| **LLM diarization** | Turn-taking analysis via GPT-4o-mini — works in any language | `--diarizer llm` |
| **PyAnnote diarization** | Neural diarization — handles overlapping speech and any speaker count | `--diarizer pyannote` |
| **MFCC voice profiling** | 40-dim embeddings for cross-call speaker re-identification | `--profile-db` |
| **ECAPA-TDNN profiling** | 192-dim neural embeddings — far more accurate on compressed phone audio | `--ecapa` |
| **LLM post-correction** | Fix homophones, restore punctuation, clean ASR artefacts | `--post-correct` |
| **Speaker name recognition** | Detect real names from greetings and introductions in the transcript | `--name-speakers` |
| **Per-speaker splitting** | Concatenated (compact) or keep-timing (sync'd silence) speaker tracks | `--keep-timing` |
| **Directory watcher** | Zero-touch automation — process files the moment they land in a folder | `watcher.py` |
| **Celery workers** | Distributed parallel processing across multiple machines | `tasks.py` |
| **Docker** | Portable, offline-capable containerised deployment | `Dockerfile` |
| **Android / Termux** | Runs natively on Android — no root, no Docker, no PC required | [docs/TERMUX.md](docs/TERMUX.md) |

---

## Quick Start

### Requirements

- Python 3.9+
- ffmpeg (system package)
- An OpenAI API key (optional — only for `--post-correct`, `--name-speakers`, `--diarizer llm`)

### Install

```bash
# Clone
git clone https://github.com/Stijnman/audio-processing-pipeline.git
cd audio-processing-pipeline

# Install Python dependencies
pip install -r requirements.txt

# Install ffmpeg (if not already installed)
# macOS:   brew install ffmpeg
# Ubuntu:  sudo apt install ffmpeg
# Windows: https://ffmpeg.org/download.html
```

### Process a file

```bash
# Basic — transcribe and diarize
python advanced_pipeline.py call.amr

# Full pipeline — studio quality + LLM post-correction + name recognition
python advanced_pipeline.py call.amr \
  --studio --post-correct --name-speakers

# GPU acceleration with large-v3 model
python advanced_pipeline.py call.amr \
  --device cuda --compute float16 --model large-v3 \
  --studio --post-correct

# Keep original timing in per-speaker output files
python advanced_pipeline.py call.amr --studio --keep-timing
```

### Interactive menu (double-click)

```bash
python AudioPipeline.py
```

Opens a numbered menu — no CLI knowledge required. Auto-installs missing packages on first run.

### Automate with the directory watcher

```bash
# Drop audio files into ./inbox — they are processed automatically
python watcher.py --inbox ./inbox --output ./output --studio --post-correct
```

### Deploy with Docker

```bash
# CPU (single container)
docker build -t audio-pipeline .
docker run -d \
  -v ./inbox:/inbox \
  -v ./output:/output \
  -e OPENAI_API_KEY=sk-... \
  audio-pipeline

# Full stack — Redis + watcher + Celery workers + Flower monitoring UI
cp .env.example .env   # fill in your API keys
docker compose up -d
# Flower UI: http://localhost:5555
```

---

## Output

For an input file `call_20260705.amr`, the pipeline produces:

```
output/
├── call_20260705_cleaned.mp3        # studio-enhanced full call
├── call_20260705_speaker_alice.mp3  # Alice's audio only
├── call_20260705_speaker_bob.mp3    # Bob's audio only
├── call_20260705_transcript.txt     # speaker-labelled transcript
└── call_20260705_diarization.json   # timestamped diarization data
```

---

## CLI Reference

```
python advanced_pipeline.py [OPTIONS] input

Arguments:
  input                   Path to input audio file (AMR, MP3, WAV, M4A, OGG, FLAC)

Audio Enhancement:
  --studio                Apply all 8 enhancement stages
  --vad                   Remove silence (sound-activated recording)
  --vad-noise-db DB       Silence threshold in dB (default: -30)
  --vad-min-silence SEC   Minimum silence duration to cut (default: 0.5)

Transcription:
  --model MODEL           Whisper model: tiny, base, small, medium, large-v2, large-v3
  --device DEVICE         Device: auto, cpu, cuda (default: auto)
  --compute TYPE          Quantization: auto, int8, float16, int8_float16 (default: auto)
  --language LANG         Language code, e.g. nl, en, de (default: auto-detect)
  --chunk-sec N           Chunk size in seconds for long files (default: 60)
  --overlap-sec N         Overlap between chunks in seconds (default: 2)

Diarization:
  --diarizer {llm,pyannote}   Diarization engine (default: llm)
  --hf-token TOKEN            HuggingFace token for PyAnnote
  --json FILE                 Use pre-existing diarization JSON (skip transcription)

Voice Profiling:
  --profile-db FILE       Path to voice profile database (JSON)
  --enroll-unknown        Enroll unmatched speakers automatically
  --ecapa                 Use ECAPA-TDNN embeddings (requires speechbrain)
  --profile-threshold F   Cosine similarity threshold (default: 0.75)

LLM Features (require OPENAI_API_KEY):
  --post-correct          Fix homophones, punctuation, ASR artefacts
  --name-speakers         Detect real names from transcript context

Output:
  --output-dir DIR        Output directory (default: same as input)
  --keep-timing           Full-length sync'd tracks (silence where other speaker talks)
  --format {mp3,wav,ogg,flac}  Output format (default: mp3)
```

---

## Performance

| Configuration | Model | Speed (realtime ×) | VRAM |
|---|---|---|---|
| CPU int8 | tiny | 8–10× | — |
| CPU int8 | base | 4–6× | — |
| CPU int8 | large-v3 | 0.3–0.5× | — |
| GPU float16 | base | 20–30× | ~1 GB |
| GPU float16 | large-v3 | 8–12× | ~6 GB |
| GPU int8_float16 | large-v3 | 12–18× | ~4 GB |

---

## Voice Profiling

The profile database (`profiles.json`) stores named speaker embeddings and grows incrementally — each time a known speaker is matched, their profile is updated with the new embedding (running average), so accuracy improves over time.

```bash
# Step 1 — First call: enroll speakers
python advanced_pipeline.py call1.amr \
  --profile-db profiles.json --enroll-unknown

# Step 2 — Edit profiles.json to rename generic labels
# Change "SPEAKER_A" → "Alice", "SPEAKER_B" → "Bob"

# Step 3 — All subsequent calls: automatic name matching
python advanced_pipeline.py call2.amr --profile-db profiles.json
# Output: speaker_alice.mp3, speaker_bob.mp3
```

> **Note on phone audio:** GSM/AMR compression (8 kHz, heavy quantization) strips spectral detail that distinguishes voices. MFCC embeddings work well for re-identifying the same speaker across different calls. For blind separation of two unknown speakers from the same compressed call, use `--ecapa` (ECAPA-TDNN) or `--diarizer pyannote`.

---

## Distributed Processing with Celery

```bash
# 1. Start Redis
docker run -d -p 6379:6379 redis:alpine

# 2. Start workers (CPU)
celery -A tasks worker --loglevel=info --concurrency=4

# 3. Start a GPU worker (optional)
celery -A tasks worker --loglevel=info --queues=gpu --concurrency=1

# 4. Submit a job
python tasks.py /inbox/call.amr --studio --post-correct --wait

# 5. Monitor
celery -A tasks flower   # http://localhost:5555
```

| Volume | Recommended approach |
|---|---|
| < 50 calls/day | `watcher.py` |
| 50–500 calls/day | Celery with 4–8 workers |
| 500+ calls/day | Celery + multiple machines + GPU workers |

---

## Optional Upgrades

### PyAnnote Neural Diarization

Handles overlapping speech and any number of speakers. Recommended for meeting recordings and multi-party calls.

```bash
pip install pyannote.audio

# Accept model license at:
# https://huggingface.co/pyannote/speaker-diarization-3.1

python advanced_pipeline.py call.amr \
  --diarizer pyannote \
  --hf-token YOUR_HUGGINGFACE_TOKEN
```

### ECAPA-TDNN Voice Profiling

192-dimensional speaker embeddings trained on VoxCeleb — significantly more discriminative than MFCC on compressed phone audio.

```bash
pip install speechbrain torchaudio torch

python advanced_pipeline.py call.amr \
  --profile-db profiles.json \
  --ecapa \
  --enroll-unknown
```

---

## Running on Android (Termux)

The pipeline runs natively on Android via [Termux](https://termux.dev) — no root, no Docker, no PC required.

```bash
# In Termux:
pkg install git -y
git clone https://github.com/Stijnman/audio-processing-pipeline.git
cd audio-processing-pipeline
bash termux_install.sh        # installs all dependencies (~3–8 min)
bash process.sh call.amr --studio
```

Process recordings directly from your phone's storage:

```bash
termux-setup-storage   # grant storage access (run once)
bash process.sh ~/storage/downloads/call.amr --studio --post-correct
```

See **[docs/TERMUX.md](docs/TERMUX.md)** for the full guide, including recommended models for Android CPUs, background process management, and troubleshooting.

---

## Environment Variables

| Variable | Required for |
|---|---|
| `OPENAI_API_KEY` | `--post-correct`, `--name-speakers`, `--diarizer llm` |
| `HF_TOKEN` | `--diarizer pyannote` |
| `CELERY_BROKER_URL` | Celery workers (default: `redis://localhost:6379/0`) |
| `CELERY_RESULT_URL` | Celery result backend (default: `redis://localhost:6379/1`) |

Copy `.env.example` to `.env` and fill in your values.

---

## File Reference

| File | Purpose |
|---|---|
| `AudioPipeline.py` | Single-file interactive launcher with auto-install menu |
| `advanced_pipeline.py` | Core pipeline — all processing logic, full CLI |
| `watcher.py` | Directory watcher for zero-touch automation |
| `tasks.py` | Celery task definitions for distributed processing |
| `Dockerfile` | Container image definition |
| `docker-compose.yml` | Full stack: Redis + watcher + workers + Flower UI |
| `requirements.txt` | Python dependencies |
| `.env.example` | Environment variable template |
| `termux_install.sh` | One-command Termux/Android installer |
| `docs/ARCHITECTURE.md` | Pipeline stages, design decisions, embedding strategy |
| `docs/TERMUX.md` | Android/Termux setup, usage, and troubleshooting |

---

## Documentation

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full pipeline architecture, stage-by-stage breakdown, design decisions |
| [docs/TERMUX.md](docs/TERMUX.md) | Android/Termux setup, storage access, model recommendations, troubleshooting |

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a full history of changes.

---

## License

[MIT](LICENSE) © Stijnman
