# Audio Processing Pipeline  v2.0

Production-grade audio processing pipeline for phone calls and recordings. Converts, cleans, transcribes, diarizes, and splits audio by speaker — automatically.

---

## Features

| Feature | Description | Flag |
|---|---|---|
| Studio enhancement | 8-stage ffmpeg chain (highpass, anlmdn, afftdn, agate, acompressor, speechnorm, alimiter) | `--studio` |
| Silero VAD | Silence removal before transcription (built into faster-whisper) | `--vad-threshold` |
| faster-whisper | CTranslate2-optimised Whisper with int8/float16 quantization | `--model`, `--compute` |
| Dynamic chunking | Rolling chunks with SequenceMatcher overlap stitching for long files | `--chunk-sec`, `--overlap-sec` |
| LLM diarization | Turn-taking analysis via GPT-4o-mini | `--diarizer llm` |
| PyAnnote diarization | Neural diarization — handles overlap and any speaker count | `--diarizer pyannote` |
| MFCC voice profiling | 40-dim MFCC embeddings for cross-call speaker re-identification | `--profile-db` |
| ECAPA-TDNN profiling | 192-dim neural embeddings (far more accurate on phone audio) | `--ecapa` |
| LLM post-correction | Fix homophones, restore punctuation, clean ASR artefacts | `--post-correct` |
| Speaker name recognition | Scan transcript for real names (greetings, introductions) | `--name-speakers` |
| Per-speaker splitting | Concatenated or keep-timing (sync'd silence) speaker tracks | `--keep-timing` |
| Directory watcher | Zero-touch automation — process files the moment they land | `watcher.py` |
| Celery workers | Distributed parallel processing across multiple machines | `tasks.py` |
| Docker | Portable, offline-capable containerised deployment | `Dockerfile` |

---

## Quick Start

### 1. Install dependencies

```bash
pip install faster-whisper onnxruntime scipy openai watchdog celery redis
```

### 2. Process a single call

```bash
# Basic: transcribe + diarize
python advanced_pipeline.py call.amr

# Full pipeline: studio quality + post-correction + name recognition
python advanced_pipeline.py call.amr \
  --studio --post-correct --name-speakers

# GPU acceleration (large-v3 model)
python advanced_pipeline.py call.amr \
  --device cuda --compute float16 --model large-v3 \
  --studio --post-correct
```

### 3. Automate with the directory watcher

```bash
# Drop audio files into ./inbox — they are processed automatically
python watcher.py --inbox ./inbox --output ./output --studio --post-correct
```

### 4. Deploy with Docker

```bash
# Build image
docker build -t audio-pipeline .

# Run (CPU)
docker run -d \
  -v ./inbox:/inbox \
  -v ./output:/output \
  -v ./processed:/processed \
  -e OPENAI_API_KEY=sk-... \
  audio-pipeline

# Full stack with Celery + monitoring UI
docker compose up -d
# Flower UI: http://localhost:5555
```

---

## File Reference

| File | Purpose |
|---|---|
| `advanced_pipeline.py` | Core pipeline — all processing logic |
| `watcher.py` | Directory watcher for zero-touch automation |
| `tasks.py` | Celery task definitions for distributed processing |
| `Dockerfile` | Container image definition |
| `docker-compose.yml` | Full stack: Redis + watcher + workers + Flower UI |
| `requirements.txt` | Python dependencies |

---

## CLI Reference: `advanced_pipeline.py`

```
usage: advanced_pipeline.py [-h] [--output-dir DIR] [--format {mp3,wav,ogg,flac}]
                             [--studio] [--vad] [--vad-noise-db DB]
                             [--model {tiny,base,small,medium,large-v2,large-v3}]
                             [--device {auto,cpu,cuda}]
                             [--compute {auto,int8,float16,int8_float16,float32}]
                             [--language LANG] [--chunk-sec N] [--overlap-sec N]
                             [--diarizer {llm,pyannote}] [--hf-token TOKEN]
                             [--json FILE] [--profile-db FILE]
                             [--profile-threshold FLOAT] [--enroll-unknown]
                             [--ecapa] [--post-correct] [--name-speakers]
                             [--keep-timing]
                             input
```

### Key flag groups

**Audio enhancement**
- `--studio` — Apply all 8 enhancement stages
- `--vad` — Remove silence (sound-activated recording)
- `--vad-noise-db -30` — Silence threshold in dB (lower = more aggressive)

**Transcription**
- `--model base` — Whisper model size (tiny/base/small/medium/large-v2/large-v3)
- `--device auto` — Device: auto (detects GPU), cpu, cuda
- `--compute auto` — Quantization: auto, int8 (CPU), float16 (GPU), int8_float16

**Diarization**
- `--diarizer llm` — LLM turn-taking (default, no extra deps)
- `--diarizer pyannote --hf-token TOKEN` — Neural diarization (best accuracy)

**Voice profiling**
- `--profile-db profiles.json` — Path to voice profile database
- `--enroll-unknown` — Enroll new speakers automatically
- `--ecapa` — Use ECAPA-TDNN embeddings (requires speechbrain)
- `--profile-threshold 0.75` — Cosine similarity threshold for matching

**LLM features** (require `OPENAI_API_KEY`)
- `--post-correct` — Fix homophones, punctuation, ASR artefacts
- `--name-speakers` — Detect real names from transcript context

**Output**
- `--keep-timing` — Full-length sync'd tracks (silence where other speaker talks)
- `--format mp3` — Output format: mp3, wav, ogg, flac

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

The profile database (`profiles.json`) stores named speaker embeddings. It grows incrementally — each time a known speaker is matched, their profile is updated with the new embedding (running average).

```bash
# First call: enroll speakers
python advanced_pipeline.py call1.amr --profile-db profiles.json --enroll-unknown

# Edit profiles.json to rename SPEAKER_A → "Alice", SPEAKER_B → "Bob"

# Subsequent calls: automatic name matching
python advanced_pipeline.py call2.amr --profile-db profiles.json
# Output: speaker_alice.mp3, speaker_bob.mp3
```

**Note on phone audio:** GSM/AMR compression (8 kHz, heavy quantization) strips spectral detail that distinguishes voices. MFCC embeddings work well for re-identifying the same speaker across different calls, but may not reliably separate two unknown speakers from the same compressed call. Use `--ecapa` (ECAPA-TDNN) for significantly better accuracy, or `--diarizer pyannote` for best-in-class blind diarization.

---

## Distributed Processing with Celery

```bash
# 1. Start Redis
docker run -d -p 6379:6379 redis:alpine

# 2. Start workers
celery -A tasks worker --loglevel=info --concurrency=4

# 3. Submit jobs
python tasks.py /inbox/call.amr --studio --post-correct --wait

# 4. Monitor
celery -A tasks flower   # http://localhost:5555
```

**When to use Celery vs the watcher:**

| Volume | Approach |
|---|---|
| < 50 calls/day | `watcher.py` |
| 50–500 calls/day | Celery with 4–8 workers |
| 500+ calls/day | Celery + multiple machines + GPU workers |

---

## Optional: PyAnnote Neural Diarization

```bash
pip install pyannote.audio

# Accept model license at:
# https://huggingface.co/pyannote/speaker-diarization-3.1

python advanced_pipeline.py call.amr \
  --diarizer pyannote \
  --hf-token YOUR_HUGGINGFACE_TOKEN
```

PyAnnote handles overlapping speech and any number of speakers — the LLM heuristic cannot. It is the recommended upgrade for meeting recordings and multi-party calls.

---

## Optional: ECAPA-TDNN Voice Profiling

```bash
pip install speechbrain torchaudio torch

python advanced_pipeline.py call.amr \
  --profile-db profiles.json \
  --ecapa \
  --enroll-unknown
```

ECAPA-TDNN produces 192-dimensional speaker embeddings trained on VoxCeleb — significantly more discriminative than MFCC on compressed phone audio.

---

## Environment Variables

| Variable | Required for |
|---|---|
| `OPENAI_API_KEY` | `--post-correct`, `--name-speakers`, `--diarizer llm` |
| `HF_TOKEN` | `--diarizer pyannote` |
| `CELERY_BROKER_URL` | Celery workers (default: `redis://localhost:6379/0`) |
| `CELERY_RESULT_URL` | Celery result backend (default: `redis://localhost:6379/1`) |
