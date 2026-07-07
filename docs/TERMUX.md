# Running on Android with Termux

This guide explains how to run the Audio Processing Pipeline on an Android device using [Termux](https://termux.dev). No root access is required. The pipeline runs entirely on-device — no cloud, no Docker, no PC needed.

---

## What works on Android

| Feature | Available | Notes |
|---|---|---|
| AMR / MP3 / WAV conversion | Yes | ffmpeg is available in Termux |
| Studio audio enhancement | Yes | All 8 ffmpeg filter stages work |
| faster-whisper transcription | Yes | CPU only; `tiny` and `base` models recommended |
| Silero VAD | Yes | Built into faster-whisper |
| Dynamic chunking + stitching | Yes | |
| LLM diarization | Yes | Requires `OPENAI_API_KEY` |
| LLM post-correction | Yes | Requires `OPENAI_API_KEY` |
| Speaker name recognition | Yes | Requires `OPENAI_API_KEY` |
| MFCC voice profiling | Yes | |
| Per-speaker audio splitting | Yes | |
| Directory watcher | Yes | `watch.sh` convenience script |
| ECAPA-TDNN profiling | **No** | PyTorch not available in Termux |
| PyAnnote diarization | **No** | PyTorch not available in Termux |
| GPU acceleration | **No** | CPU only on Android |
| Celery workers | **No** | Redis not available in Termux |

---

## Prerequisites

Install the following two apps from **[F-Droid](https://f-droid.org)** (not the Play Store versions, which are outdated):

| App | Purpose |
|---|---|
| [Termux](https://f-droid.org/packages/com.termux/) | Linux terminal emulator and package manager |
| [Termux:API](https://f-droid.org/packages/com.termux.api/) | Grants Termux access to Android hardware |

> **Important:** Install both apps from F-Droid. The Play Store versions of Termux are no longer maintained and will not work correctly with modern packages.

---

## Quick Install

Open Termux and run:

```bash
# 1. Install git
pkg install git -y

# 2. Clone the repository
git clone https://github.com/Stijnman/audio-processing-pipeline.git
cd audio-processing-pipeline

# 3. Run the installer
bash termux_install.sh

# 4. Process a file
bash process.sh call.amr --studio
```

The installer takes 3–8 minutes on a modern Android device depending on your connection speed. It installs Python, ffmpeg, faster-whisper, and all required packages.

---

## Manual Installation

```bash
# Update package index
pkg update -y

# Install system packages
pkg install python ffmpeg libsndfile openssl -y

# Install Python packages
pip install faster-whisper onnxruntime scipy numpy

# Optional: LLM features
pip install openai
export OPENAI_API_KEY=sk-...

# Process a file
python AudioPipeline.py process call.amr --studio
```

---

## Usage

### Process a single file

```bash
# Basic transcription and diarization
bash process.sh call.amr

# Full pipeline: studio quality + LLM post-correction + name recognition
bash process.sh call.amr --studio --post-correct --name-speakers

# Keep original timing in per-speaker output files
bash process.sh call.amr --studio --keep-timing

# Use a larger Whisper model for better accuracy (slower)
bash process.sh call.amr --studio --model small
```

### Watch a folder automatically

```bash
# Drop audio files into ./inbox — they are processed automatically
bash watch.sh

# Custom inbox and output folders
INBOX=/sdcard/Recordings OUTPUT=/sdcard/Transcripts bash watch.sh
```

### Use the full CLI directly

```bash
python AudioPipeline.py --help
python AudioPipeline.py process --help
python AudioPipeline.py watch --help
```

---

## Accessing files from Android storage

By default, Termux can only access files in its own home directory (`~/`). To process audio files from your phone's Downloads or Recordings folder, grant Termux storage access:

```bash
termux-setup-storage
```

This creates symlinks in `~/storage/` to your Android folders:

| Symlink | Android folder |
|---|---|
| `~/storage/downloads` | Downloads |
| `~/storage/dcim` | Camera / DCIM |
| `~/storage/music` | Music |
| `~/storage/shared` | Internal storage root |

**Example — process a call recording from Downloads:**
```bash
bash process.sh ~/storage/downloads/call_recording.mp3 --studio --post-correct
```

---

## Recommended Whisper models for Android

Larger models are more accurate but slower. On a modern Android CPU (e.g. Snapdragon 8 Gen 2):

| Model | Size | Speed (realtime ×) | Accuracy | Recommendation |
|---|---|---|---|---|
| `tiny` | 75 MB | ~4–6× | Good | Best for quick processing |
| `base` | 145 MB | ~2–3× | Better | Recommended default |
| `small` | 465 MB | ~0.8–1.2× | Very good | For important recordings |
| `medium` | 1.5 GB | ~0.3× | Excellent | Only if you have time |

Set the model with `--model base` (or `tiny`, `small`, etc.).

---

## Setting your OpenAI API key

LLM features (`--post-correct`, `--name-speakers`, `--diarizer llm`) require an OpenAI API key.

**Set for the current session:**
```bash
export OPENAI_API_KEY=sk-...
```

**Set permanently (survives Termux restarts):**
```bash
echo 'export OPENAI_API_KEY=sk-...' >> ~/.bashrc
source ~/.bashrc
```

---

## Keeping processes running in the background

Android kills background processes when the screen is off. To prevent this:

**Option 1 — Wake lock (recommended):**
```bash
termux-wake-lock
bash process.sh call.amr --studio &
```

**Option 2 — Disable battery optimization:**
Go to Android **Settings → Battery → Battery Optimization**, find **Termux**, and set it to **Unrestricted**.

**Option 3 — Run in a Termux session (keeps running while Termux is open):**
```bash
# Use nohup to keep running even if the terminal is backgrounded
nohup bash process.sh call.amr --studio > processing.log 2>&1 &
tail -f processing.log   # watch progress
```

---

## Updating

```bash
cd audio-processing-pipeline
git pull
# Re-run the installer if requirements changed
bash termux_install.sh
```

---

## Troubleshooting

**`pkg: command not found`**
You are not in Termux. Make sure you opened the Termux app.

**`ffmpeg: command not found` after install**
Run `pkg install ffmpeg -y` again, then restart Termux.

**`faster_whisper` import error**
Run `pip install faster-whisper onnxruntime --upgrade`.

**Processing is very slow**
Use a smaller model: `--model tiny`. On a 5-minute call, `tiny` takes ~1 minute, `base` takes ~2–3 minutes on a mid-range Android CPU.

**`OPENAI_API_KEY` not set error**
Run `export OPENAI_API_KEY=sk-...` before processing, or add it to `~/.bashrc`.

**Out of storage space**
Whisper models are cached in `~/.cache/huggingface/`. To clear them:
```bash
rm -rf ~/.cache/huggingface/hub/
```
They will be re-downloaded on next use.

**`termux-setup-storage` permission denied**
Go to Android **Settings → Apps → Termux → Permissions** and grant **Files and media** permission, then run `termux-setup-storage` again.
