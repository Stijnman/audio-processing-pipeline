# Architecture & Design Decisions

This document explains the internal architecture of the Audio Processing Pipeline, the reasoning behind key design decisions, and the trade-offs considered during development.

---

## Pipeline Overview

The pipeline is a sequential processing chain. Each stage is independently optional and controlled by CLI flags. The output of each stage feeds directly into the next.

```
Input audio (AMR, MP3, WAV, M4A, OGG, FLAC, …)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  Stage 1: Format Conversion                                   │
│  ffmpeg → 16-bit mono WAV at 16 kHz                          │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│  Stage 2: Studio Enhancement  (--studio)                      │
│  8-stage ffmpeg chain:                                        │
│  highpass → lowpass → anlmdn → afftdn →                      │
│  agate → acompressor → speechnorm → alimiter                  │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│  Stage 3: Transcription                                       │
│  faster-whisper (CTranslate2) with built-in Silero VAD        │
│  Dynamic chunking with SequenceMatcher overlap stitching      │
│  Quantization: int8 (CPU) or float16 (GPU)                    │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│  Stage 4: LLM Post-Correction  (--post-correct)               │
│  GPT-4o-mini fixes homophones, restores punctuation,          │
│  cleans ASR artefacts. Batched in groups of 30 segments.      │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│  Stage 5: Speaker Diarization                                 │
│  Option A: LLM turn-taking (GPT-4o-mini) — default           │
│  Option B: PyAnnote 3.1 neural diarization — --diarizer pyannote │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│  Stage 6: Voice Profiling  (--profile-db)                     │
│  MFCC (40-dim, CPU) or ECAPA-TDNN (192-dim, GPU)              │
│  Incremental running-average ProfileDB (JSON)                 │
│  Cross-call speaker re-identification                         │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│  Stage 7: Speaker Name Recognition  (--name-speakers)         │
│  LLM scans transcript for direct address, introductions,      │
│  greetings. Maps SPEAKER_A/B to real names if found.          │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│  Stage 8: Per-Speaker Audio Splitting                         │
│  Mode A: Concatenated — speech only, no gaps (default)        │
│  Mode B: Keep-timing — silence where other speaker talks      │
│  ffmpeg atrim+concat (Mode A) or apad+amix (Mode B)           │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
Output:
  ├── {name}_cleaned.mp3           (studio-enhanced full call)
  ├── {name}_speaker_alice.mp3     (Alice's audio only)
  ├── {name}_speaker_bob.mp3       (Bob's audio only)
  ├── {name}_transcript.txt        (speaker-labelled transcript)
  └── {name}_diarization.json      (timestamped diarization data)
```

---

## Stage 1: Format Conversion

All input audio is normalised to a common intermediate format before any processing:
- **Format:** 16-bit signed PCM
- **Sample rate:** 16,000 Hz
- **Channels:** 1 (mono)

This is done via ffmpeg and ensures consistent behaviour regardless of the input format. AMR (used by Android phone call recorders) is particularly important — it is an 8 kHz narrowband codec that ffmpeg upsamples to 16 kHz.

---

## Stage 2: Studio Enhancement

The 8-stage ffmpeg filter chain is designed for voice clarity on phone call audio:

| Stage | Filter | Purpose |
|---|---|---|
| 1 | `highpass=f=80` | Removes sub-bass rumble (handling noise, wind, electrical hum) |
| 2 | `lowpass=f=12000` | Removes hiss above the voice frequency range |
| 3 | `anlmdn` | Non-local means broadband denoising |
| 4 | `afftdn` | FFT spectral subtraction for stationary noise (fans, hum) |
| 5 | `agate=threshold=-45dB` | Noise gate — silences track between words |
| 6 | `acompressor=threshold=-20dB` | Dynamic range compression |
| 7 | `speechnorm=p=0.25:r=0.05` | Speech-specific loudness normalisation |
| 8 | `alimiter=limit=-1dB` | True-peak limiter — prevents clipping |

All stages run in a single ffmpeg pass, so there is no intermediate file and no generational quality loss.

---

## Stage 3: Transcription

### faster-whisper

The pipeline uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper), a CTranslate2-optimised port of OpenAI Whisper. It provides:
- **int8 quantization on CPU:** 4–8× faster than standard Whisper with no meaningful accuracy loss.
- **float16 on GPU:** 8–18× faster than standard Whisper.
- **Built-in Silero VAD:** Silence is stripped before sending audio to the model, preventing hallucination loops and reducing compute.

### Dynamic chunking with overlap stitching

For files longer than `--chunk-sec` (default: 60 seconds), the audio is split into overlapping chunks. Each chunk has a `--overlap-sec` (default: 2 seconds) overlap with the next. After transcription, the overlapping segments are stitched using Python's `difflib.SequenceMatcher`:

1. The last `overlap_sec` of chunk N and the first `overlap_sec` of chunk N+1 are compared token by token.
2. The longest common subsequence is found.
3. The duplicate tokens are removed and the chunks are joined at the midpoint of the match.

This prevents words from being clipped at chunk boundaries, which is a common artefact in naive chunked transcription.

---

## Stage 4: LLM Post-Correction

The raw ASR output is sent to GPT-4o-mini in batches of 30 segments with a system prompt instructing it to:
- Fix common homophone errors (e.g. "their" vs "there" in context)
- Restore sentence-ending punctuation
- Correct technical terms and proper nouns
- Format numbers, dates, and currency consistently
- Remove filler words only if they are clearly ASR artefacts (not genuine speech)

The LLM is instructed to return only the corrected text, preserving the original segment structure. A fallback returns the original text if the API call fails or returns malformed output.

---

## Stage 5: Speaker Diarization

### LLM turn-taking (default)

The full transcript is sent to GPT-4o-mini with a system prompt that explains the task: assign `SPEAKER_A` or `SPEAKER_B` to each numbered segment based on conversational turn-taking patterns, content context, and the natural rhythm of a two-person phone call. Short affirmations ("Ja", "OK", "Mhmm") are typically the listener; longer explanatory passages are typically the main speaker.

**Strengths:** Works on any language, handles code-switching, no additional dependencies.
**Limitations:** Assumes exactly two speakers; cannot handle overlapping speech.

### PyAnnote neural diarization (`--diarizer pyannote`)

Uses [pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio), a state-of-the-art neural diarization system. It handles:
- Any number of speakers
- Overlapping speech
- Speaker changes within a single sentence

Requires a HuggingFace token (`HF_TOKEN`) to download the model weights.

---

## Stage 6: Voice Profiling

### MFCC (default)

Mel-Frequency Cepstral Coefficients. A 40-dimensional voice fingerprint derived from the spectral shape of a speaker's voice. Computed entirely in NumPy/SciPy — no GPU required, ~8 ms per segment.

### ECAPA-TDNN (`--ecapa`)

A deep neural network trained on VoxCeleb via SpeechBrain. Produces a 192-dimensional embedding that is significantly more discriminative on compressed audio. Requires `pip install speechbrain torchaudio torch` and a GPU for practical use.

### ProfileDB

The profile database is a JSON file. Each entry stores the speaker's name and their mean embedding vector, computed as an incremental running average:

```
new_mean = (old_mean × n + new_embedding) / (n + 1)
```

This means profiles improve with every new sample enrolled, without storing any raw audio. Matching uses cosine similarity with a configurable threshold (`--profile-threshold`, default: `0.75`).

---

## Stage 8: Per-Speaker Audio Splitting

### Concatenated mode (default)

For each speaker, all their segments are extracted from the audio using ffmpeg `atrim` filters and concatenated into a single output file. The result is a compact file containing only that speaker's speech, with no silence gaps.

### Keep-timing mode (`--keep-timing`)

For each speaker, their segments are kept at their original timestamps and silence is inserted where the other speaker was talking. Both output files are the same length as the original call and can be played side by side in sync. This is useful for further audio editing or alignment.

---

## Automation Architecture

### Directory watcher (`watcher.py`)

Uses the [watchdog](https://github.com/gorakhargosh/watchdog) library to monitor a directory for new audio files. When a file appears, it waits for the write to complete (file stability check), then processes it using the same pipeline as the CLI. Processed files are moved to a `processed/` subdirectory to prevent reprocessing.

On Linux (including Termux), watchdog uses inotify for zero-latency file detection. On macOS it uses FSEvents. A polling fallback is available for network filesystems.

### Celery workers (`tasks.py`)

For high-volume deployments, the pipeline can be distributed across multiple machines using Celery with a Redis broker. Each audio file is submitted as a Celery task and processed by the next available worker. The `batch_process` task accepts a list of files and submits them all in parallel.

---

## Design Decisions

### Why ffmpeg for audio processing instead of a Python library?

ffmpeg is a battle-tested, highly optimised C library with native support for every audio format used in phone calls (AMR, GSM, MP3, AAC, OGG). Python audio libraries (pydub, librosa, soundfile) are wrappers around the same underlying codecs, but with additional Python overhead. For a pipeline that processes many files, the ffmpeg CLI is faster and more reliable.

### Why not store raw audio in the voice profile database?

Storing raw audio raises significant privacy concerns, especially for phone call recordings. The MFCC/ECAPA embedding approach stores only a compact numerical vector that cannot be used to reconstruct the original audio. The profile database can be shared or backed up without exposing any voice recordings.

### Why LLM diarization instead of a dedicated diarization model?

Dedicated diarization models (pyannote, resemblyzer, ECAPA) require significant compute resources and perform poorly on compressed phone audio (GSM/AMR). The LLM approach leverages the conversational context of the transcript — which is far richer than acoustic features alone — to assign speakers. For two-speaker phone calls, it achieves comparable accuracy to neural models at a fraction of the compute cost.
