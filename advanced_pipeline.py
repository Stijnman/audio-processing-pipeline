#!/usr/bin/env python3
"""
advanced_pipeline.py  v2.0
==========================
Production-grade audio processing pipeline.

Features:
  - 8-stage studio-quality audio enhancement (ffmpeg)
  - Silero VAD pre-filter (built into faster-whisper)
  - faster-whisper transcription with int8/float16 quantization
  - Dynamic chunking with SequenceMatcher overlap stitching
  - LLM-based speaker diarization (turn-taking analysis)
  - PyAnnote 3.1 neural diarization (optional, best accuracy)
  - ECAPA-TDNN voice profiling (optional, replaces MFCC)
  - MFCC voice profiling (default, no extra deps)
  - LLM post-correction (homophones, punctuation, technical terms)
  - LLM speaker name recognition
  - Per-speaker audio splitting (concatenated or keep-timing)
  - Profile database with incremental learning

Usage:
  python advanced_pipeline.py call.amr --studio --post-correct --name-speakers
  python advanced_pipeline.py call.amr --device cuda --compute float16 --model large-v3
  python advanced_pipeline.py call.amr --diarizer pyannote --hf-token YOUR_TOKEN
  python advanced_pipeline.py call.amr --profile-db profiles.json --enroll-unknown
  python advanced_pipeline.py call.amr --studio --vad-threshold 0.4 --keep-timing

Install:
  pip install faster-whisper onnxruntime scipy openai
  # Optional GPU:
  pip install faster-whisper onnxruntime-gpu scipy openai
  # Optional PyAnnote:
  pip install pyannote.audio
  # Optional ECAPA-TDNN:
  pip install speechbrain torchaudio
"""

import argparse
import difflib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.fftpack import dct

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from pyannote.audio import Pipeline as PyannotePipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

try:
    import torch
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier
    ECAPA_AVAILABLE = True
except ImportError:
    ECAPA_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# ══════════════════════════════════════════════════════════════════════════════
# Audio Utilities
# ══════════════════════════════════════════════════════════════════════════════

def convert_to_wav(input_path: Path, output_path: Path, sample_rate: int = 16000) -> Path:
    """Convert any audio format to a 16 kHz mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ar", str(sample_rate), "-ac", "1",
        "-sample_fmt", "s16",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed:\n{result.stderr[-800:]}")
    return output_path


def studio_enhance(input_path: Path, output_path: Path) -> Path:
    """
    8-stage studio-quality audio enhancement chain:
      1. highpass=f=80       — remove sub-bass rumble
      2. lowpass=f=12000     — remove hiss above voice range
      3. anlmdn              — non-local means broadband denoising
      4. afftdn=nt=w:om=o    — FFT spectral subtraction
      5. agate               — noise gate (silence between words)
      6. acompressor         — dynamic range compression
      7. speechnorm          — speech-specific loudness normalisation
      8. alimiter            — true-peak limiter (-1 dBFS)
    """
    log.info("Applying 8-stage studio enhancement...")
    filter_chain = (
        "highpass=f=80,"
        "lowpass=f=12000,"
        "anlmdn=s=7:p=0.005:r=0.002,"
        "afftdn=nt=w:om=o,"
        "agate=threshold=-45dB:ratio=2:attack=20:release=100,"
        "acompressor=threshold=-20dB:ratio=3:attack=5:release=50,"
        "speechnorm=p=0.25:r=0.05,"
        "alimiter=level_in=1:level_out=1:limit=-1dB:attack=5:release=80:asin=0.03"
    )
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", filter_chain,
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Studio enhancement failed:\n{result.stderr[-800:]}")
    log.info("Studio enhancement complete: %s", output_path.name)
    return output_path


def vad_trim(input_path: Path, output_path: Path,
             noise_db: float = -30.0, min_silence: float = 0.5,
             pad: float = 0.15) -> Path:
    """
    Sound-activated recording: strip silence using ffmpeg silencedetect.
    Returns a gap-free WAV containing only speech segments.
    """
    log.info("Running VAD silence removal (threshold=%s dB)...", noise_db)
    # Step 1: detect silence boundaries
    detect_cmd = [
        "ffmpeg", "-i", str(input_path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
        "-f", "null", "-",
    ]
    result = subprocess.run(detect_cmd, capture_output=True, text=True)
    stderr = result.stderr

    # Parse silence_start / silence_end pairs
    import re
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", stderr)]
    ends   = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", stderr)]

    # Get total duration
    dur_match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", stderr)
    if dur_match:
        h, m, s = dur_match.groups()
        total_dur = int(h) * 3600 + int(m) * 60 + float(s)
    else:
        total_dur = 99999.0

    # Build speech intervals (inverse of silence)
    silence_regions = []
    for i, s_start in enumerate(starts):
        s_end = ends[i] if i < len(ends) else total_dur
        silence_regions.append((s_start, s_end))

    speech_intervals = []
    cursor = 0.0
    for s_start, s_end in silence_regions:
        seg_start = max(0.0, cursor - pad)
        seg_end   = s_start + pad
        if seg_end > seg_start + 0.05:
            speech_intervals.append((seg_start, seg_end))
        cursor = s_end
    if cursor < total_dur:
        speech_intervals.append((max(0.0, cursor - pad), total_dur))

    if not speech_intervals:
        log.warning("VAD found no speech — returning original audio.")
        import shutil
        shutil.copy(str(input_path), str(output_path))
        return output_path

    # Build ffmpeg concat filter
    filter_parts = []
    for i, (start, end) in enumerate(speech_intervals):
        duration = end - start
        filter_parts.append(
            f"[0:a]atrim=start={start:.4f}:duration={duration:.4f},asetpts=PTS-STARTPTS[s{i}]"
        )
    n = len(filter_parts)
    concat_inputs = "".join(f"[s{i}]" for i in range(n))
    filter_complex = ";".join(filter_parts) + f";{concat_inputs}concat=n={n}:v=0:a=1[out]"

    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.warning("VAD trim failed, using original: %s", result.stderr[-400:])
        import shutil
        shutil.copy(str(input_path), str(output_path))
    else:
        original_dur = total_dur
        trimmed_dur  = sum(e - s for s, e in speech_intervals)
        log.info("VAD: %.1fs → %.1fs (%.0f%% silence removed)",
                 original_dur, trimmed_dur,
                 100 * (1 - trimmed_dur / max(original_dur, 0.001)))
    return output_path


def export_mp3(input_wav: Path, output_mp3: Path, quality: int = 2) -> Path:
    """Export a WAV to MP3 using VBR quality (0=best, 9=worst)."""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_wav),
        "-codec:a", "libmp3lame", f"-q:a", str(quality),
        str(output_mp3),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"MP3 export failed:\n{result.stderr[-400:]}")
    return output_mp3


# ══════════════════════════════════════════════════════════════════════════════
# Transcription
# ══════════════════════════════════════════════════════════════════════════════

def auto_device() -> tuple[str, str]:
    """Automatically select the best available device and compute type."""
    try:
        import torch
        if torch.cuda.is_available():
            log.info("GPU detected — using cuda/float16")
            return "cuda", "float16"
    except ImportError:
        pass
    log.info("No GPU detected — using cpu/int8")
    return "cpu", "int8"


def stitch_chunks(base: list[dict], append: list[dict], overlap_sec: float) -> list[dict]:
    """
    Merge two overlapping transcript segment lists using SequenceMatcher.
    Finds the longest common token run in the overlap zone and splices cleanly.
    """
    if not base:
        return append
    if not append:
        return base

    # Slice the overlap window from both sides
    overlap_words = int(overlap_sec * 3)  # ~3 words/sec estimate
    slice1 = [seg["text"].strip() for seg in base[-overlap_words:]]
    slice2 = [seg["text"].strip() for seg in append[:overlap_words]]

    matcher = difflib.SequenceMatcher(None, slice1, slice2)
    match   = matcher.find_longest_match(0, len(slice1), 0, len(slice2))

    if match.size > 0:
        # Splice at the midpoint of the longest identical token run
        cut1 = len(base) - overlap_words + match.a + match.size
        cut2 = match.b + match.size
        return base[:cut1] + append[cut2:]

    return base + append


def transcribe(wav_path: Path, args) -> list[dict]:
    """
    Transcribe audio using faster-whisper with Silero VAD and dynamic chunking.
    Returns a list of {start, end, text} dicts.
    """
    if not FASTER_WHISPER_AVAILABLE:
        raise RuntimeError("faster-whisper is not installed. Run: pip install faster-whisper")

    device   = args.device
    compute  = args.compute

    if device == "auto":
        device, compute = auto_device()
        if args.compute != "auto":
            compute = args.compute  # honour explicit override

    log.info("Loading Whisper model '%s' on %s (%s)...", args.model, device, compute)
    model = WhisperModel(args.model, device=device, compute_type=compute)

    # Get audio duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(wav_path)],
        capture_output=True, text=True,
    )
    try:
        total_dur = float(probe.stdout.strip())
    except ValueError:
        total_dur = 99999.0

    chunk_sec   = args.chunk_sec
    overlap_sec = args.overlap_sec
    all_segments: list[dict] = []

    if total_dur <= chunk_sec * 1.5:
        # Short file — transcribe in one pass
        log.info("Transcribing (single pass, %.1fs)...", total_dur)
        segments, _ = model.transcribe(
            str(wav_path),
            vad_filter=True,
            vad_parameters=dict(
                threshold=args.vad_threshold,
                min_silence_duration_ms=args.vad_min_silence,
            ),
            language=args.language if args.language != "auto" else None,
        )
        all_segments = [{"start": s.start, "end": s.end, "text": s.text.strip()}
                        for s in segments]
    else:
        # Long file — dynamic chunking
        log.info("Transcribing in chunks (chunk=%.0fs, overlap=%.0fs)...",
                 chunk_sec, overlap_sec)
        cursor = 0.0
        while cursor < total_dur:
            end = min(cursor + chunk_sec, total_dur)
            log.info("  Chunk %.1fs – %.1fs", cursor, end)

            # Extract chunk to a temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            subprocess.run([
                "ffmpeg", "-y", "-i", str(wav_path),
                "-ss", str(cursor), "-t", str(end - cursor),
                "-ar", "16000", "-ac", "1", str(tmp_path),
            ], capture_output=True)

            segments, _ = model.transcribe(
                str(tmp_path),
                vad_filter=True,
                vad_parameters=dict(
                    threshold=args.vad_threshold,
                    min_silence_duration_ms=args.vad_min_silence,
                ),
                language=args.language if args.language != "auto" else None,
            )
            chunk_segs = [
                {"start": cursor + s.start, "end": cursor + s.end, "text": s.text.strip()}
                for s in segments
            ]
            tmp_path.unlink(missing_ok=True)

            all_segments = stitch_chunks(all_segments, chunk_segs, overlap_sec)
            cursor += chunk_sec - overlap_sec

    log.info("Transcription complete: %d segments", len(all_segments))
    return all_segments


# ══════════════════════════════════════════════════════════════════════════════
# Diarization
# ══════════════════════════════════════════════════════════════════════════════

def diarize_llm(segments: list[dict], openai_client) -> list[dict]:
    """
    Assign SPEAKER_A / SPEAKER_B to each segment using LLM turn-taking analysis.
    Falls back gracefully if the API is unavailable.
    """
    if not OPENAI_AVAILABLE or openai_client is None:
        log.warning("OpenAI not available — assigning alternating speakers.")
        for i, seg in enumerate(segments):
            seg["speaker"] = "SPEAKER_A" if i % 2 == 0 else "SPEAKER_B"
        return segments

    log.info("Running LLM diarization...")
    lines = [f"[{i}] [{s['start']:.1f}s–{s['end']:.1f}s] {s['text']}"
             for i, s in enumerate(segments)]
    transcript_text = "\n".join(lines)

    system_prompt = (
        "You are an expert phone call analyst. You receive a numbered, timestamped "
        "transcript of a two-person phone call. Assign each segment to SPEAKER_A or "
        "SPEAKER_B based on conversational turn-taking, content context, and natural "
        "dialogue flow.\n\n"
        "Rules:\n"
        "- Speakers alternate in a phone call; use natural turn-taking as the primary signal.\n"
        "- Short affirmations ('Yes', 'OK', 'Mhmm') are typically the listening speaker.\n"
        "- Longer explanatory segments are typically the main speaker.\n"
        "- Return ONLY a JSON array of objects: "
        '[{"index": 0, "speaker": "SPEAKER_A"}, ...]\n'
        "- Include every segment index. No extra text."
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": transcript_text},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)
        # Accept both {"assignments": [...]} and bare [...]
        assignments = data if isinstance(data, list) else data.get("assignments", [])
        label_map = {item["index"]: item["speaker"] for item in assignments}
        for i, seg in enumerate(segments):
            seg["speaker"] = label_map.get(i, "SPEAKER_A")
    except Exception as e:
        log.warning("LLM diarization error (%s) — falling back to alternating.", e)
        for i, seg in enumerate(segments):
            seg["speaker"] = "SPEAKER_A" if i % 2 == 0 else "SPEAKER_B"

    return segments


def diarize_pyannote(wav_path: Path, hf_token: str) -> list[dict]:
    """
    Neural speaker diarization using PyAnnote 3.1.
    Handles overlapping speech and any number of speakers.
    Requires: pip install pyannote.audio
    """
    if not PYANNOTE_AVAILABLE:
        raise RuntimeError(
            "pyannote.audio is not installed. Run: pip install pyannote.audio\n"
            "Also ensure you have accepted the model license at:\n"
            "  https://huggingface.co/pyannote/speaker-diarization-3.1"
        )
    log.info("Running PyAnnote 3.1 neural diarization...")
    pipeline = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    diarization = pipeline(str(wav_path))
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start":   turn.start,
            "end":     turn.end,
            "speaker": speaker,  # e.g. "SPEAKER_00", "SPEAKER_01"
            "text":    "",       # text will be merged in from transcript
        })
    log.info("PyAnnote found %d turns.", len(segments))
    return segments


def merge_transcript_with_diarization(transcript: list[dict],
                                      diar_turns: list[dict]) -> list[dict]:
    """
    Assign speaker labels from PyAnnote diarization turns to transcript segments
    by finding the diarization turn with the maximum overlap for each segment.
    """
    for seg in transcript:
        seg_start = seg["start"]
        seg_end   = seg["end"]
        best_speaker = "SPEAKER_00"
        best_overlap = 0.0
        for turn in diar_turns:
            overlap = min(seg_end, turn["end"]) - max(seg_start, turn["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
        seg["speaker"] = best_speaker
    return transcript


# ══════════════════════════════════════════════════════════════════════════════
# Voice Profiling
# ══════════════════════════════════════════════════════════════════════════════

def extract_mfcc_embedding(wav_path: Path, start: float, end: float,
                            n_mfcc: int = 40) -> np.ndarray:
    """
    Compute a 40-dimensional MFCC embedding for a speaker segment.
    Uses a proper Mel filterbank + DCT pipeline.
    """
    try:
        sample_rate, signal = wavfile.read(str(wav_path))
        if signal.ndim > 1:
            signal = signal[:, 0]
        signal = signal.astype(np.float32)

        s_idx = int(start * sample_rate)
        e_idx = int(end   * sample_rate)
        segment = signal[s_idx:e_idx]

        if len(segment) < 512:
            return np.zeros(n_mfcc)

        # Frame the signal
        frame_len  = int(0.025 * sample_rate)  # 25 ms
        frame_step = int(0.010 * sample_rate)  # 10 ms
        n_fft      = 512
        n_mels     = 40

        frames = []
        for i in range(0, len(segment) - frame_len, frame_step):
            frame = segment[i:i + frame_len] * np.hamming(frame_len)
            frames.append(frame)
        if not frames:
            return np.zeros(n_mfcc)

        # Mel filterbank
        low_freq_mel  = 0
        high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
        mel_points    = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
        hz_points     = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points    = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

        fbank = np.zeros((n_mels, n_fft // 2 + 1))
        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m       = bin_points[m]
            f_m_plus  = bin_points[m + 1]
            for k in range(f_m_minus, f_m):
                if f_m != f_m_minus:
                    fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus != f_m:
                    fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        # Compute MFCCs
        mfcc_frames = []
        for frame in frames[:200]:  # cap at 200 frames for speed
            spectrum   = np.abs(np.fft.rfft(frame, n=n_fft)) ** 2
            filter_out = np.dot(fbank, spectrum)
            log_filter = np.log(filter_out + 1e-8)
            mfcc       = dct(log_filter, type=2, norm="ortho")[:n_mfcc]
            mfcc_frames.append(mfcc)

        return np.mean(mfcc_frames, axis=0)

    except Exception as e:
        log.debug("MFCC extraction error: %s", e)
        return np.zeros(n_mfcc)


def extract_ecapa_embedding(wav_path: Path, start: float, end: float) -> np.ndarray:
    """
    Compute a 192-dimensional ECAPA-TDNN speaker embedding.
    Far more discriminative than MFCC on compressed phone audio.
    Requires: pip install speechbrain torchaudio
    """
    if not ECAPA_AVAILABLE:
        raise RuntimeError(
            "SpeechBrain is not installed. Run: pip install speechbrain torchaudio"
        )
    # Lazy-load model (cached after first call)
    if not hasattr(extract_ecapa_embedding, "_model"):
        log.info("Loading ECAPA-TDNN model (first call — downloads ~80 MB)...")
        extract_ecapa_embedding._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(Path.home() / ".cache" / "speechbrain" / "ecapa"),
        )
    model = extract_ecapa_embedding._model

    signal, sr = torchaudio.load(str(wav_path))
    s_idx = int(start * sr)
    e_idx = int(end   * sr)
    segment = signal[:, s_idx:e_idx]

    if segment.shape[1] < 160:
        return np.zeros(192)

    if sr != 16000:
        segment = torchaudio.functional.resample(segment, sr, 16000)

    with torch.no_grad():
        embedding = model.encode_batch(segment)
    return embedding.squeeze().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SpeakerProfileDB:
    """
    Persistent speaker voice profile database.
    Stores named embeddings and supports incremental learning.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.profiles: dict[str, dict] = {}
        if db_path.exists():
            with open(db_path) as f:
                raw = json.load(f)
            for name, entry in raw.items():
                self.profiles[name] = {
                    "embedding":    np.array(entry["embedding"]),
                    "sample_count": entry.get("sample_count", 1),
                    "enrolled_at":  entry.get("enrolled_at", "unknown"),
                }
            log.info("Loaded %d voice profiles from %s", len(self.profiles), db_path)

    def match(self, embedding: np.ndarray, threshold: float) -> Optional[str]:
        """Return the name of the best-matching profile, or None."""
        if not self.profiles or np.all(embedding == 0):
            return None
        best_name  = None
        best_score = 0.0
        for name, entry in self.profiles.items():
            score = cosine_similarity(embedding, entry["embedding"])
            if score > best_score:
                best_score = score
                best_name  = name
        if best_score >= threshold:
            log.debug("Profile match: %s (score=%.3f)", best_name, best_score)
            return best_name
        return None

    def enroll(self, name: str, embedding: np.ndarray) -> None:
        """Enroll a new speaker or update an existing profile with incremental averaging."""
        if np.all(embedding == 0):
            return
        if name in self.profiles:
            n = self.profiles[name]["sample_count"]
            old_emb = self.profiles[name]["embedding"]
            # Running average: new_mean = (old_mean * n + new) / (n + 1)
            self.profiles[name]["embedding"]    = (old_emb * n + embedding) / (n + 1)
            self.profiles[name]["sample_count"] = n + 1
            log.info("Updated profile '%s' (n=%d)", name, n + 1)
        else:
            from datetime import datetime
            self.profiles[name] = {
                "embedding":    embedding,
                "sample_count": 1,
                "enrolled_at":  datetime.now().isoformat(),
            }
            log.info("Enrolled new profile '%s'", name)

    def save(self) -> None:
        """Persist the database to disk."""
        serialisable = {
            name: {
                "embedding":    entry["embedding"].tolist(),
                "sample_count": entry["sample_count"],
                "enrolled_at":  entry["enrolled_at"],
            }
            for name, entry in self.profiles.items()
        }
        with open(self.db_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        log.info("Saved %d voice profiles to %s", len(self.profiles), self.db_path)


def apply_voice_profiling(segments: list[dict], wav_path: Path,
                           profile_db: SpeakerProfileDB,
                           threshold: float, use_ecapa: bool,
                           enroll_unknown: bool) -> list[dict]:
    """
    Match each speaker's aggregated embedding against the profile database.
    Optionally enroll unrecognised speakers.
    """
    # Aggregate segments by speaker label
    speaker_segments: dict[str, list[dict]] = {}
    for seg in segments:
        spk = seg.get("speaker", "SPEAKER_A")
        speaker_segments.setdefault(spk, []).append(seg)

    speaker_name_map: dict[str, str] = {}

    for spk_label, spk_segs in speaker_segments.items():
        # Build a single embedding from all segments for this speaker
        embeddings = []
        for seg in spk_segs:
            if seg["end"] - seg["start"] < 0.5:
                continue
            if use_ecapa and ECAPA_AVAILABLE:
                emb = extract_ecapa_embedding(wav_path, seg["start"], seg["end"])
            else:
                emb = extract_mfcc_embedding(wav_path, seg["start"], seg["end"])
            if not np.all(emb == 0):
                embeddings.append(emb)

        if not embeddings:
            speaker_name_map[spk_label] = spk_label
            continue

        mean_embedding = np.mean(embeddings, axis=0)
        matched_name   = profile_db.match(mean_embedding, threshold)

        if matched_name:
            speaker_name_map[spk_label] = matched_name
            # Update profile with new data (incremental learning)
            profile_db.enroll(matched_name, mean_embedding)
        else:
            speaker_name_map[spk_label] = spk_label
            if enroll_unknown:
                profile_db.enroll(spk_label, mean_embedding)

    # Apply name map to segments
    for seg in segments:
        spk = seg.get("speaker", "SPEAKER_A")
        seg["speaker"] = speaker_name_map.get(spk, spk)

    return segments


# ══════════════════════════════════════════════════════════════════════════════
# LLM Post-Processing
# ══════════════════════════════════════════════════════════════════════════════

def llm_call(client, system_prompt: str, user_content: str,
             model: str = "gpt-4o-mini", fallback=None):
    """
    Call the LLM with safe fallback. Returns parsed JSON or fallback value.
    """
    if client is None:
        return fallback
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        log.warning("LLM call failed: %s", e)
        return fallback


def post_correct(segments: list[dict], client, model: str) -> list[dict]:
    """
    LLM post-correction: fix homophones, restore punctuation, clean ASR artefacts.
    Processes in batches of 30 segments to stay within context limits.
    """
    if client is None:
        return segments

    log.info("Running LLM post-correction...")
    system_prompt = (
        "You are a professional transcript editor. Fix the following transcript segments:\n"
        "1. Correct homophone errors (e.g. 'their' vs 'there', 'to' vs 'too')\n"
        "2. Restore missing punctuation and capitalisation\n"
        "3. Fix technical terms, product names, and proper nouns\n"
        "4. Remove filler repetitions (e.g. 'I I I think' → 'I think')\n"
        "5. Preserve the original meaning and speaker labels exactly\n\n"
        "Return ONLY a JSON array with the same structure as the input. "
        "Each object must have 'start', 'end', 'text', and 'speaker' fields."
    )

    batch_size = 30
    corrected  = []

    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        raw   = llm_call(client, system_prompt, json.dumps(batch), model=model,
                         fallback=None)
        if raw is None:
            corrected.extend(batch)
            continue
        try:
            # Strip markdown code fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])
            result = json.loads(raw)
            if isinstance(result, list):
                corrected.extend(result)
            else:
                corrected.extend(batch)
        except json.JSONDecodeError:
            log.warning("Post-correction JSON parse error — keeping original batch.")
            corrected.extend(batch)

    log.info("Post-correction complete.")
    return corrected


def extract_speaker_names(segments: list[dict], client, model: str) -> dict[str, str]:
    """
    Scan the transcript for spoken names (greetings, self-introductions, direct address).
    Returns a mapping of {speaker_label: real_name}.
    """
    if client is None:
        return {}

    log.info("Scanning transcript for speaker names...")
    transcript_text = "\n".join(
        f"[{s.get('speaker', '?')}] {s['text']}" for s in segments[:80]
    )
    system_prompt = (
        "You are analysing a phone call transcript to identify real speaker names.\n"
        "Look for: greetings ('Hi, it's Alice'), self-introductions ('This is John'), "
        "direct address ('Thanks Bob'), and contextual clues.\n\n"
        "Return a JSON object mapping speaker labels to real names. "
        "Only include mappings where you are >80%% confident.\n"
        "Example: {\"SPEAKER_A\": \"Alice\", \"SPEAKER_B\": \"Bob\"}\n"
        "If no names are found, return an empty object: {}"
    )
    raw = llm_call(client, system_prompt, transcript_text, model=model, fallback="{}")
    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1])
        mapping = json.loads(raw)
        if mapping:
            log.info("Speaker names found: %s", mapping)
        else:
            log.info("No speaker names found in transcript.")
        return mapping
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# Audio Splitting
# ══════════════════════════════════════════════════════════════════════════════

def merge_intervals(intervals: list[tuple[float, float]],
                    gap: float = 0.3) -> list[tuple[float, float]]:
    """Merge adjacent intervals separated by less than `gap` seconds."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def split_audio_by_speaker(wav_path: Path, segments: list[dict],
                            output_dir: Path, keep_timing: bool,
                            fmt: str = "mp3") -> dict[str, Path]:
    """
    Split audio into per-speaker tracks.

    keep_timing=False (default):
        Concatenate only the speaker's segments — shorter, no silence.

    keep_timing=True:
        Full-length file with silence where the other speaker talks.
        Both speaker files are the same length and stay in sync.
    """
    # Group intervals by speaker
    speaker_intervals: dict[str, list[tuple[float, float]]] = {}
    for seg in segments:
        spk = seg.get("speaker", "SPEAKER_A")
        speaker_intervals.setdefault(spk, []).append((seg["start"], seg["end"]))

    output_files: dict[str, Path] = {}

    for spk, intervals in speaker_intervals.items():
        intervals = merge_intervals(intervals, gap=0.3)
        safe_name = spk.lower().replace(" ", "_").replace("/", "_")
        out_wav   = output_dir / f"speaker_{safe_name}.wav"
        out_file  = output_dir / f"speaker_{safe_name}.{fmt}"

        if keep_timing:
            # Get total duration
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(wav_path)],
                capture_output=True, text=True,
            )
            try:
                total_dur = float(probe.stdout.strip())
            except ValueError:
                total_dur = max(e for _, e in intervals) + 1.0

            # Build silence-filled track: keep speech, silence everything else
            filter_parts = []
            prev_end = 0.0
            for i, (start, end) in enumerate(intervals):
                if start > prev_end + 0.01:
                    # Silence gap
                    dur = start - prev_end
                    filter_parts.append(
                        f"[0:a]atrim=start={prev_end:.4f}:duration={dur:.4f},"
                        f"asetpts=PTS-STARTPTS,volume=0[sil{i}]"
                    )
                # Speech segment
                dur = end - start
                filter_parts.append(
                    f"[0:a]atrim=start={start:.4f}:duration={dur:.4f},"
                    f"asetpts=PTS-STARTPTS[sp{i}]"
                )
                prev_end = end
            if prev_end < total_dur - 0.01:
                dur = total_dur - prev_end
                filter_parts.append(
                    f"[0:a]atrim=start={prev_end:.4f}:duration={dur:.4f},"
                    f"asetpts=PTS-STARTPTS,volume=0[tail]"
                )

            all_labels = []
            for part in filter_parts:
                label = part.split("[")[-1].rstrip("]")
                all_labels.append(f"[{label}]")

            n = len(all_labels)
            concat_in = "".join(all_labels)
            filter_complex = ";".join(filter_parts) + f";{concat_in}concat=n={n}:v=0:a=1[out]"

        else:
            # Concatenate speech segments only
            filter_parts = []
            for i, (start, end) in enumerate(intervals):
                dur = end - start
                filter_parts.append(
                    f"[0:a]atrim=start={start:.4f}:duration={dur:.4f},"
                    f"asetpts=PTS-STARTPTS[s{i}]"
                )
            n = len(filter_parts)
            concat_in = "".join(f"[s{i}]" for i in range(n))
            filter_complex = ";".join(filter_parts) + f";{concat_in}concat=n={n}:v=0:a=1[out]"

        cmd = [
            "ffmpeg", "-y", "-i", str(wav_path),
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-ar", "16000", "-ac", "1",
            str(out_wav),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error("Speaker split failed for %s: %s", spk, result.stderr[-400:])
            continue

        if fmt == "mp3":
            export_mp3(out_wav, out_file)
            out_wav.unlink(missing_ok=True)
        else:
            out_file = out_wav

        output_files[spk] = out_file
        log.info("Speaker track: %s → %s", spk, out_file.name)

    return output_files


# ══════════════════════════════════════════════════════════════════════════════
# Output Writers
# ══════════════════════════════════════════════════════════════════════════════

def write_transcript(segments: list[dict], output_path: Path) -> None:
    """Write a human-readable speaker-labelled transcript."""
    with open(output_path, "w", encoding="utf-8") as f:
        current_speaker = None
        for seg in segments:
            spk  = seg.get("speaker", "SPEAKER_A")
            text = seg.get("text", "").strip()
            ts   = f"[{seg['start']:.1f}s – {seg['end']:.1f}s]"
            if spk != current_speaker:
                f.write(f"\n{spk}:\n")
                current_speaker = spk
            f.write(f"  {ts} {text}\n")
    log.info("Transcript written: %s", output_path.name)


def write_diarization_json(segments: list[dict], output_path: Path) -> None:
    """Write the structured diarization JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    log.info("Diarization JSON written: %s", output_path.name)


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args: argparse.Namespace) -> None:
    t_start = time.time()

    input_path = Path(args.input)
    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── OpenAI client ─────────────────────────────────────────────────────────
    openai_client = None
    if OPENAI_AVAILABLE and (args.post_correct or args.name_speakers or
                              args.diarizer == "llm"):
        try:
            openai_client = openai.OpenAI()
            log.info("OpenAI client initialised.")
        except Exception as e:
            log.warning("OpenAI init failed: %s", e)

    # ── Step 1: Convert to WAV ─────────────────────────────────────────────────
    log.info("Step 1/7: Converting to WAV...")
    raw_wav = output_dir / "raw.wav"
    convert_to_wav(input_path, raw_wav)

    # ── Step 2: Studio enhancement ────────────────────────────────────────────
    if args.studio:
        log.info("Step 2/7: Studio enhancement...")
        enhanced_wav = output_dir / "enhanced.wav"
        studio_enhance(raw_wav, enhanced_wav)
        work_wav = enhanced_wav
    else:
        work_wav = raw_wav

    # ── Step 3: VAD trim ──────────────────────────────────────────────────────
    if args.vad:
        log.info("Step 3/7: VAD silence removal...")
        vad_wav = output_dir / "vad.wav"
        vad_trim(work_wav, vad_wav,
                 noise_db=args.vad_noise_db,
                 min_silence=args.vad_min_silence_sec,
                 pad=args.vad_pad)
        vad_out = vad_wav
    else:
        vad_out = work_wav

    # Export clean full-call MP3
    clean_mp3 = output_dir / f"{input_path.stem}_cleaned.mp3"
    export_mp3(vad_out, clean_mp3)
    log.info("Clean MP3 saved: %s", clean_mp3.name)

    # ── Step 4: Transcription ─────────────────────────────────────────────────
    if args.json:
        log.info("Step 4/7: Loading pre-supplied diarization JSON...")
        with open(args.json) as f:
            raw_data = json.load(f)
        segments = raw_data if isinstance(raw_data, list) else raw_data.get("segments", [])
        # Normalise timestamps to float seconds
        for seg in segments:
            for key in ("start", "end"):
                val = seg.get(key, 0)
                if isinstance(val, str):
                    # Parse HH:MM:SS,mmm or HH:MM:SS.mmm
                    import re
                    m = re.match(r"(\d+):(\d+):([\d.,]+)", val)
                    if m:
                        h, mi, s = m.groups()
                        seg[key] = int(h) * 3600 + int(mi) * 60 + float(s.replace(",", "."))
                    else:
                        seg[key] = float(val)
    else:
        log.info("Step 4/7: Transcribing audio...")
        segments = transcribe(vad_out, args)

    # ── Step 5: Diarization ───────────────────────────────────────────────────
    if not args.json or not any("speaker" in s for s in segments):
        log.info("Step 5/7: Diarizing speakers (%s)...", args.diarizer)
        if args.diarizer == "pyannote":
            if not args.hf_token:
                log.error("--hf-token is required for PyAnnote diarization.")
                sys.exit(1)
            diar_turns = diarize_pyannote(vad_out, args.hf_token)
            segments   = merge_transcript_with_diarization(segments, diar_turns)
        else:
            segments = diarize_llm(segments, openai_client)
    else:
        log.info("Step 5/7: Speaker labels already present — skipping diarization.")

    # ── Step 6: Voice profiling ───────────────────────────────────────────────
    profile_db = SpeakerProfileDB(Path(args.profile_db))
    if len(profile_db.profiles) > 0 or args.enroll_unknown:
        log.info("Step 6/7: Voice profiling (embedder=%s)...",
                 "ecapa" if (args.ecapa and ECAPA_AVAILABLE) else "mfcc")
        segments = apply_voice_profiling(
            segments, vad_out, profile_db,
            threshold=args.profile_threshold,
            use_ecapa=(args.ecapa and ECAPA_AVAILABLE),
            enroll_unknown=args.enroll_unknown,
        )
        profile_db.save()
    else:
        log.info("Step 6/7: No profiles loaded and --enroll-unknown not set — skipping profiling.")

    # ── Step 6b: Name recognition ─────────────────────────────────────────────
    if args.name_speakers and openai_client:
        name_map = extract_speaker_names(segments, openai_client, args.post_correct_model)
        if name_map:
            for seg in segments:
                spk = seg.get("speaker", "")
                if spk in name_map:
                    seg["speaker"] = name_map[spk]
            # Also update profile DB keys if names were found
            for old_name, new_name in name_map.items():
                if old_name in profile_db.profiles and new_name not in profile_db.profiles:
                    profile_db.profiles[new_name] = profile_db.profiles.pop(old_name)
            profile_db.save()

    # ── Step 7: LLM post-correction ───────────────────────────────────────────
    if args.post_correct and openai_client:
        log.info("Step 7/7: LLM post-correction...")
        segments = post_correct(segments, openai_client, args.post_correct_model)
    else:
        log.info("Step 7/7: Skipping post-correction.")

    # ── Outputs ───────────────────────────────────────────────────────────────
    write_transcript(segments, output_dir / "transcript.txt")
    write_diarization_json(segments, output_dir / "diarization.json")

    speaker_files = split_audio_by_speaker(
        vad_out, segments, output_dir,
        keep_timing=args.keep_timing,
        fmt=args.format,
    )

    # Clean up intermediate WAVs
    if not args.keep_wav:
        for f in [raw_wav, output_dir / "enhanced.wav", output_dir / "vad.wav"]:
            f.unlink(missing_ok=True)

    elapsed = time.time() - t_start
    log.info("=" * 60)
    log.info("Pipeline complete in %.1fs", elapsed)
    log.info("Output directory: %s", output_dir)
    log.info("  Clean MP3:      %s", clean_mp3.name)
    log.info("  Transcript:     transcript.txt")
    log.info("  Diarization:    diarization.json")
    for spk, path in speaker_files.items():
        log.info("  Speaker track:  %s", path.name)
    log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Advanced audio processing pipeline v2.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input / Output
    p.add_argument("input", help="Input audio file (AMR, MP3, WAV, M4A, OGG, ...)")
    p.add_argument("--output-dir", "-o", default="output",
                   help="Output directory")
    p.add_argument("--format", choices=["mp3", "wav", "ogg", "flac"], default="mp3",
                   help="Output audio format for speaker tracks")
    p.add_argument("--keep-wav", action="store_true",
                   help="Keep intermediate WAV files")

    # Audio enhancement
    p.add_argument("--studio", action="store_true",
                   help="Apply 8-stage studio audio enhancement")
    p.add_argument("--vad", action="store_true",
                   help="Apply VAD silence removal (sound-activated recording)")
    p.add_argument("--vad-noise-db", type=float, default=-30.0,
                   help="VAD silence threshold in dB")
    p.add_argument("--vad-min-silence-sec", type=float, default=0.5,
                   help="Minimum silence duration to remove (seconds)")
    p.add_argument("--vad-pad", type=float, default=0.15,
                   help="Padding to keep around speech bursts (seconds)")

    # Transcription
    p.add_argument("--model", default="base",
                   choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                   help="Whisper model size")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="Compute device (auto detects GPU)")
    p.add_argument("--compute", default="auto",
                   choices=["auto", "int8", "float16", "int8_float16", "float32"],
                   help="Quantization type (auto selects based on device)")
    p.add_argument("--language", default="auto",
                   help="Audio language code (e.g. 'nl', 'en') or 'auto' for detection")
    p.add_argument("--chunk-sec", type=int, default=300,
                   help="Chunk size for long files (seconds)")
    p.add_argument("--overlap-sec", type=int, default=5,
                   help="Overlap between chunks for stitching (seconds)")
    p.add_argument("--vad-threshold", type=float, default=0.5,
                   help="Silero VAD speech probability threshold (0–1)")
    p.add_argument("--vad-min-silence", type=int, default=500,
                   help="Silero VAD minimum silence duration (ms)")

    # Diarization
    p.add_argument("--diarizer", choices=["llm", "pyannote"], default="llm",
                   help="Speaker diarization method")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace token (required for --diarizer pyannote)")
    p.add_argument("--json", default=None,
                   help="Pre-supplied diarization JSON (skips transcription)")

    # Voice profiling
    p.add_argument("--profile-db", default="profiles.json",
                   help="Path to voice profile database JSON")
    p.add_argument("--profile-threshold", type=float, default=0.75,
                   help="Cosine similarity threshold for speaker matching")
    p.add_argument("--enroll-unknown", action="store_true",
                   help="Enroll unrecognised speakers into the profile database")
    p.add_argument("--ecapa", action="store_true",
                   help="Use ECAPA-TDNN embeddings instead of MFCC (requires speechbrain)")

    # LLM features
    p.add_argument("--post-correct", action="store_true",
                   help="Apply LLM post-correction to the transcript")
    p.add_argument("--post-correct-model", default="gpt-4o-mini",
                   help="LLM model to use for post-correction and diarization")
    p.add_argument("--name-speakers", action="store_true",
                   help="Scan transcript for real speaker names")

    # Output options
    p.add_argument("--keep-timing", action="store_true",
                   help="Keep original timeline in speaker tracks (silence where other speaks)")

    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
