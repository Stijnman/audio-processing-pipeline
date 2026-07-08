#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║          AUDIO PROCESSING PIPELINE  v2.0  —  Single File        ║
║                                                                  ║
║  Double-click this file (or run: python AudioPipeline.py)        ║
║  It will auto-install all dependencies and show a menu.          ║
╚══════════════════════════════════════════════════════════════════╝

Features:
  • 8-stage studio audio enhancement (ffmpeg)
  • Silero VAD silence removal
  • faster-whisper transcription (int8 CPU / float16 GPU)
  • Dynamic chunking with overlap stitching for long files
  • LLM speaker diarization (GPT-4o-mini)
  • PyAnnote neural diarization (optional)
  • MFCC / ECAPA-TDNN voice profiling
  • LLM post-correction & speaker name recognition
  • Per-speaker MP3 track splitting
  • Zero-touch directory watcher mode
"""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — Bootstrap: auto-install missing packages
# ══════════════════════════════════════════════════════════════════════════════

import subprocess
import sys

REQUIRED_PACKAGES = {
    "faster_whisper": "faster-whisper",
    "onnxruntime":    "onnxruntime",
    "scipy":          "scipy",
    "numpy":          "numpy",
    "openai":         "openai",
}

def _bootstrap():
    missing = []
    for module, package in REQUIRED_PACKAGES.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print("╔══════════════════════════════════════════════════════╗")
        print("║  First run: installing required packages...          ║")
        print("╚══════════════════════════════════════════════════════╝")
        print(f"  Installing: {', '.join(missing)}\n")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing
        )
        print("\n  ✓ All packages installed. Starting pipeline...\n")

_bootstrap()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Standard imports (after bootstrap)
# ══════════════════════════════════════════════════════════════════════════════

import argparse
import difflib
import json
import logging
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
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

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

AUDIO_EXTENSIONS = {
    ".amr", ".mp3", ".wav", ".m4a", ".ogg", ".flac",
    ".aac", ".wma", ".opus", ".webm", ".mp4",
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Audio Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _ffmpeg(*args) -> subprocess.CompletedProcess:
    """Run an ffmpeg command, raise on failure."""
    result = subprocess.run(["ffmpeg", "-y"] + list(args),
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error:\n{result.stderr[-800:]}")
    return result


def convert_to_wav(src: Path, dst: Path, sr: int = 16000) -> Path:
    _ffmpeg("-i", str(src), "-ar", str(sr), "-ac", "1", "-sample_fmt", "s16", str(dst))
    return dst


def studio_enhance(src: Path, dst: Path) -> Path:
    """8-stage studio enhancement chain."""
    log.info("Applying studio enhancement...")
    chain = (
        "highpass=f=80,"
        "lowpass=f=12000,"
        "anlmdn=s=7:p=0.005:r=0.002,"
        "afftdn=nt=w:om=o,"
        "agate=threshold=-45dB:ratio=2:attack=20:release=100,"
        "acompressor=threshold=-20dB:ratio=3:attack=5:release=50,"
        "speechnorm=p=0.25:r=0.05,"
        "alimiter=level_in=1:level_out=1:limit=-1dB:attack=5:release=80:asin=0.03"
    )
    _ffmpeg("-i", str(src), "-af", chain,
            "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(dst))
    log.info("Studio enhancement complete.")
    return dst


def vad_trim(src: Path, dst: Path,
             noise_db: float = -30.0, min_silence: float = 0.5,
             pad: float = 0.15) -> Path:
    """Remove silence using ffmpeg silencedetect."""
    import re
    log.info("Running VAD silence removal...")
    result = subprocess.run(
        ["ffmpeg", "-i", str(src),
         "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    stderr = result.stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", stderr)]
    ends   = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", stderr)]
    dur_m  = re.search(r"Duration: (\d+):(\d+):([\d.]+)", stderr)
    total  = (int(dur_m.group(1)) * 3600 + int(dur_m.group(2)) * 60
              + float(dur_m.group(3))) if dur_m else 99999.0

    speech = []
    cursor = 0.0
    for i, ss in enumerate(starts):
        se = ends[i] if i < len(ends) else total
        if ss + pad > cursor - pad + 0.05:
            speech.append((max(0.0, cursor - pad), ss + pad))
        cursor = se
    if cursor < total:
        speech.append((max(0.0, cursor - pad), total))

    if not speech:
        shutil.copy(str(src), str(dst))
        return dst

    parts  = [f"[0:a]atrim=start={s:.4f}:duration={e-s:.4f},asetpts=PTS-STARTPTS[s{i}]"
              for i, (s, e) in enumerate(speech)]
    concat = "".join(f"[s{i}]" for i in range(len(parts)))
    fc     = ";".join(parts) + f";{concat}concat=n={len(parts)}:v=0:a=1[out]"

    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-filter_complex", fc,
         "-map", "[out]", "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(dst)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        shutil.copy(str(src), str(dst))
    else:
        trimmed = sum(e - s for s, e in speech)
        log.info("VAD: %.1fs → %.1fs (%.0f%% removed)", total, trimmed,
                 100 * (1 - trimmed / max(total, 0.001)))
    return dst


def export_mp3(src: Path, dst: Path, quality: int = 2) -> Path:
    _ffmpeg("-i", str(src), "-codec:a", "libmp3lame", "-q:a", str(quality), str(dst))
    return dst


def audio_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 99999.0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Transcription
# ══════════════════════════════════════════════════════════════════════════════

def auto_device() -> tuple[str, str]:
    try:
        import torch
        if torch.cuda.is_available():
            log.info("GPU detected — using cuda/float16")
            return "cuda", "float16"
    except ImportError:
        pass
    log.info("No GPU — using cpu/int8")
    return "cpu", "int8"


def stitch_chunks(base: list[dict], append: list[dict], overlap_sec: float) -> list[dict]:
    if not base:   return append
    if not append: return base
    n = int(overlap_sec * 3)
    s1 = [s["text"].strip() for s in base[-n:]]
    s2 = [s["text"].strip() for s in append[:n]]
    m  = difflib.SequenceMatcher(None, s1, s2).find_longest_match(0, len(s1), 0, len(s2))
    if m.size > 0:
        cut1 = len(base) - len(s1) + m.a + m.size
        cut2 = m.b + m.size
        return base[:cut1] + append[cut2:]
    return base + append


def transcribe(wav: Path, args) -> list[dict]:
    if not FASTER_WHISPER_AVAILABLE:
        raise RuntimeError("faster-whisper not installed. Run: pip install faster-whisper")

    device, compute = args.device, args.compute
    if device == "auto":
        device, compute = auto_device()
        if args.compute != "auto":
            compute = args.compute

    log.info("Loading Whisper '%s' on %s/%s...", args.model, device, compute)
    model    = WhisperModel(args.model, device=device, compute_type=compute)
    total    = audio_duration(wav)
    vad_kw   = dict(threshold=args.vad_threshold, min_silence_duration_ms=args.vad_min_silence)
    lang     = args.language if args.language != "auto" else None

    if total <= args.chunk_sec * 1.5:
        log.info("Transcribing (%.1fs)...", total)
        segs, _ = model.transcribe(str(wav), vad_filter=True,
                                   vad_parameters=vad_kw, language=lang)
        return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segs]

    log.info("Transcribing in chunks (%.0fs / %.0fs overlap)...",
             args.chunk_sec, args.overlap_sec)
    all_segs: list[dict] = []
    cursor = 0.0
    while cursor < total:
        end = min(cursor + args.chunk_sec, total)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        subprocess.run(["ffmpeg", "-y", "-i", str(wav),
                        "-ss", str(cursor), "-t", str(end - cursor),
                        "-ar", "16000", "-ac", "1", str(tmp_path)],
                       capture_output=True)
        segs, _ = model.transcribe(str(tmp_path), vad_filter=True,
                                   vad_parameters=vad_kw, language=lang)
        chunk = [{"start": cursor + s.start, "end": cursor + s.end,
                  "text": s.text.strip()} for s in segs]
        tmp_path.unlink(missing_ok=True)
        all_segs = stitch_chunks(all_segs, chunk, args.overlap_sec)
        cursor  += args.chunk_sec - args.overlap_sec

    log.info("Transcription complete: %d segments", len(all_segs))
    return all_segs


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Diarization
# ══════════════════════════════════════════════════════════════════════════════

def diarize_llm(segments: list[dict], client) -> list[dict]:
    if client is None:
        for i, s in enumerate(segments):
            s["speaker"] = "SPEAKER_A" if i % 2 == 0 else "SPEAKER_B"
        return segments

    log.info("Running LLM diarization...")
    lines = [f"[{i}] [{s['start']:.1f}s–{s['end']:.1f}s] {s['text']}"
             for i, s in enumerate(segments)]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "Assign each segment to SPEAKER_A or SPEAKER_B based on "
                    "turn-taking. Return ONLY JSON: "
                    '[{"index":0,"speaker":"SPEAKER_A"},...]'
                )},
                {"role": "user", "content": "\n".join(lines)},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        items = data if isinstance(data, list) else data.get("assignments", [])
        lmap  = {x["index"]: x["speaker"] for x in items}
        for i, s in enumerate(segments):
            s["speaker"] = lmap.get(i, "SPEAKER_A")
    except Exception as e:
        log.warning("LLM diarization failed (%s) — using alternating fallback.", e)
        for i, s in enumerate(segments):
            s["speaker"] = "SPEAKER_A" if i % 2 == 0 else "SPEAKER_B"
    return segments


def diarize_pyannote(wav: Path, hf_token: str) -> list[dict]:
    if not PYANNOTE_AVAILABLE:
        raise RuntimeError("pyannote.audio not installed: pip install pyannote.audio")
    log.info("Running PyAnnote 3.1 neural diarization...")
    pipe = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    diar = pipe(str(wav))
    return [{"start": t.start, "end": t.end, "speaker": spk, "text": ""}
            for t, _, spk in diar.itertracks(yield_label=True)]


def merge_diar(transcript: list[dict], turns: list[dict]) -> list[dict]:
    for seg in transcript:
        best_spk, best_ov = "SPEAKER_00", 0.0
        for t in turns:
            ov = min(seg["end"], t["end"]) - max(seg["start"], t["start"])
            if ov > best_ov:
                best_ov, best_spk = ov, t["speaker"]
        seg["speaker"] = best_spk
    return transcript


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Voice Profiling
# ══════════════════════════════════════════════════════════════════════════════

def mfcc_embedding(wav: Path, start: float, end: float, n: int = 40) -> np.ndarray:
    try:
        sr, sig = wavfile.read(str(wav))
        if sig.ndim > 1: sig = sig[:, 0]
        sig = sig.astype(np.float32)[int(start*sr):int(end*sr)]
        if len(sig) < 512: return np.zeros(n)
        fl, fs, nf, nm = int(0.025*sr), int(0.010*sr), 512, 40
        frames = [sig[i:i+fl] * np.hamming(fl)
                  for i in range(0, len(sig)-fl, fs)]
        if not frames: return np.zeros(n)
        lf, hf = 0, 2595 * np.log10(1 + (sr/2)/700)
        mp = np.linspace(lf, hf, nm+2)
        hp = (700 * (10**(mp/2595) - 1))
        bp = np.floor((nf+1) * hp / sr).astype(int)
        fb = np.zeros((nm, nf//2+1))
        for m in range(1, nm+1):
            for k in range(bp[m-1], bp[m]):
                if bp[m] != bp[m-1]: fb[m-1,k] = (k-bp[m-1])/(bp[m]-bp[m-1])
            for k in range(bp[m], bp[m+1]):
                if bp[m+1] != bp[m]: fb[m-1,k] = (bp[m+1]-k)/(bp[m+1]-bp[m])
        mfccs = []
        for f in frames[:200]:
            spec = np.abs(np.fft.rfft(f, n=nf))**2
            mfccs.append(dct(np.log(np.dot(fb, spec)+1e-8), type=2, norm="ortho")[:n])
        return np.mean(mfccs, axis=0)
    except Exception:
        return np.zeros(n)


def ecapa_embedding(wav: Path, start: float, end: float) -> np.ndarray:
    if not ECAPA_AVAILABLE: raise RuntimeError("speechbrain not installed")
    if not hasattr(ecapa_embedding, "_model"):
        log.info("Loading ECAPA-TDNN model...")
        ecapa_embedding._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(Path.home()/".cache"/"speechbrain"/"ecapa"))
    sig, sr = torchaudio.load(str(wav))
    seg = sig[:, int(start*sr):int(end*sr)]
    if seg.shape[1] < 160: return np.zeros(192)
    if sr != 16000: seg = torchaudio.functional.resample(seg, sr, 16000)
    with torch.no_grad():
        return ecapa_embedding._model.encode_batch(seg).squeeze().numpy()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0


class ProfileDB:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        if path.exists():
            raw = json.loads(path.read_text())
            for name, e in raw.items():
                self.data[name] = {"emb": np.array(e["emb"]), "n": e.get("n", 1)}
            log.info("Loaded %d voice profiles.", len(self.data))

    def match(self, emb: np.ndarray, thr: float) -> Optional[str]:
        if not self.data or np.all(emb == 0): return None
        best, score = None, 0.0
        for name, e in self.data.items():
            s = cosine_sim(emb, e["emb"])
            if s > score: score, best = s, name
        return best if score >= thr else None

    def enroll(self, name: str, emb: np.ndarray) -> None:
        if np.all(emb == 0): return
        if name in self.data:
            n = self.data[name]["n"]
            self.data[name] = {"emb": (self.data[name]["emb"]*n + emb)/(n+1), "n": n+1}
        else:
            self.data[name] = {"emb": emb, "n": 1}
        log.info("Profile '%s' updated (n=%d).", name, self.data[name]["n"])

    def save(self) -> None:
        self.path.write_text(json.dumps(
            {k: {"emb": v["emb"].tolist(), "n": v["n"]} for k, v in self.data.items()},
            indent=2))
        log.info("Saved %d profiles → %s", len(self.data), self.path.name)


def apply_profiling(segs: list[dict], wav: Path, db: ProfileDB,
                    thr: float, use_ecapa: bool, enroll: bool) -> list[dict]:
    by_spk: dict[str, list] = {}
    for s in segs:
        by_spk.setdefault(s.get("speaker", "SPEAKER_A"), []).append(s)

    name_map: dict[str, str] = {}
    for spk, ss in by_spk.items():
        embs = []
        for s in ss:
            if s["end"] - s["start"] < 0.5: continue
            e = (ecapa_embedding(wav, s["start"], s["end"])
                 if use_ecapa and ECAPA_AVAILABLE
                 else mfcc_embedding(wav, s["start"], s["end"]))
            if not np.all(e == 0): embs.append(e)
        if not embs:
            name_map[spk] = spk
            continue
        mean = np.mean(embs, axis=0)
        matched = db.match(mean, thr)
        if matched:
            name_map[spk] = matched
            db.enroll(matched, mean)
        else:
            name_map[spk] = spk
            if enroll: db.enroll(spk, mean)

    for s in segs:
        s["speaker"] = name_map.get(s.get("speaker", "SPEAKER_A"), s.get("speaker"))
    return segs


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — LLM Post-Processing
# ══════════════════════════════════════════════════════════════════════════════

def _llm(client, system: str, user: str, model: str, fallback=None):
    if client is None: return fallback
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            temperature=0.1,
        )
        return r.choices[0].message.content
    except Exception as e:
        log.warning("LLM call failed: %s", e)
        return fallback


def post_correct(segs: list[dict], client, model: str) -> list[dict]:
    if client is None: return segs
    log.info("Running LLM post-correction...")
    sys_p = (
        "Fix this transcript JSON: correct homophones, restore punctuation, "
        "fix technical terms, remove filler repetitions. "
        "Return ONLY the same JSON array structure."
    )
    corrected = []
    for i in range(0, len(segs), 30):
        batch = segs[i:i+30]
        raw   = _llm(client, sys_p, json.dumps(batch), model, fallback=None)
        if raw is None:
            corrected.extend(batch); continue
        try:
            raw = raw.strip()
            if raw.startswith("```"): raw = "\n".join(raw.split("\n")[1:-1])
            result = json.loads(raw)
            corrected.extend(result if isinstance(result, list) else batch)
        except Exception:
            corrected.extend(batch)
    return corrected


def extract_names(segs: list[dict], client, model: str) -> dict[str, str]:
    if client is None: return {}
    log.info("Scanning transcript for speaker names...")
    text = "\n".join(f"[{s.get('speaker','?')}] {s['text']}" for s in segs[:80])
    raw  = _llm(client,
                "Find real speaker names in this transcript. Return JSON mapping "
                'speaker labels to names, e.g. {"SPEAKER_A":"Alice"}. '
                "Only include mappings where confidence > 80%. Return {} if none found.",
                text, model, fallback="{}")
    try:
        raw = raw.strip()
        if raw.startswith("```"): raw = "\n".join(raw.split("\n")[1:-1])
        m = json.loads(raw)
        if m: log.info("Names found: %s", m)
        return m
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Audio Splitting
# ══════════════════════════════════════════════════════════════════════════════

def merge_intervals(ivs: list[tuple], gap: float = 0.3) -> list[tuple]:
    if not ivs: return []
    ivs = sorted(ivs)
    out = [ivs[0]]
    for s, e in ivs[1:]:
        if s - out[-1][1] <= gap: out[-1] = (out[-1][0], max(out[-1][1], e))
        else: out.append((s, e))
    return out


def split_speakers(wav: Path, segs: list[dict], out_dir: Path,
                   keep_timing: bool, fmt: str) -> dict[str, Path]:
    by_spk: dict[str, list] = {}
    for s in segs:
        by_spk.setdefault(s.get("speaker", "SPEAKER_A"), []).append(
            (s["start"], s["end"]))

    files: dict[str, Path] = {}
    for spk, ivs in by_spk.items():
        ivs     = merge_intervals(ivs)
        safe    = spk.lower().replace(" ", "_").replace("/", "_")
        out_wav = out_dir / f"speaker_{safe}.wav"
        out_f   = out_dir / f"speaker_{safe}.{fmt}"

        if keep_timing:
            total = audio_duration(wav)
            parts, prev = [], 0.0
            for i, (s, e) in enumerate(ivs):
                if s > prev + 0.01:
                    parts.append(f"[0:a]atrim=start={prev:.4f}:duration={s-prev:.4f},"
                                 f"asetpts=PTS-STARTPTS,volume=0[sil{i}]")
                parts.append(f"[0:a]atrim=start={s:.4f}:duration={e-s:.4f},"
                             f"asetpts=PTS-STARTPTS[sp{i}]")
                prev = e
            if prev < total - 0.01:
                parts.append(f"[0:a]atrim=start={prev:.4f}:duration={total-prev:.4f},"
                             f"asetpts=PTS-STARTPTS,volume=0[tail]")
            labels = ["[" + p.split("[")[-1].rstrip("]") + "]" for p in parts]
            fc = ";".join(parts) + f";{''.join(labels)}concat=n={len(labels)}:v=0:a=1[out]"
        else:
            parts = [f"[0:a]atrim=start={s:.4f}:duration={e-s:.4f},"
                     f"asetpts=PTS-STARTPTS[s{i}]"
                     for i, (s, e) in enumerate(ivs)]
            fc = ";".join(parts) + f";{''.join(f'[s{i}]' for i in range(len(parts)))}concat=n={len(parts)}:v=0:a=1[out]"

        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav), "-filter_complex", fc,
             "-map", "[out]", "-ar", "16000", "-ac", "1", str(out_wav)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            log.error("Split failed for %s", spk); continue

        if fmt == "mp3":
            export_mp3(out_wav, out_f)
            out_wav.unlink(missing_ok=True)
        else:
            out_f = out_wav

        files[spk] = out_f
        log.info("Speaker track: %s → %s", spk, out_f.name)

    return files


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args: argparse.Namespace) -> None:
    t0 = time.time()
    inp = Path(args.input)
    if not inp.exists():
        log.error("File not found: %s", inp); sys.exit(1)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # OpenAI client
    client = None
    if OPENAI_AVAILABLE and (args.post_correct or args.name_speakers
                              or args.diarizer == "llm"):
        try:
            client = openai.OpenAI()
        except Exception as e:
            log.warning("OpenAI init failed: %s", e)

    # Step 1: Convert
    log.info("━━ Step 1/7  Convert to WAV")
    raw = out / "raw.wav"
    convert_to_wav(inp, raw)

    # Step 2: Studio
    if args.studio:
        log.info("━━ Step 2/7  Studio enhancement")
        enh = out / "enhanced.wav"
        studio_enhance(raw, enh)
        work = enh
    else:
        work = raw

    # Step 3: VAD
    if args.vad:
        log.info("━━ Step 3/7  VAD silence removal")
        vad = out / "vad.wav"
        vad_trim(work, vad, noise_db=args.vad_noise_db,
                 min_silence=args.vad_min_silence_sec, pad=args.vad_pad)
        work = vad

    # Export clean MP3
    clean = out / f"{inp.stem}_cleaned.mp3"
    export_mp3(work, clean)
    log.info("Clean MP3: %s", clean.name)

    # Step 4: Transcribe
    if args.json:
        log.info("━━ Step 4/7  Loading pre-supplied JSON")
        raw_data = json.loads(Path(args.json).read_text())
        segs = raw_data if isinstance(raw_data, list) else raw_data.get("segments", [])
        import re
        for s in segs:
            for k in ("start", "end"):
                v = s.get(k, 0)
                if isinstance(v, str):
                    m = re.match(r"(\d+):(\d+):([\d.,]+)", v)
                    s[k] = (int(m.group(1))*3600 + int(m.group(2))*60
                            + float(m.group(3).replace(",", "."))) if m else float(v)
    else:
        log.info("━━ Step 4/7  Transcribing")
        segs = transcribe(work, args)

    # Step 5: Diarize
    if not args.json or not any("speaker" in s for s in segs):
        log.info("━━ Step 5/7  Diarizing (%s)", args.diarizer)
        if args.diarizer == "pyannote":
            if not args.hf_token:
                log.error("--hf-token required for PyAnnote"); sys.exit(1)
            turns = diarize_pyannote(work, args.hf_token)
            segs  = merge_diar(segs, turns)
        else:
            segs = diarize_llm(segs, client)
    else:
        log.info("━━ Step 5/7  Speaker labels present — skipping diarization")

    # Step 6: Voice profiling
    db = ProfileDB(Path(args.profile_db))
    if db.data or args.enroll_unknown:
        log.info("━━ Step 6/7  Voice profiling")
        segs = apply_profiling(segs, work, db, args.profile_threshold,
                               args.ecapa and ECAPA_AVAILABLE, args.enroll_unknown)
        db.save()
    else:
        log.info("━━ Step 6/7  No profiles — skipping profiling")

    # Step 6b: Name recognition
    if args.name_speakers and client:
        nm = extract_names(segs, client, args.post_correct_model)
        if nm:
            for s in segs:
                s["speaker"] = nm.get(s.get("speaker", ""), s.get("speaker", ""))
            for old, new in nm.items():
                if old in db.data and new not in db.data:
                    db.data[new] = db.data.pop(old)
            db.save()

    # Step 7: Post-correct
    if args.post_correct and client:
        log.info("━━ Step 7/7  LLM post-correction")
        segs = post_correct(segs, client, args.post_correct_model)
    else:
        log.info("━━ Step 7/7  Skipping post-correction")

    # Write outputs
    transcript_path = out / "transcript.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        cur = None
        for s in segs:
            spk = s.get("speaker", "SPEAKER_A")
            if spk != cur:
                f.write(f"\n{spk}:\n"); cur = spk
            f.write(f"  [{s['start']:.1f}s – {s['end']:.1f}s] {s.get('text','').strip()}\n")

    (out / "diarization.json").write_text(
        json.dumps(segs, indent=2, ensure_ascii=False))

    speaker_files = split_speakers(work, segs, out, args.keep_timing, args.format)

    if not args.keep_wav:
        for f in [out/"raw.wav", out/"enhanced.wav", out/"vad.wav"]:
            f.unlink(missing_ok=True)

    elapsed = time.time() - t0
    print("\n" + "━"*60)
    print(f"  ✓  Pipeline complete in {elapsed:.1f}s")
    print(f"  📁 Output: {out}")
    print(f"  🎵 Clean MP3:   {clean.name}")
    print("  📝 Transcript:  transcript.txt")
    print("  📊 Diarization: diarization.json")
    for spk, p in speaker_files.items():
        print(f"  🎤 {spk}:  {p.name}")
    print("━"*60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Directory Watcher
# ══════════════════════════════════════════════════════════════════════════════

def _is_stable(path: Path, wait: float = 1.0) -> bool:
    try:
        s1 = path.stat().st_size
        time.sleep(wait)
        return path.stat().st_size == s1 and s1 > 0
    except OSError:
        return False


def _process_one(f: Path, args: argparse.Namespace) -> bool:
    sub = argparse.Namespace(**vars(args))
    sub.input      = str(f)
    sub.output_dir = str(Path(args.output_dir) / f.stem)
    sub.profile_db = str(Path(args.output_dir) / "profiles.json")
    try:
        run_pipeline(sub)
        return True
    except Exception as e:
        log.error("Pipeline error on %s: %s", f.name, e)
        return False


def run_watcher(args: argparse.Namespace) -> None:
    inbox     = Path(args.inbox)
    processed = Path(args.processed)
    inbox.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'━'*60}")
    print(f"  👁  Watching: {inbox}")
    print(f"  📁 Output:   {args.output_dir}")
    print(f"  ✅ Done:     {processed}")
    print("  Press Ctrl+C to stop")
    print(f"{'━'*60}\n")

    executor = ThreadPoolExecutor(max_workers=args.workers)

    def handle(f: Path):
        if not _is_stable(f): return
        log.info("New file: %s", f.name)
        ok = _process_one(f, args)
        dest = processed / ("failed" if not ok else "") / f.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(dest))

    # Process existing files first
    for f in sorted(inbox.iterdir()):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
            executor.submit(handle, f)

    if WATCHDOG_AVAILABLE and not args.poll:
        class Handler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    p = Path(event.src_path)
                    if p.suffix.lower() in AUDIO_EXTENSIONS:
                        time.sleep(2)
                        executor.submit(handle, p)

        obs = Observer()
        obs.schedule(Handler(), str(inbox), recursive=False)
        obs.start()
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            obs.stop()
        obs.join()
    else:
        try:
            while True:
                for f in sorted(inbox.iterdir()):
                    if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
                        executor.submit(handle, f)
                time.sleep(args.poll_interval)
        except KeyboardInterrupt:
            pass

    executor.shutdown(wait=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Interactive Menu (double-click mode)
# ══════════════════════════════════════════════════════════════════════════════

def interactive_menu() -> None:
    """Simple terminal menu shown when the script is double-clicked."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          AUDIO PROCESSING PIPELINE  v2.0                        ║
╠══════════════════════════════════════════════════════════════════╣
║  1.  Process a single audio file                                 ║
║  2.  Watch a folder (auto-process new files)                     ║
║  3.  Process a single file — GPU mode (large-v3 / float16)       ║
║  4.  Process a single file — full quality (studio + LLM)         ║
║  5.  Enroll speakers into voice profile database                 ║
║  6.  Exit                                                        ║
╚══════════════════════════════════════════════════════════════════╝
""")

    choice = input("  Select option [1-6]: ").strip()

    if choice == "6":
        sys.exit(0)

    if choice in ("1", "3", "4", "5"):
        audio = input("  Audio file path: ").strip().strip('"').strip("'")
        if not audio:
            print("  No file specified."); return
        out = input(f"  Output folder [output/{Path(audio).stem}]: ").strip()
        if not out:
            out = f"output/{Path(audio).stem}"

        args = _default_args()
        args.input      = audio
        args.output_dir = out

        if choice == "3":
            args.device  = "cuda"
            args.compute = "float16"
            args.model   = "large-v3"
            args.studio  = True
        elif choice == "4":
            args.studio        = True
            args.post_correct  = True
            args.name_speakers = True
        elif choice == "5":
            args.studio         = True
            args.enroll_unknown = True
            db_path = input("  Profile DB path [profiles.json]: ").strip()
            args.profile_db = db_path or "profiles.json"

        run_pipeline(args)

    elif choice == "2":
        inbox = input("  Inbox folder to watch [inbox]: ").strip() or "inbox"
        out   = input("  Output folder [output]: ").strip() or "output"
        proc  = input("  Processed folder [processed]: ").strip() or "processed"

        args = _default_args()
        args.inbox      = inbox
        args.output_dir = out
        args.processed  = proc
        args.studio     = True
        args.post_correct  = True
        args.name_speakers = True
        args.workers    = 2
        args.poll       = False
        args.poll_interval = 5.0

        run_watcher(args)

    else:
        print("  Invalid option.")

    input("\n  Press Enter to exit...")


def _default_args() -> argparse.Namespace:
    return argparse.Namespace(
        input="",
        output_dir="output",
        format="mp3",
        keep_wav=False,
        studio=False,
        vad=False,
        vad_noise_db=-30.0,
        vad_min_silence_sec=0.5,
        vad_pad=0.15,
        model="base",
        device="auto",
        compute="auto",
        language="auto",
        chunk_sec=300,
        overlap_sec=5,
        vad_threshold=0.5,
        vad_min_silence=500,
        diarizer="llm",
        hf_token=None,
        json=None,
        profile_db="profiles.json",
        profile_threshold=0.75,
        enroll_unknown=False,
        ecapa=False,
        post_correct=False,
        post_correct_model="gpt-4o-mini",
        name_speakers=False,
        keep_timing=False,
        # watcher
        inbox="inbox",
        processed="processed",
        workers=2,
        poll=False,
        poll_interval=5.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Audio Processing Pipeline v2.0 — single file edition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    # ── process ──────────────────────────────────────────────────────────────
    proc = sub.add_parser("process", help="Process a single audio file")
    proc.add_argument("input")
    proc.add_argument("--output-dir",  "-o", default="output")
    proc.add_argument("--format",      choices=["mp3","wav","ogg","flac"], default="mp3")
    proc.add_argument("--keep-wav",    action="store_true")
    proc.add_argument("--studio",      action="store_true")
    proc.add_argument("--vad",         action="store_true")
    proc.add_argument("--vad-noise-db",      type=float, default=-30.0)
    proc.add_argument("--vad-min-silence-sec", type=float, default=0.5)
    proc.add_argument("--vad-pad",     type=float, default=0.15)
    proc.add_argument("--model",       default="base",
                      choices=["tiny","base","small","medium","large-v2","large-v3"])
    proc.add_argument("--device",      default="auto", choices=["auto","cpu","cuda"])
    proc.add_argument("--compute",     default="auto",
                      choices=["auto","int8","float16","int8_float16","float32"])
    proc.add_argument("--language",    default="auto")
    proc.add_argument("--chunk-sec",   type=int, default=300)
    proc.add_argument("--overlap-sec", type=int, default=5)
    proc.add_argument("--vad-threshold",   type=float, default=0.5)
    proc.add_argument("--vad-min-silence", type=int,   default=500)
    proc.add_argument("--diarizer",    default="llm", choices=["llm","pyannote"])
    proc.add_argument("--hf-token",    default=None)
    proc.add_argument("--json",        default=None)
    proc.add_argument("--profile-db",  default="profiles.json")
    proc.add_argument("--profile-threshold", type=float, default=0.75)
    proc.add_argument("--enroll-unknown",    action="store_true")
    proc.add_argument("--ecapa",       action="store_true")
    proc.add_argument("--post-correct",      action="store_true")
    proc.add_argument("--post-correct-model", default="gpt-4o-mini")
    proc.add_argument("--name-speakers",     action="store_true")
    proc.add_argument("--keep-timing",       action="store_true")

    # ── watch ─────────────────────────────────────────────────────────────────
    watch = sub.add_parser("watch", help="Watch a folder for new audio files")
    watch.add_argument("--inbox",      default="inbox")
    watch.add_argument("--output-dir", default="output")
    watch.add_argument("--processed",  default="processed")
    watch.add_argument("--workers",    type=int, default=2)
    watch.add_argument("--poll",       action="store_true")
    watch.add_argument("--poll-interval", type=float, default=5.0, dest="poll_interval")
    watch.add_argument("--studio",     action="store_true")
    watch.add_argument("--vad",        action="store_true")
    watch.add_argument("--post-correct",  action="store_true")
    watch.add_argument("--name-speakers", action="store_true")
    watch.add_argument("--keep-timing",   action="store_true")
    watch.add_argument("--enroll-unknown", action="store_true")
    watch.add_argument("--ecapa",         action="store_true")
    watch.add_argument("--model",         default="base")
    watch.add_argument("--device",        default="auto")
    watch.add_argument("--compute",       default="auto")
    watch.add_argument("--diarizer",      default="llm", choices=["llm","pyannote"])
    watch.add_argument("--hf-token",      default=None)
    watch.add_argument("--format",        default="mp3")
    watch.add_argument("--profile-db",    default="profiles.json")
    watch.add_argument("--profile-threshold", type=float, default=0.75)
    watch.add_argument("--post-correct-model", default="gpt-4o-mini")
    # shared defaults needed by run_pipeline when called from watcher
    watch.add_argument("--vad-noise-db",       type=float, default=-30.0)
    watch.add_argument("--vad-min-silence-sec", type=float, default=0.5)
    watch.add_argument("--vad-pad",            type=float, default=0.15)
    watch.add_argument("--chunk-sec",          type=int,   default=300)
    watch.add_argument("--overlap-sec",        type=int,   default=5)
    watch.add_argument("--vad-threshold",      type=float, default=0.5)
    watch.add_argument("--vad-min-silence",    type=int,   default=500)
    watch.add_argument("--language",           default="auto")
    watch.add_argument("--json",               default=None)
    watch.add_argument("--keep-wav",           action="store_true")

    return p


def main() -> None:
    # If no arguments are given (double-click), show the interactive menu
    if len(sys.argv) == 1:
        interactive_menu()
        return

    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "process":
        run_pipeline(args)
    elif args.command == "watch":
        run_watcher(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
