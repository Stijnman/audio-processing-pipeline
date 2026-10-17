import json
import os
import sys
import types

# Add parent directory to sys.path to import AudioPipeline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AudioPipeline import (
    stitch_chunks,
    speech_intervals_from_silence,
    diarize_llm,
    build_parser,
    _claim,
    _release,
    _in_flight,
)
import AudioPipeline

# ── stitch_chunks ─────────────────────────────────────────────────────────────

def test_stitch_chunks_empty_base():
    """Passing an empty base returns append unchanged."""
    append = [{"start": 0, "end": 2, "text": "hello"}]
    assert stitch_chunks([], append, 1.0) == append


def test_stitch_chunks_empty_append():
    """Passing an empty append returns base unchanged."""
    base = [{"start": 0, "end": 2, "text": "hello"}]
    assert stitch_chunks(base, [], 1.0) == base


def test_stitch_chunks_overlap_deduplication():
    """
    When overlap_words (int(overlap_sec * 3)) is larger than the actual number
    of segments in base, the window still deduplicates the re-transcribed
    tail and keeps the new content.
    """
    base = [
        {"start": 0, "end": 1, "text": "this"},
        {"start": 1, "end": 2, "text": "is"},
        {"start": 2, "end": 3, "text": "a"},
        {"start": 3, "end": 4, "text": "test"},
    ]
    append = [
        {"start": 2.5, "end": 3.5, "text": "a"},
        {"start": 3.5, "end": 4.5, "text": "test"},
        {"start": 4.5, "end": 5.5, "text": "of"},
        {"start": 5.5, "end": 6.5, "text": "stitching"},
    ]
    result = stitch_chunks(base, append, overlap_sec=2.0)
    texts = [seg["text"] for seg in result]
    assert texts == ["this", "is", "a", "test", "of", "stitching"]


def test_stitch_chunks_no_overlap():
    """When there is no common token run, the chunks are simply concatenated."""
    base   = [{"start": 0, "end": 1, "text": "alpha"}]
    append = [{"start": 1, "end": 2, "text": "beta"}]
    result = stitch_chunks(base, append, overlap_sec=1.0)
    texts  = [seg["text"] for seg in result]
    assert texts == ["alpha", "beta"]


def test_stitch_chunks_exact_overlap_window():
    """
    When overlap_words exactly equals the number of segments in base,
    the fix still produces the correct deduplicated result.
    """
    base = [
        {"start": 0, "end": 1, "text": "one"},
        {"start": 1, "end": 2, "text": "two"},
        {"start": 2, "end": 3, "text": "three"},
    ]
    append = [
        {"start": 2.5, "end": 3.5, "text": "two"},
        {"start": 3.5, "end": 4.5, "text": "three"},
        {"start": 4.5, "end": 5.5, "text": "four"},
    ]
    result = stitch_chunks(base, append, overlap_sec=1.0)
    texts  = [seg["text"] for seg in result]
    assert texts == ["one", "two", "three", "four"]

# ── stitch_chunks: chunk-boundary regressions ─────────────────────────────────

def test_stitch_chunks_dedup_real_chunk_boundary():
    """
    Realistic chunk boundary: chunk N's tail (you, bye at 295-300s) is
    re-transcribed at the head of chunk N+1 (which starts at 295s).
    The duplicated run must be removed exactly once and new content kept.
    """
    base = [
        {"start": 280.0, "end": 290.0, "text": "hello there"},
        {"start": 295.0, "end": 298.0, "text": "you"},
        {"start": 298.0, "end": 300.0, "text": "bye"},
    ]
    append = [
        {"start": 295.0, "end": 298.0, "text": "you"},
        {"start": 298.0, "end": 300.0, "text": "bye"},
        {"start": 300.0, "end": 305.0, "text": "next topic"},
    ]
    result = stitch_chunks(base, append, overlap_sec=5.0)
    texts  = [seg["text"] for seg in result]
    assert texts == ["hello there", "you", "bye", "next topic"]


def test_stitch_chunks_no_data_loss_on_ambiguous_repetition():
    """
    Regression: a repeated token early in the base overlap window must not
    cause real base segments after it to be silently dropped. Chunk 2's
    head re-transcribed only 'you'; the base's 'thank you.' is unique
    content and must survive even though 'you' matches mid-window.
    """
    base = [
        {"start": 0.0,   "end": 5.0,   "text": "so about the contract"},
        {"start": 295.0, "end": 298.0, "text": "you"},
        {"start": 298.0, "end": 300.0, "text": "thank you."},
    ]
    append = [
        {"start": 295.0, "end": 298.0, "text": "you"},
        {"start": 298.0, "end": 300.0, "text": "okay next topic"},
    ]
    result = stitch_chunks(base, append, overlap_sec=5.0)
    texts  = [seg["text"] for seg in result]
    assert "thank you." in texts
    assert "okay next topic" in texts
    # Nothing from the base may disappear.
    for seg in base:
        assert seg["text"] in texts


def test_stitch_chunks_base_outside_zone_never_cut():
    """
    Base segments that end before the next chunk starts are outside the
    duplicated audio and must never be removed, regardless of text matches.
    """
    base = [
        {"start": 0.0, "end": 10.0, "text": "alpha"},
        {"start": 10.0, "end": 20.0, "text": "beta"},
    ]
    append = [
        {"start": 295.0, "end": 296.0, "text": "alpha"},
        {"start": 296.0, "end": 297.0, "text": "gamma"},
    ]
    result = stitch_chunks(base, append, overlap_sec=5.0)
    texts  = [seg["text"] for seg in result]
    assert texts == ["alpha", "beta", "alpha", "gamma"]

# ── speech_intervals_from_silence ─────────────────────────────────────────────

def test_speech_intervals_basic():
    silence = [(10.0, 15.0), (30.0, 34.0)]
    result  = speech_intervals_from_silence(silence, total_dur=180.0, pad=0.15)
    assert result == [(0.0, 10.15), (14.85, 30.15), (33.85, 180.0)]


def test_speech_intervals_leading_silence():
    """
    Regression: silence at t=0 has no speech before it — the old code kept
    a pad-length sliver of pure silence at the start of the trimmed audio.
    """
    silence = [(0.0, 12.0), (60.0, 65.0)]
    result  = speech_intervals_from_silence(silence, total_dur=180.0, pad=0.15)
    assert result[0] == (11.85, 60.15)
    assert all(start > 0.0 for start, _ in result[:1])
    assert result == [(11.85, 60.15), (64.85, 180.0)]


def test_speech_intervals_no_silence():
    result = speech_intervals_from_silence([], total_dur=120.0, pad=0.15)
    assert result == [(0.0, 120.0)]

# ── diarize_llm ────────────────────────────────────────────────────────────────

class _FakeCompletions:
    def __init__(self, parent):
        self.parent = parent

    def create(self, **kwargs):
        self.parent.calls.append(kwargs)
        if self.parent.raise_exc is not None:
            raise self.parent.raise_exc
        payload = self.parent.payload
        content = payload if isinstance(payload, str) else json.dumps(payload)
        message = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


class _FakeClient:
    def __init__(self, payload=None, raise_exc=None):
        self.payload   = payload
        self.raise_exc = raise_exc
        self.calls     = []
        self.chat = types.SimpleNamespace(completions=_FakeCompletions(self))


def _segments(n=6):
    return [{"start": float(i), "end": float(i) + 1, "text": f"seg {i}"}
            for i in range(n)]


def test_diarize_llm_no_client_falls_back(monkeypatch):
    monkeypatch.setattr(AudioPipeline, "OPENAI_AVAILABLE", True)
    segs = _segments()
    out  = diarize_llm(segs, None, "test-model")
    assert [s["speaker"] for s in out] == ["SPEAKER_A", "SPEAKER_B"] * 3


def test_diarize_llm_valid_response(monkeypatch):
    monkeypatch.setattr(AudioPipeline, "OPENAI_AVAILABLE", True)
    payload = {"assignments": [
        {"index": i, "speaker": "SPEAKER_A" if i % 2 == 0 else "SPEAKER_B"}
        for i in range(6)
    ]}
    client = _FakeClient(payload=payload)
    segs   = _segments()
    out    = diarize_llm(segs, client, "test-model")
    assert [s["speaker"] for s in out] == ["SPEAKER_A", "SPEAKER_B"] * 3


def test_diarize_llm_garbled_response_falls_back(monkeypatch):
    """A non-JSON response must fall back to alternating, not SPEAKER_A-everything."""
    monkeypatch.setattr(AudioPipeline, "OPENAI_AVAILABLE", True)
    client = _FakeClient(payload="sure! here are the speakers: ...")
    segs   = _segments()
    out    = diarize_llm(segs, client, "test-model")
    assert [s["speaker"] for s in out] == ["SPEAKER_A", "SPEAKER_B"] * 3


def test_diarize_llm_low_coverage_falls_back(monkeypatch):
    """Labels for a small minority of segments must not be trusted."""
    monkeypatch.setattr(AudioPipeline, "OPENAI_AVAILABLE", True)
    payload = {"assignments": [{"index": 0, "speaker": "SPEAKER_A"}]}
    client = _FakeClient(payload=payload)
    segs   = _segments(6)
    out    = diarize_llm(segs, client, "test-model")
    assert [s["speaker"] for s in out] == ["SPEAKER_A", "SPEAKER_B"] * 3


def test_diarize_llm_uses_model_param(monkeypatch):
    """--post-correct-model must be honoured for diarization too."""
    monkeypatch.setattr(AudioPipeline, "OPENAI_AVAILABLE", True)
    payload = {"assignments": [
        {"index": i, "speaker": "SPEAKER_A" if i % 2 == 0 else "SPEAKER_B"}
        for i in range(6)
    ]}
    client = _FakeClient(payload=payload)
    diarize_llm(_segments(), client, "my-custom-model")
    assert client.calls[0]["model"] == "my-custom-model"


def test_diarize_llm_out_of_range_indices_ignored(monkeypatch):
    monkeypatch.setattr(AudioPipeline, "OPENAI_AVAILABLE", True)
    payload = {"assignments": [
        {"index": i, "speaker": "SPEAKER_A" if i % 2 == 0 else "SPEAKER_B"}
        for i in range(6)
    ] + [{"index": 99, "speaker": "SPEAKER_Z"}]}
    client = _FakeClient(payload=payload)
    out    = diarize_llm(_segments(6), client, "test-model")
    assert all(s["speaker"] in ("SPEAKER_A", "SPEAKER_B") for s in out)

# ── CLI parser ─────────────────────────────────────────────────────────────────

def test_build_parser_process_defaults():
    p = build_parser()
    args = p.parse_args(["process", "call.amr"])
    assert args.input == "call.amr"
    assert args.chunk_sec == 300
    assert args.overlap_sec == 5
    assert args.vad_threshold == 0.5
    assert args.vad_min_silence == 500
    assert args.post_correct_model == "gpt-4o-mini"
    assert args.output_dir == "output"
    assert args.format == "mp3"


def test_build_parser_watch_env_defaults(monkeypatch):
    monkeypatch.setenv("INBOX_DIR", "/env/inbox")
    monkeypatch.setenv("OUTPUT_DIR", "/env/output")
    monkeypatch.setenv("PROCESSED_DIR", "/env/processed")
    args = build_parser().parse_args(["watch"])
    assert args.inbox == "/env/inbox"
    assert args.output_dir == "/env/output"
    assert args.processed == "/env/processed"


def test_build_parser_watch_flags_override_env(monkeypatch):
    monkeypatch.setenv("INBOX_DIR", "/env/inbox")
    args = build_parser().parse_args(
        ["watch", "--inbox", "/flag/inbox", "--studio"])
    assert args.inbox == "/flag/inbox"
    assert args.studio is True


def test_build_parser_watch_defaults_without_env(monkeypatch):
    for var in ("INBOX_DIR", "OUTPUT_DIR", "PROCESSED_DIR"):
        monkeypatch.delenv(var, raising=False)
    args = build_parser().parse_args(["watch"])
    assert args.inbox == "inbox"
    assert args.output_dir == "output"
    assert args.processed == "processed"

# ── watcher claim/release ──────────────────────────────────────────────────────

def test_claim_release_exclusive():
    _in_flight.clear()
    assert _claim("file1.amr") is True
    # A second claim of the same file must be rejected while in flight.
    assert _claim("file1.amr") is False
    # A different file can still be claimed.
    assert _claim("file2.amr") is True
    _release("file1.amr")
    # After release the file can be claimed again.
    assert _claim("file1.amr") is True
    _release("file1.amr")
    _release("file2.amr")
    assert not _in_flight


def test_release_unknown_name_is_noop():
    _in_flight.clear()
    _release("never-claimed.amr")
    assert not _in_flight
