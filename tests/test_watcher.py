import pytest
import argparse
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from watcher import (is_file_stable, AUDIO_EXTENSIONS, move_file,
                  build_parser)


# ââ AUDIO_EXTENSIONS ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def test_audio_extensions_present():
    """All expected audio formats are in the supported set."""
    for ext in [".amr", ".mp3", ".wav", ".m4a", ".ogg", ".flac",
                ".aac", ".opus", ".webm", ".mp4"]:
        assert ext in AUDIO_EXTENSIONS, f"{ext} missing from AUDIO_EXTENSIONS"


def test_audio_extensions_absent():
    """Non-audio extensions are not in the supported set."""
    for ext in [".txt", ".pdf", ".json", ".py", ".sh"]:
        assert ext not in AUDIO_EXTENSIONS, f"{ext} should not be in AUDIO_EXTENSIONS"


# ââ is_file_stable ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def test_is_file_stable_existing(tmp_path):
    """A file that is not being written should be reported as stable."""
    f = tmp_path / "test.mp3"
    f.write_text("dummy content")
    assert is_file_stable(f, wait=0.05) is True


def test_is_file_stable_missing(tmp_path):
    """A non-existent file should return False."""
    assert is_file_stable(tmp_path / "missing.mp3", wait=0.05) is False


def test_is_file_stable_empty(tmp_path):
    """An empty file (size 0) should return False."""
    f = tmp_path / "empty.mp3"
    f.touch()
    assert is_file_stable(f, wait=0.05) is False


# ââ move_file âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def test_move_file_success(tmp_path):
    """A successfully processed file is moved to the dest directory."""
    src = tmp_path / "call.mp3"
    src.write_text("audio")
    dest_dir = tmp_path / "processed"
    move_file(src, dest_dir, success=True)
    assert (dest_dir / "call.mp3").exists()
    assert not src.exists()


def test_move_file_collision(tmp_path):
    """If a file with the same name already exists, a numbered suffix is used."""
    src1 = tmp_path / "call.mp3"
    src1.write_text("audio1")
    dest_dir = tmp_path / "processed"
    dest_dir.mkdir()
    (dest_dir / "call.mp3").write_text("existing")

    src2 = tmp_path / "call.mp3"
    src2.write_text("audio2")
    move_file(src2, dest_dir, success=True)

    assert (dest_dir / "call.mp3").exists()
    assert (dest_dir / "call_1.mp3").exists()



def test_watcher_cli_profile_db_default():
    """watcher.py's actual parser must default --profile-db to profiles.json."""
    args = build_parser().parse_args([])
    assert args.profile_db == "profiles.json"


def test_watcher_cli_profile_db_custom():
    args = build_parser().parse_args(["--profile-db", "/profiles/custom.json"])
    assert args.profile_db == "/profiles/custom.json"


def test_watcher_cli_forwards_all_pipeline_flags():
    args = build_parser().parse_args([
        "--studio", "--vad", "--post-correct", "--name-speakers",
        "--keep-timing", "--enroll-unknown", "--ecapa",
    ])
    assert args.studio and args.vad and args.post_correct
    assert args.name_speakers and args.keep_timing
    assert args.enroll_unknown and args.ecapa


def test_watcher_cli_directory_defaults_from_env(monkeypatch):
    """INBOX_DIR / OUTPUT_DIR / PROCESSED_DIR env vars must set defaults."""
    monkeypatch.setenv("INBOX_DIR", "/data/inbox")
    monkeypatch.setenv("OUTPUT_DIR", "/data/out")
    monkeypatch.setenv("PROCESSED_DIR", "/data/done")
    args = build_parser().parse_args([])
    assert args.inbox == "/data/inbox"
    assert args.output == "/data/out"
    assert args.processed == "/data/done"


def test_watcher_cli_flags_override_env(monkeypatch):
    monkeypatch.setenv("INBOX_DIR", "/data/inbox")
    args = build_parser().parse_args(["--inbox", "/other/inbox"])
    assert args.inbox == "/other/inbox"


def test_watcher_in_flight_claim_release(tmp_path):
    """A file claimed for processing must not be claimable a second time."""
    import watcher as watcher_mod

    f = tmp_path / "call.mp3"
    f.write_text("audio")
    assert watcher_mod._claim(f) is True
    assert watcher_mod._claim(f) is False  # duplicate submission blocked
    watcher_mod._release(f)
    assert watcher_mod._claim(f) is True
    watcher_mod._release(f)
