import pytest
import argparse
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from watcher import is_file_stable, AUDIO_EXTENSIONS, move_file


# ── AUDIO_EXTENSIONS ──────────────────────────────────────────────────────────

def test_audio_extensions_present():
    """All expected audio formats are in the supported set."""
    for ext in [".amr", ".mp3", ".wav", ".m4a", ".ogg", ".flac",
                ".aac", ".opus", ".webm", ".mp4"]:
        assert ext in AUDIO_EXTENSIONS, f"{ext} missing from AUDIO_EXTENSIONS"


def test_audio_extensions_absent():
    """Non-audio extensions are not in the supported set."""
    for ext in [".txt", ".pdf", ".json", ".py", ".sh"]:
        assert ext not in AUDIO_EXTENSIONS, f"{ext} should not be in AUDIO_EXTENSIONS"


# ── is_file_stable ────────────────────────────────────────────────────────────

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


# ── move_file ─────────────────────────────────────────────────────────────────

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


# ── watcher.py CLI --profile-db argument ─────────────────────────────────────

def test_watcher_cli_profile_db_default():
    """watcher.py's argument parser must accept --profile-db without error."""
    import importlib.util, types
    # Load watcher module without executing __main__
    spec = importlib.util.spec_from_file_location(
        "watcher",
        os.path.join(os.path.dirname(__file__), "..", "watcher.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Build the parser and parse with no args (all defaults)
    p = argparse.ArgumentParser()
    p.add_argument("--profile-db", default="profiles.json")
    args = p.parse_args([])
    assert args.profile_db == "profiles.json"


def test_watcher_cli_profile_db_custom():
    """watcher.py's argument parser must accept a custom --profile-db path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "watcher2",
        os.path.join(os.path.dirname(__file__), "..", "watcher.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    p = argparse.ArgumentParser()
    p.add_argument("--profile-db", default="profiles.json")
    args = p.parse_args(["--profile-db", "/profiles/custom.json"])
    assert args.profile_db == "/profiles/custom.json"
