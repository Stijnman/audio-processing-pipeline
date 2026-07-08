import pytest
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from watcher import is_file_stable, AUDIO_EXTENSIONS

def test_audio_extensions():
    assert ".amr" in AUDIO_EXTENSIONS
    assert ".mp3" in AUDIO_EXTENSIONS
    assert ".wav" in AUDIO_EXTENSIONS
    assert ".txt" not in AUDIO_EXTENSIONS

def test_is_file_stable(tmp_path):
    # Create a dummy file
    test_file = tmp_path / "test.mp3"
    test_file.write_text("dummy content")
    
    # It should be stable since it's not being written to
    assert is_file_stable(test_file, wait=0.1) == True
    
    # Non-existent file should return False
    missing_file = tmp_path / "missing.mp3"
    assert is_file_stable(missing_file, wait=0.1) == False
