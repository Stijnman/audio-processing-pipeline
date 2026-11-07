"""Regression checks: watcher never destroys originals or reprocesses unchanged audio."""
import tempfile
import unittest
from pathlib import Path

from watcher import is_archived, move_file


class WatcherPreservationTests(unittest.TestCase):
    def test_original_preserved_and_skipped_after_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "recording.webm"
            original.write_bytes(b"example recording bytes")
            archive = root / "processed"
            self.assertFalse(is_archived(original, archive))

            move_file(original, archive, success=True)

            self.assertEqual(original.read_bytes(), b"example recording bytes")
            self.assertEqual((archive / original.name).read_bytes(), original.read_bytes())
            self.assertTrue(is_archived(original, archive))

    def test_modified_source_requires_reprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "recording.webm"
            original.write_bytes(b"initial")
            archive = root / "processed"
            move_file(original, archive, success=True)
            original.write_bytes(b"modified recording")
            self.assertFalse(is_archived(original, archive))

    def test_failed_archives_keep_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "recording.webm"
            original.write_bytes(b"test")
            failed = root / "processed" / "failed"
            move_file(original, failed, success=False)
            self.assertTrue(original.exists())
            self.assertTrue(is_archived(original, root / "processed"))


if __name__ == "__main__":
    unittest.main()
