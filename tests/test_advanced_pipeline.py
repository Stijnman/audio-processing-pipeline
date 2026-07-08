import pytest
from pathlib import Path
import sys
import os

# Add parent directory to sys.path to import advanced_pipeline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from advanced_pipeline import stitch_chunks, merge_intervals


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
    of segments in base, the slice s1 is shorter than n.  The fix uses
    len(s1) instead of n so the match tokens are included in the output
    rather than silently dropped.
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
    # overlap_words = int(2.0 * 3) = 6, but base only has 4 items.
    # s1 = base[-6:] = all 4 items; the match "a", "test" must be preserved.
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
    # overlap_words = int(1.0 * 3) = 3; base has exactly 3 items.
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


# ── merge_intervals ───────────────────────────────────────────────────────────

def test_merge_intervals_basic():
    """Intervals within gap distance are merged; those outside are not."""
    intervals = [(0.0, 1.0), (1.1, 2.0), (3.0, 4.0), (4.2, 5.0)]
    merged = merge_intervals(intervals, gap=0.3)
    assert merged == [(0.0, 2.0), (3.0, 5.0)]


def test_merge_intervals_no_merge():
    """With a very small gap, no intervals merge."""
    intervals = [(0.0, 1.0), (1.1, 2.0), (3.0, 4.0), (4.2, 5.0)]
    assert merge_intervals(intervals, gap=0.05) == intervals


def test_merge_intervals_empty():
    """An empty list returns an empty list."""
    assert merge_intervals([], gap=0.3) == []


def test_merge_intervals_single():
    """A single interval is returned unchanged."""
    assert merge_intervals([(1.0, 2.0)], gap=0.3) == [(1.0, 2.0)]


def test_merge_intervals_all_merge():
    """Overlapping intervals all collapse into one."""
    intervals = [(0.0, 1.5), (1.0, 2.5), (2.0, 3.5)]
    merged = merge_intervals(intervals, gap=0.0)
    assert merged == [(0.0, 3.5)]


def test_merge_intervals_unsorted():
    """Unsorted input is sorted before merging."""
    intervals = [(3.0, 4.0), (0.0, 1.0), (1.1, 2.0)]
    merged = merge_intervals(intervals, gap=0.3)
    assert merged == [(0.0, 2.0), (3.0, 4.0)]
