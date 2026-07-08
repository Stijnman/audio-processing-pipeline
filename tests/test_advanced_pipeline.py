import pytest
from pathlib import Path
import sys
import os

# Add parent directory to sys.path to import advanced_pipeline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from advanced_pipeline import stitch_chunks, merge_intervals

def test_stitch_chunks_empty():
    base = [{"start": 0, "end": 2, "text": "hello"}]
    assert stitch_chunks(base, [], 1.0) == base
    assert stitch_chunks([], base, 1.0) == base

def test_stitch_chunks_overlap():
    base = [
        {"start": 0, "end": 1, "text": "this"},
        {"start": 1, "end": 2, "text": "is"},
        {"start": 2, "end": 3, "text": "a"},
        {"start": 3, "end": 4, "text": "test"}
    ]
    append = [
        {"start": 2.5, "end": 3.5, "text": "a"},
        {"start": 3.5, "end": 4.5, "text": "test"},
        {"start": 4.5, "end": 5.5, "text": "of"},
        {"start": 5.5, "end": 6.5, "text": "stitching"}
    ]
    
    # overlap_sec is used to determine how many words to look at.
    # In advanced_pipeline.py, overlap_words = int(overlap_sec * 3).
    # If overlap_sec = 2.0, overlap_words = 6.
    # With base having 4 items and append having 4 items, the match is "a", "test"
    # size = 2.
    # cut1 = 4 - 4 + 2 + 2 = 4 (actually, overlap_words is 6, but slice1 length is 4)
    # The actual result will be exactly deduplicated.
    result = stitch_chunks(base, append, overlap_sec=2.0)
    
    texts = [seg["text"] for seg in result]
    assert texts == ["this", "is", "a", "test", "of", "stitching"]

def test_merge_intervals():
    intervals = [(0.0, 1.0), (1.1, 2.0), (3.0, 4.0), (4.2, 5.0)]
    
    # With gap=0.3, (0.0, 1.0) and (1.1, 2.0) should merge.
    # (3.0, 4.0) and (4.2, 5.0) should merge.
    merged = merge_intervals(intervals, gap=0.3)
    assert merged == [(0.0, 2.0), (3.0, 5.0)]
    
    # With gap=0.05, nothing should merge.
    merged_small_gap = merge_intervals(intervals, gap=0.05)
    assert merged_small_gap == intervals
