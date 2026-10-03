"""Offline tests for index drift detection.

An index degrades after it is built: documents are deleted, the same passage is indexed
twice, a body is re-chunked at a different offset. None of those is visible in a chunk
count, and each needs a different detector. These tests pin the detectors and the report
shape, using injected path existence so nothing touches a real filesystem.
"""
from __future__ import annotations

import pytest

from laya_rag.chunking import Chunk
from laya_rag.integrity import (
    audit,
    duplicate_clusters,
    fingerprint,
    jaccard,
    near_duplicate_pairs,
    shingles,
    stale_paths,
    summarize,
)


def chunk(chunk_id: str, text: str, path: str | None = None, index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=(path or chunk_id).split("#")[0],
        path=path or chunk_id.split("#")[0],
        text=text,
        index=index,
        start_char=0,
        end_char=len(text),
    )


# ── fingerprints and shingles ────────────────────────────────────────────

def test_fingerprint_ignores_whitespace_but_not_case():
    assert fingerprint("a  b\n c") == fingerprint("a b c")
    assert fingerprint("Refund") != fingerprint("refund")


def test_shingles_of_a_short_text_is_itself():
    assert shingles("refunds take five days", size=10) == frozenset({"refunds take five days"})


def test_shingles_of_empty_text_is_empty():
    assert shingles("", size=3) == frozenset()


def test_shingles_rejects_a_zero_size():
    with pytest.raises(ValueError, match="shingle size"):
        shingles("a b", size=0)


def test_jaccard_handles_the_extremes():
    assert jaccard(frozenset({"a", "b"}), frozenset({"a", "b"})) == 1.0
    assert jaccard(frozenset({"a"}), frozenset({"b"})) == 0.0
    assert jaccard(frozenset(), frozenset()) == 1.0


# ── exact duplicates ─────────────────────────────────────────────────────

def test_duplicate_clusters_finds_identical_text_across_paths():
    chunks = [
        chunk("a#0", "Refunds take five days.", path="a.md"),
        chunk("b#0", "Refunds take five days.", path="b.md"),
        chunk("c#0", "Something else entirely.", path="c.md"),
    ]
    clusters = duplicate_clusters(chunks)
    assert len(clusters) == 1
    assert clusters[0].chunk_ids == ["a#0", "b#0"]
    assert clusters[0].paths == ["a.md", "b.md"]
    assert clusters[0].redundant == 1


def test_duplicate_clusters_is_empty_for_distinct_content():
    chunks = [chunk("a#0", "one"), chunk("b#0", "two")]
    assert duplicate_clusters(chunks) == []


def test_duplicate_clusters_sorts_the_largest_cluster_first():
    chunks = [
        chunk("a#0", "x"),
        chunk("b#0", "x"),
        chunk("c#0", "y"),
        chunk("d#0", "y"),
        chunk("e#0", "y"),
    ]
    clusters = duplicate_clusters(chunks)
    assert [len(c.chunk_ids) for c in clusters] == [3, 2]


# ── near duplicates ──────────────────────────────────────────────────────

def test_near_duplicate_pairs_finds_a_rechunked_body():
    body = " ".join(f"word{i}" for i in range(60))
    chunks = [
        chunk("a#0", body),
        chunk("a#1", body + " trailing sentence"),   # near-identical
        chunk("b#0", " ".join(f"other{i}" for i in range(60))),
    ]
    pairs = near_duplicate_pairs(chunks, threshold=0.6, shingle_size=4)
    assert pairs, "a body that differs by one sentence must be caught"
    left, right, score = pairs[0]
    assert {left, right} == {"a#0", "a#1"}
    assert score >= 0.6


def test_near_duplicate_pairs_respects_the_threshold():
    body = " ".join(f"word{i}" for i in range(60))
    chunks = [
        chunk("a#0", body),
        chunk("a#1", body + " " + " ".join(f"tail{i}" for i in range(20))),
    ]
    assert near_duplicate_pairs(chunks, threshold=0.99, shingle_size=4) == []
    assert near_duplicate_pairs(chunks, threshold=0.5, shingle_size=4)


def test_near_duplicate_pairs_rejects_an_impossible_band_schedule():
    with pytest.raises(ValueError, match="bands"):
        near_duplicate_pairs([chunk("a#0", "x")], signature=4, bands=8)
    with pytest.raises(ValueError, match="threshold"):
        near_duplicate_pairs([chunk("a#0", "x")], threshold=0.0)


def test_near_duplicate_pairs_is_bounded_by_max_chunks():
    chunks = [chunk(f"p{i}#0", f"unique body number {i}") for i in range(10)]
    assert near_duplicate_pairs(chunks, max_chunks=0) == []


def test_a_mistuned_band_schedule_is_rejected_not_quietly_empty():
    """Too few bands at a high threshold cannot see the pairs, and must say so.

    A schedule that finds nothing reports a clean index, which reads as a result. The
    guard turns that silent failure into a configuration error.
    """
    chunks = [chunk("a#0", " ".join(f"w{i}" for i in range(50)))]
    with pytest.raises(ValueError, match="of pairs at"):
        near_duplicate_pairs(chunks, threshold=0.6, signature=32, bands=2)


def test_band_recall_rises_with_similarity_and_with_bands():
    from laya_rag.integrity import band_recall

    assert band_recall(8, 2, 0.9) > band_recall(8, 2, 0.5)
    assert band_recall(16, 2, 0.6) > band_recall(4, 2, 0.6)
    assert band_recall(8, 2, 0.0) == 0.0


def test_minhash_agreement_estimates_jaccard():
    """The property the LSH depends on: positional agreement ~ the shingle Jaccard."""
    from laya_rag.integrity import _minhash_signature

    left = frozenset(f"w{i}" for i in range(100))
    right = frozenset([f"w{i}" for i in range(50)] + [f"x{i}" for i in range(50)])
    expected = jaccard(left, right)  # 50 / 150
    signature_left = _minhash_signature(left, 256)
    signature_right = _minhash_signature(right, 256)
    observed = sum(a == b for a, b in zip(signature_left, signature_right)) / 256
    assert abs(observed - expected) < 0.08, (observed, expected)


# ── stale paths ──────────────────────────────────────────────────────────

def test_stale_paths_uses_the_injected_existence_check():
    present = {"keep.md"}
    assert stale_paths(["keep.md", "gone.md"], exists=lambda p: p in present) == ["gone.md"]


# ── the report ───────────────────────────────────────────────────────────

def test_audit_reports_a_clean_index():
    chunks = [chunk("a#0", "one", path="a.md"), chunk("b#0", "two", path="b.md")]
    report = audit(chunks, exists=lambda path: True)
    assert report["clean"] is True
    assert report["stale_paths"]["count"] == 0
    assert report["duplicate_content"]["clusters"] == 0
    assert "clean" in summarize(report)


def test_audit_reports_each_drift_class_separately():
    chunks = [
        chunk("a#0", "same text", path="a.md"),
        chunk("b#0", "same text", path="b.md"),   # duplicate content
        chunk("gone#0", "orphan", path="gone.md"),  # deleted document
    ]
    report = audit(chunks, exists=lambda path: path != "gone.md")
    assert report["clean"] is False
    assert report["stale_paths"]["count"] == 1
    assert report["stale_paths"]["stale_chunks"] == 1
    assert report["duplicate_content"]["clusters"] == 1
    assert report["duplicate_content"]["redundant_chunks"] == 1
    text = summarize(report)
    assert "deleted documents: 1" in text
    assert "duplicate content: 1" in text


def test_audit_skips_the_path_check_when_asked():
    chunks = [chunk("gone#0", "orphan", path="gone.md")]
    report = audit(chunks, check_paths=False, exists=lambda path: False)
    assert report["stale_paths"]["count"] == 0
    assert report["clean"] is True


def test_audit_reports_near_duplicates_only_when_requested():
    body = " ".join(f"w{i}" for i in range(50))
    chunks = [chunk("a#0", body), chunk("a#1", body + " extra")]
    without = audit(chunks, exists=lambda path: True)
    assert without["near_duplicate_content"] is None
    assert "not scanned" in summarize(without)
    with_scan = audit(
        chunks, exists=lambda path: True,
        near_duplicates=True, threshold=0.5, shingle_size=4,
    )
    assert with_scan["near_duplicate_content"]["pairs"] == 1


def test_audit_marks_a_truncated_scan_rather_than_reporting_it_complete():
    chunks = [chunk(f"p{i}#0", f"body {i}") for i in range(5)]
    report = audit(chunks, exists=lambda path: True, max_chunks=2)
    assert report["truncated"] is True
    assert report["scanned_chunks"] == 2
    assert report["chunks"] == 5


def test_audit_of_an_empty_index_is_clean():
    report = audit([], exists=lambda path: True)
    assert report["chunks"] == 0
    assert report["clean"] is True
