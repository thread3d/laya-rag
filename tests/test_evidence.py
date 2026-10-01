"""Tests for grouping retrieved passages into one evidence entry per document.

The grouping decides what ``[n]`` means for the composer, the interface and the
citation check, so these tests pin the ordering rule, the overlap trim, and the fact
that a passage with no path is never merged into another.
"""
from __future__ import annotations

from laya_rag.evidence import MIN_OVERLAP_CHARS, group_passages_by_document


# -- one entry per document --------------------------------------------------

def test_one_entry_per_document_joined_in_rank_order():
    grouped = group_passages_by_document([
        ("b.md", "b1"),
        ("a.md", "a1"),
        ("b.md", "b2"),
    ])
    assert grouped == [("b.md", "b1\n\nb2"), ("a.md", "a1")]


def test_a_document_keeps_the_position_of_its_first_passage():
    grouped = group_passages_by_document([
        ("a.md", "a1"),
        ("b.md", "b1"),
        ("a.md", "a2"),
    ])
    assert [path for path, _ in grouped] == ["a.md", "b.md"]


def test_a_single_passage_is_untouched():
    assert group_passages_by_document([("a.md", "text")]) == [("a.md", "text")]


def test_whitespace_around_a_path_does_not_split_a_document():
    assert group_passages_by_document([("a.md", "1"), (" a.md ", "2")]) == [
        ("a.md", "1\n\n2"),
    ]


# -- overlap trimming --------------------------------------------------------

def test_overlapping_chunks_are_joined_without_repeating_the_overlap():
    """Adjacent chunks overlap by the chunker's overlap_chars, so the repeated span
    must be fed once rather than twice."""
    head = "x" * 300
    overlap = "shared " * 20  # 140 chars, well past MIN_OVERLAP_CHARS
    grouped = group_passages_by_document([
        ("a.md", head + overlap),
        ("a.md", overlap + "tail"),
    ])
    assert grouped == [("a.md", head + overlap + "tail")]


def test_a_chunk_already_contained_is_dropped_entirely():
    text = "y" * 100
    assert group_passages_by_document([("a.md", text), ("a.md", text)]) == [("a.md", text)]


def test_a_short_coincidental_match_is_not_treated_as_overlap():
    grouped = group_passages_by_document([
        ("a.md", "ends with abc"),
        ("a.md", "abc and more"),
    ])
    assert grouped == [("a.md", "ends with abc\n\nabc and more")]


def test_the_overlap_threshold_cannot_catch_trivial_matches():
    assert MIN_OVERLAP_CHARS >= 32


# -- passages without a document --------------------------------------------

def test_passages_without_a_path_are_never_merged():
    assert group_passages_by_document([("", "one"), ("", "two")]) == [
        ("", "one"),
        ("", "two"),
    ]


def test_pathless_passages_keep_their_position_among_documents():
    grouped = group_passages_by_document([
        ("a.md", "a1"),
        ("", "loose"),
        ("a.md", "a2"),
        ("", "loose2"),
    ])
    assert grouped == [("a.md", "a1\n\na2"), ("", "loose"), ("", "loose2")]


def test_none_is_treated_as_no_path():
    assert group_passages_by_document([(None, "one"), (None, "two")]) == [
        ("", "one"),
        ("", "two"),
    ]


# -- edges -------------------------------------------------------------------

def test_nothing_in_nothing_out():
    assert group_passages_by_document([]) == []
