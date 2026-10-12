"""BM25F's IDF input cannot be derived from per-field document frequencies.

The on-disk BM25F design composes one index per field, and the tempting shortcut is to derive the
union document frequency from the per-field ones. It is wrong in both available directions — `max`
under-counts a term that appears in *different* documents in each field, the sum over-counts one that
appears in the *same* document in both — so the count has to be accumulated while building.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from laya_rag.ondisk import union_document_frequency  # noqa: E402


def test_a_term_in_different_documents_counts_each_document_once():
    """The case `max` gets wrong: title and body mention the term in *different* documents."""
    documents = [
        (["refund"], []),          # doc 1: title only
        ([], ["refund"]),          # doc 2: body only
    ]
    assert union_document_frequency(documents) == {"refund": 2}


def test_a_term_in_both_fields_of_one_document_counts_once():
    """The case the sum gets wrong: both fields of the *same* document."""
    documents = [(["refund"], ["refund"])]
    assert union_document_frequency(documents) == {"refund": 1}


def test_tokens_are_deduplicated_within_and_across_fields():
    documents = [(["a", "a", "b"], ["b", "c"])]
    assert union_document_frequency(documents) == {"a": 1, "b": 1, "c": 1}


def test_the_lower_and_upper_bounds_are_both_wrong_here():
    """A direct statement of the trap, so a future shortcut has to argue with this test."""
    documents = [(["x"], []), ([], ["x"]), (["x"], ["x"])]
    union = union_document_frequency(documents)
    assert union["x"] == 3
    df_title, df_body = 2, 2                 # docs 1 and 3 mention it in the title; 2 and 3 in body
    assert max(df_title, df_body) == 2 != union["x"]      # lower bound
    assert df_title + df_body == 4 != union["x"]          # upper bound


def test_nothing_indexed_is_an_empty_union():
    assert union_document_frequency([]) == {}
