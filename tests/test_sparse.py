"""Tests for the learned-sparse channel's encoder handle and inverted index.

The encoder is lazy and its model is large, so these cover the index arithmetic and the
persistence contract; the checkpoint itself is exercised by the BEIR harness runs.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from laya_rag.sparse import SPARSE_FILE, SparseEncoder, SparseIndex


def _index() -> SparseIndex:
    index = SparseIndex()
    index.build(
        ["a", "b", "c"],
        [
            {1: 1.0, 2: 2.0},
            {2: 1.0, 3: 4.0},
            {9: 1.0},
        ],
    )
    return index


def test_search_scores_the_sparse_dot_product():
    index = _index()
    assert index.search({2: 1.0}, k=5) == [("a", 2.0), ("b", 1.0)]
    assert index.search({3: 1.0}, k=5) == [("b", 4.0)]
    assert index.search({1: 2.0}, k=5) == [("a", 2.0)]
    # No matching term is no result, not an error.
    assert index.search({99: 1.0}, k=5) == []


def test_search_respects_top_k_and_breaks_ties_by_index():
    index = SparseIndex()
    index.build(["first", "second", "third"], [{1: 1.0}, {1: 1.0}, {1: 1.0}])
    assert [doc_id for doc_id, _ in index.search({1: 1.0}, k=2)] == ["first", "second"]


def test_search_ignores_terms_no_document_carries():
    index = _index()
    assert index.search({2: 1.0, 99: 5.0}, k=5) == [("a", 2.0), ("b", 1.0)]


def test_save_load_round_trip_preserves_search(tmp_path):
    index = _index()
    path = tmp_path / SPARSE_FILE
    index.save(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["ids"] == ["a", "b", "c"]

    loaded = SparseIndex.load(path)
    assert len(loaded) == 3
    assert loaded.ids == ["a", "b", "c"]
    assert loaded.terms == 4
    assert loaded.search({2: 1.0}, k=5) == index.search({2: 1.0}, k=5)
    assert loaded.search({3: 1.0}, k=5) == [("b", 4.0)]


def test_load_rejects_an_unknown_version(tmp_path):
    path = tmp_path / SPARSE_FILE
    path.write_text(json.dumps({"version": 999, "ids": [], "postings": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        SparseIndex.load(path)


def test_an_empty_index_searches_to_nothing():
    index = SparseIndex()
    assert len(index) == 0
    assert index.search({1: 1.0}, k=5) == []
    assert "documents=0" in repr(index)


def test_sparse_encoder_is_lazy_and_reports_its_configuration():
    encoder = SparseEncoder("example/model", top_k=32)
    assert encoder.loaded is False
    info = encoder.info()
    assert info == {"model": "example/model", "top_k": 32, "max_length": 256, "loaded": False}
