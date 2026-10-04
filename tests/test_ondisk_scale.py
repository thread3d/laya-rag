"""Offline tests for the on-disk memory sweep.

The sweep's whole purpose is to separate two axes, so the tests pin that the synthetic corpus really
holds a *fixed* vocabulary when asked and really is deterministic — a corpus whose vocabulary drifts
with document count could not tell the dictionary's cost from the id list's.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def _load():
    name = "laya_rag_ondisk_scale"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "ondisk_scale.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scale = _load()


def test_the_corpus_is_deterministic():
    first = list(scale.synthetic_corpus(5, 100, 10, seed=7))
    second = list(scale.synthetic_corpus(5, 100, 10, seed=7))
    assert first == second


def test_the_document_count_and_token_count_are_exact():
    docs = list(scale.synthetic_corpus(7, 50, 12))
    assert len(docs) == 7
    assert all(len(text.split()) == 12 for _, text in docs)
    assert all(doc_id == f"doc-{i}" for i, (doc_id, _) in enumerate(docs))


def test_the_vocabulary_size_is_a_bound_not_a_byproduct():
    """Every term must fall inside the requested vocabulary, whatever the document count."""
    for count in (10, 100):
        terms = {token for _, text in scale.synthetic_corpus(count, 200, 30) for token in text.split()}
        assert max(int(t[1:]) for t in terms) < 200


def test_a_larger_vocabulary_uses_more_distinct_terms():
    small = {t for _, text in scale.synthetic_corpus(200, 10, 20) for t in text.split()}
    large = {t for _, text in scale.synthetic_corpus(200, 1000, 20) for t in text.split()}
    assert len(large) > len(small)


def test_a_seed_change_changes_the_corpus():
    a = list(scale.synthetic_corpus(3, 100, 10, seed=1))
    b = list(scale.synthetic_corpus(3, 100, 10, seed=2))
    assert a != b


def test_build_once_reports_the_measured_fields(tmp_path):
    report = scale.build_once(200, 100, 20)
    assert report["docs"] == 200 and report["indexed_docs"] == 200
    assert report["terms"] <= 100
    assert report["peak_rss_mb"] > 0
    # A 200-document index is a few tens of KB, which rounds to 0.0 at MB precision.
    assert report["index_mb"] >= 0.0
    assert report["build_s"] >= 0.0


def test_the_default_sweep_holds_one_axis_fixed_each_way():
    points = scale.default_sweep()
    docs_axis = [v for d, v in points if v == 50_000]
    vocab_axis = [d for d, v in points if d == 50_000]
    assert len(docs_axis) >= 3 and len(vocab_axis) >= 3
