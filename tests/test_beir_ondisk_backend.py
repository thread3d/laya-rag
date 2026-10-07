"""Offline tests for the BEIR harness's on-disk lexical backend.

`--lexical-backend ondisk` exists for corpora whose *residency* is the problem, not the scoring:
`BM25Index` keeps a `Counter` per document, which is what puts a 14.9M-document corpus out of reach.
`OnDiskBM25` spills postings to a memory-mapped file instead. That switch is only legitimate if it
changes nothing about the answer, so these tests assert the two backends return **identical
rankings** — not merely similar scores — across the settings the harness actually uses.
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
    name = "laya_rag_beir_ondisk"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "beir.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORPUS = {
    "d1": "gut microbiota profiles altered by irradiation in mice",
    "d2": "irradiation profoundly impacts the gut microbiota of mice and humans",
    "d3": "a study of fungal proteins in the journal of general physiology",
    "d4": "depressive disorder is a multifactorial disease with many causes",
    "d5": "irradiation",
}
QUERIES = {
    "q1": "are gut microbiota profiles altered by irradiation?",
    "q2": "fungal proteins general physiology",
    # Real words would match even with stopword filtering off ("a", "in", "at" all appear below),
    # so the no-match case needs tokens that are in no document under either setting.
    "q3": "zzqqxx yywwvv",
}


@pytest.mark.parametrize("drop_stopwords,stem", [(False, False), (True, True), (True, False)])
def test_the_two_backends_return_identical_rankings(tmp_path, drop_stopwords, stem):
    beir = _load()
    memory = beir.bm25_rankings(CORPUS, QUERIES, 5, k1=0.9, b=0.4,
                                drop_stopwords=drop_stopwords, stem=stem)
    disk = beir.bm25_rankings(CORPUS, QUERIES, 5, k1=0.9, b=0.4,
                              drop_stopwords=drop_stopwords, stem=stem,
                              backend="ondisk", index_dir=tmp_path / "idx")
    assert memory == disk
    # A query matching nothing must rank nothing, on either backend.
    assert memory["q3"] == disk["q3"] == []


def test_the_index_is_reused_only_for_the_same_settings(tmp_path):
    """A directory left by a different stem/stopword/b setting must not be read as this run's."""
    import json

    beir = _load()
    index_dir = tmp_path / "idx"
    beir.bm25_rankings(CORPUS, QUERIES, 5, k1=0.9, b=0.4, drop_stopwords=True, stem=True,
                       backend="ondisk", index_dir=index_dir)
    first_meta = json.loads((index_dir / "meta.json").read_text())

    # Different settings: the index must be rebuilt, and must still match the memory backend.
    second = beir.bm25_rankings(CORPUS, QUERIES, 5, k1=0.9, b=0.4, drop_stopwords=False, stem=False,
                                backend="ondisk", index_dir=index_dir)
    expected = beir.bm25_rankings(CORPUS, QUERIES, 5, k1=0.9, b=0.4, drop_stopwords=False,
                                  stem=False)
    assert second == expected

    # The property under test is that the directory was rewritten, not that the rankings differ —
    # on five documents the settings may well rank identically, which would prove nothing either way.
    second_meta = json.loads((index_dir / "meta.json").read_text())
    assert (first_meta["stem"], first_meta["drop_stopwords"]) == (True, True)
    assert (second_meta["stem"], second_meta["drop_stopwords"]) == (False, False)


def test_a_different_document_count_forces_a_rebuild(tmp_path):
    beir = _load()
    index_dir = tmp_path / "idx"
    beir.bm25_rankings(CORPUS, QUERIES, 5, k1=0.9, b=0.4, drop_stopwords=True, stem=True,
                       backend="ondisk", index_dir=index_dir)
    smaller = {k: v for k, v in list(CORPUS.items())[:3]}
    rebuilt = beir.bm25_rankings(smaller, QUERIES, 5, k1=0.9, b=0.4, drop_stopwords=True, stem=True,
                                 backend="ondisk", index_dir=index_dir)
    expected = beir.bm25_rankings(smaller, QUERIES, 5, k1=0.9, b=0.4, drop_stopwords=True, stem=True)
    assert rebuilt == expected


def test_the_backend_needs_a_directory(tmp_path):
    beir = _load()
    with pytest.raises(ValueError, match="index"):
        beir.bm25_rankings(CORPUS, QUERIES, 5, backend="ondisk", index_dir=None)
