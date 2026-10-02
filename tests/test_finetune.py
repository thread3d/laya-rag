"""Offline tests for the fine-tuner's pure pair-building helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    spec = importlib.util.spec_from_file_location("laya_rag_finetune", ROOT / "tools" / "finetune.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


finetune = _load()


def test_training_pairs_follows_qrels_and_honours_the_cap():
    corpus = {"d1": "one", "d2": "two"}
    queries = {"q1": "q one", "q2": "q two"}
    qrels = {"q1": {"d1": 1, "d2": 0}, "q2": {"d2": 2}}
    assert finetune.training_pairs(queries, qrels, corpus, 0) == [("q one", "one"), ("q two", "two")]
    assert finetune.training_pairs(queries, qrels, corpus, 1) == [("q one", "one")]


def test_ensemble_positive_pairs_fuses_the_teachers_by_rank():
    corpus = {"d1": "a", "d2": "b", "d3": "c"}
    queries = {"q1": "q"}
    teachers = {"m1": {"q1": ["d1", "d2"]}, "m2": {"q1": ["d2", "d3"]}}
    pairs = finetune.ensemble_positive_pairs(corpus, queries, teachers, per_query=2)
    # d2 appears in both teachers, so RRF puts it first; d1 and d3 tie and d1 wins on rank.
    assert pairs == [("q", "b"), ("q", "a")]


def test_ensemble_positive_pairs_skips_queries_and_docs_it_cannot_resolve():
    corpus = {"d1": "a"}
    queries = {"q1": "q", "q2": "unranked"}
    teachers = {"m1": {"q1": ["d1", "missing"]}}
    pairs = finetune.ensemble_positive_pairs(corpus, queries, teachers, per_query=5)
    assert pairs == [("q", "a")]
