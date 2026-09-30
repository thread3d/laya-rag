"""Unit tests for the standard IR metrics and the BM25 settings the benchmark uses.

``tools/beir.py`` produces numbers meant to be read beside published IR results, so the
metric arithmetic is checked against values computed by hand. The BM25 configuration
tests are here too: a benchmark row is only comparable if ``k1``/``b`` and the stopword
behaviour are what they claim to be.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import evalkit  # noqa: E402
from laya_rag.lexical import BM25Index  # noqa: E402


# ── nDCG ─────────────────────────────────────────────────────────────────

def test_ndcg_is_one_for_a_perfect_ranking():
    relevant = {"a": 1, "c": 1}
    assert evalkit.ndcg_at_k(["a", "c", "b"], relevant, 3) == pytest.approx(1.0)


def test_ndcg_matches_a_hand_computed_value():
    # gains [1, 0, 1] -> 1/log2(2) + 1/log2(4) = 1.5; ideal [1, 1] -> 1 + 1/log2(3) = 1.6309
    relevant = {"a": 1, "c": 1}
    assert evalkit.ndcg_at_k(["a", "b", "c"], relevant, 3) == pytest.approx(1.5 / 1.63093, abs=1e-5)


def test_ndcg_uses_exponential_gain_for_graded_relevance():
    assert evalkit.ndcg_at_k(["a"], {"a": 2}, 1) == pytest.approx(1.0)
    assert evalkit.dcg_at_k([2], 1) == pytest.approx(3.0)


def test_ndcg_is_zero_when_the_query_has_no_relevant_document():
    assert evalkit.ndcg_at_k(["a", "b"], {}, 10) == 0.0


def test_ndcg_is_zero_when_no_relevant_document_was_retrieved():
    assert evalkit.ndcg_at_k(["x", "y"], {"a": 1}, 10) == 0.0


def test_ndcg_only_looks_at_the_top_k():
    # "a" sits at rank 3, so it is invisible to @2. At @3 it is found but discounted:
    # DCG = 1/log2(4) = 0.5 against an ideal of 1.0.
    relevant = {"a": 1}
    assert evalkit.ndcg_at_k(["x", "y", "a"], relevant, 2) == 0.0
    assert evalkit.ndcg_at_k(["x", "y", "a"], relevant, 3) == pytest.approx(0.5)


# ── MRR / Recall / MAP ───────────────────────────────────────────────────

def test_mrr_is_the_reciprocal_rank_of_the_first_hit():
    assert evalkit.mrr_at_k(["x", "a", "b"], {"a": 1, "b": 1}, 10) == pytest.approx(0.5)
    assert evalkit.mrr_at_k(["a", "x"], {"a": 1}, 10) == pytest.approx(1.0)
    assert evalkit.mrr_at_k(["x", "y"], {"a": 1}, 10) == 0.0


def test_mrr_respects_the_cutoff():
    assert evalkit.mrr_at_k(["x", "a"], {"a": 1}, 1) == 0.0


def test_recall_counts_the_fraction_of_all_relevant_documents_found():
    assert evalkit.recall_at_k(["a", "z"], {"a": 1, "b": 1}, 2) == pytest.approx(0.5)
    assert evalkit.recall_at_k(["a", "b"], {"a": 1, "b": 1}, 2) == pytest.approx(1.0)
    assert evalkit.recall_at_k(["a"], {}, 2) == 0.0


def test_average_precision_matches_a_hand_computed_value():
    # hits at ranks 1 and 3 -> (1/1 + 2/3) / 2 = 0.8333
    assert evalkit.average_precision_at_k(["a", "b", "c"], {"a": 1, "c": 1}, 3) == pytest.approx(0.83333, abs=1e-5)


def test_average_precision_penalises_a_late_hit():
    early = evalkit.average_precision_at_k(["a", "b"], {"a": 1}, 2)
    late = evalkit.average_precision_at_k(["b", "a"], {"a": 1}, 2)
    assert early == pytest.approx(1.0)
    assert late == pytest.approx(0.5)


# ── aggregation ──────────────────────────────────────────────────────────

def test_evaluate_rankings_macro_averages_over_scored_queries():
    rankings = {"q1": ["a", "b"], "q2": ["x", "b"]}
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    report = evalkit.evaluate_rankings(rankings, qrels, k_values=(10,))
    assert report["queries"] == 2
    # q1: nDCG 1.0, MRR 1.0. q2: b is at rank 2 -> nDCG 1/log2(3), MRR 0.5.
    assert report["summary"]["nDCG@10"] == pytest.approx((1.0 + 1 / 1.58496) / 2, abs=1e-4)
    assert report["summary"]["MRR@10"] == pytest.approx(0.75)


def test_evaluate_rankings_skips_queries_with_no_relevant_document():
    rankings = {"q1": ["a"], "q2": ["x"]}
    qrels = {"q1": {"a": 1}, "q2": {"x": 0}}
    report = evalkit.evaluate_rankings(rankings, qrels, k_values=(10,))
    assert report["queries"] == 1
    assert report["summary"]["nDCG@10"] == pytest.approx(1.0)


def test_evaluate_rankings_treats_a_missing_ranking_as_empty():
    report = evalkit.evaluate_rankings({}, {"q1": {"a": 1}}, k_values=(10,))
    assert report["queries"] == 1
    assert report["summary"]["nDCG@10"] == 0.0


# ── BM25 configuration (the benchmark's baseline) ────────────────────────

class _Doc:
    def __init__(self, chunk_id: str, text: str) -> None:
        self.chunk_id = chunk_id
        self.text = text


def test_bm25_defaults_to_the_engine_settings():
    index = BM25Index()
    assert index.k1 == pytest.approx(1.2)
    assert index.b == pytest.approx(0.75)
    assert index.drop_stopwords is True


def test_bm25_accepts_the_published_beir_settings():
    index = BM25Index(k1=0.9, b=0.4)
    assert index.k1 == pytest.approx(0.9)
    assert index.b == pytest.approx(0.4)


def test_bm25_configuration_survives_a_save_and_load(tmp_path):
    index = BM25Index(k1=0.9, b=0.4, drop_stopwords=False)
    index.build([_Doc("d1", "the refund policy")])
    path = tmp_path / "lexical.json"
    index.save(path)

    loaded = BM25Index.load(path)
    assert loaded.k1 == pytest.approx(0.9)
    assert loaded.b == pytest.approx(0.4)
    assert loaded.drop_stopwords is False


def test_query_stopwords_are_dropped_by_default_and_kept_for_beir():
    docs = [_Doc("d1", "the the the")]
    dropped = BM25Index(drop_stopwords=True)
    dropped.build(docs)
    assert dropped.search("the", k=5) == []

    kept = BM25Index(drop_stopwords=False)
    kept.build(docs)
    assert [hit.chunk_id for hit in kept.search("the", k=5)] == ["d1"]
