"""Offline tests for the second-stage selection rules.

The experiment itself needs the cached cross-encoder scores; these pin the rules it
compares, because a selection rule that is subtly wrong would be read as a result.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "laya_rag_rerank_experiment", ROOT / "tools" / "multihop_rerank_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp = _load()


def test_ce_order_sorts_descending_and_breaks_ties_by_pool_order():
    pool = ["a", "b", "c", "d"]
    assert exp.ce_order(pool, [0.1, 0.9, 0.5, 0.9]) == ["b", "d", "c", "a"]
    assert exp.ce_order(pool, [0.0, 0.0, 0.0, 0.0]) == pool


def test_blend_order_endpoints_are_the_two_stages():
    pool = ["a", "b", "c"]
    scores = [0.1, 0.5, 0.9]
    assert exp.blend_order(pool, scores, 0.0) == ["a", "b", "c"]
    assert exp.blend_order(pool, scores, 1.0) == ["c", "b", "a"]
    with_exp = exp.blend_order(pool, scores, 0.5)
    assert sorted(with_exp) == sorted(pool)
    assert with_exp[0] in {"a", "c"}  # both stages are represented at the top


def test_blend_order_rejects_an_out_of_range_weight():
    try:
        exp.blend_order(["a"], [0.5], 1.5)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("blend > 1 should raise")


def test_guardrail_order_pins_the_first_stage_top_one():
    pool = ["a", "b", "c", "d"]
    scores = [0.1, 0.2, 0.3, 0.4]  # CE order is d, c, b, a
    assert exp.guardrail_order(pool, scores, pin=1)[0] == "a"
    assert exp.guardrail_order(pool, scores, pin=1) == ["a", "d", "c", "b"]
    assert exp.guardrail_order(pool, scores, pin=2)[:2] == ["a", "b"]
    # pin=0 is exactly the full cross-encoder order.
    assert exp.guardrail_order(pool, scores, pin=0) == exp.ce_order(pool, scores)


def test_guardrail_order_does_not_pin_more_than_the_pool():
    pool = ["a", "b"]
    assert sorted(exp.guardrail_order(pool, [0.1, 0.9], pin=99)) == sorted(pool)


def test_coverage_order_prefers_new_atoms_over_a_redundant_high_score():
    scores = [0.9, 0.8, 0.1]
    atoms = [[0, 1], [1], [2]]
    weight = [1.0, 1.0, 1.0]
    # Candidate 0 wins slot 1; with its atoms covered, candidate 2 (a new atom) beats the
    # redundant candidate 1 despite the lower score.
    assert exp.coverage_order(scores, atoms, weight, top_k=2, lam=0.5) == [0, 2]


def test_coverage_order_with_pure_relevance_is_the_ce_order():
    scores = [0.1, 0.9, 0.5]
    atoms = [[0], [1], [2]]
    weight = [1.0, 1.0, 1.0]
    assert exp.coverage_order(scores, atoms, weight, top_k=3, lam=1.0) == [1, 2, 0]


def test_coverage_order_respects_top_k():
    scores = [0.1, 0.2, 0.3]
    atoms = [[0], [1], [2]]
    assert len(exp.coverage_order(scores, atoms, [1.0] * 3, top_k=2, lam=0.5)) == 2


def test_strategy_rankings_returns_every_named_rule():
    cache = {
        "_settings": {"atom_weight": [1.0, 1.0, 1.0]},
        "queries": {
            "0": {"pool": ["a", "b", "c"], "ce": [0.1, 0.9, 0.5], "atoms": [[0], [1], [2]]},
        },
    }
    rankings = exp.strategy_rankings(cache, top_k=2, rrf_k=60)
    # Strategies return pool indices so they can be scored against cached matches.
    assert rankings["fused"]["0"] == [0, 1]
    assert rankings["ce"]["0"] == [1, 2]
    assert rankings["guardrail-1"]["0"][0] == 0
    for name in ("blend-0.5", "coverage-0.5", "guardrail-2"):
        assert len(rankings[name]["0"]) == 2


def test_metrics_from_matches_matches_the_official_definition():
    # Two gold facts; candidate 0 covers fact 0, candidate 1 covers fact 1.
    matches = [{0}, {1}, set()]
    metrics = exp.metrics_from_matches([0, 1], matches, gold_count=2, top_k=10)
    assert metrics["Hits@10"] == 1.0
    assert metrics["Hits@4"] == 1.0
    # AP = 1/1 + 1/2 = 1.5, normalized by min(2, 10) = 2.
    assert metrics["MAP@10"] == pytest.approx(0.75)
    assert metrics["MRR@10"] == 1.0


def test_metrics_from_matches_penalizes_a_late_fact():
    matches = [set(), set(), {0}]
    metrics = exp.metrics_from_matches([0, 1, 2], matches, gold_count=1, top_k=10)
    assert metrics["Hits@10"] == 1.0
    assert metrics["Hits@4"] == 1.0
    assert metrics["MAP@10"] == pytest.approx(1 / 3)


def test_metrics_from_matches_is_zero_when_nothing_matches():
    metrics = exp.metrics_from_matches([0, 1], [set(), set()], gold_count=2, top_k=10)
    assert metrics == {"Hits@10": 0.0, "Hits@4": 0.0, "MAP@10": 0.0, "MRR@10": 0.0}


def test_gold_matches_for_pool_uses_the_benchmark_normalization():
    corpus = {"a": "The quick brown fox", "b": "a slow turtle"}
    matches = exp.gold_matches_for_pool(corpus, ["a", "b"], ["quick brown", "turtle"])
    assert matches == [{0}, {1}]


# ── bridge-linked rescue ─────────────────────────────────────────────────

def test_rescue_keeps_ce_seeds_then_readmits_a_bridge_linked_candidate():
    scores = [0.9, 0.8, 0.2, 0.1, 0.6]
    tokens = [
        {"refund", "policy"},
        {"shipping", "rates"},
        {"company", "news"},
        {"policy", "deadline"},   # shares the rare bridge term "policy" with seed 0
        {"generic", "filler"},
    ]
    order = exp.rescue_order(scores, tokens, top_k=3, seed_k=2, beta=0.7)
    # Seeds are the CE top two; slot 3 goes to the bridge-linked candidate 3, not the
    # higher-scoring but unrelated candidate 4.
    assert order[:2] == [0, 1]
    assert order[2] == 3


def test_rescue_with_beta_zero_is_the_ce_order():
    scores = [0.1, 0.9, 0.5]
    tokens = [{"a"}, {"b"}, {"c"}]
    assert exp.rescue_order(scores, tokens, top_k=3, seed_k=1, beta=0.0) == [1, 2, 0]


def test_rescue_respects_top_k_and_handles_empty_input():
    assert exp.rescue_order([], [], top_k=5) == []
    scores = [0.9, 0.5, 0.1]
    tokens = [{"a"}, {"b"}, {"c"}]
    order = exp.rescue_order(scores, tokens, top_k=2, seed_k=1)
    assert len(order) == 2
    assert order[0] == 0


def test_rescue_rankings_returns_the_three_bridge_weights():
    cache = {
        "queries": {"0": {"pool": ["a", "b"], "ce": [0.9, 0.1], "atoms": [[0], [1]]}},
    }
    tokens = {"0": [{"refund"}, {"shipping"}]}
    rankings = exp.rescue_rankings(cache, tokens, top_k=2, seed_k=1)
    assert set(rankings) == {"rescue-0.3", "rescue-0.5", "rescue-0.7"}
    for name in rankings:
        assert len(rankings[name]["0"]) == 2
