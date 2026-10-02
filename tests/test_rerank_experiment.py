"""Offline tests for the second-stage selection rules.

The experiment itself needs the cached cross-encoder scores; these pin the rules it
compares, because a selection rule that is subtly wrong would be read as a result.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
    assert rankings["fused"]["0"] == ["a", "b"]
    assert rankings["ce"]["0"] == ["b", "c"]
    assert rankings["guardrail-1"]["0"][0] == "a"
    for name in ("blend-0.5", "coverage-0.5", "guardrail-2"):
        assert len(rankings[name]["0"]) == 2
