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


# ── document-aware selection ─────────────────────────────────────────────

def test_document_of_splits_the_article_from_the_chunk_position():
    assert exp.document_of("example.com_news#3") == "example.com_news"
    assert exp.document_of("no-position") == "no-position"


def test_doc_quota_forces_distinct_articles_into_the_top_slots():
    scores = [0.9, 0.8, 0.7, 0.6]
    docs = ["A", "A", "A", "B"]
    order = exp.doc_quota_order(scores, docs, top_k=3, min_docs=2)
    # Slot 1 is A's best; slot 2 must be a new article, so B displaces A's second chunk.
    assert order[0] == 0
    assert order[1] == 3
    assert sorted(order) == [0, 1, 3]


def test_doc_best_puts_a_second_article_best_chunk_ahead_of_a_first_article_third():
    scores = [0.9, 0.8, 0.7, 0.85]
    docs = ["A", "A", "A", "B"]
    order = exp.doc_best_order(scores, docs, top_k=3)
    # A's best (0.9) first, then B's best (0.85) before A's second (0.8).
    assert order[:2] == [0, 3]


def test_doc_rankings_returns_the_document_strategies():
    cache = {
        "_settings": {"atom_weight": [1.0, 1.0, 1.0]},
        "queries": {"0": {"pool": ["a#0", "a#1", "b#0"], "ce": [0.9, 0.7, 0.8],
                          "atoms": [[0], [1], [2]]}},
    }
    rankings = exp.doc_rankings(cache, top_k=2)
    assert set(rankings) == {
        "docquota-3", "docquota-4", "docbest", "covquota-3-0.5", "covquota-3-0.7"
    }
    for name in rankings:
        assert len(rankings[name]["0"]) == 2


def test_coverage_doc_quota_seeds_distinct_articles_then_fills_by_coverage():
    scores = [0.9, 0.8, 0.7, 0.6]
    docs = ["A", "A", "B", "C"]
    atoms = [[0], [1], [2], [3]]
    order = exp.coverage_doc_quota_order(
        scores, atoms, [1.0, 1.0, 1.0, 1.0], docs, top_k=3, lam=0.7, min_docs=3
    )
    # Phase 1 takes A's best, then B, then C: three distinct articles in three slots.
    assert [docs[i] for i in order[:3]] == ["A", "B", "C"]


def test_coverage_doc_quota_fills_beyond_the_quota_by_coverage():
    scores = [0.9, 0.8, 0.2, 0.1]
    docs = ["A", "A", "B", "C"]
    atoms = [[0], [0], [1], [2]]  # candidate 1 repeats candidate 0's only atom
    order = exp.coverage_doc_quota_order(
        scores, atoms, [1.0, 1.0, 1.0, 1.0], docs, top_k=4, lam=0.5, min_docs=2
    )
    assert order[0] == 0 and order[1] == 2  # A's best, then B
    # Candidate 1 is a repeat of candidate 0's atom, so C's new atom beats it.
    assert order[2] == 3


# ── held-out selection ───────────────────────────────────────────────────

def test_holdout_selection_picks_on_a_and_reports_on_b():
    per_query = {
        "ce": {str(i): 0.2 for i in range(12)},
        "good": {str(i): 0.3 for i in range(12)},
    }
    holdout = exp.holdout_selection(per_query, seed=3)
    assert holdout["selected_on_a"] == "good"
    assert holdout["selected_b_score"] == 0.3
    assert holdout["baseline_b"]["ce"] == 0.2
    assert holdout["queries_a"] + holdout["queries_b"] == 12


def test_holdout_selection_reports_the_winners_held_out_score_even_if_it_loses():
    import random

    per_query = {
        "steady": {str(i): 0.4 for i in range(20)},
        "lucky": {str(i): (0.9 if i < 10 else 0.0) for i in range(20)},
    }
    holdout = exp.holdout_selection(per_query, seed=1)
    # Independently reconstruct half B from the same seed and check the reported number.
    qids = sorted(per_query["steady"])
    shuffled = list(qids)
    random.Random(1).shuffle(shuffled)
    set_b = shuffled[len(shuffled) // 2:]
    name = holdout["selected_on_a"]
    expected = round(sum(per_query[name][q] for q in set_b) / len(set_b), 4)
    assert holdout["selected_b_score"] == expected


def test_holdout_selection_handles_an_empty_input():
    assert exp.holdout_selection({}, seed=1) == {}


def test_truncate_pools_limits_every_member_of_the_cache():
    cache = {
        "queries": {
            "0": {"pool": ["a", "b", "c"], "ce": [0.1, 0.2, 0.3], "atoms": [[0], [1], [2]]},
        }
    }
    out = exp.truncate_pools(cache, 2)
    assert out["queries"]["0"]["pool"] == ["a", "b"]
    assert out["queries"]["0"]["ce"] == [0.1, 0.2]
    assert out["queries"]["0"]["atoms"] == [[0], [1]]
    # A falsy limit is a no-op, so the default path is untouched.
    assert exp.truncate_pools(cache, None) is cache


# ── query-aware coverage ─────────────────────────────────────────────────

def test_query_aware_coverage_prefers_a_new_query_relevant_atom():
    scores = [0.9, 0.1, 0.1]
    atoms = [[0], [1], [2]]
    atom_weight = [1.0, 5.0, 1.0]           # candidate 1 is longer/generic
    atom_tokens = [{"a"}, {"x"}, {"q"}]     # candidate 2 is query-relevant
    query_terms = {"q"}
    # Without the boost the longer atom wins; with a strong boost the query-relevant one does.
    plain = exp.coverage_query_order(
        scores, atoms, atom_weight, atom_tokens, query_terms, top_k=2, lam=0.5, qboost=0.0
    )
    boosted = exp.coverage_query_order(
        scores, atoms, atom_weight, atom_tokens, query_terms, top_k=2, lam=0.5, qboost=10.0
    )
    assert plain[1] == 1
    assert boosted[1] == 2


def test_query_aware_coverage_with_zero_boost_matches_plain_coverage():
    scores = [0.9, 0.5, 0.2]
    atoms = [[0], [1], [2]]
    weight = [1.0, 2.0, 3.0]
    tokens = [{"a"}, {"b"}, {"c"}]
    assert exp.coverage_query_order(
        scores, atoms, weight, tokens, {"c"}, top_k=3, lam=0.7, qboost=0.0
    ) == exp.coverage_order(scores, atoms, weight, top_k=3, lam=0.7)


def test_coverage_query_rankings_returns_the_boost_strategies():
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from multihop_coverage import atom_index, split_atoms

    corpus = {"a#0": "Refunds are processed within five business days for every customer."}
    atoms_by_doc = {key: split_atoms(text) for key, text in corpus.items()}
    ids, weight, _ = atom_index(list(atoms_by_doc.values()))
    cache = {
        "_settings": {"atom_weight": weight},
        "queries": {"0": {"pool": ["a#0"], "ce": [0.9], "atoms": [ids[0]]}},
    }
    rankings = exp.coverage_query_rankings(cache, corpus, {"0": "refund policy"}, top_k=1)
    assert set(rankings) == {"coverage-q-1.0", "coverage-q-3.0"}
    assert len(rankings["coverage-q-3.0"]["0"]) == 1
