"""Offline tests for the MultiHop depth/retention diagnostics.

These pin the definitions the report rests on: what a gold match is, how a fact's
grounding is computed, and which facts count as the second hop. The experiment itself
needs the real corpus and encoder and is not run here; only the pure helpers are.
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
        "laya_rag_multihop_diagnostics", ROOT / "tools" / "multihop_diagnostics.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diag = _load()


def test_chunk_text_uses_the_benchmark_window_convention():
    text = " ".join(f"w{i}" for i in range(10))
    assert diag.chunk_text(text, 4, 0) == [
        "w0 w1 w2 w3",
        "w4 w5 w6 w7",
        "w8 w9",
    ]
    # Overlap 2 steps by 2 tokens, so consecutive windows share two words.
    windows = diag.chunk_text(text, 4, 2)
    assert windows[0] == "w0 w1 w2 w3"
    assert windows[1] == "w2 w3 w4 w5"
    assert windows[-1].endswith("w9")


def test_normalize_fact_matches_the_benchmark_rule():
    assert diag.normalize_fact("the cat\nsat") == "thecatsat"


def test_gold_matches_is_substring_over_normalized_text():
    corpus = {"a": "The quick brown fox", "b": "a slow turtle", "c": "fox and turtle"}
    matches = diag.gold_matches(corpus, ["a", "b", "c"], ["quick brown", "turtle"])
    assert matches == [{0}, {1}, {1}]
    # Both facts land in c: "fox and turtle" contains "turtle" only.
    assert diag.gold_matches(corpus, ["c"], ["fox", "turtle"]) == [{0, 1}]


def test_gold_matches_ignores_empty_facts():
    corpus = {"a": "hello world"}
    assert diag.gold_matches(corpus, ["a"], ["", "world"]) == [{1}]


def test_candidate_overlap_is_the_share_of_query_terms_present():
    corpus = {"a": "refund policy refund", "b": "shipping rates", "c": ""}
    tokens = diag.tokenize("refund policy", stem=True)
    overlap = diag.candidate_overlap(corpus, ["a", "b", "c"], set(tokens))
    assert overlap[0] == 1.0
    assert overlap[1] == 0.0
    assert overlap[2] == 0.0
    # No query terms -> no grounding rather than a division error.
    assert diag.candidate_overlap(corpus, ["a"], set()) == [0.0]


def test_fact_grounding_takes_the_max_over_containing_candidates():
    matches = [{0}, {0, 1}, set()]
    grounding = diag.fact_grounding(matches, [0.25, 0.75, 0.0])
    assert grounding == [0.75, 0.75]
    assert diag.fact_grounding([], []) == []


def test_split_hops_makes_the_lowest_grounding_fact_the_second_hop():
    # fact 0 is grounded 1.0 (first hop); fact 1 only 0.2 (second hop).
    matches = [{0}, {0, 1}]
    first, second = diag.split_hops(matches, [1.0, 0.2])
    assert first == {0}
    assert second == {1}


def test_split_hops_needs_a_relevance_gap():
    # Both facts share the same grounding: no gap, so no hop split.
    assert diag.split_hops([{0}, {0, 1}], [0.5, 0.5]) == (set(), set())
    # A single present fact cannot be split.
    assert diag.split_hops([{0}], [1.0]) == (set(), set())
    assert diag.split_hops([], []) == (set(), set())


def test_split_hops_treats_all_minimum_facts_as_second_hop():
    matches = [{0}, {0, 1}, {1, 2}]
    first, second = diag.split_hops(matches, [1.0, 0.3, 0.3])
    assert second == {1, 2}
    assert first == {0}


def test_recall_at_depth_is_none_for_an_empty_wanted_set():
    matches = [{0}, {1}, set()]
    assert diag.recall_at_depth(matches, 2) == 1.0
    assert diag.recall_at_depth(matches, 1) == 0.5
    assert diag.recall_at_depth(matches, 3, {0}) == 1.0
    assert diag.recall_at_depth(matches, 1, {1}) == 0.0
    assert diag.recall_at_depth(matches, 3, set()) is None


def test_recall_at_depth_divides_by_the_true_gold_count_when_given_one():
    """The absolute normalisation, and why the relative one is not a substitute.

    Three gold facts, one reachable in the top-1. Relative to the facts the pool matched
    (one) that is 1.0; relative to the query's true three it is 1/3. Only the second is
    stable if the pool is incomplete — the `pool_ceiling_map` failure mode.
    """
    matches = [{0}, set(), set()]
    assert diag.recall_at_depth(matches, 1) == 1.0
    assert diag.recall_at_depth(matches, 1, gold_count=3) == pytest.approx(1 / 3)
    # The denominator does not move when the pool deepens, which is the point.
    assert diag.recall_at_depth(matches + [{1}, {2}], 1, gold_count=3) == pytest.approx(1 / 3)


def test_recall_at_depth_with_a_gold_count_of_zero_is_undefined():
    assert diag.recall_at_depth([{0}], 1, gold_count=0) is None


def test_hop_grounding_gap_reports_zero_without_facts():
    assert diag.hop_grounding_gap([{0}, {0, 1}], [1.0, 0.2]) == 0.8
    assert diag.hop_grounding_gap([], []) == 0.0


def _row(q, ceiling10, ceiling100, hop2_10, hop2_100, retention10=0.5, retention100=1.0):
    return {
        "query": q,
        "fused_MAP@10": 0.25,
        "fused_Hits@10": 1.0,
        "fused_Hits@4": 0.5,
        "hop2_size": 1,
        "pool_fact_recall@10": 0.5,
        "pool_ceiling_MAP@10@10": ceiling10,
        "hop1_fact_recall@10": 1.0,
        "hop2_fact_recall@10": hop2_10,
        "hop2_retention@10": retention10,
        "pool_fact_recall@100": 1.0,
        "pool_ceiling_MAP@10@100": ceiling100,
        "hop1_fact_recall@100": 1.0,
        "hop2_fact_recall@100": hop2_100,
        "hop2_retention@100": retention100,
    }


def test_summarize_aggregates_and_pairs_depths():
    s = diag.summarize([_row("0", 0.5, 0.8, 0.0, 1.0), _row("1", 0.6, 0.9, 1.0, 1.0)], [10, 100])
    assert s["queries"] == 2
    assert s["depth"]["10"]["pool_ceiling_MAP@10"]["mean"] == 0.55
    assert s["depth"]["100"]["pool_ceiling_MAP@10"]["mean"] == 0.85
    assert s["hop_queries"] == 2
    # Widening 10 -> 100 raised the ceiling on both queries; the paired interval must
    # exclude zero.
    pair = s["paired"]["pool_ceiling_MAP@10:100-10"]
    assert pair["delta"] == 0.3
    assert pair["significant"] is True


def test_summarize_skips_missing_hop_values_without_crashing():
    row = _row("0", 0.5, 0.8, 0.0, 1.0)
    row["hop2_retention@10"] = None
    s = diag.summarize([row], [10, 100])
    assert s["depth"]["10"]["hop2_retention"]["n"] == 0
    assert s["depth"]["10"]["hop2_retention"]["mean"] is None


def test_render_markdown_includes_every_depth_and_pair():
    report = {
        "_settings": {"depths": [10, 100]},
        "summary": diag.summarize([_row("0", 0.5, 0.8, 0.0, 1.0)], [10, 100]),
    }
    text = diag.render_markdown(report)
    assert "| 10 |" in text and "| 100 |" in text
    assert "pool_ceiling_MAP@10:100-10" in text
