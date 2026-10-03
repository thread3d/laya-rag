"""Offline tests for the pool-duplication measurement.

The result is a negative — removing duplicates from the MultiHop pool changes nothing — and
a negative is only worth recording if the detector could have found something. These tests
pin that it does: a planted duplicate is removed, and a non-duplicate is not.
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
        "laya_rag_pool_dedup", ROOT / "tools" / "multihop_pool_dedup.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dedup = _load()

CORPUS = {
    "a": "Refunds are processed within five business days to the original payment method.",
    "b": "Refunds are processed within five business days to the original payment method.",  # identical to a
    "c": "Shipping is free on orders over fifty dollars in the continental United States.",
    "d": "Refunds are processed within five business days to the original payment method, and sometimes longer.",
}


def test_rank_by_ce_is_descending_with_index_tie_break():
    assert dedup.rank_by_ce(["a", "b", "c"], [0.1, 0.9, 0.9]) == [1, 2, 0]


def test_exact_dedup_removes_the_identical_copy_and_keeps_the_better_ranked_one():
    pool = ["a", "b", "c"]
    assert dedup.dedup_exact(pool, CORPUS, [0, 1, 2]) == [0, 2]
    # The survivor is whichever ranked first, not a fixed pool position.
    assert dedup.dedup_exact(pool, CORPUS, [1, 0, 2]) == [1, 2]


def test_exact_dedup_leaves_a_corpus_without_duplicates_untouched():
    pool = ["a", "c"]
    assert dedup.dedup_exact(pool, CORPUS, [0, 1]) == [0, 1]


def test_near_dedup_removes_a_near_copy_but_not_a_distinct_passage():
    pool = ["a", "c", "d"]
    kept = dedup.dedup_near(pool, CORPUS, [0, 1, 2], threshold=0.6, shingle_size=3)
    assert 0 in kept and 1 in kept
    assert 2 not in kept, "d is a near copy of a and should be dropped"


def test_near_dedup_respects_a_threshold_it_cannot_reach():
    pool = ["a", "d"]
    assert dedup.dedup_near(pool, CORPUS, [0, 1], threshold=1.0, shingle_size=3) == [0, 1]
    assert dedup.dedup_near(pool, CORPUS, [0, 1], threshold=0.5, shingle_size=3) == [0]


def test_as_chunks_carries_the_pool_id_and_text():
    chunks = dedup.as_chunks(["a#0", "c#1"], {"a#0": "one", "c#1": "two"})
    assert [c.chunk_id for c in chunks] == ["a#0", "c#1"]
    assert [c.text for c in chunks] == ["one", "two"]
    assert chunks[0].doc_id == "a"


def test_render_markdown_reports_prevalence_and_every_comparison():
    result = {
        "_settings": {
            "queries": 400, "top_k": 10, "near_threshold": 0.85, "shingle_size": 5,
        },
        "prevalence": {
            "queries": 400, "with_exact_duplicate": 0, "with_near_duplicate": 6,
            "exact_redundant_chunks": 0, "near_redundant_chunks": 7, "pool_chunks": 40000,
            "exact_in_top10": 0, "near_in_top10": 1,
        },
        "summary": {"ce": {"Hits@10": 0.86, "Hits@4": 0.72, "MAP@10": 0.30, "MRR@10": 0.60}},
        "comparisons": {
            "ce-dedup-near vs ce MAP@10": {
                "bootstrap": {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "significant": False},
                "permutation": {"p_value": 1.0},
            }
        },
    }
    text = dedup.render_markdown(result)
    assert "1.5%" in text
    assert "near **1**" in text
    assert "ce-dedup-near vs ce MAP@10" in text
