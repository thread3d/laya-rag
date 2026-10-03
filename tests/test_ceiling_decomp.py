"""Offline tests for the signal-vs-algorithm ceiling decomposition.

The decomposition is what decides whether the next investment goes into a better scorer or a
better set selector, so the instrument has to be right: the prefix oracle must be the oracle
over a *prefix*, and both must use the query's true gold count as the denominator.
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
        "laya_rag_ceiling_decomp", ROOT / "tools" / "multihop_ceiling_decomp.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


decomp = _load()


def test_oracle_ceiling_reaches_one_when_every_fact_is_reachable_in_order():
    # Three facts, three candidates, top_k 3: each rank finds one new fact.
    assert decomp.oracle_ceiling([{0}, {1}, {2}], 3, 3) == pytest.approx(
        (1 + 1 / 2 + 1 / 3) / 3
    )


def test_oracle_ceiling_is_capped_by_the_true_gold_count():
    """The denominator is the query's gold count, not the facts the candidates match."""
    # Only one of three gold facts is present: 1/1 over min(3, 10).
    assert decomp.oracle_ceiling([{0}, set(), set()], 3, 10) == pytest.approx(1 / 3)


def test_oracle_ceiling_prefers_the_candidate_covering_two_facts_first():
    # Candidate 1 covers two facts, so it belongs at rank 1: AP = 2/1 / min(2, 10).
    assert decomp.oracle_ceiling([{0}, {0, 1}], 2, 10) == pytest.approx(2 / 2)


def test_oracle_ceiling_of_nothing_is_zero():
    assert decomp.oracle_ceiling([], 3) == 0.0
    assert decomp.oracle_ceiling([set(), set()], 0) == 0.0


def test_restrict_takes_the_oracle_over_a_prefix_not_the_whole_pool():
    """The prefix oracle must be bounded by what the prefix actually contains."""
    matches = [{0}, {1}, {2}]
    order = [2, 1, 0]
    assert decomp.restrict(matches, order, 1) == [{2}]
    assert decomp.restrict(matches, order, 2) == [{2}, {1}]
    # A prefix holding one fact cannot score the two-fact ceiling.
    assert decomp.oracle_ceiling(decomp.restrict(matches, order, 1), 3, 10) == pytest.approx(1 / 3)


def test_prefix_oracle_is_monotone_in_the_prefix_size():
    """A superset of candidates can only raise the greedy ceiling."""
    matches = [{0}, set(), {1}, {2}]
    values = [decomp.oracle_ceiling(decomp.restrict(matches, [0, 1, 2, 3], n), 3, 10)
              for n in (1, 2, 3, 4)]
    assert values == sorted(values)


def test_render_markdown_reports_all_three_headrooms():
    result = {
        "_settings": {"queries": 2255, "top_k": 10, "coverage_lambda": 0.7, "min_docs": 3},
        "summary": {
            "ce order (achieved)": 0.2948,
            "covquota-3-0.7 (achieved)": 0.3192,
            "oracle over the full pool": 0.6072,
            "oracle over CE top-20": 0.5080,
        },
        "decomposition": {
            "best_achieved": 0.3192,
            "ordering_headroom": 0.1254,
            "selection_headroom": 0.0634,
            "signal_headroom": 0.0992,
            "reference_prefix": 10,
            "deeper_prefix": 20,
            "read": "the first two are algorithm work, the third is model work",
        },
        "comparisons": {
            "covquota vs ce order MAP@10": {
                "bootstrap": {"delta": 0.0244, "ci_low": 0.0219, "ci_high": 0.0270, "significant": True},
                "permutation": {"p_value": 0.0005},
            }
        },
    }
    text = decomp.render_markdown(result)
    assert "ordering headroom" in text
    assert "selection headroom" in text
    assert "+0.1254" in text
    assert "+0.0634" in text
    assert "+0.0992" in text
    assert "covquota vs ce order MAP@10" in text
