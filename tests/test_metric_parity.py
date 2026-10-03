"""Parity between the two implementations of the MultiHop-RAG metric.

The benchmark's metric is implemented twice — `multihop_official.official_metrics` for the
harness and `multihop_coverage.official_metrics_one` for the coverage sweep. Two copies of a
metric definition drift, and a metric that is subtly different between the tool that reports
the headline and the tool that selects the rule would silently compare incomparable numbers.
This is the audit for that: the same inputs must give the same outputs.

It is also where the `pool_ceiling_map` bug class is pinned, because that function is scored
*against* these metrics: any normalisation difference shows up as a parity failure here.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


official = _load("laya_rag_multihop_official", "multihop_official.py")
coverage = _load("laya_rag_multihop_coverage", "multihop_coverage.py")

METRICS = ("Hits@10", "Hits@4", "MAP@10", "MRR@10")

#: Cases chosen to hit the boundaries: rank 11 (outside top-10), multi-fact candidates,
#: facts that repeat across candidates, a short list, and no gold at all.
CASES = [
    (["alpha", "beta", "gamma"], ["alpha"]),
    (["filler"] * 10 + ["alpha"], ["alpha"]),                      # hit at rank 11
    (["filler"] * 9 + ["alpha"], ["alpha"]),                       # hit at rank 10
    (["alpha"], ["alpha", "beta"]),                                # one of two facts
    (["alpha beta", "gamma"], ["alpha", "beta", "gamma"]),         # multi-fact candidate
    (["alpha", "alpha", "beta"], ["alpha", "beta"]),               # repeated fact
    (["alpha", "beta"], []),                                       # no gold
    ([], ["alpha"]),                                               # empty ranking
    (["alpha beta gamma", "delta"], ["alpha", "beta", "gamma", "delta"]),
]


@pytest.mark.parametrize("retrieved,gold", CASES)
def test_both_metric_implementations_agree(retrieved, gold):
    summary, rows = official.official_metrics([retrieved], [gold])
    one = coverage.official_metrics_one(retrieved, gold)
    for metric in METRICS:
        assert one[metric] == pytest.approx(rows[0][metric]), (metric, retrieved, gold)
        assert one[metric] == pytest.approx(summary[metric]), (metric, retrieved, gold)


def test_rank_eleven_never_scores_in_either_implementation():
    """The boundary both implementations used to examine and then discard."""
    retrieved = ["filler"] * 10 + ["alpha"]
    gold = ["alpha"]
    summary, rows = official.official_metrics([retrieved], [gold])
    one = coverage.official_metrics_one(retrieved, gold)
    for metric in METRICS:
        assert rows[0][metric] == 0.0, metric
        assert one[metric] == 0.0, metric


def test_average_precision_uses_the_true_gold_count_in_both():
    """One of two gold facts retrieved at rank 1 is AP 1/2, not AP 1/1."""
    retrieved, gold = ["alpha"], ["alpha", "beta"]
    summary, _ = official.official_metrics([retrieved], [gold])
    assert summary["MAP@10"] == pytest.approx(0.5)
    assert coverage.official_metrics_one(retrieved, gold)["MAP@10"] == pytest.approx(0.5)


def test_metrics_are_whitespace_insensitive_in_both():
    retrieved = ["alpha\n  beta"]
    gold = ["alphabeta"]
    summary, _ = official.official_metrics([retrieved], [gold])
    assert summary["MAP@10"] == pytest.approx(1.0)
    assert coverage.official_metrics_one(retrieved, gold)["MAP@10"] == pytest.approx(1.0)
