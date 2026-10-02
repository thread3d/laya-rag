"""Offline tests for the experimental evidence-shape gate features."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "laya_rag_gate_extra", ROOT / "tools" / "gate_extra_features.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extra = _load()


def test_max_passage_coverage_finds_the_best_single_passage():
    passages = ["alpha beta here", "gamma only"]
    assert extra.max_passage_coverage("alpha beta gamma", passages) == pytest.approx(2 / 3)
    assert extra.max_passage_coverage("nothing matches", passages) == 0.0
    assert extra.max_passage_coverage("", passages) == 0.0


def test_coverage_spread_counts_the_greedy_set_cover():
    passages = ["alpha beta here", "gamma only"]
    assert extra.coverage_spread("alpha beta gamma", passages) == 2.0
    assert extra.coverage_spread("alpha", passages) == 1.0
    assert extra.coverage_spread("nothing matches", passages) == 0.0
    assert extra.coverage_spread("alpha beta gamma", []) == 0.0


def test_pairwise_redundancy_is_mean_jaccard():
    assert extra.pairwise_redundancy(["alpha beta", "alpha beta"]) == pytest.approx(1.0)
    assert extra.pairwise_redundancy(["alpha beta", "gamma delta"]) == pytest.approx(0.0)
    assert extra.pairwise_redundancy(["alpha beta"]) == 0.0
    assert extra.pairwise_redundancy([]) == 0.0


def test_extra_features_accepts_objects_dicts_and_strings():
    objects = [SimpleNamespace(text="alpha beta"), SimpleNamespace(text="gamma")]
    dicts = [{"text": "alpha beta"}, {"text": "gamma"}]
    strings = ["alpha beta", "gamma"]
    for passages in (objects, dicts, strings):
        values = extra.extra_features("alpha beta gamma", passages)
        assert list(values.shape) == [3]
        assert values[0] == pytest.approx(2 / 3)  # max_passage_coverage
        assert values[1] == pytest.approx(2.0)    # coverage_spread
        assert values[2] == pytest.approx(0.0)    # pairwise_redundancy (disjoint)
