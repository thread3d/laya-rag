"""Offline tests for the harness-vs-engine comparison.

The comparison's only dangerous step is the pairing: the engine report carries query keys, the
harness report carries rows in subset order, and nothing in the harness report says which
questions those rows are. These tests pin the rule that makes the pairing safe — the subset is
re-derived from the seed the engine report recorded, and a mismatch is refused.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def _load():
    name = "laya_rag_engine_vs_harness"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "engine_vs_harness.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evh = _load()


def _items(n=12, nulls=()):
    return [
        {"question_type": "null_query" if i in nulls else "inference_query", "query": f"q{i}"}
        for i in range(n)
    ]


def test_subset_keys_skips_null_queries_and_keeps_original_indices():
    items = _items(6, nulls=(1, 4))
    keys = evh.subset_keys(items, limit=4, sample_seed=17)
    # limit 4 covers all four answerable records, so every answerable index appears in order.
    assert keys == ["0", "2", "3", "5"]


def test_subset_keys_samples_positions_like_the_harness():
    import random as _random

    items = _items(40, nulls=(0, 7, 13))
    answerable = [i for i, x in enumerate(items) if x["question_type"] != "null_query"]
    positions = sorted(_random.Random(17).sample(range(len(answerable)), 5))
    assert evh.subset_keys(items, limit=5, sample_seed=17) == [str(answerable[p]) for p in positions]


def test_compare_reports_the_engine_minus_harness_delta():
    keys = ["0", "1"]
    engine = {
        "queries": {k: f"q{k}" for k in keys},
        "per_query": {
            "0": {"metrics": {"MAP@10": 0.5, "Hits@4": 1.0, "Hits@10": 1.0, "MRR@10": 1.0}},
            "1": {"metrics": {"MAP@10": 0.1, "Hits@4": 0.0, "Hits@10": 1.0, "MRR@10": 0.5}},
        },
    }
    harness_arm = {
        "_per_query": [
            {"MAP@10": 0.4, "Hits@4": 1.0, "Hits@10": 1.0, "MRR@10": 1.0},
            {"MAP@10": 0.0, "Hits@4": 0.0, "Hits@10": 1.0, "MRR@10": 0.5},
        ]
    }
    rows = {row["metric"]: row for row in evh.compare(engine, harness_arm, keys, samples=200)}
    assert rows["MAP@10"]["delta"] == pytest.approx(0.1)
    assert rows["MAP@10"]["engine"] == pytest.approx(0.3)
    assert rows["MAP@10"]["harness"] == pytest.approx(0.2)
    assert rows["Hits@4"]["delta"] == pytest.approx(0.0)


def test_compare_refuses_reports_from_different_subsets():
    engine = {"queries": {"0": "q0"}, "per_query": {"0": {"metrics": {m: 0.0 for m in evh.METRICS}}}}
    harness_arm = {"_per_query": [{"MAP@10": 0.0}] * 3}
    with pytest.raises(ValueError, match="not run on the same subset"):
        evh.compare(engine, harness_arm, ["0"])


def test_compare_requires_every_metric_in_both_reports():
    engine = {"per_query": {"0": {"metrics": {"MAP@10": 0.0}}}}
    harness_arm = {"_per_query": [{"MAP@10": 0.0}]}
    with pytest.raises(KeyError):
        evh.compare(engine, harness_arm, ["0"], samples=10)


def test_main_refuses_a_report_without_a_recorded_seed(tmp_path, monkeypatch):
    engine_path = tmp_path / "engine.json"
    engine_path.write_text(json.dumps({
        "_settings": {"limit": 400, "sample_seed": None},
        "queries": {"0": "q0"},
        "per_query": {},
    }))
    with pytest.raises(SystemExit, match="limit/--sample-seed"):
        evh.main(["--engine", str(engine_path)])


def test_main_refuses_engine_keys_that_are_not_the_reconstructed_subset(tmp_path):
    items_path = tmp_path / "items.json"
    items_path.write_text(json.dumps(_items(6, nulls=(1, 4))))
    engine_path = tmp_path / "engine.json"
    engine_path.write_text(json.dumps({
        "_settings": {"limit": 2, "sample_seed": 17},
        "queries": {"0": "q0", "99": "not in the subset"},
        "per_query": {m: {} for m in evh.METRICS},
    }))
    harness_path = tmp_path / "harness.json"
    harness_path.write_text(json.dumps({"arm": {"_per_query": [{}, {}]}}))
    with pytest.raises(SystemExit, match="not the limit"):
        evh.main([
            "--engine", str(engine_path), "--harness", str(harness_path),
            "--arm", "arm", "--items", str(items_path),
        ])
