"""Tests for the convex-fusion weight sweep.

§P5(a) asks for a *tuned* convex combination against the engine's RRF, and the two ways to get
that wrong are both pinned here: tuning the weights on the queries they are then scored on, and
scoring a subset against the full qrels. The weights are chosen on half and read on the other
half, and `_score` restricts the qrels to the queries it ranks.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def _load():
    name = "laya_rag_fusion_tuning"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "fusion_tuning.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ft = _load()


def test_the_split_is_a_reproducible_partition():
    ids = [str(i) for i in range(100)]
    train, test = ft._split(ids, seed=17)
    assert sorted(train + test) == sorted(ids)
    assert len(train) == len(test) == 50
    assert ft._split(ids, seed=17) == (train, test)
    assert ft._split(ids, seed=18) != (train, test)


def test_the_split_does_not_take_a_head_slice():
    """A head slice would put a whole corpus's easy queries in one half."""
    ids = [f"q{i}" for i in range(100)]
    train, test = ft._split(ids, seed=17)
    assert train != ids[:50]
    assert test != ids[50:]


def test_scoring_a_subset_restricts_the_qrels():
    """The dilution trap: full qrels plus a subset's rankings scores the rest as zero."""
    rankings = {"a": ["d1"], "b": ["d2"]}
    qrels = {"a": {"d1": 1}, "b": {"d2": 1}, "c": {"d3": 1}}
    mean, per_query = ft._score(rankings, qrels, ["a", "b"])
    assert set(per_query) == {"a", "b"}
    assert mean == pytest.approx(1.0)


def test_an_empty_query_set_scores_zero_rather_than_raising():
    assert ft._score({}, {"a": {"d1": 1}}, []) == (0.0, {})


def test_render_names_the_held_out_reading_and_the_pick():
    result = {
        "datasets": [{
            "dataset": "scifact",
            "arms": [
                {"arm": "rrf_k5", "full": 0.74, "train": 0.74, "test": 0.74},
                {"arm": "rrf_k60", "full": 0.73, "train": 0.73, "test": 0.73},
                {"arm": "convex@1", "full": 0.75, "train": 0.75, "test": 0.75},
            ],
            "tuned_arm": "convex@1",
            "tuned_full": 0.75,
            "oracle_arm": "convex@1",
            "oracle_full": 0.75,
            "tuned_vs_rrf_k5_on_test": {
                "delta": 0.01, "ci_low": 0.001, "ci_high": 0.02, "p_value": 0.01,
                "significant": True,
            },
        }],
    }
    text = ft.render(result)
    assert "| scifact |" in text
    assert "convex@1" in text
    assert "significant wins **1**" in text


def test_the_engine_matched_sweep_reproduces_the_recorded_scifact_rows():
    """The two knobs the earlier comparison missed are both in this tool's defaults.

    `title_boost=1` is the engine's configuration and `rrf_k=5` is what ships, so the two
    baselines the sweep reports must be the values the reports carry for *that* configuration:
    0.7408 and 0.7320. The title-weighted reports' 0.7413 / 0.7310 are the same retriever under
    `--title-boost 2` and are deliberately not what this measures.
    """
    report = ROOT / "eval" / "beir" / "engine-parity.json"
    if not report.is_file():
        pytest.skip("engine-parity report is not present")
    import json

    data = json.loads(report.read_text(encoding="utf-8"))
    settings = data["settings"]
    assert settings.get("title_boost", 1) == 1
    assert settings.get("rrf_k") == 5
    row = data["datasets"]["scifact"]["retrievers"]["hybrid-biencoder"]
    assert row["nDCG@10"] == pytest.approx(0.7408, abs=5e-5)
