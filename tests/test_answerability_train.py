"""Tests for the answerability-gate trainer.

The trainer is the part that turns the MultiHop gate run into a usable artefact, so these
tests pin its plumbing (feature order, split, threshold selection, persistence) on
synthetic data — no engine and no model needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from laya_rag.answerability import FEATURE_NAMES, LogisticGate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import answerability_train as train  # noqa: E402


def _record(label: int, strong: bool, head: float = 0.5) -> dict:
    features = {name: 0.0 for name in FEATURE_NAMES}
    features["max_relevance"] = 1.0 if strong else 0.1
    features["mean_relevance"] = 0.9 if strong else 0.05
    features["query_term_coverage"] = 0.8 if strong else 0.1
    if strong:
        # Two distinct documents for a spread-out answerable set.
        return {"q": "q", "label": label, "p": head, "features": features}
    return {"q": "q", "label": label, "p": head, "features": features}


def _synthetic(n: int = 40) -> list[dict]:
    rows = [_record(1, True) for _ in range(n)] + [_record(0, False) for _ in range(n)]
    return rows


def _write_gate_file(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "gate.json"
    path.write_text(
        json.dumps({"null": [r for r in rows if r["label"] == 0],
                    "positive": [r for r in rows if r["label"] == 1]}),
        encoding="utf-8",
    )
    return path


def test_load_records_flattens_both_groups(tmp_path):
    rows = _synthetic(5)
    records = train.load_records(_write_gate_file(tmp_path, rows))
    assert len(records) == 10
    assert {r["label"] for r in records} == {0, 1}


def test_load_records_rejects_a_run_without_features(tmp_path):
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"null": [{"q": "x", "p": 0.1}], "positive": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        train.load_records(path)


def test_build_matrix_respects_the_documented_feature_order():
    X, y, head = train.build_matrix(_synthetic(3))
    assert X.shape == (6, len(FEATURE_NAMES))
    assert list(y) == [1, 1, 1, 0, 0, 0]
    assert np.allclose(head, 0.5)
    column = FEATURE_NAMES.index("max_relevance")
    assert X[0, column] == 1.0 and X[-1, column] == 0.1


def test_stratified_split_keeps_the_class_ratio():
    _, y, _ = train.build_matrix(_synthetic(10))
    train_idx, test_idx = train.stratified_split(y, seed=1)
    assert set(train_idx).isdisjoint(set(test_idx))
    for half in (train_idx, test_idx):
        assert int(y[half].sum()) == len(half) // 2


def test_best_threshold_separates_perfectly():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    threshold, value = train.best_threshold(y, scores)
    assert value == pytest.approx(1.0)
    assert 0.2 < threshold <= 0.8


def test_evaluate_split_finds_signal_the_scalar_lacks():
    X, y, head = train.build_matrix(_synthetic(30))
    reading = train.evaluate_split(X, y, head, seed=3)
    assert reading["gate_auc"] > 0.9
    assert reading["head_auc"] == pytest.approx(0.5)  # all ties -> chance


def test_summarize_reports_mean_and_interval():
    summary = train.summarize([0.4, 0.5, 0.6])
    assert summary["mean"] == pytest.approx(0.5)
    assert summary["lo"] <= summary["mean"] <= summary["hi"]


def test_selective_metrics_reports_the_operating_point():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    metrics = train.selective_metrics(labels, scores, threshold=0.5)
    assert set(metrics) == {
        "aurc", "augrc", "risk@80", "risk@90", "brier", "nll", "ece", "coverage"
    }
    assert metrics["coverage"] == 0.5
    assert metrics["risk@80"] == 0.5
    assert metrics["aurc"] == pytest.approx(0.1458, abs=1e-4)
    assert 0.0 <= metrics["ece"] <= 1.0


def test_evaluate_split_carries_the_selective_metrics_for_both_scorers():
    X, y, head = train.build_matrix(_synthetic(30))
    reading = train.evaluate_split(X, y, head, seed=5)
    for prefix in ("gate", "head"):
        for name in ("aurc", "augrc", "risk@80", "brier", "nll", "ece", "coverage"):
            assert f"{prefix}_{name}" in reading
    assert 0.0 <= reading["gate_coverage"] <= 1.0
    assert 0.0 <= reading["head_risk@80"] <= 1.0


def test_main_writes_a_report_and_a_loadable_gate(tmp_path):
    rows = _synthetic(25)
    gate_file = _write_gate_file(tmp_path, rows)
    out = tmp_path / "report.json"
    code = train.main(["--features", str(gate_file), "--out", str(out), "--splits", "5"])
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["n"] == 50
    assert report["held_out"]["gate_auc"]["mean"] > 0.9
    loaded = LogisticGate.load(out.with_suffix(".json.gate"))
    assert loaded.predict_proba(np.zeros((1, len(FEATURE_NAMES)))).shape == (1,)
