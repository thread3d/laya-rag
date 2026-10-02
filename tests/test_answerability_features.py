"""Offline tests for the answerability feature-engineering experiment."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "laya_rag_answerability_features", ROOT / "tools" / "answerability_features.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


feat = _load()


def _base_row():
    # In FEATURE_NAMES order: max/mean/min relevance, max/mean recall,
    # distinct_document_ratio, passage_count, log_evidence_chars, query_term_coverage.
    return np.array([[1.0, 0.5, 0.2, 0.9, 0.4, 0.5, 4.0, 5.0, 0.3]])


def test_augment_appends_columns_in_the_documented_order():
    augmented = feat.augment(_base_row())
    assert augmented.shape == (1, 9 + len(feat.DERIVED))
    expected = [
        0.5,                                   # relevance_margin
        0.8,                                   # relevance_spread
        -0.1,                                  # recall_relevance_agreement
        0.3,                                   # coverage_times_max_relevance
        2.0,                                   # distinct_documents
        5.0 - np.log1p(4.0),                   # chars_per_passage
        1.0,                                   # max_relevance_squared
        0.25,                                  # mean_relevance_squared
        0.9,                                   # max_recall_times_max_relevance
    ]
    assert np.allclose(augmented[0, 9:], expected)


def test_augment_preserves_the_base_columns():
    base = _base_row()
    assert np.allclose(feat.augment(base)[:, :9], base)


def test_augment_handles_several_rows():
    base = np.vstack([_base_row()[0], _base_row()[0] * 2])
    augmented = feat.augment(base)
    assert augmented.shape == (2, 18)
    # Doubling the base doubles the squared terms (not the linear margin).
    margin = 9  # relevance_margin column
    spread = 10
    assert augmented[1, margin] == 2 * augmented[0, margin]
    assert augmented[1, spread] == 2 * augmented[0, spread]


def test_evaluate_reports_the_expected_metric_keys():
    rng = np.random.default_rng(0)
    X = np.column_stack([
        rng.normal(1.0, 0.3, 60), rng.normal(0.5, 0.2, 60), rng.normal(0.1, 0.1, 60),
        rng.normal(0.8, 0.2, 60), rng.normal(0.4, 0.2, 60), rng.uniform(0.2, 1.0, 60),
        rng.uniform(1, 6, 60), rng.uniform(3, 7, 60), rng.uniform(0.0, 1.0, 60),
    ])
    y = np.array([1] * 30 + [0] * 30, dtype=float)
    readings = feat.evaluate(feat.augment(X), y, splits=3, seed=1)
    assert set(readings) == {"auc", "balanced_accuracy", "aurc", "augrc", "risk@80", "ece"}
    for value in readings.values():
        assert value["lo"] <= value["mean"] <= value["hi"]


def test_load_extras_reads_the_recorded_columns_in_record_order(tmp_path):
    import json

    payload = {
        "null": [
            {"extra": {"max_passage_coverage": 0.1, "coverage_spread": 2.0,
                       "pairwise_redundancy": 0.3}},
        ],
        "positive": [
            {"extra": {"max_passage_coverage": 0.9, "coverage_spread": 1.0,
                       "pairwise_redundancy": 0.0}},
            {"extra": {}},  # a record with no extras is zero, not an error
        ],
    }
    path = tmp_path / "gate-extra.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    extras = feat.load_extras(path)

    assert extras.shape == (3, 3)
    assert list(extras[0]) == [0.1, 2.0, 0.3]   # null first
    assert list(extras[1]) == [0.9, 1.0, 0.0]   # then positive
    assert list(extras[2]) == [0.0, 0.0, 0.0]


def test_evaluate_splits_returns_one_reading_per_split():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = np.array([1, 0] * 40, dtype=float)
    readings = feat.evaluate_splits(X, y, splits=5, seed=3)
    assert 0 < len(readings) <= 5
    assert set(readings[0]) == {"auc", "balanced_accuracy", "aurc", "augrc", "risk@80", "ece"}


def test_paired_across_splits_reports_a_positive_paired_delta():
    base = [{"auc": 0.50}, {"auc": 0.60}, {"auc": 0.70}]
    augmented = [{"auc": 0.60}, {"auc": 0.70}, {"auc": 0.80}]
    paired = feat.paired_across_splits(base, augmented, samples=500)
    assert paired["auc"]["delta"] == pytest.approx(0.1, abs=1e-9)
    assert paired["auc"]["ci_low"] > 0.0


def test_prefix_mode_compares_the_first_n_columns_with_the_full_vector(tmp_path):
    import json

    from laya_rag.answerability import FEATURE_NAMES

    rng = np.random.default_rng(0)
    rows = {"null": [], "positive": []}
    for group, label in (("null", 0), ("positive", 1)):
        for _ in range(20):
            values = {name: float(rng.normal(label, 0.4)) for name in FEATURE_NAMES}
            rows[group].append({"features": values, "label": label})
    path = tmp_path / "gate12.json"
    path.write_text(json.dumps(rows), encoding="utf-8")

    code = feat.main([
        "--features", str(path), "--prefix", "9", "--splits", "3",
        "--out", str(tmp_path / "prefix-out"),
    ])

    assert code == 0
    payload = json.loads((tmp_path / "prefix-out.json").read_text(encoding="utf-8"))
    assert payload["derived_names"] == list(FEATURE_NAMES[9:])
    assert "paired" in payload and "auc" in payload["paired"]
