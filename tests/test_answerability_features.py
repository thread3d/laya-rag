"""Offline tests for the answerability feature-engineering experiment."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

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
