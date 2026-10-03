"""Offline tests for the learned rank-fusion prototype.

The result is a null — a fitted scorer over the cached channels does not beat the hand-built
blend — and a null is only trustworthy if the fit could have found an effect. These pin the
trainer and, more importantly, the no-leakage property of the standardisation, because
standardising over both halves would let the held-out distribution into the model and turn a
null into a spurious win.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "laya_rag_ltr", ROOT / "tools" / "multihop_ltr.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ltr = _load()


def test_standardize_uses_training_statistics_only():
    """The held-out half must not shift the transform it is scored under."""
    train = np.array([[1.0, 10.0], [3.0, 30.0]])
    other = np.array([[100.0, 1000.0]])
    train_z, other_z = ltr.standardize(train, other)
    assert train_z.mean(axis=0) == pytest.approx([0.0, 0.0])
    assert train_z.std(axis=0) == pytest.approx([1.0, 1.0])
    # The held-out row keeps the training scale: (100 - 2) / 1 = 98 for the first column.
    assert other_z[0, 0] == pytest.approx(98.0)


def test_standardize_survives_a_constant_column():
    train = np.array([[5.0, 1.0], [5.0, 3.0]])
    train_z, other_z = ltr.standardize(train, np.array([[5.0, 9.0]]))
    assert np.isfinite(train_z).all()
    assert np.isfinite(other_z).all()


def test_fit_logistic_learns_the_sign_of_a_separating_feature():
    features = np.array([[2.0], [2.2], [-2.0], [-2.2]])
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    weights = ltr.fit_logistic(features, labels)
    assert weights[0] > 0, "a feature that is high on positives must get a positive weight"


def test_fit_logistic_is_deterministic():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(64, 3))
    labels = (features[:, 0] + rng.normal(scale=0.5, size=64) > 0).astype(float)
    assert (ltr.fit_logistic(features, labels) == ltr.fit_logistic(features, labels)).all()


def test_fit_logistic_handles_a_single_class_without_blowing_up():
    features = np.array([[1.0], [2.0], [3.0]])
    weights = ltr.fit_logistic(features, np.ones(3))
    assert np.isfinite(weights).all()


def test_rank_by_is_descending_with_index_tie_break():
    assert ltr.rank_by([0.1, 0.9, 0.9, 0.5]) == [1, 2, 3, 0]


def test_render_markdown_states_the_supervision_caveat():
    """A supervised number quoted without its caveat is the failure mode being avoided."""
    result = {
        "_settings": {
            "features": ["ce", "fused_rank", "tri"], "weights": [0.7, 0.46, 0.08],
            "train_queries": 200, "test_queries": 200, "seed": 17,
            "caveat": "supervised on the benchmark's own labels",
        },
        "summary": {
            "ce": {"Hits@10": 0.85, "Hits@4": 0.72, "MAP@10": 0.30, "MRR@10": 0.58},
            "ltr": {"Hits@10": 0.81, "Hits@4": 0.73, "MAP@10": 0.31, "MRR@10": 0.62},
        },
        "comparisons": {
            "ltr vs ce MAP@10": {
                "bootstrap": {"delta": 0.014, "ci_low": -0.0004, "ci_high": 0.029, "significant": False},
                "permutation": {"p_value": 0.074},
            }
        },
    }
    text = ltr.render_markdown(result)
    assert "supervised on the benchmark's own labels" in text
    assert "held-out" in text
    assert "+0.0140" in text
    assert "ltr vs ce MAP@10" in text
