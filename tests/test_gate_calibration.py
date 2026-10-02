"""Offline tests for the gate threshold/calibration experiment."""
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
        "laya_rag_gate_calibration", ROOT / "tools" / "gate_calibration.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cal = _load()


def test_sigmoid_inverts_logit():
    probabilities = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
    np.testing.assert_allclose(cal.sigmoid(cal.logit(probabilities)), probabilities, atol=1e-6)


def test_negative_log_likelihood_rewards_confident_correct_predictions():
    labels = [1, 0]
    confident = cal.negative_log_likelihood(labels, [0.99, 0.01])
    unsure = cal.negative_log_likelihood(labels, [0.6, 0.4])
    assert confident < unsure


def test_fit_temperature_cools_overconfident_wrong_probabilities():
    # The model is confidently wrong on every point, so the fitted temperature must exceed 1.
    labels = [0, 0, 1, 1]
    probabilities = [0.9, 0.9, 0.1, 0.1]
    temperature = cal.fit_temperature(labels, probabilities)
    assert temperature > 1.0
    # And it must reduce NLL on those very data.
    cooled = cal.sigmoid(cal.logit(probabilities) / temperature)
    assert cal.negative_log_likelihood(labels, cooled) < cal.negative_log_likelihood(
        labels, probabilities
    )


def test_fit_temperature_is_one_grid_point_for_a_perfect_prediction():
    temperature = cal.fit_temperature([0, 1], [0.0, 1.0], grid=(1.0,))
    assert temperature == 1.0


def test_render_markdown_reports_both_arms():
    report = {
        "splits": {"mean": 200.0, "sd": 0.0},
        "threshold": {"mean": 0.47, "sd": 0.06, "min": 0.333, "max": 0.6},
        "temperature": {"mean": 1.01, "sd": 0.1},
        "fixed_accuracy": {"mean": 0.7424, "sd": 0.03},
        "learned_accuracy": {"mean": 0.7428, "sd": 0.03},
        "fixed_coverage": {"mean": 0.5312, "sd": 0.07},
        "learned_coverage": {"mean": 0.5486, "sd": 0.08},
        "raw_ece": {"mean": 0.0683, "sd": 0.02},
        "scaled_ece": {"mean": 0.0719, "sd": 0.02},
        "raw_brier": {"mean": 0.1706, "sd": 0.01},
        "scaled_brier": {"mean": 0.1717, "sd": 0.01},
        "raw_nll": {"mean": 0.5164, "sd": 0.04},
        "scaled_nll": {"mean": 0.5242, "sd": 0.05},
    }
    text = cal.render_markdown(report)
    assert "learned threshold: mean 0.47" in text
    assert "0.7424 | 0.7428" in text
    assert "0.0683 | 0.0719" in text
