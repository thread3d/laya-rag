"""Tests for the calibration row.

The metrics come from `evalkit`, which other reports already use, so these tests pin the two
things specific to this tool: that `gate.json`'s groups become the labels (null is the negative
class), and that a known-miscalibrated decision produces the ECE/Brier the definition implies.
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
    name = "laya_rag_calibration_report"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "calibration_report.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cal = _load()


def test_a_perfectly_calibrated_pole_predictor_is_perfect():
    result = cal.calibration([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0])
    assert result["ece_equal_width"] == 0.0
    assert result["brier"] == 0.0
    assert result["auc"] == 1.0


def test_an_overconfident_predictor_has_the_gap_it_implies():
    # Four items, all predicted 0.9, two correct: ECE is exactly 0.4 and Brier 0.41.
    result = cal.calibration([0, 0, 1, 1], [0.9, 0.9, 0.9, 0.9])
    assert result["ece_equal_width"] == pytest.approx(0.4, abs=1e-4)
    assert result["brier"] == pytest.approx(0.41, abs=1e-4)
    assert result["auc"] == 0.5


def test_positive_rate_is_reported():
    assert cal.calibration([0, 0, 0, 1], [0.1, 0.2, 0.3, 0.9])["positive_rate"] == 0.25


def test_load_head_treats_null_as_the_negative_class(tmp_path):
    path = tmp_path / "gate.json"
    path.write_text(
        '{"null": [{"p": 0.1}, {"p": 0.2}], "positive": [{"p": 0.8}, {"p": 0.9}]}'
    )
    probabilities, labels = cal.load_head(path)
    assert probabilities == [0.1, 0.2, 0.8, 0.9]
    assert labels == [0, 0, 1, 1]


def test_build_report_reads_the_committed_artifacts():
    if not cal.GATE.is_file():
        pytest.skip("gate.json not generated")
    report = cal.build_report()
    assert report["corpus"] == "MultiHop-RAG"
    assert report["shipped_head"]["n"] == 601
    assert "feature_gate" in report


def test_render_markdown_has_both_sections():
    if not cal.GATE.is_file():
        pytest.skip("gate.json not generated")
    text = cal.render_markdown(cal.build_report())
    assert "# Answerability calibration" in text
    assert "## Reliability (equal-width bins)" in text
    assert "## What this does not cover" in text
    for corpus in ("BEIR-10", "MuSiQue", "FRAMES"):
        assert corpus in text


def test_main_writes_the_report(tmp_path):
    if not cal.GATE.is_file():
        pytest.skip("gate.json not generated")
    out = tmp_path / "calibration"
    assert cal.main(["--out", str(out)]) == 0
    assert out.with_suffix(".json").is_file()
    assert out.with_suffix(".md").is_file()
