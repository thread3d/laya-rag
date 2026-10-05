"""Offline tests for the added cross-system charts.

Two of these charts carry numbers that could mislead if the arithmetic drifted: the cost axis is
numeric because "335M" and "1.2B" have to become the same units, and the reversal chart plots a
*percentage* precisely because the corpora it spans do not share a metric. Both are pinned here.
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
    name = "laya_rag_compare_more"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "compare_more.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cm = _load()


def test_parameters_are_converted_to_one_unit():
    assert cm.parse_params("335M") == 335.0
    assert cm.parse_params("1.2B") == pytest.approx(1200.0)
    assert cm.parse_params("110M") == 110.0


def test_bm25_has_no_encoder_parameters_and_sits_at_zero():
    assert cm.parse_params("—") == 0.0
    assert cm.parse_params("") == 0.0


def test_the_reversal_delta_is_a_percentage_of_the_lexical_arm():
    rows = cm.reversal_rows()
    for row in rows:
        expected = 100.0 * (row["dense"] - row["lexical"]) / row["lexical"]
        assert row["delta_pct"] == pytest.approx(expected)


def test_the_reversal_chart_spans_both_signs():
    """The whole point of the figure: some corpora favour dense, some favour lexical."""
    deltas = [row["delta_pct"] for row in cm.reversal_rows()]
    assert max(deltas) > 0 and min(deltas) < 0


def test_every_reversal_row_carries_its_own_metric():
    """BEIR and FRAMES are nDCG@10 while the engine rows are MAP@10, so each row names its metric."""
    rows = cm.reversal_rows()
    assert {row["metric"] for row in rows} == {"nDCG@10", "MAP@10"}


def test_the_frames_arms_are_ranked_as_measured():
    summary = cm.chart_frames_arms.__doc__ is None  # keep the import used
    report = __import__("json").loads(cm.FRAMES_REPORT.read_text())["summary"]
    assert report["ensemble"]["recall@1"] > report["dense"]["recall@1"] > report["lexical"]["recall@1"]
