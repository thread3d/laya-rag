"""Offline tests for the significance/FDR post-processor.

The permutation and BH machinery is tested in ``test_evalkit.py``; these pin the report
plumbing — which rows are comparable, positional pairing, baseline choice, and that a
consistent win survives FDR while an identical system does not.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "laya_rag_eval_significance", ROOT / "tools" / "eval_significance.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sig = _load()


def _series(values, metric="MAP@10"):
    return {"_per_query": [{metric: v} for v in values]}


def _report():
    zeros = [0.0] * 10
    wins = [0.2] * 10
    return {
        "_settings": {"overlap": 200},
        "bge-small hybrid@ov200": _series(zeros),
        "bge-small dense@ov200": _series(wins),
        "bge-small same@ov200": _series(zeros),
        "broken": {"MAP@10": 0.5},
    }


def test_compared_rows_keeps_only_rows_with_a_per_query_series():
    rows = sig.compared_rows(_report())
    assert rows == ["bge-small hybrid@ov200", "bge-small dense@ov200", "bge-small same@ov200"]


def test_per_query_series_keys_positionally_and_raises_on_missing_metric():
    series = sig.per_query_series(_report(), "bge-small dense@ov200", "MAP@10")
    assert list(series) == [str(i) for i in range(10)]
    assert series["3"] == 0.2
    try:
        sig.per_query_series(_report(), "bge-small dense@ov200", "MRR@10")
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("a missing metric should raise")


def test_choose_baseline_prefers_hybrid_then_respects_an_explicit_request():
    report = _report()
    assert sig.choose_baseline(report, None) == "bge-small hybrid@ov200"
    assert sig.choose_baseline(report, "bge-small dense@ov200") == "bge-small dense@ov200"


def test_a_consistent_win_survives_fdr_and_an_identical_row_does_not():
    report = _report()
    family = ["bge-small dense@ov200", "bge-small same@ov200"]
    cells = sig.comparison_cells(
        report, "bge-small hybrid@ov200", family, ["MAP@10"], samples=200, seed=1
    )
    adjusted = sig.apply_fdr(cells, q=0.05)
    by_row = {cell["row"]: cell for cell in cells}
    assert by_row["bge-small dense@ov200"]["rejected_fdr"] is True
    assert by_row["bge-small dense@ov200"]["delta"] == 0.2
    assert by_row["bge-small same@ov200"]["rejected_fdr"] is False
    assert by_row["bge-small same@ov200"]["p_value"] == 1.0
    assert adjusted["tests"] == 2
    assert sig.summarize(cells, adjusted)["cells"] == 2


def test_render_markdown_lists_every_cell_and_the_summary():
    report = _report()
    cells = sig.comparison_cells(
        report, "bge-small hybrid@ov200", ["bge-small dense@ov200"], ["MAP@10"],
        samples=200, seed=1,
    )
    adjusted = sig.apply_fdr(cells, q=0.05)
    text = sig.render_markdown({
        "report": "x.json",
        "baseline": "bge-small hybrid@ov200",
        "family": ["bge-small dense@ov200"],
        "summary": sig.summarize(cells, adjusted),
        "cells": cells,
    })
    assert "bge-small dense@ov200" in text
    assert "BH p" in text
    assert "significant after BH-FDR: **1**" in text
