"""Tests for the generated lexical-control summary.

This report exists because the sentence summarising the control chart was wrong twice. The counts it
produces are now what the prose is pinned to, so the counting rule itself needs pinning: a gap exactly
on the tolerance counts as inside it, and the outliers are only those beyond the loose one.
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
    name = "laya_rag_lexical_control"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "lexical_control.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lc = _load()


def _rows(*gaps):
    return [{"dataset": f"d{i}", "gap": gap} for i, gap in enumerate(gaps)]


def test_counts_are_split_by_tolerance():
    summary = lc.summarise(_rows(0.001, -0.005, 0.015, -0.030))
    assert summary["corpora"] == 4
    assert summary["within_tight"] == 2      # 0.001, -0.005
    assert summary["within_loose"] == 3      # plus 0.015


def test_a_gap_exactly_on_the_tolerance_counts_as_inside():
    """`<=` not `<`: the boundary is the claim being made."""
    assert lc.summarise(_rows(lc.TIGHT))["within_tight"] == 1
    assert lc.summarise(_rows(lc.LOOSE))["within_loose"] == 1


def test_outliers_are_reported_by_side():
    summary = lc.summarise(_rows(0.05, -0.05, 0.001))
    assert summary["in_our_favour"] == ["d0"]
    assert summary["against_us"] == ["d1"]


def test_an_empty_or_perfect_set_is_handled():
    assert lc.summarise([])["corpora"] == 0
    summary = lc.summarise(_rows(0.0, 0.0))
    assert summary["within_tight"] == 2 and summary["in_our_favour"] == []


def test_the_report_shape_matches_what_claim_check_pins():
    """The claims read counts.corpora / counts.within_0.02 / counts.within_0.008."""
    summary = lc.summarise(_rows(0.001, 0.05))
    counts = {"corpora": summary["corpora"], "within_0.008": summary["within_tight"],
              "within_0.02": summary["within_loose"]}
    assert counts == {"corpora": 2, "within_0.008": 1, "within_0.02": 1}
