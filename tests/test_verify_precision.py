"""Offline tests for the deterministic verifier's precision harness.

The harness carries a shipping bar — advisory checks must be precise enough not to become noise —
so the tests pin three things: the labelled set is balanced and reproducible, the metric arithmetic
counts a false positive as a false positive, and the **known hard near-misses stay known**. The
last one matters: a test that only asserted "precision is 1.0" would pass forever while quietly
losing the cases the detector actually struggles with.
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
    name = "laya_rag_verify_precision"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "verify_precision.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


vp = _load()


def test_the_set_is_balanced_and_deterministic():
    cases = vp.build_cases(10)
    for kind in vp.KINDS:
        positives = [c for c in cases if c.kind == kind and c.positive]
        negatives = [c for c in cases if c.kind == kind and not c.positive]
        assert len(positives) == len(negatives) == 10
    assert vp.build_cases(10) == cases


def test_clear_mutations_are_all_detected():
    report = vp.summarize(vp.build_cases(20))
    for kind, row in report["per_kind"].items():
        assert row["recall"] == 1.0, f"{kind} missed a clear mutation"
        assert row["precision"] == 1.0, f"{kind} false-fired on a faithful control"
    assert report["control_false_alarm_rate"] == 0.0


def test_a_wrong_flag_on_a_control_lowers_precision():
    """A faithful claim that fires anyway is a false positive, and must move the number."""
    bad = vp.Case(kind="number_not_in_evidence", positive=False,
                  claim="The plant produced 999 widgets.",
                  evidence="The plant produced 1 widget.")
    good = vp.Case(kind="number_not_in_evidence", positive=True,
                   claim="The plant produced 999 widgets.",
                   evidence="The plant produced 12 widgets.")
    report = vp.summarize([bad, good])
    row = report["per_kind"]["number_not_in_evidence"]
    assert row["false_positives"] == 1
    assert row["precision"] == pytest.approx(0.5)   # one hit, one false alarm
    assert row["recall"] == 1.0


def test_the_known_hard_near_misses_stay_known():
    """Cross-format dates and rounding are the detector's honest limits, not silent passes.

    Every hard case is a documented "legitimate exception" and should ideally be quiet. Two are
    not: a date in another format is read as a different date *and* as absent numbers, and a
    rounded figure is read as an absent number. If a future change fixes either, this test fails
    and the limits in the write-up must be updated with it.
    """
    rows = {row["label"]: row for row in vp.hard_case_results()}
    assert rows["same date, another format"]["quiet"] is False
    assert "date_not_in_evidence" in rows["same date, another format"]["kinds"]
    assert rows["rounding"]["quiet"] is False
    for label in ("range endpoint", "number spelled out", "percent as decimal",
                  "negation in both texts"):
        assert rows[label]["quiet"] is True, f"{label} should not be flagged"


def test_the_reported_hard_case_rate_is_what_the_cases_show():
    report = vp.summarize(vp.build_cases(5))
    flagged = sum(1 for row in report["hard_cases"] if not row["quiet"])
    expected = round(flagged / len(report["hard_cases"]), 4)  # the report stores 4 dp
    assert report["hard_case_false_alarm_rate"] == pytest.approx(expected)


def test_check_passes_at_the_default_bar(tmp_path):
    assert vp.main(["--size", "10", "--check", "--out", str(tmp_path / "r")]) == 0


def test_check_fails_below_an_impossible_bar(tmp_path):
    assert vp.main(["--size", "10", "--check", "--min-precision", "1.01",
                    "--out", str(tmp_path / "r")]) == 1
