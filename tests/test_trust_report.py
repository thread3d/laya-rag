"""Offline tests for the aggregated trust report.

The report's job is to state claims that are already measured, so the properties worth pinning are
the ones that could turn a missing measurement into a passing one: every check is pass/fail/skip,
a skipped input is never counted as a pass, and the point estimate of a `{mean, lo, hi}` metric is
read correctly rather than mistaken for a number.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def _load():
    name = "laya_rag_trust_report"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "trust_report.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tr = _load()

SELECTIVE = ROOT / "eval" / "benchmarks" / "multihop-rag" / "answerability-selective.json"


def test_mean_reads_the_point_estimate_of_an_interval():
    assert tr._mean({"mean": 0.8146, "lo": 0.76, "hi": 0.86}) == pytest.approx(0.8146)
    assert tr._mean(0.5) == pytest.approx(0.5)
    assert tr._mean(None) == 0.0


def test_summarize_separates_failures_from_skips():
    summary = tr.summarize({"checks": {"a": True, "b": False, "c": None}})
    assert summary["passed"] == 1
    assert summary["failed"] == ["b"]
    assert summary["skipped"] == ["c"]
    assert summary["ok"] is False


def test_a_skipped_input_is_not_a_pass():
    """A missing selective suite must never read as success."""
    summary = tr.summarize({"checks": {"abstention_beats_chance": None}})
    assert summary["ok"] is True          # nothing *failed*
    assert summary["passed"] == 0          # but nothing passed either
    assert summary["skipped"] == ["abstention_beats_chance"]


def _synthetic_report():
    return {
        "audit": {
            "settings": {"k": 5, "poison_path": "poison/x.md"},
            "trust_off": {"gold_hit_rate": 1.0, "poison_leak_rate": 0.2,
                          "negative_reject_rate": 1.0, "injection_flag_rate": 1.0},
            "trust_on": {"gold_hit_rate": 1.0, "poison_leak_rate": 0.0,
                         "negative_reject_rate": 1.0, "injection_flag_rate": 1.0},
            "tenants": {"alpha_sees_beta": False, "beta_sees_alpha": False,
                        "anonymous_passages": 0},
            "false_refusal_rate": 0.0,
            "checks": {"poison_excluded": True},
            "poison_dose": {"k": 5, "answerable": 2,
                            "leak_rate_by_dose": {"0": 0.0, "1": 0.5}},
        },
        "verification": {"citation_mechanics_clean": True, "number_flagged": True,
                         "injection_flagged": True, "content_kinds": ["number_not_in_evidence"]},
        "abstention": {"gate_auc": {"mean": 0.8}, "gate_risk@80": {"mean": 0.4},
                       "head_auc": {"mean": 0.5}, "head_risk@80": {"mean": 0.5}},
        "checks": {"audit_poison_excluded": True},
    }


def test_render_markdown_shows_both_abstention_arms():
    report = _synthetic_report()
    report["summary"] = tr.summarize(report)
    text = tr.render_markdown(report)
    assert "gate (12 features)" in text and "scalar head (shipped)" in text
    assert "0.4000" in text and "0.5000" in text
    assert "`audit_poison_excluded`: pass" in text


def test_render_markdown_marks_a_missing_suite_as_not_measured():
    report = _synthetic_report()
    report["abstention"] = None
    report["summary"] = tr.summarize(report)
    assert "not measured" in tr.render_markdown(report)


def test_collect_skips_abstention_when_the_suite_is_absent(tmp_path):
    report = tr.collect(tmp_path / "absent.json")
    assert report["abstention"] is None
    assert report["checks"]["abstention_beats_chance"] is None
    assert report["checks"]["abstention_beats_scalar_risk@80"] is None
    # The deterministic halves still ran and passed.
    assert report["checks"]["verification_citation_mechanics_clean"] is True


@pytest.mark.skipif(not SELECTIVE.is_file(), reason="run `make multihop-answerability` first")
def test_collect_scores_the_real_selective_suite():
    report = tr.collect(SELECTIVE)
    assert report["abstention"] is not None
    assert report["checks"]["abstention_beats_chance"] is True
    assert report["checks"]["abstention_beats_scalar_risk@80"] is True
    assert tr.summarize(report)["ok"] is True
