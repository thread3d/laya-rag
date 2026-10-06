"""Tests for the leakage audit.

The dilution signature is decidable from a report alone: per-query entries must not outnumber the
recorded query limit. The held-out check is weaker but catches a tuning block that was recorded
without the reading that makes it checkable.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    name = "laya_rag_leakage_audit"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "leakage_audit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


la = _load()


def test_per_query_lengths_finds_nested_arrays():
    report = {
        "datasets": {"scifact": {"per_query_nDCG@10": {"a": 1.0, "b": 2.0}}},
        "per_query": {"x": 1.0},
    }
    lengths = dict(la.per_query_lengths(report))
    assert lengths["/per_query"] == 1
    assert lengths["/datasets/scifact/per_query_nDCG@10"] == 2


def test_a_diluted_report_is_flagged(tmp_path):
    report = {"_settings": {"limit_queries": 10}, "per_query_nDCG@10": {str(i): 0.0 for i in range(20)}}
    findings = la.audit_report(tmp_path / "r.json", report)
    assert [f["kind"] for f in findings] == ["dilution"]


def test_a_consistent_limited_report_passes(tmp_path):
    report = {"_settings": {"limit_queries": 10}, "per_query_nDCG@10": {str(i): 0.0 for i in range(10)}}
    assert la.audit_report(tmp_path / "r.json", report) == []


def test_a_held_out_block_without_a_checkable_field_is_flagged(tmp_path):
    report = {"holdout": {"seed": 17}}
    findings = la.audit_report(tmp_path / "r.json", report)
    assert [f["kind"] for f in findings] == ["held_out_missing"]


def test_a_held_out_block_with_a_field_passes(tmp_path):
    report = {"holdout": {"selected_on_a": "arm", "selected_b_score": 0.3}}
    assert la.audit_report(tmp_path / "r.json", report) == []


def test_audit_tree_scans_eval_reports(tmp_path):
    (tmp_path / "eval" / "sub").mkdir(parents=True)
    (tmp_path / "eval" / "sub" / "clean.json").write_text(
        json.dumps({"_settings": {"limit_queries": 5}, "per_query": {"a": 1.0}})
    )
    assert la.audit_tree(tmp_path)["ok"] is True
    (tmp_path / "eval" / "sub" / "diluted.json").write_text(
        json.dumps({"_settings": {"limit_queries": 1}, "per_query": {"a": 1.0, "b": 2.0}})
    )
    result = la.audit_tree(tmp_path)
    assert result["ok"] is False and len(result["findings"]) == 1


def test_check_mode_exits_one_on_a_finding(tmp_path):
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "diluted.json").write_text(
        json.dumps({"_settings": {"limit_queries": 1}, "per_query": {"a": 1.0, "b": 2.0}})
    )
    assert la.main(["--root", str(tmp_path), "--check"]) == 1
