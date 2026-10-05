"""Behavioural tests for the claim registry's multi-report support.

The registry exists because a hand-copied number drifts and a pinned one cannot. This session added
claims sourced from reports *other* than the headline diagnostics file — the FRAMES arms and the
touché comparison — so the machinery for that has to be shown working, not assumed: a wrong value
must fail, a missing report must skip rather than pass, and a claim must resolve against its own
report rather than the default one.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    name = "laya_rag_claim_check_reports"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "claim_check.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cc = _load()


def _doc(tmp_path: Path, text: str) -> None:
    (tmp_path / "d.md").write_text(text, encoding="utf-8")


def test_a_matching_value_passes(tmp_path):
    _doc(tmp_path, "| row | 0.5000 |")
    claim = cc.Claim("x", "d.md", r"^\| row \|", 0.5000, ("v",))
    assert cc.run_checks([claim], [], root=tmp_path, report={"v": 0.5}) == []


def test_a_value_the_document_does_not_state_fails(tmp_path):
    _doc(tmp_path, "| row | 0.5000 |")
    wrong = cc.Claim("x", "d.md", r"^\| row \|", 0.6000, ("v",))
    assert cc.run_checks([wrong], [], root=tmp_path, report={"v": 0.5})


def test_a_value_the_report_contradicts_fails(tmp_path):
    _doc(tmp_path, "| row | 0.5000 |")
    wrong = cc.Claim("x", "d.md", r"^\| row \|", 0.5000, ("v",))
    assert cc.run_checks([wrong], [], root=tmp_path, report={"v": 0.9})


def test_a_claim_resolves_against_its_own_report_not_the_default(tmp_path):
    _doc(tmp_path, "| row | 0.2000 |")
    claim = cc.Claim("x", "d.md", r"^\| row \|", 0.2000, ("a", "b"), report="other.json")
    default = {"a": {"b": 0.9}}
    assert cc.run_checks([claim], [], root=tmp_path, report=default,
                         reports={"other.json": {"a": {"b": 0.2}}}) == []
    assert cc.run_checks([claim], [], root=tmp_path, report=default,
                         reports={"other.json": {"a": {"b": 0.9}}})


def test_a_claim_whose_report_is_missing_is_skipped_not_failed(tmp_path):
    """eval/ is gitignored, so a fresh checkout has no reports and must still pass."""
    _doc(tmp_path, "| row | 0.1111 |")
    claim = cc.Claim("x", "d.md", r"^\| row \|", 0.1111, ("v",), report="absent.json")
    assert cc.run_checks([claim], [], root=tmp_path, report={}, reports={}) == []


def test_an_anchor_that_matches_nothing_fails(tmp_path):
    _doc(tmp_path, "nothing relevant here")
    claim = cc.Claim("x", "d.md", r"^\| row \|", 0.5000, ("v",))
    assert cc.run_checks([claim], [], root=tmp_path, report={"v": 0.5})
