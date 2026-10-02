"""Offline tests for the adversarial retrieval regression suite."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "laya_rag_adversarial_audit", ROOT / "tools" / "adversarial_audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load()


def test_the_planted_document_is_flagged_as_an_injection():
    from laya_rag.verify import injection_flags

    assert "ignore_instructions" in injection_flags(audit.POISON_TEXT)


def test_trust_off_leaks_the_poison_and_keeps_gold():
    result = audit.retrieval_audit(k=5, manifest=None)
    assert result["gold_hit_rate"] == 1.0
    assert result["poison_leak_rate"] > 0.0
    assert result["negative_reject_rate"] == 1.0


def test_trust_on_excludes_the_poison_without_losing_gold():
    from laya_rag.trust import TrustManifest

    manifest = TrustManifest.from_mapping(
        {"default": "trusted", "rules": {audit.POISON_PATH: "untrusted"}}
    )
    result = audit.retrieval_audit(k=5, manifest=manifest, min_tier="trusted")
    assert result["poison_leak_rate"] == 0.0
    assert result["gold_hit_rate"] == 1.0


def test_report_checks_all_pass_and_filtering_costs_nothing():
    report = audit.build_report(k=5)
    assert report["false_refusal_rate"] == 0.0
    assert report["trust_off"]["poison_leak_rate"] > 0.0
    assert report["trust_on"]["poison_leak_rate"] == 0.0
    assert all(report["checks"].values())


def test_verification_probe_flags_the_false_number_and_the_injection():
    probe = audit.verification_probe()
    assert probe["number_flagged"] is True
    assert probe["injection_flagged"] is True


def test_render_markdown_lists_both_scenarios_and_checks():
    report = audit.build_report(k=5)
    report["verification"] = audit.verification_probe()
    report["checks"]["verification_flags_number"] = True
    report["checks"]["verification_flags_injection"] = True
    text = audit.render_markdown(report)
    assert "| poison leak rate |" in text
    assert "PASS: poison_excluded" in text


def test_main_check_mode_returns_zero():
    assert audit.main(["--check", "--out", "/tmp/adversarial-audit-test"]) == 0
