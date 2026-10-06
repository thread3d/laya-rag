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


# ── tenant isolation ─────────────────────────────────────────────────────

def test_tenant_probe_finds_no_cross_tenant_leak():
    probe = audit.tenant_probe(k=5)
    assert probe["alpha_sees_beta"] is False
    assert probe["beta_sees_alpha"] is False
    # Once a tenant block exists, no identity retrieves nothing.
    assert probe["anonymous_passages"] == 0
    # A shared document is still visible to a listed tenant.
    assert probe["alpha_shared_visible"] is True


def test_report_includes_the_tenant_checks_and_they_pass():
    report = audit.build_report(k=5)
    for name in ("tenants_isolated", "anonymous_denied", "shared_visible_to_a_tenant"):
        assert name in report["checks"]
        assert report["checks"][name] is True
    assert report["tenants"]["alpha_sees_beta"] is False


# ── seeded noise ─────────────────────────────────────────────────────────

def test_noise_documents_are_deterministic_for_a_seed():
    assert audit.noise_documents(5, seed=7) == audit.noise_documents(5, seed=7)
    assert audit.noise_documents(5, seed=7) != audit.noise_documents(5, seed=8)


def test_noise_dose_zero_is_the_control():
    probe = audit.noise_probe(k=5, doses=(0,))
    assert probe["noise_in_top_k_by_dose"]["0"] == 0.0
    assert probe["gold_hit_rate_by_dose"]["0"] == 1.0


def test_noise_curve_is_recorded_per_dose():
    probe = audit.noise_probe(k=5, doses=(0, 5))
    assert set(probe["gold_hit_rate_by_dose"]) == {"0", "5"}
    assert probe["seed"] == audit.NOISE_SEED


def test_shared_vocabulary_noise_displaces_gold_at_high_density():
    """The measured finding: unrelated-looking distractor text that shares the query vocabulary
    eventually outranks a gold document. Dose 0 is clean; dose 100 is not."""
    probe = audit.noise_probe(k=5, doses=(0, 100))
    assert probe["gold_hit_rate_by_dose"]["0"] == 1.0
    assert probe["gold_hit_rate_by_dose"]["100"] < 1.0


# ── injection variants ───────────────────────────────────────────────────

def test_injection_probe_flags_every_payload_and_no_control():
    probe = audit.injection_probe()
    assert probe["recall"] == 1.0
    assert probe["control_false_positives"] == 0
    assert len(probe["markers_covered"]) == 6


def test_injection_probe_carries_a_benign_near_miss():
    probe = audit.injection_probe()
    near_miss = [row for row in probe["variants"] if row["name"] == "benign_near_miss"][0]
    assert near_miss["found"] == [] and near_miss["ok"] is True


def test_report_includes_the_noise_and_injection_checks():
    report = audit.build_report(k=5)
    for name in ("injection_variants_flagged", "injection_controls_clean", "noise_control_clean"):
        assert report["checks"][name] is True
    text = audit.render_markdown(report)
    assert "## Seeded noise" in text
    assert "## Injection variants" in text

