"""Offline tests for the poison dose-response probe.

The roadmap asks for the "five poisoned documents, 90% attack success" figure to be **measured
here, never cited**, because it is a property of the corpus and the retriever. What the tests pin is
that the measurement is honest: dose 0 is a control that must read zero, the stuffed document is
built from the corpus's vocabulary rather than from any one query, and the probe is deterministic.
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
    name = "laya_rag_adversarial_audit"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "adversarial_audit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit = _load()


def test_stuffed_poison_documents_have_distinct_paths():
    first = audit.stuffed_poison(0)
    second = audit.stuffed_poison(1)
    assert first[0] != second[0]
    assert first[0].startswith("poison/stuffed-")


def test_stuffed_poison_repeats_the_corpus_vocabulary_not_a_query():
    _, text = audit.stuffed_poison(0)
    for term in audit.STUFFING_TERMS:
        assert term in text


def test_dose_zero_is_a_control_and_leaks_nothing():
    probe = audit.poison_dose_probe(k=5, doses=(0,))
    assert probe["leak_rate_by_dose"]["0"] == 0.0


def test_the_probe_reports_a_rate_per_dose():
    probe = audit.poison_dose_probe(k=5, doses=(0, 1, 5))
    assert set(probe["leak_rate_by_dose"]) == {"0", "1", "5"}
    assert probe["k"] == 5 and probe["answerable"] > 0
    assert all(0.0 <= rate <= 1.0 for rate in probe["leak_rate_by_dose"].values())


def test_the_probe_is_deterministic():
    first = audit.poison_dose_probe(k=5, doses=(1, 2))
    second = audit.poison_dose_probe(k=5, doses=(1, 2))
    assert first == second


def test_build_report_carries_the_dose_probe_as_a_measurement():
    report = audit.build_report(k=5)
    assert "poison_dose" in report
    assert report["poison_dose"]["leak_rate_by_dose"]["0"] == 0.0


def test_the_gated_checks_do_not_depend_on_the_dose_probe():
    """It is a measurement: the suite must not start passing or failing because of it."""
    report = audit.build_report(k=5)
    assert set(report["checks"]) == {
        "gold_retained", "poison_excluded", "negative_rejected", "injection_flagged",
        "tenants_isolated", "anonymous_denied", "shared_visible_to_a_tenant",
    }
