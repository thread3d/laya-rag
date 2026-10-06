"""Tests for the cross-document consistency gate.

The gate exists because the same quantity drifted between hand-written documents: the BEIR coverage
count as 5/7/10, the published `voyage-02` row round-truncated in one table and rounded in another,
the recalibration half-split count as 300 in its report and 500 in two documents. These tests pin the
two rules that make the gate able to catch that: the rendering is round-**half-up** (so 0.6625 is
0.663, not 0.662), and a superseded report may not be read as current.
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
    name = "laya_rag_consistency_check"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "consistency_check.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cc = _load()


# ── the rendering rule ───────────────────────────────────────────────────

def test_rendering_is_half_up_not_half_even():
    """This is the drift: 0.6625 is 0.663 half-up and 0.662 half-even/truncated."""
    assert cc.half_up(0.6625, 3) == 0.663
    assert cc.half_up(0.4795, 3) == 0.480
    assert cc.half_up(0.4515, 3) == 0.452


def test_rendering_leaves_an_unambiguous_value_alone():
    assert cc.half_up(0.6624, 3) == 0.662
    assert cc.half_up(10, 0) == 10.0
    assert cc.half_up(0.7467, 3) == 0.747


# ── coverage is computed, not typed ──────────────────────────────────────

def _write_report(directory: Path, stem: str, datasets) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.json").write_text(json.dumps({"datasets": {d: {} for d in datasets}}))


def test_coverage_excludes_bounded_and_superseded_reports(tmp_path):
    beir = tmp_path / "beir"
    _write_report(beir, "five-small-fixed", ["scifact", "nfcorpus"])
    _write_report(beir, "hotpotqa-subset", ["hotpotqa"])
    _write_report(beir, "trec-covid-bounded", ["trec-covid"])
    _write_report(beir, "five-small", ["quora"])          # a superseded stem
    (beir / "coverage.json").write_text(json.dumps({"count": 99}))  # our own output, not a report

    measured = cc.measured_beir(beir)
    assert measured["count"] == 3
    assert measured["measured"] == ["hotpotqa", "nfcorpus", "scifact"]
    assert "trec-covid" in measured["bounded"] and "trec-covid" not in measured["measured"]
    assert "quora" not in measured["measured"]
    assert measured["total"] == 18


def test_a_bounded_run_does_not_count_toward_the_total(tmp_path):
    beir = tmp_path / "beir"
    _write_report(beir, "trec-covid-bounded", ["trec-covid"])
    assert cc.measured_beir(beir)["count"] == 0


# ── superseded reports ───────────────────────────────────────────────────

def test_a_superseded_report_without_a_marker_fails_until_fixed(tmp_path):
    report = tmp_path / "eval" / "beir" / "three-base.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("{}")

    failures, notes = cc.check_superseded(root=tmp_path, markers=False)
    assert any("three-base" in f for f in failures)
    assert not (report.with_suffix(".superseded.json")).exists()

    failures, notes = cc.check_superseded(root=tmp_path, markers=True)
    assert failures == [] and notes
    marker = report.with_suffix(".superseded.json")
    assert marker.is_file() and json.loads(marker.read_text())["superseded_by"] == "large-splade-fixed"

    again, _ = cc.check_superseded(root=tmp_path, markers=False)
    assert again == []


def test_a_marker_naming_the_wrong_replacement_fails(tmp_path):
    report = tmp_path / "eval" / "beir" / "three-base.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("{}")
    report.with_suffix(".superseded.json").write_text(
        json.dumps({"superseded_by": "something-else"})
    )
    failures, _ = cc.check_superseded(root=tmp_path, markers=False)
    assert any("registry says" in f for f in failures)


def test_citing_a_superseded_report_is_a_failure(tmp_path):
    (tmp_path / "README.md").write_text("see eval/beir/three-base.json for the old row\n")
    (tmp_path / "docs").mkdir()
    failures = cc.check_references(root=tmp_path)
    assert any("three-base" in f for f in failures)


# ── the registry's external authorities ──────────────────────────────────

def test_voyage_row_reads_the_pinned_paper_table():
    row = cc.voyage02()
    assert row == {"Hits@10": 0.7467, "Hits@4": 0.6625, "MAP@10": 0.4795, "MRR@10": 0.586}


def test_recalibration_splits_reads_the_report():
    if not cc.RECALIBRATION.is_file():
        pytest.skip("recalibration.json not generated")
    assert cc.recalibration_splits() == 300.0


# ── the doc-vs-doc check itself ──────────────────────────────────────────

def test_a_disagreeing_document_is_caught(tmp_path):
    (tmp_path / "A.md").write_text("| row | 0.663 |\n")
    (tmp_path / "B.md").write_text("| row | 0.662 |\n")
    quantity = cc.Quantity(
        name="x",
        sites=(cc.Site("A.md", r"^\| row", 0.663), cc.Site("B.md", r"^\| row", 0.662)),
        authority=lambda: {"v": 0.6625},
        key="v",
        provenance="test",
    )
    failures = cc.run_checks([quantity], root=tmp_path)
    assert len(failures) == 1 and "B.md" in failures[0]


def test_agreeing_documents_pass(tmp_path):
    (tmp_path / "A.md").write_text("| row | 0.663 |\n")
    quantity = cc.Quantity(
        name="x", sites=(cc.Site("A.md", r"^\| row", 0.663),),
        authority=lambda: {"v": 0.6625}, key="v", provenance="test",
    )
    assert cc.run_checks([quantity], root=tmp_path) == []


def test_a_missing_authority_report_is_a_skip_not_a_failure(tmp_path):
    quantity = cc.Quantity(
        name="x", sites=(cc.Site("A.md", r"^never", 1.0),),
        authority=lambda: None, provenance="absent",
    )
    assert cc.run_checks([quantity], root=tmp_path) == []


def test_the_committed_documents_agree():
    """The integration guard: the repository's own docs pass the gate when eval/ is present."""
    if cc.measured_beir()["count"] == 0:
        pytest.skip("no eval/beir reports to check against")
    assert cc.main([]) == 0
