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


def test_coverage_excludes_bounded_and_superseded_reports(tmp_path, monkeypatch):
    beir = tmp_path / "beir"
    _write_report(beir, "five-small-fixed", ["scifact", "nfcorpus"])
    _write_report(beir, "hotpotqa-subset", ["hotpotqa"])
    _write_report(beir, "robust04-bounded", ["robust04"])
    _write_report(beir, "five-small", ["quora"])          # a superseded stem
    (beir / "coverage.json").write_text(json.dumps({"count": 99}))  # our own output, not a report
    # The shipped registry has no bounded dataset left (TREC-COVID's subsample is superseded by
    # the full run), so the mechanism is pinned by injecting one rather than by leaving a
    # placeholder in the registry.
    monkeypatch.setattr(cc, "BOUNDED", {"robust04": "40,000-document subsample"})

    measured = cc.measured_beir(beir)
    assert measured["count"] == 3
    assert measured["measured"] == ["hotpotqa", "nfcorpus", "scifact"]
    assert "robust04" in measured["bounded"] and "robust04" not in measured["measured"]
    assert "quora" not in measured["measured"]
    assert measured["total"] == 18


def test_a_bounded_run_does_not_count_toward_the_total(tmp_path, monkeypatch):
    beir = tmp_path / "beir"
    _write_report(beir, "robust04-bounded", ["robust04"])
    monkeypatch.setattr(cc, "BOUNDED", {"robust04": "40,000-document subsample"})
    assert cc.measured_beir(beir)["count"] == 0


def test_only_the_eighteen_members_count_and_the_rest_are_extras(tmp_path):
    """Membership is stated: MS MARCO is downloadable but is not one of BEIR's 18."""
    beir = tmp_path / "beir"
    _write_report(beir, "msmarco", ["msmarco"])
    measured = cc.measured_beir(beir)
    assert measured["count"] == 0
    assert measured["extras"] == ["msmarco"]
    assert "cqadupstack" in measured["missing"] and "bioasq" in measured["missing"]
    assert len(cc.BEIR_18) == 18 and "msmarco" not in cc.BEIR_18 and "specter" not in cc.BEIR_18
    assert "cqadupstack" in cc.BEIR_18


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


# ── the audit set above the repository (best-in-the-world/) ──────────────

def _audit_doc(tmp_path: Path, name: str, body: str, *, banner: bool = True) -> Path:
    audit = tmp_path / "best-in-the-world"
    audit.mkdir(parents=True, exist_ok=True)
    text = ("> **Currency:** pinned by `make consistency-check`.\n\n" if banner else "") + body
    (audit / name).write_text(text, encoding="utf-8")
    return audit


def test_an_audit_document_must_carry_a_currency_banner(tmp_path):
    audit = _audit_doc(tmp_path, "a.md", "coverage is 12 of 18\n", banner=False)
    failures = cc.check_audit_documents(root=tmp_path, coverage_count=12, audit_dir=audit)
    assert any("Currency" in f for f in failures)


def test_an_audit_count_must_match_the_computed_value(tmp_path):
    audit = _audit_doc(tmp_path, "a.md", "coverage is 10 of 18\n")
    failures = cc.check_audit_documents(root=tmp_path, coverage_count=12, audit_dir=audit)
    assert len(failures) == 1 and "computed 12 of 18" in failures[0]


def test_a_fenced_dated_finding_may_keep_a_superseded_count(tmp_path):
    """The point of the fence: history is allowed, but only when it is dated as history."""
    body = (
        "<!-- consistency: snapshot 2026-10-06 -->\n"
        "the audit found the count stated as 5, 7 and 10 of 18\n"
        "<!-- consistency: /snapshot -->\n\n"
        "coverage is 12 of 18\n"
    )
    audit = _audit_doc(tmp_path, "a.md", body)
    assert cc.check_audit_documents(root=tmp_path, coverage_count=12, audit_dir=audit) == []


def test_the_audit_guard_is_not_vacuous_at_full_coverage(tmp_path):
    """At 18 of 18, "is the number 18 on this line?" is true of every line that says "N of 18".

    That is a real hole rather than a hypothetical: the guard tests for the presence of the computed
    value, and the string "of 18" *is* the computed value once coverage is complete. So the stated
    count has to be compared, not merely detected — otherwise a fully covered repository could say
    anything at all and the gate would agree.
    """
    audit = _audit_doc(tmp_path, "a.md", "coverage is 17 of 18\n")
    failures = cc.check_audit_documents(root=tmp_path, coverage_count=18, audit_dir=audit)
    assert len(failures) == 1 and "computed 18 of 18" in failures[0]

    # A second document stating the computed count must pass, alongside the failing one.
    _audit_doc(tmp_path, "b.md", "coverage is 18 of 18\n")
    failures = cc.check_audit_documents(root=tmp_path, coverage_count=18, audit_dir=audit)
    assert [f for f in failures if "b.md" in f] == []


def test_every_count_on_an_audit_line_has_to_agree(tmp_path):
    """A line naming several counts is only current if each of them is the computed one."""
    audit = _audit_doc(
        tmp_path, "a.md", "the documents said 15 of 18, and now say 18 of 18\n"
    )
    failures = cc.check_audit_documents(root=tmp_path, coverage_count=18, audit_dir=audit)
    assert len(failures) == 1, "a stale count hiding beside a current one must still fail"


def test_a_snapshot_fence_must_name_a_date(tmp_path):
    """An undated fence is not a fence, so its historical count is read as a current claim."""
    body = (
        "<!-- consistency: snapshot -->\n"
        "the audit found 10 of 18\n"
        "<!-- consistency: /snapshot -->\n"
    )
    audit = _audit_doc(tmp_path, "a.md", body)
    failures = cc.check_audit_documents(root=tmp_path, coverage_count=12, audit_dir=audit)
    assert any("computed 12 of 18" in f for f in failures)


def test_an_unclosed_snapshot_fence_fails(tmp_path):
    audit = _audit_doc(
        tmp_path / "unclosed", "a.md",
        "<!-- consistency: snapshot 2026-10-06 -->\nold 10 of 18\n",
    )
    failures = cc.check_audit_documents(
        root=tmp_path / "unclosed", coverage_count=12, audit_dir=audit
    )
    assert any("never closed" in f for f in failures)


def test_a_snapshot_closure_with_nothing_open_fails(tmp_path):
    audit = _audit_doc(tmp_path, "a.md", "<!-- consistency: /snapshot -->\n")
    failures = cc.check_audit_documents(root=tmp_path, coverage_count=12, audit_dir=audit)
    assert any("nothing open" in f for f in failures)


def test_a_missing_audit_directory_skips_the_check(tmp_path):
    """The audit set is not committed here: a checkout without it must still pass."""
    assert cc.check_audit_documents(
        root=tmp_path, coverage_count=12, audit_dir=tmp_path / "absent"
    ) == []


def test_an_optional_site_skips_a_missing_companion_document(tmp_path):
    quantity = cc.Quantity(
        name="x",
        sites=(cc.Site("best-in-the-world/absent.md", r"^row", 1.0, optional=True),),
        authority=lambda: {"v": 1.0}, key="v", provenance="test",
    )
    assert cc.run_checks([quantity], root=tmp_path) == []
    assert cc.checked_site_count([quantity], root=tmp_path) == 0


def test_a_required_site_still_fails_when_its_document_is_missing(tmp_path):
    """Only the audit sites are optional; a typo in a repo document stays a failure."""
    quantity = cc.Quantity(
        name="x",
        sites=(cc.Site("absent.md", r"^row", 1.0),),
        authority=lambda: {"v": 1.0}, key="v", provenance="test",
    )
    assert any("matched no line" in f for f in cc.run_checks([quantity], root=tmp_path))


def test_citing_a_superseded_report_from_the_audit_set_is_a_failure(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "a.md").write_text("see eval/beir/three-base.json\n")
    failures = cc.check_references(root=tmp_path, audit_dir=audit)
    assert any("three-base" in f for f in failures)


def test_the_audit_documents_are_pinned():
    """The integration guard for the audit set, when the workspace carries it."""
    if not cc.AUDIT_DIR.is_dir():
        pytest.skip("best-in-the-world/ is not present")
    count = cc.measured_beir()["count"]
    if count == 0:
        pytest.skip("no eval/beir reports to check against")
    assert cc.check_audit_documents(root=cc.ROOT, coverage_count=count) == []


def test_the_committed_documents_agree():
    """The integration guard: the repository's own docs pass the gate when eval/ is present."""
    if cc.measured_beir()["count"] == 0:
        pytest.skip("no eval/beir reports to check against")
    assert cc.main([]) == 0


# ── SCORE-LIFT.md §1 is pinned to the reports it is derived from ─────────

def _beir_report(directory: Path, stem: str, dataset: str, retrievers, documents: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.json").write_text(json.dumps({
        "datasets": {dataset: {"documents": documents, "retrievers": {r: {} for r in retrievers}}},
    }))


def test_channel_coverage_pins_the_section_one_table(tmp_path):
    """A member gaining a dense row must fail until the table is regenerated."""
    beir = tmp_path / "eval" / "beir"
    _beir_report(beir, "scifact", "scifact", ["bm25-engine"], 5_183)
    (tmp_path / "SCORE-LIFT.md").write_text(
        "# doc\n\n| scifact | 5,183 | bm25-engine | **lexical only** |\n"
    )
    failures = cc.check_channel_coverage(tmp_path)
    assert any("§1.1 does not carry" in f for f in failures), failures


def test_channel_coverage_passes_a_fully_generated_section(tmp_path):
    beir = tmp_path / "eval" / "beir"
    _beir_report(beir, "scifact", "scifact", ["bm25-engine"], 5_183)
    module = cc.channel_coverage_module()
    coverage = module.channel_map(beir)
    lines = module.coverage_table(coverage) + module.deficit_table(module.deficit())
    (tmp_path / "SCORE-LIFT.md").write_text("\n".join(lines) + "\n")
    assert cc.check_channel_coverage(tmp_path) == []


def test_channel_coverage_pins_a_deficit_cell(tmp_path):
    """The §1.2 arithmetic is a computed quantity: a moved gap on any row fails."""
    beir = tmp_path / "eval" / "beir"
    _beir_report(beir, "scifact", "scifact", ["bm25-engine"], 5_183)
    module = cc.channel_coverage_module()
    lines = module.coverage_table(module.channel_map(beir))
    # Every §1.2 row except SciFact's, which is replaced by one with a moved gap.
    lines += [ln for ln in module.deficit_table(module.deficit()) if "| SciFact |" not in ln]
    lines.append("| SciFact | 0.7670 | 0.699 | **+0.169** | full |")
    (tmp_path / "SCORE-LIFT.md").write_text("\n".join(lines) + "\n")
    failures = cc.check_channel_coverage(tmp_path)
    assert any("§1.2 row 'SciFact'" in f and "gap" in f for f in failures), failures


def test_channel_coverage_skips_when_no_reports_exist(tmp_path):
    (tmp_path / "SCORE-LIFT.md").write_text("# doc\n")
    assert cc.check_channel_coverage(tmp_path) == []


def test_channel_coverage_skips_a_missing_document(tmp_path):
    _beir_report(tmp_path / "eval" / "beir", "scifact", "scifact", ["bm25-engine"], 5_183)
    assert cc.check_channel_coverage(tmp_path) == []


def test_the_deficit_arithmetic_names_the_members_it_sums():
    """The four figures are one computation, so they cannot disagree with each other."""
    module = cc.channel_coverage_module()
    deficit = module.deficit()
    if not deficit["cells"]:
        pytest.skip("no eval/beir reports to compute the deficit from")
    assert deficit["counted"] == 9
    assert deficit["total"] == pytest.approx(0.7314, abs=5e-5)
    assert deficit["macro"] == pytest.approx(0.0406, abs=5e-5)


def test_the_ours_column_is_read_from_the_reports_not_typed():
    """The column §1.2 used to hand-enter disagreed with the reports on nine of eighteen cells."""
    module = cc.channel_coverage_module()
    computed = module.our_best()
    if not computed:
        pytest.skip("no eval/beir reports")
    import charts_compare

    authoritative = charts_compare.our_rows()
    for member, entry in computed.items():
        assert entry["value"] == pytest.approx(authoritative[member]["best"][1], abs=1e-9)
        assert entry["row"] == authoritative[member]["best"][0]


def test_the_macro_is_the_mean_of_the_rows_the_table_shows():
    """§1.2's opening number and its table have to be the same eighteen cells."""
    module = cc.channel_coverage_module()
    computed = module.our_best()
    if len(computed) < 18:
        pytest.skip("all eighteen members are not measured here")
    import charts_compare

    assert charts_compare.our_macro(charts_compare.our_rows())["mean"] == pytest.approx(
        sum(entry["value"] for entry in computed.values()) / 18, abs=5e-5
    )


# ── the harness-vs-engine parity figure is computed, not typed ───────────

def test_the_scifact_parity_is_read_from_both_reports():
    """The delta is `replica - engine`, so it cannot disagree with either of them."""
    parity = cc.scifact_harness_parity()
    if parity is None:
        pytest.skip("engine-beir or engine-parity report is not present")
    assert parity["engine"] == pytest.approx(0.7264, abs=5e-5)
    assert parity["replica"] == pytest.approx(0.7408, abs=5e-5)
    assert parity["delta"] == pytest.approx(0.0144, abs=5e-5)


def test_the_replica_row_is_the_engine_matched_configuration():
    """A BEIR row is a retriever *plus its settings*.

    `five-small-fixed.json` and the `rrf-*` sweep carry `--title-boost 2` and read 0.7413 for the
    same retriever and `rrf_k=5`; the engine indexes each title inline once, so the parity must be
    read from the un-boosted `engine-parity.json`. Reading the wrong one moved every document's
    delta by 0.0005 and produced a "correction" that was itself the error.
    """
    matched = cc.beir_report_metric(
        "engine-parity", "scifact", "nDCG@10", retriever="hybrid-biencoder"
    )()
    title_weighted = cc.beir_report_metric(
        "five-small-fixed", "scifact", "nDCG@10", retriever="hybrid-biencoder"
    )()
    if matched is None or title_weighted is None:
        pytest.skip("one of the two replica reports is not present")
    assert matched == pytest.approx(0.7408, abs=5e-5)
    assert title_weighted == pytest.approx(0.7413, abs=5e-5)
    assert matched != title_weighted


def test_the_scifact_parity_skips_without_its_reports(tmp_path):
    """A missing report is a skip, as everywhere else in this gate."""
    missing = cc.beir_report_metric("no-such-report", "scifact", "nDCG@10")
    assert missing() is None
    assert cc.engine_beir_metric("no-such-report", "nDCG@10")() is None


def test_the_replica_row_is_the_hybrid_biencoder_not_the_control():
    """The control is the published Anserini configuration; quoting a fused row against it is not
    like-for-like, so the parity must read `hybrid-biencoder`."""
    authority = cc.beir_report_metric(
        "five-small-fixed", "scifact", "nDCG@10", retriever="hybrid-biencoder"
    )
    if authority() is None:
        pytest.skip("five-small-fixed report is not present")
    assert authority() != cc.beir_report_metric("five-small-fixed", "scifact", "nDCG@10")()
