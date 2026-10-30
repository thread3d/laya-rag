"""The status page's R-series numbers are pinned to the artefacts that produced them.

`docs/PROGRAMME-STATUS.md` is the page a reader lands on, and its table quotes numbers from six
different reports. The repository's answer to prose drift is a claim ledger, so this is the same idea
scoped to one page: every registered number must appear **and** match the report field it cites, and a
missing artefact skips rather than failing — the reports are gitignored, so a fresh clone has none.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "docs" / "PROGRAMME-STATUS.md"
BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"

#: (label, report, `/`-separated field path, decimals as printed, scale) — the page prints some
#: quantities as percentages, so a pin carries the transform rather than the page being forced to an
#: unnatural unit.
PINS = (
    ("R2 shipped composer nugget recall", BENCH / "engine-extractive-rank150.json",
     "summary/answer_nugget_recall", 4),
    ("R2 objective nugget recall", BENCH / "engine-extractive-cov150.json",
     "summary/answer_nugget_recall", 4),
    ("R5 base reranker MAP@10", BENCH / "engine-extractive-ce-BAAI-bge-reranker-base.json",
     "summary/MAP@10", 4),
    ("R5 v2-m3 reranker MAP@10", BENCH / "engine-extractive-ce-BAAI-bge-reranker-v2-m3.json",
     "summary/MAP@10", 4),
    ("R6 untruncated MAP@10", BENCH / "engine-extractive-rank150.json", "summary/MAP@10", 4),
    ("R6 M=10 MAP@10", BENCH / "engine-extractive-m10.json", "summary/MAP@10", 4),
    # The page quotes the *trainer's* 40-split fit for the gate (held_out.gate_auc.mean) and the
    # *protocol's* half-split run for the deployed-threshold scalar. Two runs, two files: the pin
    # exists to stop the two being confused, which it caught on its first execution.
    ("R3 gate AUC (40 splits)", ROOT / "eval" / "gate" / "answerability-gate.json",
     "held_out/gate_auc/mean", 4),
    ("R3 gate risk@80 (40 splits)", ROOT / "eval" / "gate" / "answerability-gate.json",
     "held_out/gate_risk@80/mean", 4),
    ("R3 protocol gate AUC (half split)", ROOT / "eval" / "evidence-protocol.json",
     "abstention/held_out/auc", 4),
    ("R3 protocol scalar AUC", ROOT / "eval" / "evidence-protocol.json", "abstention/scalar/auc", 4),
    ("R3 protocol scalar false accept", ROOT / "eval" / "evidence-protocol.json",
     "abstention/scalar_false_accept_at_0.5", 1, 100.0),
)


def _dig(payload, dotted: str):
    """Walk a `/`-separated path. Not `.`: one of the report's keys is literally `…_at_0.5`."""
    node = payload
    for part in dotted.split("/"):
        node = node[part]
    return float(node)


def test_every_pinned_number_appears_on_the_status_page():
    """A number that is not on the page is either a stale pin or a stale page — both are bugs."""
    text = STATUS.read_text(encoding="utf-8")
    present = [pin for pin in PINS if pin[1].is_file()]
    if not present:
        pytest.skip("no eval/ artefacts on this checkout; run the measurement chain first")
    missing = []
    for label, report, dotted, decimals, *rest in present:
        value = _dig(json.loads(Path(report).read_text(encoding="utf-8")), dotted)
        printed = f"{value * (rest[0] if rest else 1.0):.{decimals}f}"
        if printed not in text:
            missing.append(f"{label}: {printed} (from {Path(report).name}:{dotted})")
    assert not missing, "the status page no longer quotes:\n  " + "\n  ".join(missing)


def test_the_page_still_describes_the_r_series():
    """The table is the part that keeps the October round legible; losing a row loses a result."""
    text = STATUS.read_text(encoding="utf-8")
    for item in ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"):
        assert re.search(rf"\*\*{item}\*\*", text), f"{item} is no longer listed on the status page"


def test_the_page_names_the_three_corrections():
    """The round's own summary claims three corrections rather than three wins; that must stay true."""
    text = STATUS.read_text(encoding="utf-8")
    assert "Three of the eight items ended in a correction" in text


def test_the_page_reports_the_placement_numbers_from_the_report_it_cites():
    """The subset placement is quoted on the status page; it must match the report behind it."""
    report = BENCH / "engine-extractive-ce-BAAI-bge-reranker-v2-m3.json"
    if not report.is_file():
        pytest.skip("no cross-encoder report on this checkout")
    summary = json.loads(report.read_text(encoding="utf-8"))["summary"]
    text = STATUS.read_text(encoding="utf-8")
    for field, printed in (("Hits@10", "0.940"), ("Hits@4", "0.847"), ("MRR@10", "0.688")):
        assert f"{summary[field]:.3f}" == printed, f"{field} moved: {summary[field]:.3f}"
        assert printed in text, f"the status page no longer quotes {field} {printed}"


def test_the_page_says_the_placement_is_a_subset():
    """The row is shown with its refusal, not as the dataset's number."""
    text = STATUS.read_text(encoding="utf-8")
    assert "refuses to place a subset" in text
    assert "150-query subset" in text


def test_the_page_reports_the_arguana_verdict_from_the_reports():
    """The ArguAna numbers are quoted on the status page; they must match the two reports."""
    base = ROOT / "eval" / "r1" / "arguana400-BAAI-bge-reranker-base.json"
    big = ROOT / "eval" / "r1" / "arguana400-BAAI-bge-reranker-v2-m3.json"
    if not base.is_file() or not big.is_file():
        pytest.skip("no ArguAna reports on this checkout")
    first = json.loads(big.read_text(encoding="utf-8"))["datasets"]["arguana"]["retrievers"]
    text = STATUS.read_text(encoding="utf-8")
    for field, printed in (
        ("hybrid-biencoder", "0.5664"),
        ("crossencoder", "0.4722"),
    ):
        assert f"{first[field]['nDCG@10']:.4f}" == printed, f"{field} moved"
        assert printed in text, f"the status page no longer quotes {field} {printed}"
    small = json.loads(base.read_text(encoding="utf-8"))["datasets"]["arguana"]["retrievers"]
    assert f"{small['crossencoder']['nDCG@10']:.4f}" == "0.2736"
    assert "0.2736" in text


def test_the_page_says_the_falsifier_did_not_fire_and_the_arm_is_still_untested():
    """A confirmed prediction is not a licence: the listwise arm is still open, and the page must say so."""
    text = STATUS.read_text(encoding="utf-8")
    assert "falsifier did not fire" in text
    assert "The listwise arm is still untested" in text


def test_the_assessment_no_longer_says_none_of_its_predictions_were_measured():
    """The closing section said "None has been measured" — true when written, false eight rounds later.

    That sentence is in the section headed "what this document does not claim", so leaving it stale
    under-claims in a way that misrepresents the work: it says nothing was measured when six items have
    been. The replacement is an outcome table, and this test keeps both halves of it true.
    """
    assessment = ROOT / "docs" / "best-in-the-world-2026-10.md"
    text = assessment.read_text(encoding="utf-8")
    assert "None has been measured" not in text
    assert "**When this document was written, none had been\n  measured.**" in text
    for item in ("falsified", "confirmed", "withdrawn", "half done"):
        assert item in text, f"the outcome table no longer records that an item was {item}"
    # and the two falsifications the table calls out by name
    assert "R0" in text and "R2" in text


# ── the same ledger applied to the index page ─────────────────────────────

FINDINGS = ROOT / "docs" / "FINDINGS.md"

#: (label, report, `/`-separated field, decimals, scale) — the four levers, as the index quotes them.
INDEX_PINS = (
    ("gate held-out AUC", ROOT / "eval" / "gate" / "answerability-gate.json",
     "held_out/gate_auc/mean", 4),
    ("composer shipped arm", BENCH / "engine-extractive-rank150.json", "summary/answer_nugget_recall", 4),
    ("composer objective arm", BENCH / "engine-extractive-cov150.json", "summary/answer_nugget_recall", 4),
    ("reranker v2-m3 MAP@10", BENCH / "engine-extractive-ce-BAAI-bge-reranker-v2-m3.json",
     "summary/MAP@10", 4),
    ("cascade M=10 CPU seconds", ROOT / "eval" / "cost-report.json", "cpu_seconds/p50", 1),
)


def test_the_findings_index_quotes_the_numbers_its_rows_are_about():
    """`FINDINGS.md` is the page that gets quoted; its four lever numbers must match their reports.

    The status page has had this pin since the October round and the index page had none, though it is
    the larger surface and the one a reader cites. Same registry, same rule: a number that is not on the
    page is either a stale pin or a stale page, and a missing artefact skips rather than fails.
    """
    text = FINDINGS.read_text(encoding="utf-8")
    present = [pin for pin in INDEX_PINS if pin[1].is_file()]
    if not present:
        pytest.skip("no eval/ artefacts on this checkout; run the measurement chain first")
    missing = []
    for label, report, dotted, decimals, *rest in present:
        value = _dig(json.loads(Path(report).read_text(encoding="utf-8")), dotted)
        printed = f"{value * (rest[0] if rest else 1.0):.{decimals}f}"
        if printed not in text:
            missing.append(f"{label}: {printed} (from {Path(report).name}:{dotted})")
    assert not missing, "docs/FINDINGS.md no longer quotes:\n  " + "\n  ".join(missing)


def test_the_grouped_transfer_numbers_are_quoted_where_the_gate_is():
    """The published AUC came from random halves; the grouped run is the honest transfer figure.

    Three claim-bearing pages quote the gate — the status table, the claim boundary and the README's known
    limits — and all three said `0.8377` alone. The grouped regime (hold out a whole question type, which the
    trainer's own docstring calls the only real transfer test) reads 0.7909 with ECE more than doubled, so a page
    quoting only the random-half figure is quoting the weaker regime as though it were the only one.
    """
    report = ROOT / "eval" / "gate" / "answerability-gate-grouped"
    if not report.is_file():
        pytest.skip("no grouped gate report on this checkout")
    held = json.loads(report.read_text(encoding="utf-8"))["held_out"]
    auc = f"{held['gate_auc']['mean']:.4f}"
    ece = f"{held['gate_ece']['mean']:.4f}"
    for page in (STATUS, ROOT / "BENCHMARKS.md", ROOT / "README.md"):
        text = page.read_text(encoding="utf-8")
        assert auc in text, f"{page.name} does not quote the grouped AUC {auc}"
        assert ece in text, f"{page.name} does not quote the grouped ECE {ece}"


def test_the_findings_index_quotes_the_full_set_row_it_names():
    """The index's extractive entry now quotes the full 2,255-query row; keep it tied to the report."""
    report = BENCH / "engine-extractive-full.json"
    if not report.is_file():
        pytest.skip("the full-set row is not on this checkout")
    summary = json.loads(report.read_text(encoding="utf-8"))["summary"]
    text = FINDINGS.read_text(encoding="utf-8")
    assert summary["queries"] == 2255
    for field, printed in (
        ("Hits@10", "0.9011"), ("Hits@4", "0.7561"), ("MAP@10", "0.3120"),
        ("MRR@10", "0.6166"), ("answer_nugget_recall", "0.1835"),
    ):
        assert f"{summary[field]:.4f}" == printed, f"{field} moved: {summary[field]:.4f}"
        assert printed in text, f"the index no longer quotes {field} {printed}"
    # and the caveat that keeps it from being read as a like-for-like placement
    assert "chunk-level" in text
