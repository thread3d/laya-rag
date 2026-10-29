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
