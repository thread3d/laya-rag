"""The night's four corrections must survive on every page that *makes a claim*.

Each correction is a number whose name promised more than its measurement: the gate's calibration under transfer, the
composer's gain under a length lens, the chunk-versus-document unit, and citation cleanliness meaning structure rather
than truth. They were propagated page by page over a dozen rounds, which means they can also be undone page by page —
by quoting the flattering half of each pair on some page that was not updated.

This checks the **claim-bearing pages only**: README, BENCHMARKS, COMPARISONS, PROGRAMME-STATUS, the findings index and
the assessment. Findings documents are deliberately out of scope, because there the corrected figure is often the
*subject* of the sentence rather than a claim made to a reader — `findings-extractive-row.md` defines "citation clean"
as "every claim cites in-range evidence" in the same table cell, which is clearer than the word "structural" would be.

A cross-document co-occurrence check over *all* docs was tried first and produced two false positives in one run —
including `findings-coverage-scope.md`, where `0.0521` is an ArguAna Hits@4 table cell and not the gate's ECE. That is
the lesson this file records: **a number alone is not a claim identity**, so the audit is scoped to pages whose job is
to make claims, where a bare number is one.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAIM_PAGES = (
    ROOT / "README.md",
    ROOT / "BENCHMARKS.md",
    ROOT / "COMPARISONS.md",
    ROOT / "docs" / "PROGRAMME-STATUS.md",
    ROOT / "docs" / "FINDINGS.md",
    ROOT / "docs" / "best-in-the-world-2026-10.md",
)


def present() -> list[Path]:
    return [page for page in CLAIM_PAGES if page.is_file()]


def test_the_gate_transfer_pair_travels_together():
    """Quoting the random-split ECE without the hold-out one is quoting the weaker regime alone."""
    problems = []
    for page in present():
        text = page.read_text(encoding="utf-8")
        if "0.0521" in text and "0.1361" not in text:
            problems.append(page.name)
    assert not problems, f"random-split ECE quoted alone in: {problems}"


def test_the_composer_gain_travels_with_its_length_cost():
    problems = []
    for page in present():
        text = page.read_text(encoding="utf-8")
        if "0.2610" in text and not any(w in text for w in ("length", "chars", "characters", "verbos")):
            problems.append(page.name)
    assert not problems, f"composer recall quoted without a length caveat in: {problems}"


def test_citation_cleanliness_is_never_quoted_as_bare_truth():
    """`1.0000` certifies structure; a page that says only "citation clean" implies more."""
    # The qualifier is a *concept*, not a spelling: this repository's own pages say "structural", "mechanical …
    # not semantic support", and "every claim cites in-range evidence". Requiring one word produced a false positive
    # on the assessment, whose phrasing is better than the one the test was looking for.
    QUALIFIERS = ("structural", "mechanical", "in-range evidence", "not semantic")
    problems = []
    for page in present():
        text = page.read_text(encoding="utf-8")
        mentions = ("cite-clean" in text or "citation clean" in text or "citation validity" in text)
        if mentions and not any(q in text for q in QUALIFIERS):
            problems.append(page.name)
    assert not problems, f"citation cleanliness quoted without a qualifier in: {problems}"


def test_the_chunk_versus_document_caveat_reaches_the_pages_that_show_the_row():
    """A placement row beside document-level anchors must say which unit it is over."""
    problems = []
    for page in present():
        text = page.read_text(encoding="utf-8")
        if "0.9011" in text and "chunk" not in text.lower():
            problems.append(page.name)
    assert not problems, f"the full-set row appears without its unit caveat in: {problems}"


def test_the_audit_is_not_vacuous():
    pages = present()
    assert len(pages) >= 5, f"only {len(pages)} claim pages found"
    assert any("0.9011" in page.read_text(encoding="utf-8") for page in pages)
