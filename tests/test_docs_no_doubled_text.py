"""A patch applied by substring fails silently, and nothing was catching it textually.

Three commits this session needed a follow-up for a patch that missed or duplicated its own anchor —
most visibly `**The re-takes beyond SciFact.****The re-takes beyond SciFact.**`. The numeric gates catch
a wrong number; nothing caught doubled prose, because a doubled sentence is not a false statement. This
is the smallest check that closes that class: an immediately repeated sentence inside one line.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIN_CHARS = 20


def doubled_sentences(text: str) -> list[tuple[int, str]]:
    """Lines carrying the same run of characters twice in a row.

    A *substring* search rather than a sentence comparison, because the case that actually occurred is
    `...SciFact.****The re-takes...` — the two copies are joined with **no whitespace**, so splitting on
    sentence boundaries never sees them as two sentences. Any immediate repetition of ``MIN_CHARS`` or
    more is the signature of a patch that inserted its own anchor alongside the one it matched; at that
    length it does not occur in ordinary prose.
    """
    out: list[tuple[int, str]] = []
    pattern = re.compile(r"(.{%d,}?)\1" % MIN_CHARS)
    fenced = False
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            fenced = not fenced
            continue
        # Tables and code repeat their cells and their tokens legitimately — the same tool name in
        # four rows is not a patch applied twice. The defect this looks for is doubled *prose*.
        if fenced or stripped.startswith("|"):
            continue
        match = pattern.search(line)
        if match and any(character.isalpha() for character in match.group(1)):
            out.append((number, match.group(1)))
    return out


def test_the_detector_notices_a_doubled_heading():
    text = "**The re-takes beyond SciFact.****The re-takes beyond SciFact.** The other twelve reports"
    # (prose, not a table row — which is what this check covers)
    hits = doubled_sentences(text)
    assert len(hits) == 1
    assert hits[0][0] == 1


def test_the_detector_ignores_short_repetition_and_lookalikes():
    # Short repeats are normal in prose ("No. No."), and near-repeats are not repeats.
    assert doubled_sentences("Yes. Yes. We agree.") == []
    assert doubled_sentences("The first sentence is long enough. The second one is longer still.") == []
    assert doubled_sentences("") == []


def test_the_detector_catches_a_doubled_sentence_with_a_space_too():
    """Both shapes a patch produces: joined by whitespace, and joined by nothing at all."""
    spaced = "The clause is repeated here. The clause is repeated here. Then more prose follows."
    assert len(doubled_sentences(spaced)) == 1


def test_no_tracked_document_carries_a_doubled_sentence():
    """The standing check. Skips when the documents are not checked out."""
    documents = [ROOT / "README.md", ROOT / "SCORE-LIFT.md"]
    documents += sorted((ROOT / "docs").rglob("*.md"))
    present = [path for path in documents if path.is_file()]
    if not present:
        pytest.skip("no documents to scan")
    failures = [
        f"{path.relative_to(ROOT)}:{number}: {fragment[:70]}"
        for path in present
        for number, fragment in doubled_sentences(path.read_text(encoding="utf-8", errors="replace"))
    ]
    assert not failures, "a patch doubled its own text:\n  " + "\n  ".join(failures)
