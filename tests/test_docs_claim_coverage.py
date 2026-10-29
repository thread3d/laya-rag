"""The index page states how much of the prose its gates actually cover — and the count is checked.

The page used to say "`make claim-check` resolves every quoted figure against the report that produced
it", which was false by two orders of magnitude: 54 registered claims against several thousand decimal
figures in the tracked prose. That is the failure this repository catalogues as class 13 — *a correct
number, over-generalised* — applied to a statement about its own machinery, and printed on the front page.

The sentence now states the registered count and says plainly that most figures are covered by neither
gate. This test keeps the stated count equal to the number of checks that exist, so the sentence cannot
drift back into flattery. It deliberately does **not** bound the total number of figures, which moves
every time a document is edited.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "index.md"


def registered_claims() -> int:
    sys.path.insert(0, str(ROOT / "tools"))
    sys.path.insert(0, str(ROOT / "src"))
    import claim_check

    return len(claim_check.CHECKS)


def stated_claims() -> int:
    text = INDEX.read_text(encoding="utf-8")
    # the sentence wraps, so the count and the noun may be separated by a newline
    match = re.search(r"resolves \*\*(\d+)\*\*\s+registered figures", text)
    assert match, "docs/index.md no longer states how many figures claim-check resolves"
    return int(match.group(1))


def test_the_stated_claim_count_is_the_real_one():
    assert stated_claims() == registered_claims()


def test_the_page_admits_that_most_figures_are_covered_by_neither_gate():
    """The half of the sentence that matters: 54 checks do not cover several thousand numbers."""
    text = INDEX.read_text(encoding="utf-8")
    assert "covered by neither gate" in text
    assert "every quoted figure" not in text, "the over-generalised claim has come back"
