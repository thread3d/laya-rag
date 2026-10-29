"""The objective's own rule, mechanised: a prediction ships with the measurement that would refute it.

`docs/best-in-the-world-2026-10.md` says every prediction in it "was a hypothesis with a stated falsifier,
in the form [`RESEARCH.md`](../RESEARCH.md) established". That is a claim about the *documentation practice*,
and like every other claim here it should be checkable rather than asserted — a finding that registers a
prediction without a falsifier is a prediction that cannot lose.

The check is deliberately loose about layout and strict about presence: a paragraph or the fifteen lines
after it must mention a falsifier, case-insensitively. Layout varies between documents on purpose (some use
a table, some a bolded run-in); the property does not.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

#: Lines that open a prediction block. `**Prediction:**`, `**Prediction.**`, `- **Prediction.**` …
PREDICTION = re.compile(r"^\s*(?:[-*]\s*)?\*\*Prediction", re.IGNORECASE)
FALSIFIER = re.compile(r"falsifier", re.IGNORECASE)
WINDOW = 15


def documents() -> list[Path]:
    return sorted(DOCS.glob("findings-*.md")) + [DOCS / "best-in-the-world-2026-10.md"]


def unpaired(text: str) -> list[int]:
    """1-based line numbers of prediction blocks with no falsifier near them."""
    lines = text.splitlines()
    problems: list[int] = []
    for index, line in enumerate(lines):
        if not PREDICTION.match(line):
            continue
        # the paragraph the prediction opens, plus the window after it
        paragraph: list[str] = []
        for following in lines[index:]:
            if not following.strip():
                break
            paragraph.append(following)
        window = "\n".join(paragraph + lines[index:index + WINDOW])
        if not FALSIFIER.search(window):
            problems.append(index + 1)
    return problems


def test_every_registered_prediction_carries_a_falsifier():
    offenders: list[str] = []
    for document in documents():
        if not document.is_file():
            continue
        for line in unpaired(document.read_text(encoding="utf-8")):
            offenders.append(f"{document.name}:{line}")
    assert not offenders, (
        "these predictions state no measurement that would refute them:\n  " + "\n  ".join(offenders)
    )


def test_the_check_finds_predictions_and_can_fail():
    """Both halves: the corpus is well populated, and a bare prediction is caught."""
    total = sum(
        sum(1 for line in d.read_text(encoding="utf-8").splitlines() if PREDICTION.match(line))
        for d in documents() if d.is_file()
    )
    assert total >= 8, f"only {total} predictions found — the scan may have stopped matching"
    bare = "**Prediction:** the score will double.\n\nNothing else here.\n"
    assert unpaired(bare) == [1]
    paired = "**Prediction:** the score will double.\n\n**Falsifier:** if it does not, the claim is wrong.\n"
    assert unpaired(paired) == []
