"""A documented command that fails when run is a claim nobody checked.

`docs/PROGRAMME-STATUS.md` told a reader to run `make charts && make charts-more` as part of a runbook
of checks, and **`charts-more` did not exist** — `charts` already calls `tools/compare_more.py`, so the
target was folded in and the reference was stale. That one is fixed. The rest of the first pass's
findings were four references in `docs/generation-placement.md` that the document itself calls "planned
hook" and whose own Makefile sketch says "proposed; not yet in Makefile" — a reader is told the truth
there, so they are not dead commands and are listed as *proposed*, not exempted.

Same family as a flag that is parsed and never read: a documented capability that does nothing. The
difference between the two categories is the whole point of the check.

The check looks only at `make X` written **inside backticks or a fence**, because that is how a
reference is spelled and prose is full of "make up", "make for", "make both" — twenty-one of those
turned up in a first pass that scanned everything.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: References to targets a document **says are not there yet**: `generation-placement.md` calls these
#: "planned hook" and its own Makefile sketch is headed "proposed; not yet in Makefile", so the reader
#: is told the truth and the reference is not a dead command. Kept as a named set rather than skipped by
#: a heuristic, because a checker that tries to read intent from prose is a checker nobody can trust.
PROPOSED_REFERENCES = {
    ("docs/generation-placement.md", "generation-check"),
    ("docs/generation-placement.md", "generation-list"),
    ("docs/generation-placement.md", "kilt"),
    ("docs/generation-placement.md", "ragas-context"),
}

REFERENCE = re.compile(r"\bmake\s+([a-z][a-z0-9-]+)")


def make_targets(makefile: Path) -> set[str]:
    if not makefile.is_file():
        return set()
    return set(re.findall(r"^([A-Za-z0-9_-]+):", makefile.read_text(encoding="utf-8"), re.M))


def dead_references(root: Path) -> list[tuple[str, str]]:
    """`make x` inside code, where `x` is not a target the Makefile defines."""
    targets = make_targets(root / "Makefile")
    if not targets:
        return []
    documents = [root / "README.md", root / "SCORE-LIFT.md"]
    docs = root / "docs"
    if docs.is_dir():
        documents += sorted(docs.rglob("*.md"))
    out: list[tuple[str, str]] = []
    for path in documents:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        spans = re.findall(r"`([^`\n]+)`", text) + re.findall(r"```.*?```", text, re.S)
        relative = str(path.relative_to(root))
        for span in spans:
            for match in REFERENCE.finditer(span):
                if match.group(1) not in targets:
                    out.append((relative, match.group(1)))
    return out


def test_the_detector_finds_a_reference_to_a_target_that_is_not_there(tmp_path):
    (tmp_path / "Makefile").write_text("build:\n\techo ok\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text(
        "Run `make build` then `make nonexistent`.\n\nProse: make up your mind.\n", encoding="utf-8"
    )
    found = dead_references(tmp_path)
    assert found == [("docs/a.md", "nonexistent")]


def test_no_tracked_document_names_a_target_that_does_not_exist():
    """The standing check. A new dead reference fails where it is written."""
    if not (ROOT / "Makefile").is_file():
        pytest.skip("no Makefile")
    offenders = [item for item in dead_references(ROOT) if item not in PROPOSED_REFERENCES]
    assert not offenders, (
        "these documented commands name make targets that do not exist:\n  "
        + "\n  ".join(f"{path}: make {target}" for path, target in sorted(set(offenders)))
    )
