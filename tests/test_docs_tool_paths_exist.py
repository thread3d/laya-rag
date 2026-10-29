"""A `tools/…py` path written in prose is a command a reader will try to run.

`tests/test_docs_links_exist.py` checks markdown *links*; it does not see a backticked path in a
sentence, and those are how every findings document names the tool that produced its number. Nine such
paths pointed at nothing — two of them introduced by the October round itself, as a dropped letter
(`distil_verify.py` for `distill_verify.py`) and a tool that never existed under that name.

Seven are **declared planned** in the documents that name them, and they are listed here by name rather
than skipped by a heuristic: `generation-placement.md` heads its sketch "planned adapters only" and
"planned, does NOT exist yet", and `SCORE-LIFT.md` names `tools/judge.py` under "**Specification.**".
That is the same convention `test_docs_make_targets_exist.py` uses for `PROPOSED_REFERENCES`, and for the
same reason — a checker that infers intent from prose is a checker nobody can trust.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATH = re.compile(r"`(tools/[A-Za-z0-9_./-]+\.(?:py|mjs))`")

#: Adapters the documents themselves call planned or specified. Named, not inferred.
PROPOSED_TOOLS = {
    "tools/frames.py",
    "tools/judge.py",
    "tools/kilt.py",
    "tools/legalbench_rag.py",
    "tools/lotte.py",
    "tools/rgb.py",
    "tools/twincorpus.py",
}


def referenced(documents: list[Path]) -> dict[str, set[str]]:
    """Tool paths named in prose, with the documents that name them."""
    found: dict[str, set[str]] = {}
    for document in documents:
        for match in PATH.finditer(document.read_text(encoding="utf-8", errors="replace")):
            found.setdefault(match.group(1), set()).add(document.name)
    return found


def missing(documents: list[Path], root: Path = ROOT) -> dict[str, set[str]]:
    """Referenced tool paths that do not exist and are not declared planned."""
    return {
        path: where
        for path, where in referenced(documents).items()
        if not (root / path).is_file() and path not in PROPOSED_TOOLS
    }


def test_every_tool_path_named_in_prose_exists_or_is_declared_planned():
    documents = [ROOT / "README.md", ROOT / "SCORE-LIFT.md"] + sorted((ROOT / "docs").rglob("*.md"))
    offenders = missing(documents)
    assert not offenders, "these tool paths are named in prose and resolve nowhere:\n  " + "\n  ".join(
        f"{path} — named by {', '.join(sorted(where))}" for path, where in sorted(offenders.items())
    )


def test_the_check_can_fail(tmp_path):
    """A checker that cannot fail is not a check: one bad path and one proposed path."""
    (tmp_path / "doc.md").write_text(
        "`tools/typoo.py` and `tools/lotte.py` and `tools/real.py`\n", encoding="utf-8"
    )
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "real.py").write_text("", encoding="utf-8")
    found = missing([tmp_path / "doc.md"], root=tmp_path)
    assert set(found) == {"tools/typoo.py"}      # the typo is caught, the proposed path is exempt


def test_the_reference_set_is_not_empty():
    """If the regex stopped matching, the test above would pass vacuously."""
    documents = [ROOT / "README.md"] + sorted((ROOT / "docs").rglob("*.md"))
    assert len(referenced(documents)) > 30
