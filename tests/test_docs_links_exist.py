"""A relative link in the documentation that resolves nowhere is a claim nobody clicked.

The class is crisp: for every `[text](target)` in a tracked document, `target` must exist relative to
that document. One instance shipped — `docs/findings-multihop-diagnostics.md` pointed at
`../../eval/...`, one directory above the repository root, so the report it cites was two levels away
from where a reader would look.

Links into `eval/` are **skipped when `eval/` is absent**, because it is gitignored and a fresh clone has
none of it; that is the same convention the leakage audit and `claim-check` use for their artefacts.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
SKIP_PREFIXES = ("http://", "https://", "#", "mailto:")


def broken_links(root: Path) -> list[tuple[str, str]]:
    documents = [root / "README.md", root / "SCORE-LIFT.md"]
    docs = root / "docs"
    if docs.is_dir():
        documents += sorted(docs.rglob("*.md"))
    eval_present = (root / "eval").is_dir()
    out: list[tuple[str, str]] = []
    for path in documents:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in LINK.finditer(text):
            target = match.group(1)
            if target.startswith(SKIP_PREFIXES):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            if not eval_present and target.lstrip("./").startswith("eval/"):
                continue
            if not (path.parent / target).resolve().exists():
                out.append((str(path.relative_to(root)), target))
    return out


def test_the_detector_notices_a_link_that_resolves_nowhere(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text(
        "See [the report](missing.md) and [the site](https://example.com).\n", encoding="utf-8"
    )
    found = broken_links(tmp_path)
    assert found == [("docs/a.md", "missing.md")]


def test_no_tracked_document_has_a_broken_relative_link():
    """The standing check: 195 links today, and a new broken one fails where it is written."""
    offenders = broken_links(ROOT)
    assert not offenders, (
        "these links resolve nowhere:\n  "
        + "\n  ".join(f"{path}: {target}" for path, target in sorted(set(offenders)))
    )
