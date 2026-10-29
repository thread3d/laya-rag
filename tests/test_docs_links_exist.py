"""A relative link in the documentation that resolves nowhere is a claim nobody clicked.

The class is crisp: for every `[text](target)` in a tracked document, `target` must exist relative to
that document. One instance shipped — `docs/findings-multihop-diagnostics.md` pointed at
`../../eval/...`, one directory above the repository root, so the report it cites was two levels away
from where a reader would look.

Two trees are **expected to be absent in a fresh checkout**, and a link into them is not a defect:
`eval/` is gitignored (generated reports) and `best-in-the-world/` is the audit set that lives in a
sibling checkout, outside this repository, which `SCORE-LIFT.md` and `PROGRAMME-STATUS.md` cite by
path. That is the same convention `claim-check`, `consistency-check` and the leakage audit use for
their artefacts.

The skip is decided per *target*, not per document: it asks whether the resolved path sits inside one
of those trees. Deciding it from a directory's presence was order-dependent — a test elsewhere in the
suite creates `eval/` for matplotlib's cache, and from then on every `eval/` link in the docs was
checked against a tree that has none of the reports.
"""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
SKIP_PREFIXES = ("http://", "https://", "#", "mailto:")

#: Generated, gitignored: absent from a fresh clone by design.
GENERATED_TREE = "eval"
#: The audit corpus above the repository root, in a sibling checkout.
AUDIT_TREE = "best-in-the-world"


def _expected_absent(resolved: Path, root: Path) -> bool:
    """Is this target one the documentation may cite without the file being present?"""
    generated = (root / GENERATED_TREE).resolve()
    if resolved == generated or generated in resolved.parents:
        return True
    audit = (root.parent / AUDIT_TREE).resolve()
    return resolved == audit or audit in resolved.parents


def broken_links(root: Path) -> list[tuple[str, str]]:
    documents = [root / "README.md", root / "SCORE-LIFT.md"]
    docs = root / "docs"
    if docs.is_dir():
        documents += sorted(docs.rglob("*.md"))
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
            resolved = (path.parent / target).resolve()
            if resolved.exists() or _expected_absent(resolved, root):
                continue
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
