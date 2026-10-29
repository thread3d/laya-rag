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

import pytest


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


def test_the_reproduce_page_names_a_command_for_every_findings_document_it_claims():
    """A reproduce page is only worth having if its rows point at things that exist.

    Every findings document linked from `docs/REPRODUCING.md` must exist (the link checker covers that),
    and every `make` target it tells a reader to run must be one the Makefile defines. The second half is
    the part that rots: a renamed target turns "re-derive it yourself" into "hunt for the new name".
    """
    import re

    root = Path(__file__).resolve().parents[1]
    page = root / "docs" / "REPRODUCING.md"
    if not page.is_file():
        pytest.skip("no reproduce page on this checkout")
    text = page.read_text(encoding="utf-8")
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    available = set(re.search(r"^\.PHONY: (.*)$", makefile, re.MULTILINE).group(1).split())
    invoked = set(re.findall(r"\bmake\s+([a-z][a-z0-9-]+)", text))
    assert invoked, "the reproduce page tells nobody to run anything"
    missing = sorted(invoked - available)
    assert not missing, f"docs/REPRODUCING.md names targets that do not exist: {missing}"


def test_the_reproduce_page_states_one_heading_count_per_list():
    """A page that says "four things" above five numbered items is the prose drift it exists to prevent."""
    import re

    root = Path(__file__).resolve().parents[1]
    page = root / "docs" / "REPRODUCING.md"
    if not page.is_file():
        pytest.skip("no reproduce page on this checkout")
    text = page.read_text(encoding="utf-8")
    words = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
    for heading in re.finditer(r"^## .*?\b(two|three|four|five|six|seven)\b.*$", text, re.MULTILINE | re.IGNORECASE):
        stated = words[heading.group(1).lower()]
        # count the numbered items that follow, up to the next heading
        rest = text[heading.end():]
        section = rest.split("\n## ", 1)[0]
        numbered = len(re.findall(r"^\d+\.\s", section, re.MULTILINE))
        assert numbered == stated, (
            f"'{heading.group(0).strip()}' claims {stated} items but lists {numbered}"
        )
