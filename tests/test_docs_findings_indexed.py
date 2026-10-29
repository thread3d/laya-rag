"""Every findings document is reachable from the index that claims to list them all.

`docs/FINDINGS.md` opens with "Every measurement the improvement programme produced, with the headline
result and the document that carries the detail", and `docs/index.md` points a reader at it as the place
to "see every result the measurement programme produced". A findings document that neither the index nor
an indexed document links to is therefore a result nobody can find — which is the same failure as a
number with no report behind it, one step further along.

Reachability is **one hop or fewer**: the index may point at a measurement document which itself points
at the implementation record that produced it, because that is how the programme actually writes (`the
measurement`, then `what shipped`).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
INDEX = DOCS / "FINDINGS.md"

LINK = re.compile(r"\]\(([a-z0-9./-]+\.md)\)")


def findings_documents(docs: Path = DOCS) -> set[str]:
    return {path.name for path in docs.glob("findings-*.md")}


def linked_from(document: Path) -> set[str]:
    """Basenames this document links to, ignoring anchors and remote targets."""
    text = document.read_text(encoding="utf-8", errors="replace")
    return {target.rsplit("/", 1)[-1] for target in LINK.findall(text)}


def reachable(docs: Path = DOCS) -> set[str]:
    """The index's direct links, plus what those documents link to in turn."""
    index = docs / "FINDINGS.md"
    direct = linked_from(index) if index.is_file() else set()
    one_hop: set[str] = set()
    for name in direct:
        path = docs / name
        if path.is_file():
            one_hop |= linked_from(path)
    return direct | one_hop


def test_every_findings_document_is_reachable_from_the_index():
    present = findings_documents()
    if not present:
        return  # nothing to check on a tree without findings
    orphans = sorted(present - reachable())
    assert not orphans, (
        "these findings documents are linked from neither FINDINGS.md nor any document it links to:\n  "
        + "\n  ".join(orphans)
    )


def test_the_check_can_fail(tmp_path):
    """A checker that cannot fail is not a check."""
    (tmp_path / "FINDINGS.md").write_text("| a | [one](findings-one.md) |\n", encoding="utf-8")
    (tmp_path / "findings-one.md").write_text("# one\n", encoding="utf-8")
    (tmp_path / "findings-orphan.md").write_text("# two\n", encoding="utf-8")
    assert findings_documents(tmp_path) - reachable(tmp_path) == {"findings-orphan.md"}


def test_one_hop_is_enough(tmp_path):
    """The programme writes `the measurement`, then `what shipped` — the second is reached from the first."""
    (tmp_path / "FINDINGS.md").write_text("| a | [one](findings-one.md) |\n", encoding="utf-8")
    (tmp_path / "findings-one.md").write_text("see [impl](findings-impl.md)\n", encoding="utf-8")
    (tmp_path / "findings-impl.md").write_text("# impl\n", encoding="utf-8")
    assert findings_documents(tmp_path) <= reachable(tmp_path)
