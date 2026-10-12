"""A documented command naming a flag its tool does not declare is a command that errors when run.

Same class as a dead `make` target, one step finer. Scoping matters twice over, and both scopings were
found by being wrong first:

* only `make x` inside **code** is a reference (prose says "make up", "make for");
* a **commented-out** command is not a reference either. `docs/generation-placement.md` writes
  `# .venv/bin/python tools/generation_eval.py --context-relevance --corpus <corpus>` and says in the
  next sentence that the scaffold accepts only `--list` and `--check` — the document is telling the truth
  about what does not exist yet, which is the opposite of a stale command.

A first pass with a double-quote-only regex produced four false hits on `tools/multihop_gate.py`, which
declares its flags with single quotes.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

DECLARED = re.compile(r"""add_argument\(\s*['"](--[A-Za-z0-9-]+)['"]""")
INVOCATION = re.compile(r"tools/([A-Za-z0-9_]+\.py)")
FLAG = re.compile(r"(--[A-Za-z0-9-]+)")


def declared_flags(root: Path) -> dict[str, set[str]]:
    return {
        path.name: set(DECLARED.findall(path.read_text(encoding="utf-8", errors="replace")))
        for path in (root / "tools").glob("*.py")
    }


def unknown_flags(root: Path) -> list[tuple[str, str, str]]:
    known = declared_flags(root)
    if not known:
        return []
    documents = [root / "README.md", root / "SCORE-LIFT.md"]
    docs = root / "docs"
    if docs.is_dir():
        documents += sorted(docs.rglob("*.md"))
    out: list[tuple[str, str, str]] = []
    for path in documents:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        spans = re.findall(r"`([^`\n]+)`", text) + re.findall(r"```.*?```", text, re.S)
        for span in spans:
            for line in span.splitlines():
                if line.lstrip().startswith("#"):
                    continue                      # an illustrative or proposed command, not a reference
                invocation = INVOCATION.search(line)
                if not invocation or invocation.group(1) not in known:
                    continue
                tool = invocation.group(1)
                for flag in FLAG.findall(line):
                    if flag not in known[tool]:
                        out.append((str(path.relative_to(root)), tool, flag))
    return out


def test_the_detector_notices_a_flag_the_tool_does_not_declare(tmp_path):
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "t.py").write_text(
        "ap.add_argument('--real', action='store_true')\n", encoding="utf-8"
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text(
        "Run `tools/t.py --real`.\n\nAnd `tools/t.py --imaginary`.\n\n"
        "# tools/t.py --commented-is-not-a-reference\n",
        encoding="utf-8",
    )
    assert unknown_flags(tmp_path) == [("docs/a.md", "t.py", "--imaginary")]


def test_no_tracked_command_names_a_flag_that_does_not_exist():
    """The standing check: the runbook's own commands are held to it."""
    offenders = unknown_flags(ROOT)
    assert not offenders, (
        "these documented commands name flags their tool does not declare:\n  "
        + "\n  ".join(f"{path}: {tool} {flag}" for path, tool, flag in sorted(set(offenders)))
    )
