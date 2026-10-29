"""`make help` is the first thing a reader runs, so it has to list the tasks that exist.

Fifteen of the Makefile's ninety-eight targets were missing from it — `mteb-eval`,
`calibration-report`, `nugget-recall`, `leakage-audit`, `reranker-fairness`, the two `labelling-*` queues,
and others — even though every one of them is documented somewhere in the prose. A target a reader
cannot discover from `make help` is a capability the project has and its users do not.

`help` itself is exempt: it is the target that prints the list.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"

#: The target that prints the list, and the only one it cannot plausibly list.
EXEMPT = {"help"}


def phony_targets(text: str) -> set[str]:
    match = re.search(r"^\.PHONY: (.*)$", text, re.MULTILINE)
    assert match, "the Makefile has no .PHONY line"
    return set(match.group(1).split())


def help_targets(text: str) -> set[str]:
    """Every target named by an `@echo "name ..."` line in the help recipe."""
    block = re.search(r"^help:\n((?:\t.*\n)+)", text, re.MULTILINE)
    assert block, "the Makefile has no help target"
    return {m.group(1) for m in re.finditer(r'@echo "([a-z][a-z0-9-]*)', block.group(1))}


def test_every_target_is_listed_by_make_help():
    text = MAKEFILE.read_text(encoding="utf-8")
    missing = sorted(phony_targets(text) - help_targets(text) - EXEMPT)
    assert not missing, (
        "these targets exist but `make help` never mentions them:\n  " + "\n  ".join(missing)
    )


def test_help_does_not_name_a_target_that_does_not_exist():
    text = MAKEFILE.read_text(encoding="utf-8")
    invented = sorted(help_targets(text) - phony_targets(text))
    assert not invented, f"`make help` names targets that do not exist: {invented}"


def test_the_extraction_sees_a_real_number_of_targets():
    """If the regexes stopped matching, both tests above would pass vacuously."""
    text = MAKEFILE.read_text(encoding="utf-8")
    assert len(phony_targets(text)) > 60
    assert len(help_targets(text)) > 60
