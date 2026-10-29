"""Every `make …` the CI workflow runs must be a target the Makefile has.

`tests/test_docs_make_targets_exist.py` holds the *documentation* to that contract. The workflow is the
other place a `make` invocation lives, and a renamed target there fails only on the next push — after the
change that renamed it has been committed and, worse, after it has been trusted. A CI step that cannot
run is the same class of defect as a documented command that cannot run, one step further from the reader.

The check reads the YAML as text rather than through a parser, because the commands live inside `run: |`
blocks and block scalars; a parser would hand back the same string. It also matches inside **comments**,
deliberately: a workflow comment that names a command nothing defines misleads the next maintainer by the
same mechanism as a step that cannot run, and there are few enough of them that the noise a stricter scan
would avoid does not exist here.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

MAKE = re.compile(r"\bmake\s+([a-z][a-z0-9-]+)")
PHONY = re.compile(r"^\.PHONY: (.*)$", re.MULTILINE)


def make_targets() -> set[str]:
    match = PHONY.search((ROOT / "Makefile").read_text(encoding="utf-8"))
    assert match, "the Makefile has no .PHONY line"
    return set(match.group(1).split())


def invoked_targets(directory: Path = WORKFLOWS) -> dict[str, set[str]]:
    """Targets the workflow scripts invoke, by workflow file."""
    found: dict[str, set[str]] = {}
    for path in sorted(directory.glob("*.yml")):
        found[path.name] = set(MAKE.findall(path.read_text(encoding="utf-8")))
    return found


def test_every_make_invocation_in_ci_names_a_real_target():
    available = make_targets()
    missing = {
        workflow: sorted(targets - available)
        for workflow, targets in invoked_targets().items()
    }
    missing = {workflow: targets for workflow, targets in missing.items() if targets}
    assert not missing, "the CI workflow runs targets the Makefile does not define:\n  " + "\n  ".join(
        f"{workflow}: {', '.join(targets)}" for workflow, targets in sorted(missing.items())
    )


def test_the_workflows_invoke_something():
    """If the extraction stopped matching, the test above would pass vacuously."""
    invoked = invoked_targets()
    assert invoked, "no workflow files were found"
    assert sum(len(targets) for targets in invoked.values()) >= 3


def test_the_check_can_fail(tmp_path):
    (tmp_path / "ci.yml").write_text("run: make nonexistent-target\n", encoding="utf-8")
    invoked = invoked_targets(tmp_path)
    assert invoked["ci.yml"] == {"nonexistent-target"}
    assert invoked["ci.yml"] - make_targets() == {"nonexistent-target"}
