"""Everything shipped parses under the Python the package declares, not the one this host runs.

`pyproject.toml` says `requires-python = ">=3.10"` and CI runs 3.10 and 3.13. A 3.11+ syntax slip —
`type` statements, `except*`, PEP 695 generics — compiles on 3.13 and fails the 3.10 job, which is a push
nobody can undo. The same class of gap as the two tests that needed torch while the CI job installed none:
**green on the machine where everything is installed says nothing about the machine that follows
`pyproject.toml`.**

This parses every shipped module with `ast.parse(..., feature_version=...)` at the declared floor. It cannot
catch a *runtime* API added in a later version; the CI matrix is what covers that. It catches the syntax,
which is the part that fails loudly and cheaply.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
WORKFLOWS = ROOT / ".github" / "workflows" / "ci.yml"


def declared_floor() -> tuple[int, int]:
    """The minimum Python from `requires-python`, as a grammar version."""
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'requires-python\s*=\s*">=(\d+)\.(\d+)"', text)
    assert match, "pyproject.toml has no `requires-python = \">=X.Y\"`"
    return int(match.group(1)), int(match.group(2))


def shipped_modules(directory: Path | None = None) -> list[Path]:
    root = directory or ROOT
    paths = []
    for package in ("src", "tools", "tests"):
        base = root / package
        if base.is_dir():
            paths.extend(sorted(base.rglob("*.py")))
    return paths


def incompatible(version: tuple[int, int], paths: list[Path]) -> list[str]:
    problems = []
    for path in paths:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=version)
        except SyntaxError as exc:
            problems.append(f"{path.relative_to(ROOT)}:{exc.lineno}: {exc.msg}")
    return problems


def test_every_shipped_module_parses_under_the_declared_floor():
    floor = declared_floor()
    problems = incompatible(floor, shipped_modules())
    assert not problems, (
        f"these files need a Python newer than the declared floor {floor[0]}.{floor[1]}:\n  "
        + "\n  ".join(problems)
    )


def test_the_ci_matrix_runs_at_the_declared_floor():
    """A floor the CI never runs is a floor nobody has tested."""
    floor = declared_floor()
    text = WORKFLOWS.read_text(encoding="utf-8")
    versions = re.findall(r'"(\d+\.\d+)"', text)
    assert f"{floor[0]}.{floor[1]}" in versions, (
        f"pyproject declares >= {floor[0]}.{floor[1]} but the CI matrix runs {versions}"
    )


# Syntax that arrived *after* a plausible floor, cheapest first, with the version that introduced
# it. The snippet has two jobs: the running interpreter must accept it (so a refusal below is about
# the floor, not about the snippet being nonsense), and the floor must refuse it.
NEWER_SYNTAX: tuple[tuple[tuple[int, int], str], ...] = (
    ((3, 11), "try:\n    pass\nexcept* TypeError:\n    pass\n"),   # PEP 654, `except*`
    ((3, 12), "def f[T](x: T) -> T:\n    return x\n"),              # PEP 695 generics
    ((3, 13), "type Alias = int\n"),                                # PEP 695 type statement
)


def newer_than_floor() -> tuple[tuple[int, int], str] | None:
    """A construct newer than `requires-python` that this interpreter can also parse.

    `None` when the host *is* the floor or older: on 3.10 — the floor's own row of the CI matrix —
    there is no newer syntax to refuse, and the second half has nothing to demonstrate. The first
    version of this test hard-coded the 3.12 construct and asserted it parsed, which failed on the
    3.10 job with `SyntaxError: invalid syntax` — the guard against a floor-only failure became a
    floor-only failure itself.
    """
    floor = declared_floor()
    host = sys.version_info[:2]
    return next(
        ((version, snippet) for version, snippet in NEWER_SYNTAX
         if version > floor and host >= version),
        None,
    )


def test_the_check_can_fail_and_sees_a_real_number_of_files():
    """Both halves: a newer syntax is refused, and the scan is not vacuous."""
    assert len(shipped_modules()) > 100                             # the scan is not vacuous
    candidate = newer_than_floor()
    if candidate is None:
        pytest.skip(
            f"this host is Python {sys.version_info[0]}.{sys.version_info[1]}, not newer than the "
            f"declared floor {declared_floor()[0]}.{declared_floor()[1]}: no newer syntax to refuse"
        )
    _version, newer_only = candidate
    assert ast.parse(newer_only) is not None                       # valid on this host
    try:
        ast.parse(newer_only, feature_version=declared_floor())
    except SyntaxError:
        return
    raise AssertionError("a newer-than-floor construct parsed under the declared floor")
