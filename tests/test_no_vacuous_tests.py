"""A test with no check is a test that cannot fail, and it reads as coverage.

`findings-measurement-failures.md` records a class for this at the level of *gates* ("a gate that is green
because it had nothing to do"). The same failure exists one level down and is easier to miss: a test function
that calls a helper and asserts nothing passes forever, shows green in a summary, and counts in a total.

This audits the suite itself — 1,860+ test functions — for functions containing no recognisable check: an
`assert` statement, a `pytest.raises`/`fail`/`skip`, or a `numpy.testing.assert_*`/`self.assert*` call.

**What it cannot see**, stated because a check that hides its blind spot is the thing this file is about:

* semantic vacuity — `assert result == result`, or an assertion that cannot fail because the input is empty;
* a check inside a helper the test calls.

It catches the textual case, which is the one that appears by accident.

Two legitimate exemptions are named rather than pattern-matched: tests that assert by **not raising**, where
the absence of an exception *is* the assertion.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

#: `(file, function)` pairs whose assertion is that nothing raises.
RAISES_NOTHING = {
    ("test_heavylock.py", "test_allow_concurrent_bypasses_the_lock"),
}

MARKERS = ("assert", "raises", "fail", "skip", "xfail")


def has_check(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assert):
            return True
        if isinstance(sub, ast.Attribute) and any(marker in sub.attr for marker in MARKERS):
            return True
        if isinstance(sub, ast.Call):
            func = sub.func
            name = getattr(func, "id", None) or getattr(func, "attr", None) or ""
            if any(name == marker or name.startswith(marker) for marker in MARKERS):
                return True
    return False


def _test_functions() -> list[tuple[str, str, ast.AST]]:
    found: list[tuple[str, str, ast.AST]] = []
    for path in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                found.append((path.name, node.name, node))
    return found


def test_no_test_function_is_vacuous():
    offenders = [
        f"{name}::{function}"
        for name, function, node in _test_functions()
        if not has_check(node) and (name, function) not in RAISES_NOTHING
    ]
    assert not offenders, "these tests contain no recognisable check:\n  " + "\n  ".join(offenders)


def test_the_audit_sees_a_real_suite_and_can_fail():
    """A vacuous scan would pass the test above perfectly."""
    assert len(_test_functions()) > 1000

    empty = ast.parse("def test_nothing():\n    helper()\n").body[0]
    assert not has_check(empty)
    for source in (
        "def test_a():\n    assert x == 1\n",
        "def test_b():\n    with pytest.raises(ValueError):\n        f()\n",
        "def test_c():\n    np.testing.assert_allclose(a, b)\n",
        "def test_d():\n    pytest.skip('no artefact')\n",
    ):
        assert has_check(ast.parse(source).body[0]), source
