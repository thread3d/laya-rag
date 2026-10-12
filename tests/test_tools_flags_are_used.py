"""A flag that is parsed and never used is a knob that silently does nothing.

This is not hypothetical. `tools/mteb_search.py --first-stage-only` printed a banner and set nothing on
the adapter for four rounds, so every "first-stage-only" run still ran the 322M decision on every
candidate — and when that was found, 18 recorded reports turned out to be their first stage's order
wearing a reranker's name. The class is cheap to detect: an `add_argument` whose `dest` never appears as
`args.<dest>` in the same file.

Five genuine instances exist today and are **named rather than exempted** — the test fails on any
*new* one, and the list is the work remaining, not a permission:

    (none — see DEFAULT_MODE_ALIASES for the one flag that is a spelling of its tool's default)

Three of the original five are no longer here, each by the route that fitted it: `multihop_hop2.py
--rrf-k` and `laya_finetune_probe.py --seed` **named mechanisms their tools do not have** (no fusion, no
randomness) and were removed; `multihop_diagnostics.py --bootstrap-samples` **could act** and is now
wired into `_mean_ci`, whose own default matches the flag's so nothing changes unless it is set.
"""
from __future__ import annotations

import ast
import pathlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: Flags that name the **default mode** rather than a distinct one. They are not defects — the flag
#: works by selecting the branch that already runs — but the code never reads `args.<dest>`, so the
#: detector cannot tell them from the inert ones. Kept separate from `KNOWN_INERT` on purpose: an
#: alias is a documented spelling, an inert flag is a lie.
DEFAULT_MODE_ALIASES = {
    ("generation_eval.py", "--list"),          # `if args.check: ...` else this branch prints the table
}

#: Known inert flags: (file, flag). Each is a defect with the same shape as the MTEB one, recorded
#: here so the check can be strict about everything else. **Currently empty**, which is the point.
KNOWN_INERT: set[tuple[str, str]] = set()


def inert_flags(path: Path) -> list[tuple[str, str]]:
    """Flags whose parsed value is never read in the file that declares them."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if "**vars(" in text:
        # The parser's namespace is forwarded wholesale, so a flag may be read by a callee and this
        # check cannot tell. Skipping is honest; guessing would produce a false failure.
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        flags = [a.value for a in node.args
                 if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("--")]
        if not flags:
            continue
        dest = next((kw.value.value for kw in node.keywords
                     if kw.arg == "dest" and isinstance(kw.value, ast.Constant)), None)
        if dest is None:
            dest = flags[0].lstrip("-").replace("-", "_")
        if dest in {"help", "version"}:
            continue
        if f"args.{dest}" not in text and f'args["{dest}"]' not in text:
            found.append((flags[0], dest))
    return found


def test_no_tool_gains_a_flag_it_never_reads():
    """The standing check: a new inert flag fails here, at the commit that adds it."""
    tools = sorted((ROOT / "tools").glob("*.py"))
    if not tools:
        pytest.skip("no tools/ directory to scan")
    offenders = [
        (path.name, flag)
        for path in tools
        for flag, _dest in inert_flags(path)
        if (path.name, flag) not in KNOWN_INERT | DEFAULT_MODE_ALIASES
    ]
    assert not offenders, (
        "these flags are parsed and never read, so they silently do nothing:\n  "
        + "\n  ".join(f"{name}: {flag}" for name, flag in offenders)
    )


def test_the_detector_flags_a_planted_inert_flag(tmp_path):
    """The detector has to have teeth, or the standing check above is decoration."""
    planted = tmp_path / "tool.py"
    planted.write_text(
        "import argparse\n"
        "def main():\n"
        "    ap = argparse.ArgumentParser()\n"
        "    ap.add_argument('--never-read', action='store_true')\n"
        "    args = ap.parse_args()\n"
        "    print('done')\n",
        encoding="utf-8",
    )
    assert inert_flags(planted) == [("--never-read", "never_read")]


def test_a_forwarded_namespace_is_not_guessed_at(tmp_path):
    """`**vars(args)` means a callee may read it, so the check abstains rather than failing."""
    forwarded = tmp_path / "tool.py"
    forwarded.write_text(
        "import argparse\n"
        "def main():\n"
        "    ap = argparse.ArgumentParser()\n"
        "    ap.add_argument('--maybe-used')\n"
        "    run(**vars(ap.parse_args()))\n",
        encoding="utf-8",
    )
    assert inert_flags(forwarded) == []
