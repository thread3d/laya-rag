"""Every knob the code reads is documented, and every documented knob is one the code reads.

`tests/test_docs_tool_flags_exist.py` holds the CLI flags to this contract; this is the same check for
environment variables, which are the larger surface — 91 of them, and a knob that ships undocumented is
a capability nobody can find. The interesting direction is *read but not documented*: a new variable
that appears in `config.py` and not in the README is invisible to every reader of the configuration
table, and the table is the only place they are listed.

The reverse direction is checked softly. `os.environ[...]` and `_env("...")` calls are all this can see,
and a variable read through a helper this does not model would otherwise fail the suite for the wrong
reason — so the count is bounded rather than the membership asserted.
"""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "laya_rag"
README = ROOT / "README.md"

#: `_env("X")` and its typed siblings, plus any explicit `LAYA_RAG_X` spelling in the source.
ENV_CALL = re.compile(r'_env(?:_bool|_int|_float|_path)?\(\s*"([A-Z][A-Z0-9_]*)"')
EXPLICIT = re.compile(r"LAYA_RAG_[A-Z0-9_]+")
BACKTICKED = re.compile(r"`([A-Z][A-Z0-9_]{2,})`")


def read_variables() -> set[str]:
    """Every variable the package can read, as its full `LAYA_RAG_…` spelling."""
    found: set[str] = set()
    for path in SRC.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        found |= {"LAYA_RAG_" + name for name in ENV_CALL.findall(text)}
        found |= set(EXPLICIT.findall(text))
    found.discard("LAYA_RAG_")  # the prefix itself, quoted in a docstring
    return found


def documented_variables() -> set[str]:
    """Every bare name the README's tables put in backticks (the table uses bare names)."""
    documented: set[str] = set()
    for line in README.read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            documented |= set(BACKTICKED.findall(line))
    return documented


def test_every_variable_the_code_reads_is_in_the_configuration_table():
    missing = sorted(
        variable for variable in read_variables() if variable.removeprefix("LAYA_RAG_") not in documented_variables()
    )
    assert not missing, (
        "these are read by the package but appear in no README table:\n  " + "\n  ".join(missing)
    )


def test_the_table_is_not_padded_with_names_nothing_reads():
    """A soft bound: the table may name a few variables this extractor cannot see, not a crowd."""
    documented = documented_variables()
    read = read_variables()
    unknown = sorted(name for name in documented if "LAYA_RAG_" + name not in read)
    assert len(unknown) < 8, f"the table names {len(unknown)} variables the code never reads: {unknown}"


def test_the_extraction_finds_the_variables_it_is_meant_to():
    """If the regexes stopped matching, both tests above would pass vacuously."""
    read = read_variables()
    assert len(read) > 60
    assert {"LAYA_RAG_CHECKPOINT", "LAYA_RAG_INDEX_DIR", "LAYA_RAG_SECOND_STAGE_TOP_M"} <= read
    assert documented_variables() >= {"CHECKPOINT", "INDEX_DIR"}
