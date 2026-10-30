"""The version is one value in five places, and the API publishes one of them to clients.

`docs/PROGRAMME-STATUS.md` says it plainly: "The version is what `laya_rag.__version__` reports and what
the HTTP API publishes in its OpenAPI schema, so a client can tell which engine it is talking to." That
only holds if the five copies agree:

1. `pyproject.toml` — what an install records
2. `src/laya_rag/__init__.py` — what the package reports, and what the OpenAPI schema carries
3. `README.md` — what a reader sees first
4. `docs/PROGRAMME-STATUS.md` — what the next maintainer is told is current
5. `CHANGELOG.md` — the newest heading, which is what "current" means

A divergence is not cosmetic: a service answering `/openapi.json` with one version while its changelog
claims another cannot be matched to the release notes that describe its behaviour, which is exactly the
provenance failure this repository catalogues.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import laya_rag  # noqa: E402


def declared_versions() -> dict[str, str]:
    """Every place the version is written, by file."""
    found: dict[str, str] = {}
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    assert match, "pyproject.toml has no version line"
    found["pyproject.toml"] = match.group(1)

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"\*\*Version (\d+\.\d+\.\d+)\*\*", readme)
    assert match, "README.md has no '**Version x.y.z**' line"
    found["README.md"] = match.group(1)

    status = (ROOT / "docs" / "PROGRAMME-STATUS.md").read_text(encoding="utf-8")
    match = re.search(r"`(\d+\.\d+\.\d+)` is the current tagged release", status)
    assert match, "PROGRAMME-STATUS.md no longer names the current release"
    found["docs/PROGRAMME-STATUS.md"] = match.group(1)

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## (\d+\.\d+\.\d+) — ", changelog, re.MULTILINE)
    assert headings, "CHANGELOG.md has no version heading"
    found["CHANGELOG.md (newest heading)"] = headings[0]
    found["CHANGELOG.md (ordered)"] = "descending" if headings == sorted(
        headings, key=lambda v: tuple(int(p) for p in v.split(".")), reverse=True
    ) else "unsorted"
    return found


def test_every_place_the_version_is_written_agrees():
    found = declared_versions()
    assert found["CHANGELOG.md (ordered)"] == "descending", (
        "the changelog's newest heading is not its highest version"
    )
    found.pop("CHANGELOG.md (ordered)")
    values = {**found, "laya_rag.__version__": laya_rag.__version__}
    distinct = set(values.values())
    assert len(distinct) == 1, "the version disagrees across files:\n" + "\n".join(
        f"  {name}: {value}" for name, value in values.items()
    )


def test_the_openapi_schema_publishes_the_package_version():
    """The client-facing consequence: a caller reads `info.version` and matches it to a changelog entry."""
    from laya_rag.api import create_app

    app = create_app()
    assert app.openapi()["info"]["version"] == laya_rag.__version__


def test_the_readme_does_not_state_an_exact_test_count_it_cannot_keep():
    """An exact count in the README drifts every round; '~1,900' does not.

    The page said "1896 collected" while the suite collected 1,912 — a live claim about the documented command, wrong
    by sixteen and drifting with every test added. The historical counts in `CHANGELOG.md` stay exact because they
    describe the runs that produced them; the README's is a promise about *now*, so it is stated at a precision the
    page can keep.
    """
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "tests; 5 need real weights" in text, "the README's suite note changed shape"
    assert "collected, 5 of them real-weights" not in text, "an exact, drifting count is back"
