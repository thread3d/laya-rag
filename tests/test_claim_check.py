"""Offline tests for the claim-drift check.

The check exists because a corrected number was hand-copied into the *source* finding and the
findings index and stayed wrong there while six other documents were fixed. These tests pin the
properties that make it a gate rather than a linter: a table row must carry its own number, a
prose re-wrap must not fail it, and a known-stale value is rejected even when the correct one is
present elsewhere in the same paragraph.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    name = "laya_rag_claim_check"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "claim_check.py")
    module = importlib.util.module_from_spec(spec)
    # Register before exec: dataclasses resolves `cls.__module__` through sys.modules, and an
    # unregistered module makes `@dataclass` raise AttributeError.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cc = _load()

DIAGNOSTICS = ROOT / "eval" / "benchmarks" / "multihop-rag" / "diagnostics.json"

# A miniature report with the same shape as diagnostics.json: four queries, one of them an
# incomplete depth-100 pool. Derived: count 1, share 0.25, ceiling 0.40 vs 0.70, fused 0.10 vs 0.31.
REPORT = {
    "summary": {
        "depth": {"100": {"pool_ceiling_MAP@10": {"mean": 0.6088}}},
        "paired": {"pool_ceiling_MAP@10:100-10": {"delta": 0.2118}},
    },
    "per_query": [
        {"pool_fact_recall@100": 1.0, "pool_ceiling_MAP@10@100": 0.70, "fused_MAP@10": 0.31},
        {"pool_fact_recall@100": 1.0, "pool_ceiling_MAP@10@100": 0.70, "fused_MAP@10": 0.31},
        {"pool_fact_recall@100": 1.0, "pool_ceiling_MAP@10@100": 0.70, "fused_MAP@10": 0.31},
        {"pool_fact_recall@100": 0.5, "pool_ceiling_MAP@10@100": 0.40, "fused_MAP@10": 0.10},
    ],
}

CEILING_100 = ("summary", "depth", "100", "pool_ceiling_MAP@10", "mean")


def _doc(tmp_path: Path, text: str, name: str = "README.md") -> Path:
    (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def _claim(**overrides):
    base = dict(
        label="ceiling",
        doc="README.md",
        anchor=r"corrected ceiling is",
        value=0.6088,
        source=CEILING_100,
    )
    base.update(overrides)
    return cc.Claim(**base)


def test_claim_passes_when_doc_and_report_agree(tmp_path):
    root = _doc(tmp_path, "The corrected ceiling is **0.6088** at depth 100.\n")
    assert cc.run_checks([_claim()], [], root=root, report=REPORT) == []


def test_claim_fails_when_doc_disagrees_with_the_report(tmp_path):
    root = _doc(tmp_path, "The corrected ceiling is **0.7307** at depth 100.\n")
    failures = cc.run_checks([_claim(value=0.7307)], [], root=root, report=REPORT)
    assert len(failures) == 1
    assert "the report says 0.6088" in failures[0]


def test_anchor_selects_the_block_it_matches(tmp_path):
    root = _doc(tmp_path, "The corrected ceiling is **0.6088**.\n\nElsewhere: 0.6088 again.\n")
    failures = cc.run_checks([_claim(anchor=r"Elsewhere")], [], root=root, report=REPORT)
    # The anchor matches the second paragraph, and that paragraph carries the number, so the
    # claim passes: the check is about the anchored block, not about the first occurrence.
    assert failures == []


def test_claim_fails_when_no_anchored_block_states_the_number(tmp_path):
    root = _doc(tmp_path, "The corrected ceiling is 0.6088.\n")
    failures = cc.run_checks([_claim(anchor=r"a phrase that is absent")], [], root=root,
                             report=REPORT)
    assert "matched no line" in failures[0]


def test_table_row_is_its_own_region(tmp_path):
    """A second row's number must not vouch for the row the anchor selected."""
    root = _doc(tmp_path, "| depth | ceiling |\n| --- | --- |\n| 100 | 0.397 |\n| 500 | 0.609 |\n")
    failures = cc.run_checks([_claim(anchor=r"^\|\s*100\s*\|", value=0.609)], [], root=root,
                             report=REPORT)
    assert len(failures) == 1
    assert "does not state 0.609" in failures[0]


def test_prose_rewrap_keeps_the_claim_valid(tmp_path):
    root = _doc(tmp_path, "The corrected ceiling is\n**0.6088** at depth 100.\n")
    assert cc.run_checks([_claim()], [], root=root, report=REPORT) == []


def test_forbidden_value_is_rejected_even_when_the_correct_one_is_present(tmp_path):
    root = _doc(tmp_path, "The corrected ceiling is 0.6088, but 36% of pools are incomplete.\n")
    forbid = cc.Forbidden(label="stale share", doc="README.md", anchor=r"corrected ceiling is",
                          value=36.0, decimals=0)
    failures = cc.run_checks([_claim()], [forbid], root=root, report=REPORT)
    assert len(failures) == 1
    assert "stale 36 still in README.md" in failures[0]


def test_forbidden_value_allows_a_larger_number_containing_it(tmp_path):
    """`136` and `0.36` are not the stale `36`."""
    root = _doc(tmp_path, "The corrected ceiling is 0.6088; 136 queries, 0.36 of them.\n")
    forbid = cc.Forbidden(label="stale share", doc="README.md", anchor=r"corrected ceiling is",
                          value=36.0, decimals=0)
    assert cc.run_checks([_claim()], [forbid], root=root, report=REPORT) == []


def test_scaled_percentage_matches_the_derived_share(tmp_path):
    root = _doc(tmp_path, "25.0% of depth-100 pools are incomplete.\n")
    claim = _claim(anchor=r"depth-100 pools", value=25.0, source="incomplete_100_share",
                   scale=100.0, decimals=1, tol=0.05)
    assert cc.run_checks([claim], [], root=root, report=REPORT) == []


def test_derived_statistics_read_the_per_query_rows():
    assert cc.incomplete_100_count(REPORT) == 1
    assert cc.incomplete_100_share(REPORT) == pytest.approx(0.25)
    assert cc.ceiling_100_incomplete(REPORT) == pytest.approx(0.40)
    assert cc.ceiling_100_complete(REPORT) == pytest.approx(0.70)
    assert cc.fused_map_incomplete(REPORT) == pytest.approx(0.10)
    assert cc.fused_map_complete(REPORT) == pytest.approx(0.31)
    assert cc.headroom_completing(REPORT) == pytest.approx(0.70 - 0.6088)


def test_unknown_report_path_is_a_failure_not_a_crash(tmp_path):
    root = _doc(tmp_path, "The corrected ceiling is 0.6088.\n")
    failures = cc.run_checks([_claim(source=("summary", "nope"))], [], root=root, report=REPORT)
    assert len(failures) == 1
    assert "report has no path" in failures[0]


def test_missing_report_skips_rather_than_failing(tmp_path):
    code = cc.main(["--root", str(tmp_path), "--report", str(tmp_path / "absent.json")])
    assert code == 0


@pytest.mark.skipif(
    not DIAGNOSTICS.is_file(),
    reason="diagnostics.json is gitignored; run `make multihop-diagnostics` first",
)
def test_repository_claims_resolve_against_the_diagnostics_report():
    """The real gate: every claim quoted in prose equals the report that produced it."""
    report = json.loads(DIAGNOSTICS.read_text(encoding="utf-8"))
    assert cc.run_checks(cc.CHECKS, cc.FORBIDDEN, root=ROOT, report=report) == []
