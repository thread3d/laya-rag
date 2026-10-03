"""Offline tests for the jobkit retrofit of the MultiHop diagnostics.

The diagnostics report is what the reorder-ceiling claim rests on, so the properties worth
pinning are the ones a resume could corrupt: the fingerprint must cover every setting that
changes a number, and the merged report must put the per-query rows back in their original
order rather than in whatever order the checkpoint happened to record them.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def _load():
    name = "laya_rag_multihop_diagnostics"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "multihop_diagnostics.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


diag = _load()
jobkit = sys.modules["jobkit"]

SETTINGS = {"depths": [10, 100], "model": "m", "top_k": 10}


def _row(q, ceiling):
    return {
        "query": q,
        "fused_MAP@10": 0.0, "fused_Hits@10": 0.0, "fused_Hits@4": 0.0,
        "hop2_size": 0, "hop_gap": 0.0, "facts_present": 1,
        "pool_fact_recall@10": 0.5, "pool_ceiling_MAP@10@10": ceiling,
        "hop1_fact_recall@10": 0.0, "hop2_fact_recall@10": 0.0, "hop2_retention@10": None,
        "pool_fact_recall@100": 1.0, "pool_ceiling_MAP@10@100": 0.6,
        "hop1_fact_recall@100": 0.0, "hop2_fact_recall@100": 0.0, "hop2_retention@100": None,
    }


def test_fingerprint_covers_every_scoring_setting():
    base = diag.parse_args([])
    fp = diag._fingerprint(base, "m", [10, 100])
    assert diag._fingerprint(diag.parse_args(["--overlap", "0"]), "m", [10, 100]) != fp
    assert diag._fingerprint(diag.parse_args(["--rrf-k", "60"]), "m", [10, 100]) != fp
    assert diag._fingerprint(diag.parse_args(["--top-k", "4"]), "m", [10, 100]) != fp
    assert diag._fingerprint(base, "other-model", [10, 100]) != fp
    assert diag._fingerprint(base, "m", [10, 500]) != fp


def test_fingerprint_does_not_depend_on_the_run_stem_or_shard():
    assert diag._fingerprint(diag.parse_args([]), "m", [10]) == \
        diag._fingerprint(diag.parse_args(["--shard", "2/3", "--out", "elsewhere"]), "m", [10])


def test_build_report_restores_the_original_query_order():
    """Rows are recorded in completion order; the report must follow the corpus order."""
    rows = {
        jobkit.META_KEY: {"_settings": SETTINGS, "query_order": ["2", "0", "1"]},
        "1": _row("1", 0.11),
        "2": _row("2", 0.22),
        "0": _row("0", 0.33),
    }
    report = diag._build_report(rows)
    assert [row["query"] for row in report["per_query"]] == ["2", "0", "1"]
    assert report["summary"]["depth"]["10"]["pool_ceiling_MAP@10"]["mean"] == pytest.approx(
        (0.22 + 0.33 + 0.11) / 3
    )


def test_build_report_drops_rows_that_are_not_in_the_order():
    rows = {
        jobkit.META_KEY: {"_settings": SETTINGS, "query_order": ["0"]},
        "0": _row("0", 0.5),
        "orphan": _row("orphan", 0.9),
    }
    report = diag._build_report(rows)
    assert [row["query"] for row in report["per_query"]] == ["0"]


def test_build_report_without_a_checkpoint_fails_clearly():
    with pytest.raises(RuntimeError, match="no shard checkpoints"):
        diag._build_report({})


def test_args_default_to_unsharded_and_the_official_overlap():
    args = diag.parse_args([])
    assert args.shard == "0/1" and args.merge == 0 and args.fresh is False
    assert args.overlap == 200 and args.depths == "10,100,500"
