"""Offline tests for the jobkit retrofit of the engine-level BEIR harness.

This is the harness behind the engine-level BEIR claims, and the 18-dataset scale work in Phase 4
will run it over corpora that take hours. What is pinned here is what a resume could corrupt: the
fingerprint must cover the settings that change a number, and the merged report must recompute
the metrics over *all* the rankings at once, because nDCG is not a per-query average that can be
written one row at a time.
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
    name = "laya_rag_engine_beir"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "engine_beir.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


eb = _load()
jobkit = sys.modules["jobkit"]

QRELS = {"q1": {"d1": 1}, "q2": {"d2": 2}}


def test_report_stem_is_dataset_and_tag():
    args = eb.parse_args(["--dataset", "scifact", "--tag", "sparse", "--out", "/tmp/eb"])
    assert eb.report_stem(args) == Path("/tmp/eb/scifact-sparse")


def test_fingerprint_covers_the_settings_that_change_a_number():
    base = eb.parse_args([])
    fp = eb._fingerprint(base, "i")
    assert eb._fingerprint(eb.parse_args(["--dataset", "nfcorpus"]), "i") != fp
    assert eb._fingerprint(eb.parse_args(["--tag", "other"]), "i") != fp
    assert eb._fingerprint(eb.parse_args(["--sparse"]), "i") != fp
    assert eb._fingerprint(eb.parse_args(["--chunk-chars", "800"]), "i") != fp
    assert eb._fingerprint(eb.parse_args(["--context-prefix"]), "i") != fp
    assert eb._fingerprint(eb.parse_args(["--limit", "50"]), "i") != fp


def test_fingerprint_ignores_the_shard_and_the_out_dir():
    assert eb._fingerprint(eb.parse_args([]), "i") == \
        eb._fingerprint(eb.parse_args(["--shard", "2/5", "--out", "/tmp/elsewhere"]), "i")


def test_fingerprint_covers_the_index_it_was_measured_against():
    """§4.2: the same settings over a rebuilt index is a different number, so it must not resume."""
    base = eb.parse_args([])
    assert eb._fingerprint(base, "index-a") != eb._fingerprint(base, "index-b")


def test_the_identity_free_fingerprint_is_the_legacy_one():
    """`identity=None` must reproduce the pre-index fingerprint, or no checkpoint can be migrated."""
    base = eb.parse_args([])
    assert eb._fingerprint(base, None) != eb._fingerprint(base, "index-a")


def test_build_report_recomputes_metrics_over_all_rankings():
    args = eb.parse_args([])
    rows = {
        jobkit.META_KEY: {"_settings": {"dataset": "scifact"}},
        "q1": {"ranking": ["d1", "d3"]},
        "q2": {"ranking": ["d2", "d4"]},
    }
    report = eb._build_report(rows, args, QRELS, corpus_size=4)
    assert report["queries"] == 2
    assert report["documents"] == 4
    assert report["summary"]["nDCG@10"] == pytest.approx(1.0)


def test_build_report_scores_a_missed_ranking_below_a_found_one():
    args = eb.parse_args([])
    single = {"q1": {"d1": 1}}  # one qrel, so the mean is this query's own score
    found = eb._build_report(
        {jobkit.META_KEY: {"_settings": {}}, "q1": {"ranking": ["d1"]}}, args, single, 2)
    missed = eb._build_report(
        {jobkit.META_KEY: {"_settings": {}}, "q1": {"ranking": ["d9"]}}, args, single, 2)
    assert found["summary"]["nDCG@10"] == pytest.approx(1.0)
    assert missed["summary"]["nDCG@10"] == pytest.approx(0.0)


def test_build_report_averages_over_every_qrel_query_not_only_the_ranked_ones():
    """An unranked qrel query scores zero rather than being dropped; it must not vanish."""
    args = eb.parse_args([])
    report = eb._build_report(
        {jobkit.META_KEY: {"_settings": {}}, "q1": {"ranking": ["d1"]}}, args, QRELS, 2)
    assert report["summary"]["nDCG@10"] == pytest.approx(0.5)


def test_build_report_without_a_checkpoint_fails_clearly():
    with pytest.raises(RuntimeError, match="no shard checkpoints"):
        eb._build_report({}, eb.parse_args([]), QRELS, 2)


def test_args_default_to_unsharded():
    args = eb.parse_args([])
    assert args.shard == "0/1" and args.merge == 0 and args.fresh is False
