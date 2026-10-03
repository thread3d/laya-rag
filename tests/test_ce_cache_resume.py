"""Offline tests for the jobkit retrofit of the cross-encoder cache.

The cache is the repository's first resumable long run, and the properties that matter are the
ones jobkit gives it: the fingerprint is a function of the scoring configuration and *not* of
which shard is running, and the shard merge can rebuild the exact report shape the downstream
experiments read. Running the cross-encoder itself is out of scope here; that is the long job.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def _load():
    name = "laya_rag_ce_cache"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "multihop_ce_cache.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ce = _load()
jobkit = sys.modules["jobkit"]


def _args(**overrides):
    base = dict(
        model="bge-small", overlap=200, depth=100, rrf_k=5,
        rerank_model="BAAI/bge-reranker-base", limit_queries=400, sample_seed=17,
        shard="0/1",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_fingerprint_is_a_function_of_the_scoring_configuration():
    base = ce._fingerprint(_args(), "BAAI/bge-small-en-v1.5")
    assert ce._fingerprint(_args(depth=500), "BAAI/bge-small-en-v1.5") != base
    assert ce._fingerprint(_args(sample_seed=18), "BAAI/bge-small-en-v1.5") != base
    assert ce._fingerprint(_args(), "BAAI/bge-large-en-v1.5") != base


def test_fingerprint_does_not_depend_on_which_shard_runs():
    """Every shard is the same configuration, so they must share one fingerprint."""
    assert ce._fingerprint(_args(shard="1/4"), "x") == ce._fingerprint(_args(shard="3/4"), "x")


def test_build_report_reconstructs_the_downstream_shape():
    rows = {
        jobkit.META_KEY: {
            "_settings": {"model": "m", "depth": 100},
            "golds": {"0": ["a fact"]},
        },
        "7": {"pool": ["d1"], "ce": [1.5], "atoms": [[0]]},
    }
    report = ce._build_report(rows)
    assert report["_settings"] == {"model": "m", "depth": 100}
    assert report["golds"] == {"0": ["a fact"]}
    assert report["queries"] == {"7": {"pool": ["d1"], "ce": [1.5], "atoms": [[0]]}}


def test_defaults_are_unsharded_and_do_not_merge():
    args = ce.parse_args([])
    assert args.shard == "0/1"
    assert args.merge == 0
    assert args.fresh is False
    assert jobkit.parse_shard(args.shard) == (0, 1)


def test_merge_without_any_shard_checkpoint_fails_clearly():
    import pytest

    with pytest.raises(RuntimeError, match="no shard checkpoints"):
        ce._build_report({})
