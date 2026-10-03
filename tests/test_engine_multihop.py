"""Offline tests for the engine-level MultiHop-RAG harness.

The engine run itself is the long job; what is tested here is everything that decides *what*
it measures: which records are scored, how the per-query rows aggregate, that the environment
the engine reads is derived from the run's knobs, and that the fingerprint separates the two
configurations the roadmap wants compared (shipped default vs a hybrid bi-encoder).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def _load():
    name = "laya_rag_engine_multihop"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "engine_multihop.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


em = _load()
jobkit = sys.modules["jobkit"]

ITEMS = [
    {"question_type": "inference_query", "query": "q0", "evidence_list": [{"fact": "f0"}]},
    {"question_type": "null_query", "query": "n1", "evidence_list": []},
    {"question_type": "comparison_query", "query": "q2", "evidence_list": [{"fact": "f2a"}, {"fact": "f2b"}]},
]


def test_answerable_items_excludes_null_queries_and_keeps_original_indices():
    kept = em.answerable_items(ITEMS)
    assert [i for i, _ in kept] == [0, 2]
    assert [x["query"] for _, x in kept] == ["q0", "q2"]


def test_gold_facts_reads_every_evidence_fact():
    assert em.gold_facts(ITEMS[2]) == ["f2a", "f2b"]
    assert em.gold_facts(ITEMS[1]) == []


def test_summarize_rows_averages_each_metric():
    rows = [
        {"Hits@10": 1.0, "Hits@4": 1.0, "MAP@10": 0.5, "MRR@10": 1.0},
        {"Hits@10": 0.0, "Hits@4": 0.0, "MAP@10": 0.1, "MRR@10": 0.0},
    ]
    assert em.summarize_rows(rows) == {
        "Hits@10": 0.5, "Hits@4": 0.5, "MAP@10": 0.3, "MRR@10": 0.5,
    }


def test_summarize_rows_of_nothing_is_zero_not_a_crash():
    assert em.summarize_rows([]) == {k: 0.0 for k in em.METRIC_KEYS}


def test_fingerprint_separates_the_two_configurations():
    default = em.parse_args([])
    hybrid = em.parse_args(["--embed-model", "BAAI/bge-small-en-v1.5"])
    assert em._fingerprint(default) != em._fingerprint(hybrid)


def test_fingerprint_changes_with_chunking_and_is_shard_independent():
    base = em.parse_args([])
    assert em._fingerprint(base) != em._fingerprint(em.parse_args(["--no-chunk-auto"]))
    assert em._fingerprint(base) != em._fingerprint(em.parse_args(["--context-prefix"]))
    assert em._fingerprint(base) == em._fingerprint(em.parse_args(["--shard", "2/3"]))


def test_configure_engine_sets_the_shipped_default_when_no_model_is_given(monkeypatch):
    for key in ("LAYA_RAG_EMBED_MODEL", "LAYA_RAG_RERANK_MODEL", "LAYA_RAG_SPARSE_MODEL"):
        monkeypatch.setenv(key, "stale")
    args = em.parse_args(["--out", str(ROOT / "eval" / "benchmarks" / "multihop-rag" / "engine-x")])
    index_dir = em.configure_engine(args)
    assert index_dir.name == "multihop"
    assert "LAYA_RAG_EMBED_MODEL" not in os.environ
    assert "LAYA_RAG_RERANK_MODEL" not in os.environ
    assert "LAYA_RAG_SPARSE_MODEL" not in os.environ
    # Nothing may be filtered, or the returned order is not the retrieval order.
    assert os.environ["LAYA_RAG_RELEVANCE_THRESHOLD"] == "0.0"
    assert os.environ["LAYA_RAG_ANSWERABILITY_THRESHOLD"] == "0.0"


def test_build_report_reconstructs_from_the_reserved_row():
    rows = {
        jobkit.META_KEY: {"_settings": {"top_k": 10}, "queries": {"0": "q0", "1": "q1"}},
        "0": {"metrics": {"Hits@10": 1.0, "Hits@4": 1.0, "MAP@10": 0.5, "MRR@10": 1.0}},
        "1": {"metrics": {"Hits@10": 0.0, "Hits@4": 0.0, "MAP@10": 0.0, "MRR@10": 0.0}},
    }
    report = em._build_report(rows)
    assert report["_settings"] == {"top_k": 10}
    assert report["summary"]["Hits@10"] == pytest.approx(0.5)
    assert set(report["per_query"]) == {"0", "1"}


def test_build_report_without_a_checkpoint_fails_clearly():
    with pytest.raises(RuntimeError, match="no shard checkpoints"):
        em._build_report({})


def test_args_default_to_the_benchmark_metric_and_unsharded():
    args = em.parse_args([])
    assert args.top_k == 10
    assert args.shard == "0/1" and args.merge == 0 and args.fresh is False
    assert args.embed_model is None
