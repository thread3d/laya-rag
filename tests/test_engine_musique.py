"""Offline tests for the engine-level MuSiQue harness.

The two decisions worth pinning are the ones that would silently make the number wrong: the
metric must be computed over paragraph *positions* in retrieval order, and a subset run must keep
the full paragraph corpus while narrowing only the queries — because `tools/musique.py
--limit-queries` shrinks the corpus too, and comparing against it would not be like for like.
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
    name = "laya_rag_engine_musique"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "engine_musique.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


emq = _load()


def test_score_query_rewards_finding_every_gold_paragraph_early():
    # gold at positions 1 and 3; retrieved 1 first, then 0, then 3.
    metrics = emq.score_query([1, 0, 3, 2], [1, 3], top_k=4)
    assert metrics["Hits@10"] == 1.0
    assert metrics["Hits@4"] == 1.0
    assert metrics["MRR@10"] == 1.0
    assert metrics["MAP@10"] == pytest.approx((1 / 1 + 1 / 3) / 2)


def test_score_query_with_no_gold_in_the_ranking_is_zero_not_a_crash():
    metrics = emq.score_query([0, 1, 2], [7, 8], top_k=3)
    assert metrics == {"Hits@10": 0.0, "Hits@4": 0.0, "MAP@10": 0.0, "MRR@10": 0.0}


def test_score_query_normalizes_map_by_the_gold_count():
    """Finding one of four gold paragraphs is one quarter of the answer, not a perfect hit."""
    metrics = emq.score_query([0, 1, 2], [0, 5, 6, 7], top_k=10)
    assert metrics["MAP@10"] == pytest.approx(1.0 / 4)


def test_score_query_caps_the_map_denominator_at_top_k():
    """The benchmark's denominator is `min(gold_count, top_k)`, so a short k raises MAP."""
    metrics = emq.score_query([0, 1, 2], [0, 5, 6, 7], top_k=3)
    assert metrics["MAP@10"] == pytest.approx(1.0 / 3)


def test_summarize_rows_averages_and_handles_empty():
    rows = [
        {"Hits@10": 1.0, "Hits@4": 1.0, "MAP@10": 0.5, "MRR@10": 1.0},
        {"Hits@10": 0.0, "Hits@4": 0.0, "MAP@10": 0.1, "MRR@10": 0.0},
    ]
    assert emq.summarize_rows(rows) == {"Hits@10": 0.5, "Hits@4": 0.5, "MAP@10": 0.3, "MRR@10": 0.5}
    assert emq.summarize_rows([]) == {k: 0.0 for k in emq.METRICS}


def test_pair_map10_aligns_on_shared_query_ids_only():
    engine = {
        "a": {"metrics": {"MAP@10": 0.2}},
        "b": {"metrics": {"MAP@10": 0.6}},
        "c": {"metrics": {"MAP@10": 1.0}},  # the harness never scored this one
    }
    harness = {"a": 0.1, "b": 0.5, "z": 0.9}
    base, cand = emq.pair_map10(engine, harness)
    assert base == {"a": 0.1, "b": 0.5}
    assert cand == {"a": 0.2, "b": 0.6}


def test_pair_map10_of_no_overlap_is_empty():
    assert emq.pair_map10({"a": {"metrics": {"MAP@10": 0.2}}}, {"b": 0.1}) == ({}, {})


def test_fingerprint_separates_configurations_and_ignores_the_shard():
    base = emq.parse_args([])
    assert emq._fingerprint(base) != emq._fingerprint(emq.parse_args(["--embed-model", "m"]))
    assert emq._fingerprint(base) != emq._fingerprint(emq.parse_args(["--split", "train"]))
    assert emq._fingerprint(base) != emq._fingerprint(emq.parse_args(["--limit", "200"]))
    assert emq._fingerprint(base) == emq._fingerprint(emq.parse_args(["--shard", "1/3"]))


def test_load_task_keeps_the_full_corpus_when_queries_are_limited():
    """`--limit` narrows queries; the corpus is the whole split, so a subset is comparable."""
    path = emq.ROOT_BENCH / "validation.json"
    if not path.is_file():
        pytest.skip("MuSiQue validation split is not downloaded")
    corpus_all, golds_all, order_all, queries_all = emq.load_task(path)
    corpus_sub, golds_sub, order_sub, queries_sub = emq.load_task(path, limit=5)
    assert len(queries_sub) == 5
    assert len(corpus_sub) == len(corpus_all)          # the corpus did NOT shrink
    assert order_sub == order_all and golds_sub.keys() == golds_all.keys()
    assert len(corpus_sub) > len(queries_sub)


def test_build_report_reconstructs_from_the_reserved_row():
    jobkit = sys.modules["jobkit"]
    rows = {
        jobkit.META_KEY: {"_settings": {"split": "validation"}, "queries": {"a": "q"}},
        "a": {"metrics": {"Hits@10": 1.0, "Hits@4": 1.0, "MAP@10": 0.5, "MRR@10": 1.0}},
    }
    report = emq._build_report(rows)
    assert report["_settings"] == {"split": "validation"}
    assert report["summary"]["MAP@10"] == pytest.approx(0.5)


def test_build_report_without_a_checkpoint_fails_clearly():
    with pytest.raises(RuntimeError, match="no shard checkpoints"):
        emq._build_report({})


def test_args_default_to_the_full_split_and_unsharded():
    args = emq.parse_args([])
    assert args.split == "validation" and args.limit is None
    assert args.shard == "0/1" and args.merge == 0 and args.compare is False
