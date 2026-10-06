"""Tests for the 2WikiMultiHopQA adapter.

The adapter's job is to turn a 2Wiki record into the same corpus/gold shape the MuSiQue harness
uses, so the metrics are comparable. These pin the two things that would silently corrupt a row:
paragraph de-duplication across records and gold as positional indices into the finished corpus.
"""
from __future__ import annotations

import importlib.util
import json
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def _load():
    name = "laya_rag_twowiki"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "twowiki.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tw = _load()


def _write(tmp_path, records):
    path = tmp_path / "dev.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


RECORDS = [
    {
        "_id": "q1",
        "question": "Q1",
        "context": [["A", ["a1", "a2"]], ["B", ["b1"]]],
        "supporting_facts": [["A", 0], ["B", 0]],
    },
    {
        "_id": "q2",
        "question": "Q2",
        "context": [["A", ["a1", "a2"]], ["C", ["c1"]]],
        "supporting_facts": [["C", 0]],
    },
]


def test_load_split_deduplicates_paragraphs_across_records(tmp_path):
    data = tw.load_split(_write(tmp_path, RECORDS))
    # A appears in both records with the same text, so it is one document.
    assert len(data["corpus"]) == 3
    assert sorted(data["titles"].values()) == ["A", "B", "C"]


def test_load_split_assigns_gold_as_positional_indices(tmp_path):
    data = tw.load_split(_write(tmp_path, RECORDS))
    assert len(data["golds"]["q1"]) == 2
    assert len(data["golds"]["q2"]) == 1
    for query_id, gold in data["golds"].items():
        assert all(0 <= index < len(data["corpus"]) for index in gold), query_id


def test_load_split_counts_hops_from_the_gold_set(tmp_path):
    data = tw.load_split(_write(tmp_path, RECORDS))
    assert data["hops"]["q1"] == 2
    assert data["hops"]["q2"] == 1


def test_a_title_named_twice_in_the_supporting_facts_is_one_gold(tmp_path):
    records = [{
        "_id": "q",
        "question": "Q",
        "context": [["A", ["a1", "a2"]]],
        "supporting_facts": [["A", 0], ["A", 1]],
    }]
    data = tw.load_split(_write(tmp_path, records))
    assert len(data["golds"]["q"]) == 1


def test_a_supporting_title_absent_from_the_context_is_ignored(tmp_path):
    records = [{
        "_id": "q",
        "question": "Q",
        "context": [["A", ["a1"]]],
        "supporting_facts": [["Z", 0]],
    }]
    data = tw.load_split(_write(tmp_path, records))
    assert data["golds"]["q"] == []


def test_limit_queries_takes_the_first_records(tmp_path):
    data = tw.load_split(_write(tmp_path, RECORDS), limit=1)
    assert list(data["queries"]) == ["q1"]


def test_document_ids_are_slugged_and_group_by_article(tmp_path):
    data = tw.load_split(_write(tmp_path, RECORDS))
    assert all("#" in doc_id for doc_id in data["corpus"])
    assert {doc_id.split("#")[0] for doc_id in data["corpus"]} == {"a", "b", "c"}


def test_a_perfect_ranking_scores_one_under_the_shared_metrics(tmp_path):
    from musique import gold_matches
    from multihop_rerank_experiment import metrics_from_matches

    data = tw.load_split(_write(tmp_path, RECORDS))
    query_id = "q1"
    gold = set(data["golds"][query_id])
    ranked = [doc for doc, position in sorted(data["order"].items(), key=lambda kv: kv[1])
              if position in gold]
    ranked += [doc for doc, position in sorted(data["order"].items(), key=lambda kv: kv[1])
               if position not in gold]
    matches = gold_matches(ranked, data["golds"][query_id], data["order"])
    row = metrics_from_matches(list(range(len(ranked))), matches, len(gold), 10)
    assert row["Hits@10"] == 1.0
    assert row["Hits@4"] == 1.0
    # The benchmark's AP convention: two gold facts found at ranks 1 and 2 give
    # (1/1 + 1/2) / 2 = 0.75, not 1.0 — a perfect *set* at imperfect ranks.
    assert row["MAP@10"] == 0.75


def test_rrf_scores_fuse_the_channels_and_ignore_absent_candidates():
    rankings = [{"q": ["a", "b", "c"]}, {"q": ["b", "a"]}]
    scores = tw._rrf_scores(rankings, "q", ["a", "b", "c"], rrf_k=5)
    # a and b are 1/6 + 1/7 in opposite orders; c appears in one channel only.
    assert scores[0] == pytest.approx(scores[1])
    assert scores[0] > scores[2] > 0.0
    assert tw._rrf_scores(rankings, "absent", ["a"], rrf_k=5) == [0.0]


def test_the_coverage_arm_returns_a_permutation_of_the_pool():
    from multihop_coverage import atom_index, split_atoms
    from multihop_rerank_experiment import coverage_order

    texts = {
        "a#0": "Alpha fact one. Alpha fact two.",
        "b#0": "Beta fact only.",
        "c#0": "Gamma fact here.",
    }
    ids = list(texts)
    atom_ids, weights, _texts = atom_index([split_atoms(texts[doc_id]) for doc_id in ids])
    relevance = tw._rrf_scores([{"q": ["a#0", "b#0", "c#0"]}], "q", ids, rrf_k=5)
    order = coverage_order(relevance, atom_ids, weights, top_k=3, lam=0.7)
    assert sorted(order) == [0, 1, 2]
    assert len(order) == 3
