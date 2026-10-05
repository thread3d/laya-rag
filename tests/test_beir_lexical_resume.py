"""Offline tests for resumable lexical ranking in the BEIR harness.

The corpus build inside `bm25_rankings` is the cheap part — about 4 s on 25,657 documents
against ~85% of the time in per-query scoring — so the pass is made resumable by
checkpointing each query's ranking and repeating the build. That is only safe if a stale or
foreign checkpoint cannot contaminate a report, which is what the settings fingerprint is
for, and what these tests pin.
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
sys.path.insert(0, str(ROOT / "src"))


def _load():
    name = "laya_rag_beir"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "beir.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


beir = _load()

CORPUS = {
    "d1": "refunds are processed within five business days",
    "d2": "shipping is free on orders over fifty dollars",
    "d3": "the peru office handles bulk shipping questions",
}
QUERIES = {"q1": "refunds processed", "q2": "free shipping", "q3": "peru office bulk"}


def test_no_checkpoint_is_the_plain_path():
    plain = beir.bm25_rankings(CORPUS, QUERIES, 2)
    assert set(plain) == set(QUERIES)
    assert all(len(ids) <= 2 for ids in plain.values())


def test_a_checkpointed_pass_matches_the_plain_one(tmp_path):
    plain = beir.bm25_rankings(CORPUS, QUERIES, 2)
    with_checkpoint = beir.bm25_rankings(CORPUS, QUERIES, 2, checkpoint=tmp_path / "r.jsonl")
    assert with_checkpoint == plain


def test_a_resumed_pass_reuses_the_queries_already_ranked(tmp_path):
    checkpoint = tmp_path / "r.jsonl"
    first = beir.bm25_rankings(CORPUS, {"q1": QUERIES["q1"]}, 2, checkpoint=checkpoint)
    # Now ask for all three: q1 is already checkpointed, the other two are computed.
    resumed = beir.bm25_rankings(CORPUS, QUERIES, 2, checkpoint=checkpoint)
    assert resumed["q1"] == first["q1"]
    assert set(resumed) == set(QUERIES)
    lines = [json.loads(l) for l in checkpoint.read_text().splitlines() if l.strip()]
    assert sum(1 for row in lines if "q" in row) == 3, "each query is written exactly once"


def test_only_the_missing_queries_are_ranked_on_resume(tmp_path):
    """A resumed run must not recompute work; one query at a time proves it."""
    checkpoint = tmp_path / "r.jsonl"
    beir.bm25_rankings(CORPUS, {"q1": QUERIES["q1"]}, 2, checkpoint=checkpoint)
    before = checkpoint.read_text()
    beir.bm25_rankings(CORPUS, {"q1": QUERIES["q1"], "q2": QUERIES["q2"]}, 2, checkpoint=checkpoint)
    lines = [json.loads(l) for l in checkpoint.read_text().splitlines() if l.strip()]
    assert len(lines) == 3, "one fingerprint line plus two queries"
    assert before.splitlines()[0] == checkpoint.read_text().splitlines()[0]


def test_a_checkpoint_from_different_settings_is_ignored(tmp_path):
    """top_k changes the rankings, so a stale file must not be mixed into a report."""
    checkpoint = tmp_path / "r.jsonl"
    beir.bm25_rankings(CORPUS, QUERIES, 2, checkpoint=checkpoint)
    beir.bm25_rankings(CORPUS, QUERIES, 3, checkpoint=checkpoint)
    lines = [json.loads(l) for l in checkpoint.read_text().splitlines() if l.strip()]
    assert lines[0]["fingerprint"]["top_k"] == 3, "the file must be re-anchored at the new top_k"
    assert len(lines) == 4, "one fingerprint line plus three queries, re-ranked at top_k=3"


def test_a_checkpoint_from_a_different_corpus_size_is_ignored(tmp_path):
    checkpoint = tmp_path / "r.jsonl"
    beir.bm25_rankings(CORPUS, QUERIES, 2, checkpoint=checkpoint)
    smaller = {k: v for k, v in list(CORPUS.items())[:2]}
    result = beir.bm25_rankings(smaller, QUERIES, 2, checkpoint=checkpoint)
    assert set(result) == set(QUERIES)


def test_queries_not_in_the_checkpoint_are_filled_in(tmp_path):
    checkpoint = tmp_path / "r.jsonl"
    beir.bm25_rankings(CORPUS, {"q1": QUERIES["q1"]}, 2, checkpoint=checkpoint)
    result = beir.bm25_rankings(CORPUS, {"q1": QUERIES["q1"], "q9": "refunds"}, 2,
                                checkpoint=checkpoint)
    assert "q9" in result


def test_the_control_row_has_its_own_checkpoint_path():
    """The published-settings control must not share the engine row's file: different settings,
    different fingerprint, and on NQ it is the slower of the two by far, so it needs its own."""
    import re
    from pathlib import Path as _Path

    source = (_Path(__file__).resolve().parents[1] / "tools" / "beir.py").read_text()
    assert ".lexical-control.jsonl" in source
    control = re.search(r'rankings\["bm25"\] = bm25_rankings\((.*?)\)', source, re.S).group(1)
    assert "checkpoint=" in control, "the control row must be resumable like the engine row"


def test_a_query_subset_is_seeded_and_stable():
    """A subset must be reproducible, and must not be the head of the file."""
    queries = {f"q{i}": f"text {i}" for i in range(100)}
    first = beir.sample_queries(queries, 20, 17)
    second = beir.sample_queries(queries, 20, 17)
    assert first == second, "the same seed must score the same queries"
    assert len(first) == 20
    assert set(first) <= set(queries)
    assert list(first) != list(queries)[:20], "the head is not a random sample"


def test_a_different_seed_draws_a_different_subset():
    queries = {f"q{i}": f"text {i}" for i in range(100)}
    assert beir.sample_queries(queries, 20, 1) != beir.sample_queries(queries, 20, 2)


def test_a_limit_at_or_above_the_query_count_keeps_everything():
    queries = {f"q{i}": f"text {i}" for i in range(5)}
    assert beir.sample_queries(queries, 5, 17) == queries
    assert beir.sample_queries(queries, None, 17) == queries
