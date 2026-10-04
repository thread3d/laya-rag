"""Offline tests for the late-interaction ordering head.

The pure half — MaxSim with masking, score-to-rank, padding, and the arm comparison — is what
decides whether the measured number means anything, so it is pinned here. The model itself needs
a downloaded checkpoint, so the sanity check against the model card's own example carries the
`real_weights` marker and is excluded from `make test`.
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
    name = "laya_rag_multihop_ordering"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "multihop_ordering.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mo = _load()


# ── MaxSim ───────────────────────────────────────────────────────────────

def test_maxsim_sums_the_best_match_per_query_token():
    query = [[1.0, 0.0], [0.0, 1.0]]
    doc = [[1.0, 0.0], [0.0, 2.0]]
    # token 0's best is 1.0, token 1's is 2.0
    assert mo.maxsim_scores(query, [doc]) == [pytest.approx(3.0)]


def test_maxsim_mask_excludes_padding_and_special_tokens():
    query = [[1.0, 0.0]]
    doc = [[100.0, 100.0], [1.0, 0.0]]
    assert mo.maxsim_scores(query, [doc]) == [pytest.approx(100.0)]
    assert mo.maxsim_scores(query, [doc], [[False, True]]) == [pytest.approx(1.0)]


def test_maxsim_of_a_fully_masked_document_is_negative_infinity():
    assert mo.maxsim_scores([[1.0, 0.0]], [[[1.0, 0.0]]], [[False]]) == [float("-inf")]


def test_maxsim_scores_every_document_in_order():
    query = [[1.0, 0.0]]
    docs = [[[0.5, 0.0]], [[0.9, 0.0]]]
    assert mo.maxsim_scores(query, docs) == [pytest.approx(0.5), pytest.approx(0.9)]


# ── ordering and alignment ───────────────────────────────────────────────

def test_rank_positions_descends_and_breaks_ties_by_position():
    assert mo.rank_positions([0.1, 0.9, 0.5]) == [1, 2, 0]
    assert mo.rank_positions([0.5, 0.5, 0.5]) == [0, 1, 2]


def test_pad_scores_aligns_a_candidate_prefix_to_the_full_pool():
    assert mo.pad_scores([0.3, 0.4], 4)[:2] == [0.3, 0.4]
    assert mo.pad_scores([0.3, 0.4], 4)[2:] == [float("-inf"), float("-inf")]
    assert mo.pad_scores([0.3, 0.4, 0.5], 2) == [0.3, 0.4]


def test_build_corpus_uses_the_benchmark_chunk_ids():
    raw = [{"url": "http://x/a", "title": "T", "body": " ".join(str(i) for i in range(600))}]
    corpus = mo.build_corpus(raw, overlap=200)
    assert all(doc_id.startswith("http://x/a#") for doc_id in corpus)
    assert len(corpus) > 1


# ── the arms ─────────────────────────────────────────────────────────────

def _toy():
    pool = ["a", "b", "c"]
    ce = [0.1, 0.9, 0.5]
    atoms = [[0], [1], [2]]
    atom_weight = [1.0, 1.0, 1.0]
    matches = [{0}, {1}, set()]
    colbert = [0.2, 0.1, 0.9]
    return pool, ce, atoms, atom_weight, matches, colbert


def test_covquota_groups_chunks_by_article():
    """The quota must count articles, not chunks: three chunks of one article are one document.

    This is the bug the first full run had — passing chunk ids made every chunk its own
    document, so nothing forced a second article into the top slots.
    """
    pool = ["a#0", "a#1", "a#2", "b#0"]
    ce = [0.9, 0.8, 0.7, 0.1]
    order = mo.covquota_order(pool, ce, [[0], [1], [2], [3]], [1.0] * 4, 2, 0.7, 2)
    assert 3 in order[:2], f"the second article was not forced in: {order}"


def test_arm_metrics_scores_every_arm_from_the_same_pool():
    pool, ce, atoms, weight, matches, colbert = _toy()
    arms = mo.arm_metrics(pool, ce, atoms, weight, matches, 2, colbert,
                          top_k=3, lam=0.7, min_docs=1)
    assert set(arms) == set(mo.ARMS)
    # fused keeps pool order: fact 0 at rank 1, fact 1 at rank 2 -> (1 + 1/2) / 2
    assert arms["fused"]["MAP@10"] == pytest.approx(0.75)
    # the CE order puts fact 1 first and fact 0 third -> (1 + 1/3) / 2
    assert arms["ce"]["MAP@10"] == pytest.approx(2 / 3)
    # ColBERT prefers the irrelevant chunk, so it pays for it
    assert arms["colbert"]["MAP@10"] == pytest.approx((0.5 + 1 / 3) / 2)
    assert arms["fused"]["Hits@10"] == 1.0


def test_summarize_averages_each_arm_over_queries():
    per_query = {
        "0": {arm: {m: 1.0 for m in mo.METRICS} for arm in mo.ARMS},
        "1": {arm: {m: 0.0 for m in mo.METRICS} for arm in mo.ARMS},
    }
    summary = mo.summarize(per_query)
    assert all(summary[arm]["MAP@10"] == pytest.approx(0.5) for arm in mo.ARMS)


def test_compare_to_pairs_the_two_arms_over_every_metric():
    per_query = {
        "0": {"fused": {m: 0.0 for m in mo.METRICS}, "colbert": {m: 1.0 for m in mo.METRICS}},
        "1": {"fused": {m: 0.0 for m in mo.METRICS}, "colbert": {m: 1.0 for m in mo.METRICS}},
    }
    block = mo.compare_to(per_query, "fused", "colbert", samples=200)
    assert set(block) == set(mo.METRICS)
    assert block["MAP@10"]["delta"] == pytest.approx(1.0)


def test_args_default_to_the_whole_cached_pool_and_the_official_top_k():
    args = mo.parse_args([])
    assert args.top_k == 10 and args.candidates == 100
    assert args.shard == "0/1" and args.merge == 0


@pytest.mark.real_weights
def test_colbert_reproduces_the_model_cards_ordering():
    """The card's absolute scale differs (its `similarity` normalises differently); the
    ordering is what a reranker uses, and it must match exactly."""
    docs = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    ]
    scores = mo.ColBERT(device="cpu").score("Which planet is known as the Red Planet?", docs)
    assert mo.rank_positions(scores) == [1, 3, 2, 0]  # card: 31.49, 31.31, 31.30, 30.57
