"""Offline tests for the adapted cross-encoder pipeline (ROADMAP 2.2).

The two things that would silently corrupt the result: the miner must not invent negatives from a
record with no positive (an all-negative group teaches nothing and shifts the prior), and the step
checkpoint must be tied to the run that wrote it — a smoke run's checkpoint was resumed by a full
run before the fingerprint existed.
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


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ot = _load("laya_rag_order_train", "order_train.py")
oe = _load("laya_rag_order_eval", "order_eval.py")

RECORDS = {
    "q1": {
        "question": "Q1",
        "paragraphs": [
            {"paragraph_text": "pos", "is_supporting": True},
            {"paragraph_text": "neg1", "is_supporting": False},
            {"paragraph_text": "neg2", "is_supporting": False},
        ],
    },
    "q2": {  # no supporting paragraph: must be skipped, not mined as all-negative
        "question": "Q2",
        "paragraphs": [{"paragraph_text": "x", "is_supporting": False}],
    },
}


# ── mining ───────────────────────────────────────────────────────────────

def test_mine_pairs_takes_supporting_paragraphs_as_positives():
    pairs, used = ot.mine_pairs(RECORDS, negatives=0, seed=1)
    assert pairs == [("Q1", "pos", 1.0)]
    assert used == ["q1"]


def test_mine_pairs_samples_same_record_distractors_as_negatives():
    pairs, _ = ot.mine_pairs(RECORDS, negatives=2, seed=1)
    labels = sorted(label for _, _, label in pairs)
    assert labels == [0.0, 0.0, 1.0]
    assert {text for _, text, label in pairs if label == 0.0} == {"neg1", "neg2"}


def test_mine_pairs_caps_the_negative_count():
    pairs, _ = ot.mine_pairs(RECORDS, negatives=1, seed=1)
    assert sum(1 for _, _, label in pairs if label == 0.0) == 1


def test_mine_pairs_honours_max_queries():
    many = {f"q{i}": RECORDS["q1"] for i in range(10)}
    _, used = ot.mine_pairs(many, negatives=1, seed=1, max_queries=3)
    assert len(used) == 3


def test_split_pairs_respects_the_holdout_fraction():
    pairs = [(str(i), "p", 1.0) for i in range(100)]
    train, holdout = ot.split_pairs(pairs, 0.1, seed=1)
    assert len(train) == 90 and len(holdout) == 10
    assert len(set(train) | set(holdout)) == 100


# ── checkpoints ──────────────────────────────────────────────────────────

def test_latest_checkpoint_picks_the_highest_step(tmp_path):
    for step in (5, 200, 40):
        ot.checkpoint_path(tmp_path, step).write_bytes(b"x")
    (tmp_path / "not-a-checkpoint.pt").write_bytes(b"x")
    assert ot.latest_checkpoint(tmp_path).name == "step-000200.pt"


def test_latest_checkpoint_of_a_missing_directory_is_none(tmp_path):
    assert ot.latest_checkpoint(tmp_path / "absent") is None


# ── scoring ──────────────────────────────────────────────────────────────

def test_score_arms_scores_every_arm_over_the_same_pool():
    pool = ["d0", "d1", "d2"]
    order = {"d0": 0, "d1": 1, "d2": 2}
    arms = oe.score_arms(pool, [1], order, [0.1, 0.9, 0.5], [0.2, 0.1, 0.9], top_k=10)
    assert set(arms) == set(oe.ARMS)
    # fused keeps the pool order: the gold sits at rank 2
    assert arms["fused"]["MAP@10"] == pytest.approx(0.5)
    assert arms["fused"]["MRR@10"] == pytest.approx(0.5)
    # the cached CE ranks it first; the model puts it last
    assert arms["ce-cached"]["MAP@10"] == pytest.approx(1.0)
    assert arms["ce-model"]["MAP@10"] == pytest.approx(1 / 3)


def test_summarize_and_compare_average_over_queries():
    per_query = {
        "0": {arm: {m: 0.0 for m in oe.METRICS} for arm in oe.ARMS},
        "1": {arm: {m: 1.0 for m in oe.METRICS} for arm in oe.ARMS},
    }
    summary = oe.summarize(per_query)
    assert all(summary[arm]["MAP@10"] == pytest.approx(0.5) for arm in oe.ARMS)
    block = oe.compare(per_query, "fused", "ce-model", samples=200)
    assert set(block) == set(oe.METRICS)


def test_rank_positions_is_deterministic_on_ties():
    assert oe.rank_positions([0.5, 0.5, 0.9]) == [2, 0, 1]


def test_eval_requires_a_cross_encoder_unless_comparing(monkeypatch):
    # The tool takes the heavy-job lock in `main`; this test calls it in-process.
    monkeypatch.setenv("LAYA_RAG_ALLOW_CONCURRENT", "1")
    with pytest.raises(SystemExit, match="--cross-encoder is required"):
        oe.main([])
