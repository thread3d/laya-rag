"""The second stage is optional: `SECOND_STAGE=0` returns the first stage's own order.

The mode exists because two things this repository wants to measure are impossible with a
322M-parameter decision on every candidate: MTEB's `SearchProtocol` (which is supposed to score the
retrieval *system*) and the engine-level arm of §P5a's convex-fusion question. What it must not do is
report a verdict nothing computed, so the tests pin the markers as well as the mechanics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from laya_rag.engine import RagEngine  # noqa: E402

from .support import StubModel, make_config, write_corpus  # noqa: E402


def _engine(tmp_path, **overrides):
    config = make_config(tmp_path, **overrides)
    model = StubModel()
    engine = RagEngine(config, model=model)
    engine.index_paths([str(write_corpus(tmp_path / "corpus"))], rebuild=True)
    return engine, model


def test_the_default_runs_the_second_stage(tmp_path):
    engine, model = _engine(tmp_path)
    result = engine.query("How long does a refund take?")
    assert model.predict_many_calls > 0
    assert result.answerability_measured is True
    assert result.pool.get("second_stage") is not False


def test_first_stage_only_never_calls_the_decision_model(tmp_path):
    """The whole point: no 322M forward per candidate."""
    engine, model = _engine(tmp_path, second_stage=False)
    result = engine.query("How long does a refund take?")
    assert model.predict_many_calls == 0
    assert result.passages
    assert result.rerank_seconds == 0.0
    assert result.pool["second_stage"] is False


def test_first_stage_only_does_not_claim_a_verdict(tmp_path):
    """A verdict nothing computed is worse than no verdict."""
    engine, _model = _engine(tmp_path, second_stage=False)
    result = engine.query("How long does a refund take?")
    assert result.answerable is True
    assert result.answerability is None
    assert result.answerability_measured is False
    # The key is *absent* from the payload rather than present-and-null: a caller cannot then read
    # a null as a low probability, and `answerability_measured=False` is the field that says why.
    assert "answerability" not in result.as_dict()


def test_first_stage_only_carries_the_first_stage_score_in_both_fields(tmp_path):
    """`relevance` is the field a caller orders by, so it must carry the ordering signal — and
    the pool marker is what says it is not a calibrated probability."""
    engine, _model = _engine(tmp_path, second_stage=False)
    result = engine.query("How long does a refund take?")
    for passage in result.passages:
        assert passage["relevance"] == passage["recall_score"]


def test_first_stage_only_respects_top_k_and_filters_nothing(tmp_path):
    engine, _model = _engine(tmp_path, second_stage=False, relevance_threshold=0.99)
    result = engine.query("How long does a refund take?", top_k=2)
    assert len(result.passages) <= 2
    assert result.dropped_below_threshold == 0
    assert result.kept == len(result.passages)


def test_first_stage_only_still_reports_the_pool(tmp_path):
    engine, _model = _engine(tmp_path, second_stage=False)
    pool = engine.query("How long does a refund take?").pool
    assert pool["candidates"] >= pool["lexical"] >= 0
    assert pool["requested"] > 0


def test_the_config_flag_reads_the_environment(monkeypatch):
    from laya_rag.config import EngineConfig

    monkeypatch.setenv("LAYA_RAG_SECOND_STAGE", "0")
    assert EngineConfig().second_stage is False
    monkeypatch.setenv("LAYA_RAG_SECOND_STAGE", "1")
    assert EngineConfig().second_stage is True
