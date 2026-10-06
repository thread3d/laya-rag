"""Tests for second-stage blending and the optional answerability gate plumbing.

A pointwise cross-encoder trusted fully lost nDCG@10 on all five BEIR corpora, so the
engine and the harness can now blend its order with the first stage instead of replacing
it. These tests pin the blend semantics (``1.0`` = historical behaviour, ``0.0`` = the
first-stage order) and the answerability-measurement flag that keeps a structural zero
out of calibration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from laya_rag.answerability import FEATURE_NAMES, LogisticGate
from laya_rag.chunking import Chunk
from laya_rag.config import EngineConfig
from laya_rag.engine import RagEngine
from laya_rag.index_store import Hit
from laya_rag.rerank import LayaReranker, blend_scores

from .support import StubModel, make_config, write_corpus

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import beir  # noqa: E402


def _hit(chunk_id: str, text: str, score: float, path: str = "doc.md") -> Hit:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        path=path,
        text=text,
        index=0,
        start_char=0,
        end_char=len(text),
    )
    return Hit(chunk=chunk, score=score, channels=("dense",))


class _FakeGate:
    """A stand-in for :class:`laya_rag.answerability.LogisticGate`."""

    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.seen: list[np.ndarray] = []

    def predict_proba(self, X):  # noqa: N803 - mirrors the model API
        self.seen.append(np.asarray(X))
        return np.full(len(X), self.probability, dtype=np.float64)


class _FakeCrossEncoder:
    """Scores each pair from a lookup keyed by passage text."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.name = "fake-ce"

    def score(self, pairs):
        return [self.scores[passage] for _query, passage in pairs]


# ── blend_scores ─────────────────────────────────────────────────────────

def test_blend_of_one_is_the_decision_score_unchanged():
    assert blend_scores([3.0, 2.0, 1.0], [0.6, 0.7, 0.9], 1.0) == [0.6, 0.7, 0.9]


def test_blend_of_zero_is_the_normalised_recall_order():
    assert blend_scores([3.0, 2.0, 1.0], [0.6, 0.7, 0.9], 0.0) == [1.0, 0.5, 0.0]


def test_blend_normalises_flat_recall_to_zero():
    assert blend_scores([2.0, 2.0], [0.4, 0.8], 0.5) == [0.2, 0.4]


def test_blend_rejects_out_of_range():
    with pytest.raises(ValueError):
        blend_scores([1.0], [1.0], 1.5)
    with pytest.raises(ValueError):
        blend_scores([1.0], [1.0], -0.1)


# ── rerank ordering ──────────────────────────────────────────────────────

def _three_hits() -> list[Hit]:
    # Recall prefers A > B > C; the decision prefers C > B > A.
    return [
        _hit("a", "alpha text", 3.0),
        _hit("b", "beta text", 2.0),
        _hit("c", "gamma text", 1.0),
    ]


def test_blend_one_orders_by_the_supplied_decision():
    reranker = LayaReranker(StubModel())
    result = reranker.rerank(
        "q", _three_hits(), top_k=3, relevance_threshold=0.5, scores=[0.6, 0.7, 0.9], blend=1.0
    )
    assert [p.chunk_id for p in result.passages] == ["c", "b", "a"]


def test_blend_zero_keeps_the_first_stage_order():
    reranker = LayaReranker(StubModel())
    result = reranker.rerank(
        "q", _three_hits(), top_k=3, relevance_threshold=0.5, scores=[0.6, 0.7, 0.9], blend=0.0
    )
    assert [p.chunk_id for p in result.passages] == ["a", "b", "c"]


def test_blend_still_filters_below_the_relevance_threshold():
    reranker = LayaReranker(StubModel())
    result = reranker.rerank(
        "q", _three_hits(), top_k=3, relevance_threshold=0.65, scores=[0.6, 0.7, 0.9], blend=0.5
    )
    assert "a" not in [p.chunk_id for p in result.passages]
    assert result.dropped_below_threshold == 1


# ── gate + measurement flag ──────────────────────────────────────────────

def test_a_supplied_gate_replaces_layas_scalar_decision():
    gate = _FakeGate(0.91)
    reranker = LayaReranker(StubModel())
    result = reranker.rerank(
        "q", _three_hits(), top_k=3, relevance_threshold=0.5,
        scores=[0.6, 0.7, 0.9], gate=gate,
    )
    assert result.answerability == pytest.approx(0.91)
    assert result.answerable is True
    assert result.answerability_measured is True
    # The gate saw one row per kept passage, in feature order.
    assert gate.seen and gate.seen[0].shape[0] == 1


def test_an_empty_relevance_set_is_not_a_measured_answerability_decision():
    reranker = LayaReranker(StubModel())
    result = reranker.rerank(
        "q", _three_hits(), top_k=3, relevance_threshold=0.99, scores=[0.1, 0.2, 0.3]
    )
    assert result.answerability == 0.0
    assert result.answerability_measured is False
    assert result.answerable is False


def test_a_normal_decision_is_marked_measured():
    reranker = LayaReranker(StubModel())
    result = reranker.rerank("refund", [_hit("a", "refund policy text", 1.0)], top_k=1)
    assert result.answerability_measured is True


# ── config ───────────────────────────────────────────────────────────────

def test_rerank_blend_defaults_to_the_historical_full_trust():
    assert EngineConfig().rerank_blend == 1.0


def test_answerability_gate_path_defaults_to_unset():
    assert EngineConfig().answerability_gate_path == ""


# ── engine wiring of the gate ────────────────────────────────────────────

def _saved_gate(path: Path) -> Path:
    width = len(FEATURE_NAMES)
    gate = LogisticGate(
        mean=[0.0] * width,
        scale=[1.0] * width,
        coef=[10.0] + [0.0] * (width - 1),
        intercept=0.0,
    )
    gate.save(path)
    return path


def test_engine_uses_a_configured_answerability_gate(tmp_path):
    gate_path = _saved_gate(tmp_path / "gate.json")
    config = make_config(tmp_path, answerability_gate_path=str(gate_path), dense_weight=1.0)
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    result = engine.query("How long does a refund take?")
    assert engine.answerability_gate is not None
    assert result.answerability_measured is True
    assert result.answerability is not None and 0.0 <= result.answerability <= 1.0
    assert result.as_dict()["answerability_measured"] is True


def test_a_missing_gate_falls_back_to_layas_own_decision(tmp_path):
    config = make_config(tmp_path, answerability_gate_path=str(tmp_path / "absent.json"))
    engine = RagEngine(config, model=StubModel())
    assert engine.answerability_gate is None
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    result = engine.query("How long does a refund take?")
    assert result.answerability_measured is True


def test_stats_report_whether_a_gate_is_configured_and_loaded(tmp_path):
    gate_path = _saved_gate(tmp_path / "gate.json")
    engine = RagEngine(
        make_config(tmp_path, answerability_gate_path=str(gate_path)), model=StubModel()
    )
    assert engine.stats()["answerability_gate"] == {"configured": True, "loaded": True}
    unset = RagEngine(make_config(tmp_path), model=StubModel())
    assert unset.stats()["answerability_gate"] == {"configured": False, "loaded": False}


# ── harness blending ─────────────────────────────────────────────────────

def test_crossencoder_blend_one_is_the_pure_ce_order():
    corpus = {"a": "alpha", "b": "beta", "c": "gamma"}
    queries = {"q": "query"}
    base = {"q": ["a", "b", "c"]}
    ce = _FakeCrossEncoder({"alpha": 0.6, "beta": 0.7, "gamma": 0.9})
    reranked, _ = beir.crossencoder_rankings(corpus, queries, base, ce, 3, 3, blend=1.0)
    assert reranked["q"] == ["c", "b", "a"]


def test_crossencoder_blend_zero_keeps_the_first_stage():
    corpus = {"a": "alpha", "b": "beta", "c": "gamma"}
    queries = {"q": "query"}
    base = {"q": ["a", "b", "c"]}
    ce = _FakeCrossEncoder({"alpha": 0.6, "beta": 0.7, "gamma": 0.9})
    reranked, _ = beir.crossencoder_rankings(corpus, queries, base, ce, 3, 3, blend=0.0)
    assert reranked["q"] == ["a", "b", "c"]


def test_crossencoder_blend_rejects_out_of_range():
    corpus = {"a": "alpha"}
    queries = {"q": "query"}
    base = {"q": ["a"]}
    ce = _FakeCrossEncoder({"alpha": 1.0})
    with pytest.raises(ValueError):
        beir.crossencoder_rankings(corpus, queries, base, ce, 1, 1, blend=2.0)


# ── per-dataset harness options ──────────────────────────────────────────

def test_parse_pairs_parses_a_dataset_model_map():
    assert beir._parse_pairs("scidocs=allenai/specter2_base, arguana=BAAI/bge-small-en-v1.5") == {
        "scidocs": "allenai/specter2_base",
        "arguana": "BAAI/bge-small-en-v1.5",
    }


def test_parse_pairs_treats_blank_as_empty():
    assert beir._parse_pairs("") == {}
    assert beir._parse_pairs("  ") == {}


def test_parse_pairs_rejects_a_missing_equals():
    with pytest.raises(ValueError):
        beir._parse_pairs("scidocs")


def test_parse_args_exposes_a_biencoder_pooling_override():
    assert beir.parse_args([]).biencoder_pooling == "auto"
    assert beir.parse_args(["--biencoder-pooling", "cls"]).biencoder_pooling == "cls"


# -- document quota ----------------------------------------------------------

def _four_hits():
    return [
        _hit("a", "alpha", 0.9, "/d1.md"),
        _hit("b", "beta", 0.8, "/d1.md"),   # same document as "a"
        _hit("c", "gamma", 0.7, "/d2.md"),
        _hit("d", "delta", 0.6, "/d3.md"),
    ]


def _keep(hits, min_docs):
    return LayaReranker(StubModel()).rerank(
        "query", hits, top_k=4, relevance_threshold=0.0,
        scores=[0.9, 0.8, 0.7, 0.6],
        answerability_fn=lambda query, passages: 1.0,
        min_docs=min_docs,
    )


def test_min_docs_promotes_distinct_documents_before_the_cut():
    kept = [passage.chunk_id for passage in _keep(_four_hits(), min_docs=2).passages]
    # The best passage of d1, then the best of d2, then the remaining passages in order.
    assert kept == ["a", "c", "b", "d"]


def test_min_docs_zero_keeps_the_pure_passage_order():
    kept = [passage.chunk_id for passage in _keep(_four_hits(), min_docs=0).passages]
    assert kept == ["a", "b", "c", "d"]


def test_min_docs_never_drops_a_passage():
    kept = _keep(_four_hits(), min_docs=4).passages
    assert sorted(passage.chunk_id for passage in kept) == ["a", "b", "c", "d"]


def test_bridge_feature_feeds_the_conditioned_score_into_the_gate(monkeypatch):
    """With GATE_BRIDGE_FEATURE on, the gate's vector carries `s(q, b, c)`'s best value."""
    from laya_rag import answerability

    seen: dict = {}
    real = answerability.extract_features

    def spy(query, passages, *, bridge_conditioned=None):
        seen["bridge"] = bridge_conditioned
        return real(query, passages, bridge_conditioned=bridge_conditioned)

    monkeypatch.setattr(answerability, "extract_features", spy)
    hits = [_hit("c0", "the bridge passage", 0.9), _hit("c1", "another passage", 0.5)]
    gate = _FakeGate(0.7)
    LayaReranker(StubModel()).rerank(
        "which company", hits, top_k=2, relevance_threshold=0.0,
        answerability_threshold=0.0, gate=gate, scores=[0.9, 0.5], bridge_feature=True,
    )
    assert seen.get("bridge") is not None
    assert gate.seen and gate.seen[0].shape[1] == len(FEATURE_NAMES)


def test_bridge_feature_off_leaves_the_gate_vector_unchanged(monkeypatch):
    from laya_rag import answerability

    seen: dict = {}
    real = answerability.extract_features

    def spy(query, passages, *, bridge_conditioned=None):
        seen["bridge"] = bridge_conditioned
        return real(query, passages, bridge_conditioned=bridge_conditioned)

    monkeypatch.setattr(answerability, "extract_features", spy)
    hits = [_hit("c0", "the bridge passage", 0.9), _hit("c1", "another passage", 0.5)]
    LayaReranker(StubModel()).rerank(
        "which company", hits, top_k=2, relevance_threshold=0.0,
        answerability_threshold=0.0, gate=_FakeGate(0.7), scores=[0.9, 0.5],
    )
    assert seen.get("bridge") is None
