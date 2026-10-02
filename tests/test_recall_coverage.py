"""Tests for the engine's evidence-set coverage selection (``RECALL_COVERAGE_LAMBDA``).

The benchmark measured this objective as the best MultiHop-RAG coverage pass (MAP@10
0.2546 -> 0.2835, held-out +0.0288). These tests pin the pure selection arithmetic and the
engine wiring, so the shipped path uses the same greedy rule the measurement did.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from laya_rag.config import EngineConfig
from laya_rag.engine import RagEngine, _chunk_atoms, _coverage_select

from .support import StubModel, make_config, write_corpus


class _FakeIndex:
    """The two methods ``_coverage_select`` needs: text lookup and stored vectors."""

    def __init__(self, texts: dict[str, str], vectors: dict[str, np.ndarray] | None = None):
        self.texts = texts
        self.vectors = vectors or {}

    def get(self, chunk_id: str):
        return SimpleNamespace(text=self.texts[chunk_id]) if chunk_id in self.texts else None

    def vector(self, chunk_id: str):
        return self.vectors.get(chunk_id)


# ── atomisation ──────────────────────────────────────────────────────────

def test_chunk_atoms_splits_normalises_and_dedupes():
    text = "Refunds take five business days to process. Refunds take five business days to process. Short. Shipping is free over fifty dollars."
    atoms = _chunk_atoms(text)
    assert len(atoms) == 2
    assert atoms[0].startswith("refunds take five")
    assert "shipping is free" in atoms[1]


# ── coverage selection ───────────────────────────────────────────────────

def test_coverage_prefers_a_new_fact_over_a_near_duplicate():
    dup = "Refunds are processed within five business days to the original payment method."
    other = "Shipping is free on orders over fifty dollars in the continental United States."
    index = _FakeIndex({"a": dup, "b": dup, "c": other})
    fused = [("a", 3.0), ("b", 2.0), ("c", 1.0)]

    selected = _coverage_select(index, fused, limit=2, lam=0.5)

    ids = [chunk_id for chunk_id, _ in selected]
    assert ids[0] == "a"  # highest fused relevance, plus a new atom
    assert ids[1] == "c"  # b only repeats a's atom; c adds a new one


def test_coverage_respects_relevance_when_asked():
    dup = "Refunds are processed within five business days to the original payment method."
    other = "Shipping is free on orders over fifty dollars in the continental United States."
    index = _FakeIndex({"a": dup, "b": dup, "c": other})
    fused = [("a", 3.0), ("b", 2.0), ("c", 1.0)]

    # With relevance weighted fully, the fused order is preserved.
    selected = _coverage_select(index, fused, limit=2, lam=1.0)
    assert [chunk_id for chunk_id, _ in selected] == ["a", "b"]


def test_coverage_keeps_candidates_without_vectors_or_text():
    index = _FakeIndex({"a": "A sufficiently long sentence to be an atom here."})
    fused = [("a", 1.0), ("missing", 0.5)]
    selected = _coverage_select(index, fused, limit=2, lam=0.5)
    assert {chunk_id for chunk_id, _ in selected} == {"a", "missing"}


def test_coverage_returns_the_fused_prefix_without_a_pool():
    index = _FakeIndex({})
    assert _coverage_select(index, [], limit=5, lam=0.5) == []


# ── config + engine wiring ───────────────────────────────────────────────

def test_recall_coverage_is_off_by_default():
    assert EngineConfig().recall_coverage_lambda == 0.0


# ── coverage over a cross-encoder-scored pool ────────────────────────────

def test_coverage_select_from_uses_the_supplied_relevance():
    from laya_rag.engine import _coverage_select_from

    dup = "Refunds are processed within five business days to the original payment method."
    other = "Shipping is free on orders over fifty dollars in the continental United States."
    index = _FakeIndex({"a": dup, "b": dup, "c": other})
    # Cross-encoder ranks b (a duplicate fact) highest and c (a new fact) lowest.
    fused = [("a", 1.0), ("b", 2.0), ("c", 3.0)]
    relevance = [0.1, 0.9, 0.2]

    selected = _coverage_select_from(index, fused, relevance, limit=2, lam=0.7,
                                     use_redundancy=False)

    ids = [chunk_id for chunk_id, _ in selected]
    assert ids[0] == "b"  # highest CE relevance
    assert ids[1] == "c"  # a new atom beats a's higher-than-c relevance


def test_rerank_coverage_auto_resolves_and_is_serialised():
    config = EngineConfig()
    # Raw default is the auto sentinel; without a cross-encoder it resolves to off.
    assert config.rerank_coverage_lambda == -1.0
    assert config.resolved_rerank_coverage_lambda == 0.0
    assert config.rerank_coverage_min_docs == 0
    assert "rerank_coverage_lambda" in config.as_dict()
    assert config.as_dict()["rerank_coverage_lambda_resolved"] == 0.0
    assert "rerank_coverage_min_docs" in config.as_dict()


def test_rerank_coverage_auto_turns_on_the_measured_setting_with_a_cross_encoder():
    config = EngineConfig(rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert config.resolved_rerank_coverage_lambda == 0.7
    assert config.as_dict()["rerank_coverage_lambda_resolved"] == 0.7
    # An explicit value always wins, including an explicit 0.0 that disables it.
    assert EngineConfig(rerank_model="x", rerank_coverage_lambda=0.0).resolved_rerank_coverage_lambda == 0.0
    assert EngineConfig(rerank_model="x", rerank_coverage_lambda=0.4).resolved_rerank_coverage_lambda == 0.4


def test_coverage_redundancy_defaults_off_and_is_serialised():
    config = EngineConfig()
    assert config.recall_coverage_redundancy is False
    assert "recall_coverage_redundancy" in config.as_dict()


def test_coverage_select_redundancy_penalty_is_optional():
    dup_atom = "Refunds are processed within five business days to the original method."
    other = "Shipping is free on orders over fifty dollars in the continental states."
    index = _FakeIndex(
        {"a": dup_atom, "b": other, "c": other},
        vectors={
            "a": np.array([1.0, 0.0]),
            "b": np.array([1.0, 0.0]),  # distinct atom, nearly parallel to a
            "c": np.array([0.0, 1.0]),
        },
    )
    fused = [("a", 3.0), ("b", 2.0), ("c", 1.0)]
    with_penalty = _coverage_select(index, fused, limit=2, lam=0.5)
    without = _coverage_select(index, fused, limit=2, lam=0.5, use_redundancy=False)
    assert [chunk_id for chunk_id, _ in with_penalty] == ["a", "c"]
    assert [chunk_id for chunk_id, _ in without] == ["a", "b"]


def test_coverage_select_from_seeds_distinct_documents():
    from laya_rag.engine import _coverage_select_from

    index = _FakeIndex({
        "a": "Refunds are processed within five business days to the original method.",
        "b": "Refunds are processed within five business days to the original method.",
        "c": "Shipping is free on orders over fifty dollars in the continental states.",
        "d": "Contact billing support if the refund has not arrived after ten days.",
    })
    fused = [("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 1.0)]
    relevance = [0.9, 0.8, 0.7, 0.6]
    docs = ["A", "A", "B", "C"]
    selected = _coverage_select_from(
        index, fused, relevance, limit=3, lam=0.7, use_redundancy=False,
        docs=docs, min_docs=3,
    )
    chosen_docs = [docs[[c for c, _ in fused].index(chunk_id)] for chunk_id, _ in selected]
    assert chosen_docs == ["A", "B", "C"]


def test_coverage_select_from_without_docs_is_unchanged():
    from laya_rag.engine import _coverage_select_from

    index = _FakeIndex({"a": "A sufficiently long sentence to be an atom here."})
    fused = [("a", 1.0)]
    assert _coverage_select_from(index, fused, [0.5], limit=1, lam=0.7) == [("a", 1.0)]


class _StubCrossEncoder:
    """Returns a fixed score for every pair; enough to exercise the wiring."""

    def __init__(self, score: float = 0.9) -> None:
        self.score = score
        self.calls = 0

    def score_pairs(self, pairs):
        self.calls += 1
        return [self.score for _ in pairs]


def test_engine_query_runs_with_rerank_coverage_enabled(tmp_path):
    config = make_config(
        tmp_path, dense_weight=1.0, rerank_coverage_lambda=0.7, rerank_k=3
    )
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    engine.cross_encoder = _StubCrossEncoder()
    result = engine.query("How long does a refund take?")
    assert engine.cross_encoder.calls == 1
    assert result.considered > 0
    assert result.passages


def test_engine_auto_applies_coverage_when_a_cross_encoder_is_configured(tmp_path):
    """With a CE configured and no explicit lambda, the measured 0.7 is used."""
    config = make_config(
        tmp_path, dense_weight=1.0, rerank_k=3,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    assert config.resolved_rerank_coverage_lambda == 0.7
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    engine.cross_encoder = _StubCrossEncoder()  # replaces the configured model
    result = engine.query("How long does a refund take?")
    assert engine.cross_encoder.calls == 1
    assert result.passages


def test_engine_disables_auto_coverage_when_lambda_is_explicitly_zero(tmp_path):
    config = make_config(
        tmp_path, dense_weight=1.0, rerank_k=3,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_coverage_lambda=0.0,
    )
    assert config.resolved_rerank_coverage_lambda == 0.0
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    engine.cross_encoder = _StubCrossEncoder()
    result = engine.query("How long does a refund take?")
    assert result.passages


def test_rerank_coverage_requires_a_cross_encoder(tmp_path):
    """Without a second stage there is no CE score to select on, so it is a no-op."""
    config = make_config(tmp_path, dense_weight=1.0, rerank_coverage_lambda=0.7, rerank_k=3)
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    assert engine.cross_encoder is None
    result = engine.query("How long does a refund take?")
    assert result.passages


def test_engine_query_runs_with_coverage_enabled(tmp_path):
    config = make_config(tmp_path, dense_weight=1.0, recall_coverage_lambda=0.5, rerank_k=3)
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    result = engine.query("How long does a refund take?")
    assert result.considered > 0
    assert result.passages


def _spy_on_rerank(engine):
    captured: dict = {}
    original = engine.reranker.rerank

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    engine.reranker.rerank = spy
    return captured


def test_coverage_ordering_forces_a_pure_replacement_not_a_blend(tmp_path):
    """`blend < 1` reorders by a first-stage/CE mix, which would discard the coverage
    order entirely, so the engine forces a pure replacement when coverage is applied."""
    config = make_config(
        tmp_path, dense_weight=1.0, rerank_k=3,
        rerank_coverage_lambda=0.7, rerank_blend=0.5,
    )
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    engine.cross_encoder = _StubCrossEncoder()
    captured = _spy_on_rerank(engine)
    result = engine.query("How long does a refund take?")
    assert result.passages
    assert captured["order"] == "recall"
    assert captured["blend"] == 1.0


def test_blend_passes_through_when_coverage_is_disabled(tmp_path):
    config = make_config(
        tmp_path, dense_weight=1.0, rerank_k=3,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_coverage_lambda=0.0, rerank_blend=0.5,
    )
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    engine.cross_encoder = _StubCrossEncoder()
    captured = _spy_on_rerank(engine)
    result = engine.query("How long does a refund take?")
    assert result.passages
    assert captured["order"] == "relevance"
    assert captured["blend"] == 0.5


def test_substituted_cross_encoder_uses_the_rerank_threshold(tmp_path):
    """RELEVANCE_THRESHOLD is calibrated for Laya; a cross-encoder's sigmoid is not."""
    config = make_config(
        tmp_path, dense_weight=1.0, rerank_k=3,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        relevance_threshold=0.3,
    )
    assert config.rerank_threshold == 0.0
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    # 0.2 would fail the Laya bar (0.3) and the old shared 0.5 cut, but the CE path keeps
    # the pool at RERANK_THRESHOLD 0.0 and lets coverage and top-k decide.
    engine.cross_encoder = _StubCrossEncoder(score=0.2)
    result = engine.query("How long does a refund take?")
    assert result.passages


def test_an_explicit_relevance_threshold_still_filters_a_cross_encoder(tmp_path):
    config = make_config(
        tmp_path, dense_weight=1.0, rerank_k=3,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    engine.cross_encoder = _StubCrossEncoder(score=0.2)
    result = engine.query("How long does a refund take?", relevance_threshold=0.5)
    assert result.passages == []
    assert result.answerable is False
