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


def test_rerank_coverage_is_off_by_default_and_serialised():
    config = EngineConfig()
    assert config.rerank_coverage_lambda == 0.0
    assert config.rerank_coverage_min_docs == 0
    assert "rerank_coverage_lambda" in config.as_dict()
    assert "rerank_coverage_min_docs" in config.as_dict()


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
