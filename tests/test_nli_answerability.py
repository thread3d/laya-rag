"""Tests for the NLI answerability path (``LAYA_RAG_ANSWERABILITY_NLI_MODEL``).

The NLI scorer is a different model with its own calibration, so it is an explicit
alternative to Laya's scalar and the feature gate. These tests pin the label resolution,
the injection seam used for offline testing, and the precedence order.
"""
from __future__ import annotations

import numpy as np
import pytest

from laya_rag.answerability import NLIAnswerability, _resolve_entailment_index
from laya_rag.chunking import Chunk
from laya_rag.config import EngineConfig
from laya_rag.engine import RagEngine
from laya_rag.index_store import Hit
from laya_rag.rerank import LayaReranker

from .support import StubModel, make_config, write_corpus


def _hit(chunk_id: str, text: str) -> Hit:
    return Hit(
        chunk=Chunk(chunk_id, chunk_id, "doc.md", text, 0, 0, len(text)),
        score=1.0,
        channels=("dense",),
    )


class _FakeGate:
    def predict_proba(self, X):  # noqa: N803 - mirrors the model API
        return np.full(len(X), 0.1, dtype=np.float64)


# ── label resolution ─────────────────────────────────────────────────────

def test_entailment_index_is_found_by_label_name():
    assert _resolve_entailment_index({0: "contradiction", 1: "neutral", 2: "entailment"}) == 2
    assert _resolve_entailment_index({0: "LABEL_0", 1: "ENTAILMENT"}) == 1


def test_entailment_index_falls_back_by_arity():
    # Three-way NLI is (contradiction, neutral, entailment); two-way is (not, entail).
    assert _resolve_entailment_index({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}) == 2
    assert _resolve_entailment_index({}) == 2
    assert _resolve_entailment_index({0: "only"}) == 0


# ── scorer ───────────────────────────────────────────────────────────────

def test_an_injected_scorer_is_used_without_loading_weights():
    seen: list[tuple[str, str]] = []

    def scorer(premise: str, hypothesis: str) -> float:
        seen.append((premise, hypothesis))
        return 0.87

    nli = NLIAnswerability("fake-model", scorer=scorer)
    value = nli.score("How long do refunds take?", [_hit("a", "Refunds take five business days.")])

    assert value == pytest.approx(0.87)
    assert seen and "Refunds take five" in seen[0][0]
    assert nli.info()["loaded"] is True


def test_empty_evidence_scores_zero():
    nli = NLIAnswerability("fake-model", scorer=lambda premise, hypothesis: 0.99)
    assert nli.score("q", []) == 0.0


def test_the_nli_model_is_not_loaded_until_scored():
    nli = NLIAnswerability("some/checkpoint")
    assert nli.info()["loaded"] is False
    assert nli.model_id == "some/checkpoint"


# ── precedence ───────────────────────────────────────────────────────────

def test_nli_precedence_over_the_feature_gate():
    reranker = LayaReranker(StubModel())
    result = reranker.rerank(
        "refund",
        [_hit("a", "refund policy text")],
        top_k=1,
        relevance_threshold=0.0,
        gate=_FakeGate(),
        answerability_fn=lambda query, passages: 0.75,
    )
    assert result.answerability == pytest.approx(0.75)
    assert result.answerability_measured is True


# ── config + engine wiring ───────────────────────────────────────────────

def test_nli_model_defaults_to_unset():
    assert EngineConfig().answerability_nli_model == ""


def test_engine_reports_and_uses_a_configured_nli_scorer(tmp_path):
    config = make_config(tmp_path, answerability_nli_model="fake-model", dense_weight=1.0)
    engine = RagEngine(config, model=StubModel())
    assert engine.answerability_nli is not None
    assert engine.stats()["answerability_nli"]["configured"] is True
    # Inject a scorer so no weights are fetched in the test.
    engine._answerability_nli = NLIAnswerability(
        "fake-model", scorer=lambda premise, hypothesis: 0.91
    )
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    result = engine.query("How long does a refund take?")
    assert result.answerability == pytest.approx(0.91)
    assert result.answerability_measured is True


def test_engine_without_an_nli_model_has_no_scorer(tmp_path):
    engine = RagEngine(make_config(tmp_path), model=StubModel())
    assert engine.answerability_nli is None
    assert engine.stats()["answerability_nli"] == {"configured": False, "model": None}
