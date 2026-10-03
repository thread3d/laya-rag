"""Offline tests for the bridge-conditioned relevance decision.

The bridge-conditioned question is the piece of the BridgeRAG pipeline that this engine
can implement without a generator: Laya already answers typed questions over arbitrary
state, so conditioning second-hop relevance on what the first hop found is a third state
key and a question, not a new model. These tests pin the plumbing and the wording
registry — not the retrieval effect, which only the real checkpoint and a held-out
comparison can establish.
"""
from __future__ import annotations

import pytest

from laya_rag.rerank import (
    BRIDGE_RELEVANCE_QID,
    BRIDGE_RELEVANCE_QUESTION,
    BRIDGE_RELEVANCE_WORDINGS,
    LayaReranker,
    bridge_question,
)

from .support import StubModel

QUERY = "which company acquired the maker of the widget"
BRIDGE = "Acme Corp is a widget manufacturer founded in 1998 in Ohio"
PASSAGES = [
    "Acme Corp was acquired by Globex Industries for two billion dollars.",
    "Unrelated text about weather and shipping schedules.",
]


def test_every_bridge_wording_names_the_bridge_key() -> None:
    """A wording that does not name `bridge` would silently measure the two-way question."""
    assert BRIDGE_RELEVANCE_WORDINGS, "the registry must not be empty"
    for name, question in BRIDGE_RELEVANCE_WORDINGS.items():
        assert question["type"] == "noul", name
        assert "`bridge`" in question["instructions"], name
        assert "`passage`" in question["instructions"], name
        assert "`query`" in question["instructions"], name


def test_bridge_question_returns_the_named_wording() -> None:
    for name, question in BRIDGE_RELEVANCE_WORDINGS.items():
        assert bridge_question(name) is question


def test_bridge_question_rejects_an_unknown_wording() -> None:
    """A typo must raise, not silently fall back to the default and mislabel a result."""
    with pytest.raises(ValueError, match="unknown bridge wording"):
        bridge_question("no-such-wording")


def test_default_question_is_a_registered_wording() -> None:
    assert BRIDGE_RELEVANCE_QUESTION is BRIDGE_RELEVANCE_WORDINGS["bridge"]


def test_conditioned_scoring_returns_one_score_per_passage() -> None:
    model = StubModel()
    reranker = LayaReranker(model)
    scores, usage = reranker.score_relevance_conditioned(QUERY, BRIDGE, PASSAGES)
    assert len(scores) == len(PASSAGES)
    assert all(isinstance(score, float) for score in scores)
    assert model.predict_many_calls == 1
    assert usage  # usage is merged from the model's per-result payloads


def test_conditioned_scoring_puts_the_bridge_in_the_state() -> None:
    """The bridge has to reach the model as its own key, not be prepended to the query."""
    seen: list[dict] = []

    class RecordingModel(StubModel):
        def predict_many(self, states, questions, batch_size=8):  # type: ignore[override]
            seen.extend(states)
            return super().predict_many(states, questions, batch_size=batch_size)

    reranker = LayaReranker(RecordingModel())
    reranker.score_relevance_conditioned(QUERY, BRIDGE, PASSAGES, wording="chain")
    assert seen, "no states reached the model"
    for state in seen:
        assert state["bridge"] == BRIDGE
        assert state["query"] == QUERY
        assert "passage" in state


def test_the_bridge_text_moves_the_score() -> None:
    """Conditioning must be observable: the same passage scores differently under a bridge.

    The stub scores query/state token overlap across every non-query string, so a bridge
    that shares terms with the query raises the score and one that does not leaves it.
    """
    reranker = LayaReranker(StubModel())
    with_bridge, _ = reranker.score_relevance_conditioned(QUERY, BRIDGE, PASSAGES)
    without_bridge, _ = reranker.score_relevance_conditioned(QUERY, "", PASSAGES)
    assert with_bridge != without_bridge


def test_empty_texts_short_circuit_without_calling_the_model() -> None:
    model = StubModel()
    reranker = LayaReranker(model)
    scores, usage = reranker.score_relevance_conditioned(QUERY, BRIDGE, [])
    assert scores == []
    assert usage == {}
    assert model.predict_many_calls == 0


def test_a_result_count_mismatch_is_an_error() -> None:
    """Silently zipping a short result would misalign scores with passages."""

    class ShortModel(StubModel):
        def predict_many(self, states, questions, batch_size=8):  # type: ignore[override]
            return super().predict_many(states, questions, batch_size=batch_size)[:-1]

    reranker = LayaReranker(ShortModel())
    with pytest.raises(Exception, match="results"):
        reranker.score_relevance_conditioned(QUERY, BRIDGE, PASSAGES)


def test_the_question_id_is_stable() -> None:
    """The qid is what the engine keys the calibrated probability on."""
    assert BRIDGE_RELEVANCE_QID == "relevant"
