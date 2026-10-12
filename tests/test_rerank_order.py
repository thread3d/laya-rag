"""`order="recall"` preserves the caller's order — it does not re-derive the retrieval order.

This is the defect that hid behind identical MultiHop-RAG rankings for two different cross-encoders
(`SCORE-LIFT.md` §P4). `recall_score` travels with each hit, so sorting by it does not "keep the
retrieval order"; it rebuilds the order the caller *started* from and discards any reordering applied
before the hand-off. The coverage pass permutes the whole pool by coverage and then asks for
`recall` precisely so its order survives — so the sort undid every coverage decision and made the
cross-encoder's scores, which drove that permutation, irrelevant to the result.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from laya_rag.chunking import Chunk  # noqa: E402
from laya_rag.index_store import Hit  # noqa: E402

from .support import StubModel  # noqa: E402


def _hit(chunk_id: str, text: str, score: float) -> Hit:
    return Hit(
        chunk=Chunk(chunk_id=chunk_id, doc_id=chunk_id, path=f"{chunk_id}.txt", text=text,
                    index=0, start_char=0, end_char=len(text)),
        score=score,
    )


class _Gate:
    """A stand-in for the answerability gate, so no model runs in these tests."""

    def predict_proba(self, X):  # noqa: N803 - mirrors the model API
        return np.full(len(X), 0.9, dtype=np.float64)


def _rerank(hits, relevance, order):
    from laya_rag.rerank import LayaReranker

    reranker = LayaReranker(StubModel())
    return reranker.rerank(
        "a query",
        hits,
        top_k=len(hits),
        relevance_threshold=0.0,
        answerability_threshold=0.0,
        scores=relevance,
        gate=_Gate(),
        order=order,
    )


def test_recall_order_preserves_a_permuted_pool():
    """The pool arrives already ordered; `recall` must not touch it."""
    hits = [_hit("a", "first.", 3.0), _hit("b", "second.", 2.0), _hit("c", "third.", 1.0)]
    permuted = [hits[2], hits[0], hits[1]]
    result = _rerank(permuted, [0.1, 0.9, 0.5], "recall")
    assert [p.chunk_id for p in result.passages] == ["c", "a", "b"]


def test_recall_order_is_not_the_recall_score_order_when_the_caller_reordered():
    """The exact defect: sorting by `recall_score` would rebuild `a, b, c` and discard the pool."""
    hits = [_hit("a", "first.", 3.0), _hit("b", "second.", 2.0), _hit("c", "third.", 1.0)]
    permuted = [hits[2], hits[0], hits[1]]
    returned = [p.chunk_id for p in _rerank(permuted, [0.1, 0.9, 0.5], "recall").passages]
    assert returned != ["a", "b", "c"]
    assert returned == ["c", "a", "b"]


def test_relevance_order_still_ranks_by_the_decision():
    """The default is unchanged: the supplied relevance is the ranking key."""
    hits = [_hit("a", "first.", 3.0), _hit("b", "second.", 2.0), _hit("c", "third.", 1.0)]
    result = _rerank(hits, [0.1, 0.9, 0.5], "relevance")
    assert [p.chunk_id for p in result.passages] == ["b", "c", "a"]
