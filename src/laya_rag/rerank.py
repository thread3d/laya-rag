"""Laya's decision layer: relevance re-ranking and answerability gating.

This is where Laya earns its place in the pipeline. Dense retrieval returns
*similar* passages; Laya returns *decisions* with calibrated probabilities, so the
engine can threshold them honestly instead of guessing at a cosine cut-off.

The relevance question is worded exactly as Laya's own MS MARCO relevance
benchmark words it (``research/scripts/bench_apps.py``, ``app.rag_relevance``), so
the calibrated behaviour measured there carries over instead of being re-derived
from a paraphrase.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .index_store import Hit, VectorIndex  # noqa: F401  (Hit re-exported for callers)
from .model import LayaModel

RELEVANCE_QID = "relevant"

# Verbatim from Laya's app.rag_relevance benchmark (MS MARCO v1.1, 0.625 English /
# 0.657 multilingual, in-training). Do not reword without re-measuring.
RELEVANCE_QUESTION: dict[str, Any] = {
    "type": "noul",
    "instructions": "Does `passage` help answer `query`?",
}

ANSWERABILITY_QID = "answerable"
ANSWERABILITY_QUESTION: dict[str, Any] = {
    "type": "noul",
    "instructions": "Can `query` be answered using only `evidence`?",
}


class DecisionError(RuntimeError):
    """Raised when Laya returns a result the engine cannot interpret."""


@dataclass
class ScoredPassage:
    """One candidate passage after Laya has judged it."""

    text: str
    path: str
    chunk_id: str
    recall_score: float
    relevance: float
    index: int
    start_char: int
    end_char: int
    channels: tuple[str, ...] = ()

    def as_dict(self, *, max_chars: int | None = None) -> dict[str, Any]:
        text = self.text
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        return {
            "text": text,
            "path": self.path,
            "chunk_id": self.chunk_id,
            "recall_score": round(self.recall_score, 4),
            "relevance": round(self.relevance, 4),
            "channels": list(self.channels),
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass
class RerankResult:
    """The evidence set for one query, plus Laya's verdict on whether it suffices."""

    passages: list[ScoredPassage] = field(default_factory=list)
    answerable: bool = False
    answerability: float | None = None
    considered: int = 0
    kept: int = 0
    dropped_below_threshold: int = 0
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def best(self) -> ScoredPassage | None:
        return self.passages[0] if self.passages else None


def _noul_value(result: dict[str, Any], qid: str) -> float:
    """Pull one calibrated P(true) out of a Laya predict result."""
    if not isinstance(result, dict):
        raise DecisionError(f"expected a Laya result dict, got {type(result).__name__}")
    answers = result.get("answers")
    if not isinstance(answers, dict) or qid not in answers:
        raise DecisionError(f"Laya result has no answer for question {qid!r}: keys={list(answers or [])}")
    answer = answers[qid]
    if not isinstance(answer, dict) or "noul" not in answer:
        raise DecisionError(f"answer {qid!r} is not a noul answer: {answer!r}")
    return float(answer["noul"])


def _merge_usage(target: dict[str, Any], *sources: dict[str, Any]) -> dict[str, Any]:
    """Sum Laya's token usage across several calls."""
    for source in sources:
        for key in ("input_tokens", "output_tokens"):
            value = source.get(key)
            if isinstance(value, (int, float)):
                target[key] = target.get(key, 0) + value
    return target


class LayaReranker:
    """Re-ranks retrieved passages and decides whether they answer the query."""

    def __init__(self, model: LayaModel) -> None:
        self.model = model

    def score_relevance(
        self,
        query: str,
        texts: Sequence[str],
        batch_size: int = 8,
    ) -> tuple[list[float], dict[str, Any]]:
        """Score each passage with Laya's calibrated relevance decision."""
        if not texts:
            return [], {}
        states = [{"query": query, "passage": text} for text in texts]
        results = self.model.predict_many(
            states, {RELEVANCE_QID: RELEVANCE_QUESTION}, batch_size=batch_size
        )
        if len(results) != len(texts):
            raise DecisionError(f"Laya returned {len(results)} results for {len(texts)} passages")
        usage: dict[str, Any] = {}
        for result in results:
            _merge_usage(usage, result.get("usage") or {})
        return [_noul_value(result, RELEVANCE_QID) for result in results], usage

    def check_answerability(
        self,
        query: str,
        passages: Sequence[ScoredPassage],
    ) -> tuple[float, dict[str, Any]]:
        """Ask Laya whether the assembled evidence actually answers the query."""
        if not passages:
            return 0.0, {}
        evidence = "\n\n".join(f"[{i + 1}] {passage.text}" for i, passage in enumerate(passages))
        result = self.model.predict(
            {"query": query, "evidence": evidence},
            {ANSWERABILITY_QID: ANSWERABILITY_QUESTION},
        )
        usage = result.get("usage")
        return _noul_value(result, ANSWERABILITY_QID), (usage if isinstance(usage, dict) else {})

    def rerank(
        self,
        query: str,
        hits: Sequence[Hit],
        top_k: int = 5,
        relevance_threshold: float = 0.5,
        answerability_threshold: float = 0.5,
        batch_size: int = 8,
    ) -> RerankResult:
        """Apply Laya's relevance decision to dense candidates and gate on answerability."""
        if not hits:
            return RerankResult()

        relevance, usage = self.score_relevance(
            query, [hit.chunk.text for hit in hits], batch_size=batch_size
        )
        scored = [
            ScoredPassage(
                text=hit.chunk.text,
                path=hit.chunk.path,
                chunk_id=hit.chunk.chunk_id,
                recall_score=hit.score,
                relevance=score,
                index=hit.chunk.index,
                start_char=hit.chunk.start_char,
                end_char=hit.chunk.end_char,
                channels=hit.channels,
            )
            for hit, score in zip(hits, relevance)
        ]
        # Laya's probability is the ranking key; the recall score only breaks ties.
        scored.sort(key=lambda passage: (-passage.relevance, -passage.recall_score))

        above = [passage for passage in scored if passage.relevance >= relevance_threshold]
        kept = above[:top_k]
        result = RerankResult(
            passages=kept,
            considered=len(hits),
            kept=len(kept),
            dropped_below_threshold=len(scored) - len(above),
            usage=dict(usage),
        )
        if not kept:
            result.answerable = False
            result.answerability = 0.0
            return result

        answerability, answer_usage = self.check_answerability(query, kept)
        _merge_usage(result.usage, answer_usage)
        result.answerability = answerability
        result.answerable = answerability >= answerability_threshold
        return result
