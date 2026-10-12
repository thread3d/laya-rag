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

import numpy as np

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

# Bridge-conditioned relevance: `s(q, b, c)` rather than `s(q, c)`.
#
# A second-hop passage is, by construction, only weakly aligned with the *query*; it is
# strongly aligned with the entity the first hop resolved. The two-way relevance question
# therefore cannot separate the active reasoning chain from a wrong one that shares the
# query's surface entity, which is the measured "second-hop tax"
# (docs/research/08-frontier-2026.md section 2.1, arXiv:2604.03384). These instructions
# name a third state key, `bridge`, so the decision is conditioned on what hop 1 found.
#
# UNLIKE RELEVANCE_QUESTION, **none of these wordings comes from Laya's own benchmark.**
# They are ours and unvalidated, and Laya's calibrated head is phrasing-sensitive, so a
# wording that reads better can measure worse. `bridge` is the default only because it is
# the one the sweep in tools/multihop_tri_cache.py evaluates; it has to earn its place in
# a held-out comparison before anything depends on it.
BRIDGE_RELEVANCE_QID = "relevant"

BRIDGE_RELEVANCE_WORDINGS: dict[str, dict[str, Any]] = {
    "bridge": {
        "type": "noul",
        "instructions": (
            "Does `passage` help answer `query`, given that `query` is about `bridge`?"
        ),
    },
    "conditional": {
        "type": "noul",
        "instructions": (
            "Given `bridge` as context, does `passage` help answer `query`?"
        ),
    },
    "chain": {
        "type": "noul",
        "instructions": (
            "Does `passage` contain information needed to answer `query`, "
            "in the context of `bridge`?"
        ),
    },
}

#: The wording used when a caller does not name one.
BRIDGE_RELEVANCE_QUESTION: dict[str, Any] = BRIDGE_RELEVANCE_WORDINGS["bridge"]


def bridge_question(wording: str) -> dict[str, Any]:
    """The bridge-conditioned question for a named wording; unknown names raise.

    Raising rather than falling back keeps a typo from silently measuring the default
    wording under the name of the one that was asked for.
    """
    try:
        return BRIDGE_RELEVANCE_WORDINGS[wording]
    except KeyError as exc:
        known = ", ".join(sorted(BRIDGE_RELEVANCE_WORDINGS))
        raise ValueError(f"unknown bridge wording {wording!r}; expected one of {known}") from exc


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
    #: Whether ``answerability`` came from a decision. The relevance gate short-circuits
    #: to ``0.0`` when nothing survives it, and that structural zero is *not measured*:
    #: feeding it into a calibration as if it were a head output mixes two mechanisms
    #: (the reason ``tools/multihop_recalibrate.py`` filters on ``p > 0``).
    answerability_measured: bool = False
    considered: int = 0
    kept: int = 0
    dropped_below_threshold: int = 0
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def best(self) -> ScoredPassage | None:
        return self.passages[0] if self.passages else None


def blend_scores(
    recall: Sequence[float],
    relevance: Sequence[float],
    blend: float,
) -> list[float]:
    """Interpolate the first-stage and decision scores into one ordering key.

    ``blend == 1.0`` returns the decisions unchanged (the historical behaviour).
    ``blend == 0.0`` returns the recall order. In between, each channel is min-max
    normalised per query and mixed, so a second stage can *nudge* a ranking instead of
    replacing it — which is what a fully-trusted pointwise cross-encoder did when it
    lowered nDCG@10 on all five BEIR corpora, catastrophically on ArguAna.
    """
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"blend must be in [0, 1], got {blend!r}")
    recall_array = np.asarray(recall, dtype=np.float64)
    if recall_array.size and float(recall_array.max()) > float(recall_array.min()):
        recall_array = (recall_array - recall_array.min()) / (
            recall_array.max() - recall_array.min()
        )
    else:
        recall_array = np.zeros_like(recall_array)
    relevance_array = np.asarray(relevance, dtype=np.float64)
    return (blend * relevance_array + (1.0 - blend) * recall_array).tolist()


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

    def score_relevance_conditioned(
        self,
        query: str,
        bridge: str,
        texts: Sequence[str],
        batch_size: int = 8,
        wording: str = "bridge",
    ) -> tuple[list[float], dict[str, Any]]:
        """Score passages with the bridge-conditioned decision ``s(q, b, c)``.

        The same batched call as :meth:`score_relevance`, with a third state key: the
        bridge passage resolved by the first hop. This is the one piece of the BridgeRAG
        pipeline that needs no generator — Laya already answers typed questions over
        arbitrary state, so conditioning is a question and a state key, not a model.

        ``bridge`` may be empty, which degrades to the two-way question with different
        wording; callers that want a control arm should pass a *shuffled* bridge instead,
        so the instruction text is held fixed and only the conditioning varies.
        """
        if not texts:
            return [], {}
        states = [
            {"query": query, "bridge": bridge, "passage": text} for text in texts
        ]
        results = self.model.predict_many(
            states, {BRIDGE_RELEVANCE_QID: bridge_question(wording)}, batch_size=batch_size
        )
        if len(results) != len(texts):
            raise DecisionError(f"Laya returned {len(results)} results for {len(texts)} passages")
        usage: dict[str, Any] = {}
        for result in results:
            _merge_usage(usage, result.get("usage") or {})
        return [_noul_value(result, BRIDGE_RELEVANCE_QID) for result in results], usage

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

    def score_relevance_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        batch_size: int = 8,
    ) -> tuple[list[float], dict[str, Any]]:
        """Score many ``(query, passage)`` pairs in one batched pass.

        :meth:`score_relevance` scores one query against many passages. Adjudicating a
        set of claims needs many queries against many passages, and batching them
        together is the difference between a few seconds and a minute. The question
        wording is the same one, verbatim — only the batching differs.
        """
        if not pairs:
            return [], {}
        states = [{"query": query, "passage": text} for query, text in pairs]
        results = self.model.predict_many(
            states, {RELEVANCE_QID: RELEVANCE_QUESTION}, batch_size=batch_size
        )
        if len(results) != len(pairs):
            raise DecisionError(f"Laya returned {len(results)} results for {len(pairs)} pairs")
        usage: dict[str, Any] = {}
        for result in results:
            _merge_usage(usage, result.get("usage") or {})
        return [_noul_value(result, RELEVANCE_QID) for result in results], usage

    def check_answerability_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        batch_size: int = 8,
    ) -> tuple[list[float], dict[str, Any]]:
        """Ask the answerability question for many ``(query, evidence)`` pairs at once.

        This is the already-measured ``answerability`` question, not a new
        claim-support question: the threshold attached to it is the one this project
        benchmarked, and inventing a differently-worded question would carry a
        calibration that has not been established for it.
        """
        if not pairs:
            return [], {}
        states = [{"query": query, "evidence": evidence} for query, evidence in pairs]
        results = self.model.predict_many(
            states, {ANSWERABILITY_QID: ANSWERABILITY_QUESTION}, batch_size=batch_size
        )
        if len(results) != len(pairs):
            raise DecisionError(f"Laya returned {len(results)} results for {len(pairs)} pairs")
        usage: dict[str, Any] = {}
        for result in results:
            _merge_usage(usage, result.get("usage") or {})
        return [_noul_value(result, ANSWERABILITY_QID) for result in results], usage

    def rerank(
        self,
        query: str,
        hits: Sequence[Hit],
        top_k: int = 5,
        relevance_threshold: float = 0.5,
        answerability_threshold: float = 0.5,
        batch_size: int = 8,
        order: str = "relevance",
        scores: Sequence[float] | None = None,
        blend: float = 1.0,
        gate: Any | None = None,
        answerability_fn: Any | None = None,
        min_docs: int = 0,
        bridge_feature: bool = False,
    ) -> RerankResult:
        """Apply a relevance decision to the candidates and gate on answerability.

        ``order`` decides what the returned evidence is ranked by. ``relevance`` (the
        default) uses the relevance probability, which is what the engine has always
        done. ``recall`` keeps the retrieval order and uses the probability only to
        filter and to gate — measured, Laya's decision is a better *filter* than a
        *ranker*: reordering BM25's candidates by it lowered nDCG@10 on both BEIR
        datasets (SciFact 0.654 -> 0.470, NFCorpus 0.304 -> 0.283).

        ``scores`` lets the caller supply the relevance probabilities instead of having
        Laya compute them — the engine uses this to substitute a retrieval-trained
        cross-encoder for ordering and filtering while keeping Laya's answerability gate.

        ``blend`` mixes that supplied score with the first-stage recall score before
        ordering (see :func:`blend_scores`). ``1.0`` keeps the historical behaviour of
        ordering fully by the decision; values below it bound how far a second stage can
        move the list.

        ``min_docs`` > 0 promotes the best passage of each distinct document into the first
        that many slots before the ``top_k`` cut, so a passage-level list spreads across
        documents instead of filling with one file's overlapping chunks.

        ``gate`` is an optional answerability model (anything with ``predict_proba``)
        that replaces Laya's single-scalar decision with a multi-feature one. ``None``
        keeps Laya's own question.

        ``answerability_fn`` is a higher-precedence override: a callable
        ``(query, passages) -> probability`` used for an NLI answerability model. It wins
        over ``gate`` and over Laya's question because it is the most specific scorer.
        """
        if not hits:
            return RerankResult()

        if scores is None:
            relevance, usage = self.score_relevance(
                query, [hit.chunk.text for hit in hits], batch_size=batch_size
            )
        else:
            relevance = [float(score) for score in scores]
            if len(relevance) != len(hits):
                raise DecisionError(
                    f"got {len(relevance)} scores for {len(hits)} candidates"
                )
            usage = {}
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

        above = [passage for passage in scored if passage.relevance >= relevance_threshold]
        if blend < 1.0:
            # The second stage nudges the first-stage order instead of replacing it.
            combined = blend_scores(
                [passage.recall_score for passage in scored],
                [passage.relevance for passage in scored],
                blend,
            )
            blended = {passage.chunk_id: score for passage, score in zip(scored, combined)}
            above.sort(
                key=lambda passage: (
                    -blended[passage.chunk_id],
                    -passage.relevance,
                    -passage.recall_score,
                )
            )
        elif order == "recall":
            # **The caller's order, preserved.** `recall_score` travels with each hit, so sorting by
            # it does not "keep the retrieval order" — it re-derives the order the caller *started*
            # from, silently discarding any reordering the caller applied before handing the pool
            # over. The coverage pass does exactly that (it permutes the whole pool by coverage, then
            # asks for `recall` so its order is not "re-sorted by the score again"), so the sort here
            # undid every coverage decision and made the configured cross-encoder's scores — which
            # drove that permutation — irrelevant to the result. `hits` arrive in the order the
            # caller means, so preserving it is what both this branch's own documentation and its
            # caller expect.
            pass
        else:
            # Laya's probability is the ranking key; the recall score only breaks ties.
            above.sort(key=lambda passage: (-passage.relevance, -passage.recall_score))
        if min_docs > 0 and above:
            # A file's chunks are near-duplicates of each other, so a passage-level top-k
            # can return several chunks of one document and fewer documents overall. Keep
            # the best passage of each distinct document for the first `min_docs` slots,
            # then let the rest follow in rank order. Document identity is the path, with
            # the chunk id as the fallback the gate already uses.
            head: list[Any] = []
            seen: set[str] = set()
            rest: list[Any] = []
            for passage in above:
                document = str(passage.path or passage.chunk_id)
                if len(head) < min_docs and document not in seen:
                    seen.add(document)
                    head.append(passage)
                else:
                    rest.append(passage)
            above = head + rest
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

        if answerability_fn is not None:
            # Highest precedence: a dedicated scorer (e.g. an NLI cross-encoder) that
            # looks at the evidence directly rather than at retrieval features.
            answerability = float(answerability_fn(query, kept))
        elif gate is not None:
            # A trained feature gate replaces the one-scalar decision. The features are
            # the evidence signals the engine already has, so this needs no extra model
            # call — except the optional bridge column, which does, and is off by default.
            from .answerability import extract_features

            bridge_value: float | None = None
            if bridge_feature and len(kept) >= 2:
                # BridgeRAG conditioning s(q, b, c): the bridge is the top-ranked passage
                # and the column is the best conditional decision over the rest. A model
                # failure degrades to 0.0 rather than failing the query.
                try:
                    bridge_scores, bridge_usage = self.score_relevance_conditioned(
                        query, kept[0].text, [passage.text for passage in kept[1:]]
                    )
                    bridge_value = max(bridge_scores) if bridge_scores else None
                    _merge_usage(result.usage, bridge_usage)
                except DecisionError:
                    bridge_value = None
            features = extract_features(query, kept, bridge_conditioned=bridge_value)
            answerability = float(np.asarray(gate.predict_proba([features])).reshape(-1)[0])
        else:
            answerability, answer_usage = self.check_answerability(query, kept)
            _merge_usage(result.usage, answer_usage)
        result.answerability = answerability
        result.answerability_measured = True
        result.answerable = answerability >= answerability_threshold
        return result
