"""A citation-first extractive composer: every emitted sentence is a verbatim span.

The engine's credibility hole is "no end-to-end ranking", and its measured citation mechanics are a
negative: over the pilot's 14 answers **no answer met the deterministic citation bar**, dominated by
39 of 81 sentences carrying no citation. A generative composer buys fluency with a *probabilistic*
attribution guarantee. This buys a **structural** one: a sentence that is not present verbatim in a
retrieved passage is never emitted, and with nothing to emit the composer abstains. The cost is
extractive prose, which is the honest trade for a system that cannot generate.

The output is also the structured citation contract: each emitted sentence carries the document
citation number it must be checked against, its source path and chunk id, so
:func:`laya_rag.verify.verify_citations` can be run over the result without parsing prose.

    from laya_rag.extractive import compose_extractive
    composed = compose_extractive(result.passages, query=query)
    composed.answer        # "Refunds take five days [1]. ..."
    composed.sentences[0]  # ExtractedSentence(text, citation=1, path=..., chunk_id=...)

No generator, no API, no model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .evidence import group_passages_by_document

#: A sentence run ending at terminal punctuation, or a whole line when there is none.
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]+|[^.!?\n]+$")
_WHITESPACE = re.compile(r"\s+")

#: What the composer says when no passage yields an emittable sentence.
ABSTENTION = "The retrieved evidence does not contain a sentence that answers the question."


def split_sentences(text: str) -> list[str]:
    """Sentence runs, trimmed; a run with no terminal punctuation is still a sentence."""
    return [match.group(0).strip() for match in _SENTENCE.finditer(text or "") if match.group(0).strip()]


def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip().casefold()


def _score(passage: Mapping[str, Any]) -> float:
    for key in ("relevance", "recall_score", "score"):
        value = passage.get(key)
        if value is not None:
            return float(value)
    return 0.0


@dataclass(frozen=True)
class ExtractedSentence:
    """One emitted sentence and the citation it must satisfy."""

    text: str
    citation: int
    path: str | None
    chunk_id: str | None
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "citation": self.citation,
            "path": self.path,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
        }


@dataclass
class ExtractiveAnswer:
    """The composed answer, or an abstention, plus its structured citations."""

    answer: str
    abstained: bool
    sentences: list[ExtractedSentence] = field(default_factory=list)
    considered: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "abstained": self.abstained,
            "sentences": [sentence.as_dict() for sentence in self.sentences],
            "considered": self.considered,
        }


def compose_extractive(
    passages: Sequence[Mapping[str, Any]],
    *,
    query: str | None = None,
    max_sentences: int = 5,
    min_score: float = 0.0,
) -> ExtractiveAnswer:
    """Emit up to ``max_sentences`` verbatim sentences, best-scored first, each cited.

    A candidate sentence is kept only if it appears **verbatim** in one of the passage texts —
    that is the structural guarantee, and it survives the document merge the same way
    :func:`build_messages` presents evidence. Documents are numbered in first-appearance order,
    so ``[n]`` means the same thing here, in the prompt, and in `verify_citations`; the marker
    is emitted *before* the sentence it supports, for the reason given where the answer is
    assembled below.
    """
    if max_sentences < 1:
        raise ValueError("max_sentences must be positive")
    if not passages:
        return ExtractiveAnswer(answer=ABSTENTION, abstained=True)

    # Per-document best score, so a sentence inherits its document's rank.
    best: dict[str, float] = {}
    for passage in passages:
        key = str(passage.get("path") or "")
        best[key] = max(best.get(key, float("-inf")), _score(passage))

    merged = group_passages_by_document(
        [(passage.get("path"), str(passage.get("text") or "")) for passage in passages]
    )
    candidates: list[ExtractedSentence] = []
    for index, (path, text) in enumerate(merged, start=1):
        document_score = best.get(path, 0.0)
        if document_score < min_score:
            continue
        for sentence in split_sentences(text):
            # The structural guarantee: only a span that exists verbatim in a passage.
            source = next(
                (
                    passage
                    for passage in passages
                    if sentence in str(passage.get("text") or "")
                ),
                None,
            )
            if source is None:
                continue
            candidates.append(
                ExtractedSentence(
                    text=sentence,
                    citation=index,
                    path=path or None,
                    chunk_id=source.get("chunk_id"),
                    score=document_score,
                )
            )

    candidates.sort(key=lambda sentence: -sentence.score)
    seen: set[str] = set()
    kept: list[ExtractedSentence] = []
    for sentence in candidates:
        key = _normalise(sentence.text)
        if key in seen:
            continue
        seen.add(key)
        kept.append(sentence)
        if len(kept) >= max_sentences:
            break

    if not kept:
        return ExtractiveAnswer(answer=ABSTENTION, abstained=True, considered=len(candidates))
    # The marker **leads** its sentence rather than trailing it. `verify_citations` splits an
    # answer into claims *after* the terminal punctuation, so a trailing "[1]" is orphaned onto
    # the next claim — the sentence it was meant to support reads as uncited and the final marker
    # as a dangling citation. Emitting it first is what makes the structured citation contract
    # checkable without parsing prose, and it matches how `build_messages` numbers the evidence
    # the model is shown: "[1]" opens both the prompt's document block and the sentence quoting it.
    answer = " ".join(f"[{sentence.citation}] {sentence.text}" for sentence in kept)
    return ExtractiveAnswer(
        answer=answer, abstained=False, sentences=kept, considered=len(candidates)
    )
