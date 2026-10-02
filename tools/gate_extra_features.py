"""Experimental evidence-shape features for the answerability gate.

The shipped gate has nine features, all of them about the *scores* the engine produced
(relevance, recall, document spread, length, term coverage). These three are about the
**shape of the evidence set** and are new signals rather than rescalings of the nine:

* ``max_passage_coverage`` — can any single passage answer the question by itself?
* ``coverage_spread`` — how many passages a greedy set cover needs to cover the query terms;
  a high number means the answer is distributed across passages (the multi-hop case).
* ``pairwise_redundancy`` — mean Jaccard overlap between passages; high redundancy means the
  recall budget was spent on near-duplicates rather than new facts.

They are kept out of ``laya_rag.answerability.FEATURE_NAMES`` until a grouped comparison shows
they earn a production change, so no existing gate artefact is invalidated by probing them.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np

EXTRA_NAMES: tuple[str, ...] = (
    "max_passage_coverage",
    "coverage_spread",
    "pairwise_redundancy",
)

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: Any) -> set[str]:
    return {token for token in _TOKEN.findall(str(text).lower()) if len(token) >= 3}


def _text_of(passage: Any) -> str:
    if isinstance(passage, str):
        return passage
    for attribute in ("text", "passage", "content"):
        value = getattr(passage, attribute, None)
        if isinstance(value, str):
            return value
    if isinstance(passage, dict):
        for key in ("text", "passage", "content"):
            if isinstance(passage.get(key), str):
                return passage[key]
    return ""


def max_passage_coverage(query: str, passages: Sequence[Any]) -> float:
    """The best single passage's fraction of the query's content terms (0 when empty)."""
    wanted = _tokens(query)
    if not wanted:
        return 0.0
    best = 0.0
    for passage in passages or []:
        best = max(best, len(wanted & _tokens(_text_of(passage))) / len(wanted))
    return best


def coverage_spread(query: str, passages: Sequence[Any]) -> float:
    """Passages a greedy set cover needs to cover every query term.

    ``0.0`` when the query has no content terms or the evidence covers none of them; a
    larger value means the answer is spread thin across passages. Deterministic.
    """
    remaining = _tokens(query)
    if not remaining:
        return 0.0
    texts = list(passages or [])
    used = 0
    # `texts` shrinks as passages are consumed, so the bound is the list itself, not a
    # fixed count — otherwise the loop stops one selection early.
    while remaining and texts:
        best_index, best_gain = None, 0
        for index, passage in enumerate(texts):
            gain = len(remaining & _tokens(_text_of(passage)))
            if gain > best_gain:
                best_index, best_gain = index, gain
        if best_index is None:
            break
        remaining -= _tokens(_text_of(texts[best_index]))
        texts.pop(best_index)
        used += 1
    return float(used)


def pairwise_redundancy(passages: Sequence[Any]) -> float:
    """Mean Jaccard overlap between every pair of passages (0 below two passages)."""
    texts = [_tokens(_text_of(passage)) for passage in passages or []]
    if len(texts) < 2:
        return 0.0
    total, pairs = 0.0, 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            union = texts[i] | texts[j]
            if union:
                total += len(texts[i] & texts[j]) / len(union)
            pairs += 1
    return total / pairs if pairs else 0.0


def extra_features(query: str, passages: Sequence[Any]) -> np.ndarray:
    """The three experimental features in :data:`EXTRA_NAMES` order."""
    return np.array(
        [
            max_passage_coverage(query, passages),
            coverage_spread(query, passages),
            pairwise_redundancy(passages),
        ],
        dtype=np.float64,
    )
