"""Lexical recall: BM25 over the chunk corpus, plus rank fusion.

Laya's encoder is mean-pooled, not retrieval-trained, and its own README says a
dedicated bi-encoder "will usually shortlist better". Measured on this repository's
own documentation, dense cosine scores compressed into a 0.58-0.94 band and put the
expected file outside the top 20 for 2 of 5 unambiguous queries, so Laya's re-ranker
never saw the right passage.

BM25 fixes recall without adding a model: it is exact on rare terms, which is what
"LangChain", "Docker" and "Pydantic" are. Dense and lexical results are then fused
with Reciprocal Rank Fusion, which needs no score calibration between two
incomparable scales, and the fused set goes to Laya for adjudication.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

LEXICAL_FILE = "lexical.json"
LEXICAL_VERSION = 1

# Latin/digit runs become words; CJK and kana become single characters, which is
# the standard cheap way to keep scriptio-continua languages retrievable.
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]")

STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is",
    "it", "its", "me", "my", "no", "not", "of", "on", "or", "our", "so", "that",
    "the", "their", "them", "then", "there", "these", "they", "this", "to", "up",
    "was", "we", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "you", "your",
})

K1 = 1.2
B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with CJK characters kept individually."""
    return [t for t in _TOKEN_RE.findall(str(text).lower()) if len(t) > 1 or not t.isascii()]


@dataclass
class LexicalHit:
    """One BM25 result, identified by chunk id."""

    chunk_id: str
    score: float


class BM25Index:
    """An in-memory BM25 index over chunk texts, persistable as JSON.

    ``k1`` and ``b`` default to Lucene's values, which is what this engine's own
    hybrid recall was tuned against. They are parameters rather than constants
    because the published BM25 baselines on BEIR/Anserini use ``k1=0.9, b=0.4``,
    and comparing against them means running that configuration
    (``tools/beir.py`` does).
    """

    def __init__(self, k1: float = K1, b: float = B, drop_stopwords: bool = True) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        # The engine filters stopwords out of the query because its queries are short
        # natural-language questions. The published BM25 baselines do not, so a
        # benchmark comparison turns this off.
        self.drop_stopwords = bool(drop_stopwords)
        self._ids: list[str] = []
        self._term_freqs: list[Counter[str]] = []
        self._doc_freq: Counter[str] = Counter()
        self._lengths: list[int] = []
        self._avg_len: float = 0.0

    def __len__(self) -> int:
        return len(self._ids)

    def build(self, chunks: Sequence[Any]) -> None:
        """Index ``chunks`` (anything with ``chunk_id`` and ``text``)."""
        self._ids = []
        self._term_freqs = []
        self._doc_freq = Counter()
        self._lengths = []
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            counts = Counter(tokens)
            self._ids.append(chunk.chunk_id)
            self._term_freqs.append(counts)
            self._lengths.append(len(tokens))
            for term in counts:
                self._doc_freq[term] += 1
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0

    def _idf(self, term: str) -> float:
        n = len(self._ids)
        df = self._doc_freq.get(term, 0)
        if not n or not df:
            return 0.0
        # Lucene-style BM25 idf, floored so a very common term never scores negative.
        return max(math.log(1.0 + (n - df + 0.5) / (df + 0.5)), 0.0)

    def search(self, query: str, k: int = 20) -> list[LexicalHit]:
        """Return the ``k`` highest-scoring chunks for ``query``."""
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")
        if not self._ids:
            return []
        terms = tokenize(query)
        if self.drop_stopwords:
            terms = [t for t in terms if t not in STOPWORDS]
        if not terms:
            return []
        scores = [0.0] * len(self._ids)
        for term in terms:
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for row, counts in enumerate(self._term_freqs):
                freq = counts.get(term)
                if not freq:
                    continue
                length = self._lengths[row] or 1
                denom = freq + self.k1 * (1.0 - self.b + self.b * length / (self._avg_len or 1.0))
                scores[row] += idf * (freq * (self.k1 + 1.0)) / denom
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [LexicalHit(chunk_id=self._ids[i], score=scores[i]) for i in ranked if scores[i] > 0.0]

    # -- persistence -----------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": LEXICAL_VERSION,
            "ids": self._ids,
            "term_freqs": [dict(counts) for counts in self._term_freqs],
            "doc_freq": dict(self._doc_freq),
            "lengths": self._lengths,
            "avg_len": self._avg_len,
            "k1": self.k1,
            "b": self.b,
            "drop_stopwords": self.drop_stopwords,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != LEXICAL_VERSION:
            raise ValueError(
                f"lexical index version {payload.get('version')!r} is unsupported "
                f"(expected {LEXICAL_VERSION}); rebuild the index"
            )
        index = cls()
        index._ids = list(payload["ids"])
        index._term_freqs = [Counter(counts) for counts in payload["term_freqs"]]
        index._doc_freq = Counter(payload["doc_freq"])
        index._lengths = list(payload["lengths"])
        index._avg_len = float(payload.get("avg_len", 0.0))
        index.k1 = float(payload.get("k1", K1))
        index.b = float(payload.get("b", B))
        index.drop_stopwords = bool(payload.get("drop_stopwords", True))
        return index


def reciprocal_rank_fusion(
    channels: Iterable[Sequence[str]],
    k: int = 60,
    limit: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists, returning ``(id, score)`` best-first.

    RRF scores each document by ``sum(1 / (k + rank))`` across channels. It is
    rank-based, so it never has to reconcile BM25's unbounded scores with cosine
    similarity in [-1, 1].
    """
    scores: dict[str, float] = {}
    for channel in channels:
        for rank, identifier in enumerate(channel):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:limit] if limit is not None else ordered
