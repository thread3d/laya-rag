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

import numpy as np

LEXICAL_FILE = "lexical.json"
LEXICAL_VERSION = 1
#: The eager backend writes two files: the precomputed posting arrays and their metadata.
EAGER_LEXICAL_FILE = "lexical-eager.npz"
EAGER_LEXICAL_META = "lexical-eager.meta.json"
EAGER_LEXICAL_VERSION = 1

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

# Optional English stemming. Measured on BEIR, stemming the corpus and the queries closes
# most of the gap to the published Lucene BM25 baseline, which stems: SciFact nDCG@10
# 0.654 -> 0.677 and NFCorpus 0.304 -> 0.324. It is off by default because this engine
# also indexes source code and configuration, where `user_id` and `user_ids` are
# different terms; `LAYA_RAG_LEXICAL_STEM=true` turns it on. The setting is stored with
# the index, so a query is always stemmed the way its corpus was.
_stemmer: Any | None = None


def _stem_word(token: str) -> str:
    """Stem one ASCII alphabetic token; leave identifiers, digits and CJK alone."""
    global _stemmer
    if not token.isascii() or not token.isalpha():
        return token
    if _stemmer is None:
        import snowballstemmer

        _stemmer = snowballstemmer.stemmer("english")
    return _stemmer.stemWord(token)


def tokenize(text: str, stem: bool = False) -> list[str]:
    """Lowercase word tokens with CJK characters kept individually, optionally stemmed."""
    tokens = [t for t in _TOKEN_RE.findall(str(text).lower()) if len(t) > 1 or not t.isascii()]
    if stem:
        tokens = [_stem_word(token) for token in tokens]
    return tokens


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

    def __init__(
        self,
        k1: float = K1,
        b: float = B,
        drop_stopwords: bool = True,
        stem: bool = False,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        # The engine filters stopwords out of the query because its queries are short
        # natural-language questions. The published BM25 baselines do not, so a
        # benchmark comparison turns this off.
        self.drop_stopwords = bool(drop_stopwords)
        # Stored with the index so queries are stemmed the same way the corpus was.
        self.stem = bool(stem)
        self._ids: list[str] = []
        self._term_freqs: list[Counter[str]] = []
        self._doc_freq: Counter[str] = Counter()
        self._lengths: list[int] = []
        self._avg_len: float = 0.0
        # term -> [(row, frequency)]. Scoring used to loop over every document for every
        # query term, which is O(terms x documents): fine at a few thousand chunks, but on
        # ArguAna — where a query is a ~190-word argument — it is billions of iterations
        # and minutes per query set. Postings make it O(matching documents) instead, which
        # is the same index Lucene keeps and the difference between "slow" and "usable".
        self._postings: dict[str, list[tuple[int, int]]] = {}

    def __len__(self) -> int:
        return len(self._ids)

    def _build_postings(self) -> None:
        """Invert ``_term_freqs`` into term -> [(row, frequency)]."""
        postings: dict[str, list[tuple[int, int]]] = {}
        for row, counts in enumerate(self._term_freqs):
            for term, frequency in counts.items():
                postings.setdefault(term, []).append((row, frequency))
        self._postings = postings

    def build(self, chunks: Sequence[Any]) -> None:
        """Index ``chunks`` (anything with ``chunk_id`` and ``text``)."""
        self._ids = []
        self._term_freqs = []
        self._doc_freq = Counter()
        self._lengths = []
        for chunk in chunks:
            tokens = tokenize(chunk.text, stem=self.stem)
            counts = Counter(tokens)
            self._ids.append(chunk.chunk_id)
            self._term_freqs.append(counts)
            self._lengths.append(len(tokens))
            for term in counts:
                self._doc_freq[term] += 1
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._build_postings()

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
        terms = tokenize(query, stem=self.stem)
        if self.drop_stopwords:
            terms = [t for t in terms if t not in STOPWORDS]
        if not terms:
            return []
        scores = [0.0] * len(self._ids)
        for term in terms:
            idf = self._idf(term)
            if idf == 0.0:
                continue
            postings = self._postings.get(term)
            if not postings:
                continue
            # Only the documents that actually contain the term are touched.
            for row, freq in postings:
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
            "stem": self.stem,
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
        # Absent means an index written before stemming existed, which is an unstemmed one.
        index.stem = bool(payload.get("stem", False))
        index._build_postings()
        return index


class EagerBM25Index:
    """BM25 with eager per-posting weights, scored by vectorised accumulation.

    ``BM25Index`` stores per-document term frequencies and scores by looping over the
    postings of every query term in Python. This precomputes each posting's final BM25
    weight once (BM25S's "eager scoring" idea) and stores the postings as NumPy arrays, so
    a query is a handful of slice-adds instead of a Python inner loop, and the resident
    index is a few bytes per posting rather than a tuple per posting.

    The ranking is **identical** to :class:`BM25Index` for the same ``k1``/``b``/``stem``/
    ``drop_stopwords`` settings: the weights are computed from the same formula and
    accumulated in the same query-term order, and ties break to the lower document index
    exactly as the stable sort there does. Only the accumulation mechanism differs.
    """

    def __init__(
        self,
        k1: float = K1,
        b: float = B,
        drop_stopwords: bool = True,
        stem: bool = False,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.drop_stopwords = bool(drop_stopwords)
        self.stem = bool(stem)
        self._ids: list[str] = []
        #: term -> (document rows int32, precomputed BM25 weights float64)
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._avg_len: float = 0.0

    def __len__(self) -> int:
        return len(self._ids)

    def build(self, chunks: Sequence[Any]) -> None:
        """Index ``chunks`` (anything with ``chunk_id`` and ``text``)."""
        ids: list[str] = []
        term_freqs: list[Counter[str]] = []
        lengths: list[int] = []
        doc_freq: Counter[str] = Counter()
        for chunk in chunks:
            tokens = tokenize(chunk.text, stem=self.stem)
            counts = Counter(tokens)
            ids.append(chunk.chunk_id)
            term_freqs.append(counts)
            lengths.append(len(tokens))
            for term in counts:
                doc_freq[term] += 1
        n = len(ids)
        avg_len = (sum(lengths) / n) if n else 0.0

        postings: dict[str, list[tuple[int, float]]] = {}
        for row, counts in enumerate(term_freqs):
            length = lengths[row] or 1
            norm = self.k1 * (1.0 - self.b + self.b * length / (avg_len or 1.0))
            for term, freq in counts.items():
                df = doc_freq[term]
                idf = max(math.log(1.0 + (n - df + 0.5) / (df + 0.5)), 0.0)
                weight = idf * (freq * (self.k1 + 1.0)) / (freq + norm)
                postings.setdefault(term, []).append((row, weight))

        self._ids = ids
        self._avg_len = avg_len
        self._postings = {
            term: (
                np.asarray([row for row, _ in entries], dtype=np.int32),
                np.asarray([weight for _, weight in entries], dtype=np.float64),
            )
            for term, entries in postings.items()
        }

    def search(self, query: str, k: int = 20) -> list[LexicalHit]:
        """Return the ``k`` highest-scoring chunks for ``query``."""
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")
        if not self._ids:
            return []
        terms = tokenize(query, stem=self.stem)
        if self.drop_stopwords:
            terms = [t for t in terms if t not in STOPWORDS]
        if not terms:
            return []
        scores = np.zeros(len(self._ids), dtype=np.float64)
        for term in terms:
            entry = self._postings.get(term)
            if entry is None:
                continue
            rows, weights = entry
            scores[rows] += weights
        # Stable descending sort: ties keep the lower document index, matching the
        # stable sort BM25Index applies to the same scores.
        order = np.argsort(-scores, kind="stable")[:k]
        return [
            LexicalHit(chunk_id=self._ids[int(row)], score=float(scores[int(row)]))
            for row in order
            if scores[int(row)] > 0.0
        ]

    # -- persistence -----------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        terms = list(self._postings)
        offsets = [0]
        docs: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for term in terms:
            rows, term_weights = self._postings[term]
            docs.append(rows)
            weights.append(term_weights)
            offsets.append(offsets[-1] + len(rows))
        arrays = {
            "offsets": np.asarray(offsets, dtype=np.int64),
            "docs": np.concatenate(docs) if docs else np.zeros(0, dtype=np.int32),
            "weights": np.concatenate(weights) if weights else np.zeros(0, dtype=np.float64),
        }
        meta = {
            "version": EAGER_LEXICAL_VERSION,
            "ids": self._ids,
            "terms": terms,
            "avg_len": self._avg_len,
            "k1": self.k1,
            "b": self.b,
            "drop_stopwords": self.drop_stopwords,
            "stem": self.stem,
        }
        tmp_npz = path.with_name(path.name + ".tmp.npz")
        np.savez(tmp_npz, **arrays)
        tmp_npz.replace(path)
        tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
        tmp_meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        tmp_meta.replace(meta_path)

    @classmethod
    def load(cls, path: Path) -> "EagerBM25Index":
        path = Path(path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("version") != EAGER_LEXICAL_VERSION:
            raise ValueError(
                f"eager lexical index version {meta.get('version')!r} is unsupported "
                f"(expected {EAGER_LEXICAL_VERSION}); rebuild the index"
            )
        data = np.load(path)
        offsets = data["offsets"]
        docs = data["docs"]
        weights = data["weights"]
        index = cls(
            k1=float(meta.get("k1", K1)),
            b=float(meta.get("b", B)),
            drop_stopwords=bool(meta.get("drop_stopwords", True)),
            stem=bool(meta.get("stem", False)),
        )
        index._ids = list(meta["ids"])
        index._avg_len = float(meta.get("avg_len", 0.0))
        index._postings = {}
        for position, term in enumerate(meta["terms"]):
            start, end = int(offsets[position]), int(offsets[position + 1])
            index._postings[str(term)] = (docs[start:end], weights[start:end])
        return index


def reciprocal_rank_fusion(
    channels: Iterable[Sequence[str]],
    k: int = 60,
    limit: int | None = None,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists, returning ``(id, score)`` best-first.

    RRF scores each document by ``sum(weight_c / (k + rank))`` across channels. It is
    rank-based, so it never has to reconcile BM25's unbounded scores with cosine
    similarity in [-1, 1].

    ``weights`` exists because plain RRF assumes the channels are comparably good. When
    one channel is much weaker, an equal-weight sum lets it dilute the other: measured
    on BEIR, fusing a near-chance dense channel with BM25 cost ~43% of nDCG@10. A weight
    of 0 drops a channel entirely.
    """
    channel_list = [list(channel) for channel in channels]
    if weights is None:
        weights = [1.0] * len(channel_list)
    elif len(weights) != len(channel_list):
        raise ValueError(f"got {len(channel_list)} channels but {len(weights)} weights")

    scores: dict[str, float] = {}
    for channel, weight in zip(channel_list, weights):
        if weight == 0.0:
            continue
        for rank, identifier in enumerate(channel):
            scores[identifier] = scores.get(identifier, 0.0) + weight / (k + rank + 1)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:limit] if limit is not None else ordered
