"""The RAG engine: index a corpus, then answer evidence requests over it.

The pipeline is deliberately two-stage:

1. **Dense recall.** Laya's checkpoint encoder embeds every chunk; the query is
   matched against them by cosine similarity. Cheap, broad, high recall.
2. **Laya adjudication.** Every candidate gets a calibrated relevance decision, and
   the surviving evidence set gets an answerability decision. Expensive, narrow,
   and — because Laya does not generate text — impossible to hallucinate through.

The engine returns *evidence*, not prose. Composing an answer is the caller's job —
the optional compose step, or a reading of the interface — which is why every
passage carries its source path and probabilities.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .chunking import DEFAULT_EXTENSIONS, MAX_FILE_BYTES, chunk_documents, load_documents
from .compose import Composer
from .config import EngineConfig
from .index_store import Hit, VectorIndex, wipe
from .lexical import LEXICAL_FILE, BM25Index, reciprocal_rank_fusion
from .model import LayaModel
from .rerank import ANSWERABILITY_QUESTION, LayaReranker


class IndexNotBuiltError(RuntimeError):
    """Raised when a query arrives before any index exists."""


@dataclass
class IndexReport:
    """Outcome of one indexing run."""

    documents: int = 0
    chunks: int = 0
    removed_chunks: int = 0
    total_chunks: int = 0
    paths: int = 0
    embed_seconds: float = 0.0
    save_seconds: float = 0.0
    total_seconds: float = 0.0
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "removed_chunks": self.removed_chunks,
            "total_chunks": self.total_chunks,
            "paths": self.paths,
            "embed_seconds": round(self.embed_seconds, 3),
            "save_seconds": round(self.save_seconds, 3),
            "total_seconds": round(self.total_seconds, 3),
            "skipped": [{"path": path, "reason": reason} for path, reason in self.skipped],
        }


@dataclass
class QueryResult:
    """Evidence for one query, with Laya's calibrated verdicts attached."""

    query: str
    answerable: bool = False
    answerability: float | None = None
    passages: list[dict[str, Any]] = field(default_factory=list)
    considered: int = 0
    kept: int = 0
    dropped_below_threshold: int = 0
    retrieval_seconds: float = 0.0
    rerank_seconds: float = 0.0
    total_seconds: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    checkpoint: str = ""
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": self.query,
            "answerable": self.answerable,
            "passages": self.passages,
            "considered": self.considered,
            "kept": self.kept,
            "dropped_below_threshold": self.dropped_below_threshold,
            "checkpoint": self.checkpoint,
            "timings": {
                "retrieval_seconds": round(self.retrieval_seconds, 4),
                "rerank_seconds": round(self.rerank_seconds, 4),
                "total_seconds": round(self.total_seconds, 4),
            },
            "usage": self.usage,
        }
        if self.answerability is not None:
            payload["answerability"] = round(self.answerability, 4)
        if self.note:
            payload["note"] = self.note
        return payload


def _batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class RagEngine:
    """Indexes documents with Laya embeddings and answers queries with Laya decisions."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        model: Any | None = None,
        embed_batch_texts: int = 256,
    ) -> None:
        self.config = config or EngineConfig()
        self.model = model or LayaModel(self.config)
        self.reranker = LayaReranker(self.model)
        # Optional generative step. It never runs on its own: the caller has to ask
        # for it, and everything it produces is unverified prose.
        self.composer = Composer(self.config)
        self.embed_batch_texts = embed_batch_texts
        self._index: VectorIndex | None = None
        self._lexical: BM25Index | None = None

    # -- index state -----------------------------------------------------
    @property
    def index(self) -> VectorIndex:
        if self._index is None:
            if not VectorIndex.exists_on_disk(self.config.index_dir):
                raise IndexNotBuiltError(
                    f"no index at {self.config.index_dir}. Index a directory first "
                    f"(laya-rag index <path>)."
                )
            self._index = VectorIndex.load(self.config.index_dir)
        return self._index

    def has_index(self) -> bool:
        return self._index is not None or VectorIndex.exists_on_disk(self.config.index_dir)

    @property
    def lexical(self) -> BM25Index | None:
        """The lexical recall channel, or None when it was never built."""
        if self._lexical is None:
            path = self.config.index_dir / LEXICAL_FILE
            if path.is_file():
                try:
                    self._lexical = BM25Index.load(path)
                except (OSError, ValueError, KeyError):
                    # A stale or damaged lexical index must not break vector search;
                    # the engine degrades to dense-only recall.
                    self._lexical = None
        return self._lexical

    def reset(self) -> None:
        """Delete the on-disk index and drop the in-memory copies."""
        wipe(self.config.index_dir)
        self._index = None
        self._lexical = None

    # -- indexing --------------------------------------------------------
    def index_paths(
        self,
        paths: Sequence[Path | str],
        rebuild: bool = False,
        extensions: Iterable[str] = DEFAULT_EXTENSIONS,
        max_file_bytes: int = MAX_FILE_BYTES,
    ) -> IndexReport:
        """Read, chunk and embed everything under ``paths``; then persist the index."""
        started = time.perf_counter()
        report = IndexReport()

        if rebuild:
            self.reset()

        # A first run has nothing on disk; start from an empty index rather than
        # failing, which is what the `index` property would do.
        index = VectorIndex() if rebuild or not self.has_index() else self.index

        resolved = [Path(p).expanduser() for p in paths]
        loaded = load_documents(resolved, extensions=extensions, max_file_bytes=max_file_bytes)
        report.skipped = list(loaded.skipped)

        chunks = chunk_documents(
            loaded.documents,
            chunk_chars=self.config.chunk_chars,
            overlap_chars=self.config.chunk_overlap_chars,
            min_chunk_chars=self.config.min_chunk_chars,
        )
        report.documents = len(loaded.documents)
        report.chunks = len(chunks)

        # A file that changed since the last run must not leave stale chunks behind.
        changed_paths = [doc.path for doc in loaded.documents]
        if changed_paths and len(index):
            report.removed_chunks = index.remove_paths(changed_paths)

        if chunks:
            embed_started = time.perf_counter()
            vectors: list[np.ndarray] = []
            for batch in _batched(chunks, self.embed_batch_texts):
                vectors.append(np.asarray(self.model.embed([chunk.text for chunk in batch]), dtype=np.float32))
            matrix = np.vstack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)
            report.embed_seconds = time.perf_counter() - embed_started
            index.add(chunks, matrix)

        save_started = time.perf_counter()
        index.save(self.config.index_dir)
        # The lexical channel is rebuilt from the full chunk set so it always
        # matches the vector index exactly.
        lexical = BM25Index()
        lexical.build(index.chunks)
        lexical.save(self.config.index_dir / LEXICAL_FILE)
        self._lexical = lexical
        report.save_seconds = time.perf_counter() - save_started

        report.total_chunks = len(index)
        report.paths = len(index.paths())
        report.total_seconds = time.perf_counter() - started
        self._index = index
        return report

    # -- querying --------------------------------------------------------
    def query(
        self,
        query: str,
        top_k: int | None = None,
        retrieve_k: int | None = None,
        relevance_threshold: float | None = None,
        answerability_threshold: float | None = None,
        max_passage_chars: int | None = 2000,
    ) -> QueryResult:
        """Retrieve, re-rank and gate evidence for ``query``."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        started = time.perf_counter()
        index = self.index
        if not len(index):
            return QueryResult(
                query=query,
                checkpoint=self.config.checkpoint,
                note="the index is empty; index some documents first",
                total_seconds=time.perf_counter() - started,
            )

        candidate_k = retrieve_k or self.config.retrieve_k
        retrieve_started = time.perf_counter()
        query_vector = self.model.embed_one(query)
        dense_hits = index.search(query_vector, k=candidate_k)

        # Hybrid recall: dense similarity plus BM25, fused by rank. Laya's encoder is
        # not retrieval-trained, so lexical recall is what keeps rare exact terms
        # ("LangChain", "Docker") inside the candidate set the re-ranker judges.
        lexical = self.lexical
        lexical_hits = lexical.search(query, k=candidate_k) if lexical else []
        candidates = self._fuse(index, dense_hits, lexical_hits, limit=candidate_k)
        retrieval_seconds = time.perf_counter() - retrieve_started

        # An index written before the lexical channel existed, or one whose
        # lexical.json was lost, degrades to dense-only recall. Say so rather than
        # returning quietly worse results.
        degraded = (
            "lexical recall channel is missing, so this search used dense recall only "
            "and will miss exact-term matches; re-index to rebuild it"
        )

        rerank_started = time.perf_counter()
        reranked = self.reranker.rerank(
            query,
            candidates,
            top_k=top_k or self.config.rerank_k,
            relevance_threshold=(
                self.config.relevance_threshold if relevance_threshold is None else relevance_threshold
            ),
            answerability_threshold=(
                self.config.answerability_threshold
                if answerability_threshold is None
                else answerability_threshold
            ),
        )
        rerank_seconds = time.perf_counter() - rerank_started

        return QueryResult(
            query=query,
            answerable=reranked.answerable,
            answerability=reranked.answerability,
            passages=[p.as_dict(max_chars=max_passage_chars) for p in reranked.passages],
            considered=reranked.considered,
            kept=reranked.kept,
            dropped_below_threshold=reranked.dropped_below_threshold,
            retrieval_seconds=retrieval_seconds,
            rerank_seconds=rerank_seconds,
            total_seconds=time.perf_counter() - started,
            usage=reranked.usage,
            checkpoint=self.config.checkpoint,
            note=None if lexical is not None or not len(index) else degraded,
        )

    @staticmethod
    def _fuse(
        index: VectorIndex,
        dense_hits: Sequence[Hit],
        lexical_hits: Sequence[Any],
        limit: int,
    ) -> list[Hit]:
        """Fuse dense and lexical rankings into one candidate list.

        The fused ``Hit.score`` is the RRF score, not a cosine similarity; the
        channels that produced each candidate travel with it so the evidence bundle
        can say where a passage came from.
        """
        dense_ids = [hit.chunk.chunk_id for hit in dense_hits]
        lexical_ids = [hit.chunk_id for hit in lexical_hits]
        fused = reciprocal_rank_fusion([dense_ids, lexical_ids], limit=limit)
        dense_set = set(dense_ids)
        lexical_set = set(lexical_ids)
        candidates: list[Hit] = []
        for chunk_id, rrf_score in fused:
            chunk = index.get(chunk_id)
            if chunk is None:  # pragma: no cover - a channel naming an unknown chunk
                continue
            channels = tuple(
                name
                for name, present in (("dense", chunk_id in dense_set), ("lexical", chunk_id in lexical_set))
                if present
            )
            candidates.append(Hit(chunk=chunk, score=rrf_score, channels=channels))
        return candidates

    # -- corpus introspection --------------------------------------------
    def list_documents(self) -> list[dict[str, Any]]:
        """Every indexed document, with its chunk and character counts.

        Read-only and index-optional: an unbuilt index yields an empty list rather
        than raising, because callers ask for this on every status refresh.

        ``chars`` sums each chunk's own length, so text duplicated by the chunk
        overlap is counted once per chunk — it measures indexed volume, not source
        file size.
        """
        if not self.has_index():
            return []
        grouped: dict[str, dict[str, Any]] = {}
        for chunk in self.index.chunks:
            record = grouped.setdefault(
                chunk.doc_id,
                {"doc_id": chunk.doc_id, "path": chunk.path, "chunks": 0, "chars": 0},
            )
            record["chunks"] += 1
            record["chars"] += len(chunk.text)
        return sorted(grouped.values(), key=lambda record: record["path"])

    def list_chunks(self, path: str | None = None) -> list[dict[str, Any]]:
        """Indexed chunks in document order, optionally narrowed to one path.

        The full chunk text is returned: chunks are bounded by ``chunk_chars``, so
        this stays small enough to hand straight to a viewer.
        """
        if not self.has_index():
            return []
        chunks = [chunk for chunk in self.index.chunks if path is None or chunk.path == path]
        chunks.sort(key=lambda chunk: (chunk.path, chunk.index))
        return [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "path": chunk.path,
                "index": chunk.index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "chars": len(chunk.text),
                "text": chunk.text,
            }
            for chunk in chunks
        ]

    # -- claim adjudication ----------------------------------------------
    def adjudicate_claims(
        self,
        claims: Sequence[str],
        retrieve_k: int | None = None,
        relevance_threshold: float | None = None,
        answerability_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Ask Laya whether the corpus supports each claim.

        The second half of verification. The citation check decides whether an answer
        *cites* evidence; this decides whether the corpus actually *supports* what it
        says, by re-retrieving candidates for every claim across the whole index and
        adjudicating them with Laya's calibrated decisions.

        Two decisions shape this:

        * **The corpus, not just the shown passages.** A claim can be true of the
          corpus and simply not have been in the five passages the model was handed.
          Re-retrieval is what tells those two cases apart.
        * **The already-measured answerability question.** This reuses
          ``ANSWERABILITY_QUESTION`` — "Can `query` be answered using only `evidence`?" —
          with the claim standing in as the query, rather than inventing a
          claim-support question. A new wording would carry a calibration nobody has
          measured; Laya's own docs warn that ``noul`` is sensitive to wording and that
          the question type and distribution in use must be tested. The report names the
          question it used so a caller cannot mistake this for a measured
          claim-support decision.

        Work is batched: one embedding pass for every claim, one relevance pass over
        every (claim, candidate) pair, one answerability pass over every surviving set.
        """
        started = time.perf_counter()
        index = self.index  # raises IndexNotBuiltError when nothing is indexed

        cleaned = [claim.strip() for claim in claims if claim and claim.strip()][: self.config.claim_max]
        candidate_k = retrieve_k or self.config.claim_retrieve_k
        relevance_bar = (
            self.config.relevance_threshold if relevance_threshold is None else relevance_threshold
        )
        gate = (
            self.config.answerability_threshold
            if answerability_threshold is None
            else answerability_threshold
        )
        keep_k = self.config.rerank_k

        empty: dict[str, Any] = {
            "claims": [],
            "total": 0,
            "supported": 0,
            "unsupported": 0,
            "not_measured": 0,
            "answerability_threshold": gate,
            "relevance_threshold": relevance_bar,
            "retrieve_k": candidate_k,
            "question": ANSWERABILITY_QUESTION["instructions"],
            "checkpoint": self.config.checkpoint,
            "note": "no claims to adjudicate" if not cleaned else "the index is empty",
            "timings": {"total_seconds": round(time.perf_counter() - started, 4)},
            "usage": {},
        }
        if not cleaned or not len(index):
            return empty

        # 1. Recall per claim, over the whole corpus.
        vectors = np.asarray(self.model.embed(cleaned), dtype=np.float32)
        lexical = self.lexical
        per_claim_hits: list[list[Hit]] = []
        for position, claim in enumerate(cleaned):
            dense = index.search(vectors[position], k=candidate_k)
            lexical_hits = lexical.search(claim, k=candidate_k) if lexical else []
            per_claim_hits.append(self._fuse(index, dense, lexical_hits, limit=candidate_k))

        # 2. One batched relevance pass across every (claim, candidate) pair.
        pairs: list[tuple[str, str]] = []
        for claim, hits in zip(cleaned, per_claim_hits):
            pairs.extend((claim, hit.chunk.text) for hit in hits)
        scores, usage = self.reranker.score_relevance_pairs(pairs)

        kept_per_claim: list[list[tuple[Hit, float]]] = []
        cursor = 0
        for hits in per_claim_hits:
            sliced = scores[cursor : cursor + len(hits)]
            cursor += len(hits)
            above = [
                (hit, score) for hit, score in zip(hits, sliced) if score >= relevance_bar
            ]
            above.sort(key=lambda item: (-item[1], -item[0].score))
            kept_per_claim.append(above[:keep_k])

        # 3. One batched answerability pass over the claims that have evidence. The
        #    others short-circuit, and their 0.0 is reported as unmeasured rather than
        #    drawn as a decision Laya never made.
        evidence_pairs: list[tuple[str, str]] = []
        positions: list[int] = []
        for position, kept in enumerate(kept_per_claim):
            if not kept:
                continue
            evidence = "\n\n".join(
                f"[{index + 1}] {hit.chunk.text}" for index, (hit, _score) in enumerate(kept)
            )
            evidence_pairs.append((cleaned[position], evidence))
            positions.append(position)
        probabilities, answer_usage = self.reranker.check_answerability_pairs(evidence_pairs)
        by_position = dict(zip(positions, probabilities))

        adjudicated: list[dict[str, Any]] = []
        supported = unsupported = not_measured = 0
        for position, (claim, kept) in enumerate(zip(cleaned, kept_per_claim)):
            measured = position in by_position
            probability = float(by_position[position]) if measured else 0.0
            is_supported = bool(measured and probability >= gate)
            if not measured:
                not_measured += 1
            elif is_supported:
                supported += 1
            else:
                unsupported += 1
            best = kept[0][1] if kept else 0.0
            adjudicated.append({
                "text": claim,
                "supported": is_supported,
                "measured": measured,
                "support_probability": round(probability, 4),
                "best_relevance": round(best, 4),
                "considered": len(per_claim_hits[position]),
                "kept": len(kept),
                "sources": [
                    {
                        "path": hit.chunk.path,
                        "chunk_id": hit.chunk.chunk_id,
                        "relevance": round(score, 4),
                        "channels": list(hit.channels),
                    }
                    for hit, score in kept
                ],
            })

        elapsed = time.perf_counter() - started
        merged_usage = dict(usage)
        for key in ("input_tokens", "output_tokens"):
            if key in answer_usage:
                merged_usage[key] = merged_usage.get(key, 0) + answer_usage[key]

        return {
            "claims": adjudicated,
            "total": len(adjudicated),
            "supported": supported,
            "unsupported": unsupported,
            "not_measured": not_measured,
            "answerability_threshold": gate,
            "relevance_threshold": relevance_bar,
            "retrieve_k": candidate_k,
            "question": ANSWERABILITY_QUESTION["instructions"],
            "checkpoint": self.config.checkpoint,
            "note": None if lexical is not None else (
                "lexical recall channel is missing, so claims were recalled densely only"
            ),
            "timings": {"total_seconds": round(elapsed, 4)},
            "usage": merged_usage,
        }

    # -- introspection ---------------------------------------------------
    def stats(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "config": self.config.as_dict(),
            "model": self.model.info() if hasattr(self.model, "info") else {"model": type(self.model).__name__},
            "index_dir": str(self.config.index_dir),
            "index_present": VectorIndex.exists_on_disk(self.config.index_dir),
            "lexical_present": (self._lexical is not None)
            or (self.config.index_dir / LEXICAL_FILE).is_file(),
            "compose": self.composer.describe(),
        }
        if payload["index_present"]:
            try:
                payload["index"] = self.index.stats()
            except Exception as exc:  # noqa: BLE001 - reported, not raised, for a status endpoint
                payload["index_error"] = str(exc)
        else:
            payload["index"] = {"chunks": 0, "documents": 0, "paths": 0}
        return payload
