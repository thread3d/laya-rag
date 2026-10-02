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

import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .chunking import DEFAULT_EXTENSIONS, MAX_FILE_BYTES, chunk_documents, load_documents
from .compose import Composer
from .config import EngineConfig
from .index_store import Hit, VectorIndex, wipe
from .lexical import LEXICAL_FILE, BM25Index, reciprocal_rank_fusion
from .model import BiEncoderModel, CrossEncoderModel, LayaModel
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
    #: False when the relevance gate emptied the evidence set, so ``answerability`` is a
    #: structural zero rather than a measured decision. Calibration must not mix the two.
    answerability_measured: bool = False
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
            "answerability_measured": self.answerability_measured,
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


#: Called with one counter event while an index build runs. ``stage`` is one of
#: ``reading``, ``chunking``, ``embedding`` or ``saving``; the counts are ``documents``
#: read (or chunked), ``documents_total``, ``skipped``, ``chunks`` produced (the total
#: to embed during the embedding stage), ``embedded`` so far, and ``path`` of the file
#: just read. Keys a stage does not use are present and zero, so a consumer can read
#: the same shape every time.
ProgressFn = Callable[[dict[str, Any]], None]

_PROGRESS_DEFAULTS: dict[str, Any] = {
    "documents": 0,
    "documents_total": 0,
    "skipped": 0,
    "chunks": 0,
    "embedded": 0,
    "path": "",
}


def _progress(progress: ProgressFn | None, stage: str, **counters: Any) -> None:
    if progress is None:
        return
    progress({"stage": stage, **_PROGRESS_DEFAULTS, **counters})


def _resolved(path: Path | str) -> Path:
    """Absolute, symlink-resolved form, for comparing indexed paths with roots."""
    try:
        return Path(path).expanduser().resolve()
    except OSError:  # pragma: no cover - resolve(strict=False) rarely raises
        return Path(path).expanduser().absolute()


def _stale_indexed_paths(
    indexed_paths: Iterable[str],
    roots: Sequence[Path],
    skipped_paths: Iterable[str],
) -> list[str]:
    """Indexed paths under a re-indexed directory that no longer deserve to be there.

    A deleted file is never handed to the loader, so incremental indexing replaces the
    chunks of files it *does* see and silently leaves the rest. Restricting pruning to
    directory roots (never a single-file root) keeps the operation safe: a file passed on
    its own may simply not have been part of this run, but a directory is a claim about
    its whole contents. A path counts as stale when its file is gone, or when the loader
    refused it this run (unreadable, empty, binary, over the size limit).
    """
    resolved_roots = [_resolved(root) for root in roots if Path(root).expanduser().is_dir()]
    if not resolved_roots:
        return []
    refused = {str(Path(path)) for path in skipped_paths}
    stale: list[str] = []
    for path in indexed_paths:
        candidate = Path(path)
        resolved = _resolved(candidate)
        if not any(resolved == root or resolved.is_relative_to(root) for root in resolved_roots):
            continue
        if str(candidate) in refused or not candidate.is_file():
            stale.append(path)
    return stale


def _suppress_near_duplicates(
    index: VectorIndex,
    fused: Sequence[tuple[str, float]],
    limit: int,
    threshold: float,
) -> list[tuple[str, float]]:
    """Keep the fused order but drop chunks too close to one already kept.

    Chunk overlap means a matching passage can arrive several times under different chunk
    ids. This keeps the highest-ranked representative and spends the freed slots on new
    candidates, which is what recovered MAP@10 on MultiHop-RAG. A chunk with no vector is
    kept rather than dropped, so a missing embedding can never empty the candidate list.
    """
    kept: list[tuple[str, float]] = []
    kept_vectors: list[np.ndarray] = []
    for chunk_id, score in fused:
        vector = index.vector(chunk_id)
        if vector is not None and kept_vectors:
            similarity = max(float(np.dot(vector, other)) for other in kept_vectors)
            if similarity >= threshold:
                continue
        kept.append((chunk_id, score))
        if vector is not None:
            kept_vectors.append(vector)
        if len(kept) >= limit:
            break
    return kept


_ATOM_SPLIT = re.compile(r"(?<=[.!?])|\n")
_ATOM_MIN_CHARS = 25


def _chunk_atoms(text: str) -> list[str]:
    """Sentence-like "fact atoms" of a chunk, normalised and de-duplicated in order."""
    atoms: list[str] = []
    seen: set[str] = set()
    for part in _ATOM_SPLIT.split(str(text)):
        normalised = " ".join(part.lower().split())
        if len(normalised) < _ATOM_MIN_CHARS or normalised in seen:
            continue
        seen.add(normalised)
        atoms.append(normalised)
    return atoms


def _coverage_select(
    index: VectorIndex,
    fused: Sequence[tuple[str, float]],
    limit: int,
    lam: float,
) -> list[tuple[str, float]]:
    """Spend the recall budget on new facts, not near-duplicates of kept passages.

    Greedy submodular max-coverage over sentence atoms: at each step a candidate scores
    ``lam * relevance + (1 - lam) * marginal_new_atom_weight - (1 - lam) * redundancy``,
    where relevance is the normalised fused RRF score, marginal weight is the atom
    weight not yet covered (normalised by the best available gain so the two terms are
    comparable), and redundancy is the cosine to the closest kept chunk. Measured on
    MultiHop-RAG this lifts MAP@10 from 0.2546 to 0.2835 (held-out +0.0288), the best
    coverage pass measured — better than near-duplicate suppression alone (0.2790) —
    because it targets the *set* of facts the benchmark scores rather than mere
    diversity. Missing atom weights or vectors never remove a candidate.
    """
    if limit <= 0 or not fused:
        return list(fused[:limit])
    weights: list[float] = []
    atom_id_of: dict[str, int] = {}
    candidate_atoms: list[list[int]] = []
    for chunk_id, _score in fused:
        chunk = index.get(chunk_id)
        ids: list[int] = []
        if chunk is not None:
            for atom in _chunk_atoms(chunk.text):
                atom_id = atom_id_of.get(atom)
                if atom_id is None:
                    atom_id = len(weights)
                    atom_id_of[atom] = atom_id
                    weights.append(float(len(atom)))
                ids.append(atom_id)
        candidate_atoms.append(ids)

    relevance = np.asarray([score for _chunk_id, score in fused], dtype=np.float64)
    if relevance.size and float(relevance.max()) > float(relevance.min()):
        relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min())
    else:
        relevance = np.zeros_like(relevance)
    vectors = [index.vector(chunk_id) for chunk_id, _score in fused]

    selected: list[int] = []
    covered: set[int] = set()
    candidates = list(range(len(fused)))
    while candidates and len(selected) < limit:
        gains = {
            i: float(sum(weights[a] for a in candidate_atoms[i] if a not in covered))
            for i in candidates
        }
        scale = max(gains.values()) if gains else 0.0
        scale = scale if scale > 0.0 else 1.0
        best_index, best_score = None, -1e18
        for i in candidates:
            marginal = gains[i] / scale
            redundancy = 0.0
            if selected:
                sims = [
                    float(np.dot(vectors[i], vectors[j]))
                    for j in selected
                    if vectors[i] is not None and vectors[j] is not None
                ]
                redundancy = max(sims) if sims else 0.0
            score = lam * float(relevance[i]) + (1.0 - lam) * marginal - (1.0 - lam) * redundancy
            if score > best_score:
                best_index, best_score = i, score
        selected.append(best_index)
        covered.update(candidate_atoms[best_index])
        candidates.remove(best_index)
    return [fused[i] for i in selected]


class RagEngine:
    """Indexes documents with Laya embeddings and answers queries with Laya decisions."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        model: Any | None = None,
        embedder: Any | None = None,
        embed_batch_texts: int = 64,
    ) -> None:
        self.config = config or EngineConfig()
        # Laya owns the decisions (relevance, answerability) and, by default, the dense
        # embeddings too. A configured retrieval-trained embedder replaces only the
        # embeddings, so the two concerns can be tuned independently.
        self.model = model or LayaModel(self.config)
        if embedder is not None:
            self.embedder = embedder
        elif (self.config.embed_model or "").strip():
            self.embedder = BiEncoderModel(self.config)
        else:
            self.embedder = self.model
        self.reranker = LayaReranker(self.model)
        # Optional retrieval-trained cross-encoder for ordering and filtering. Laya keeps
        # the answerability gate either way, and `adjudicate_claims` is unchanged.
        self.cross_encoder = (
            CrossEncoderModel(self.config)
            if (self.config.rerank_model or "").strip()
            else None
        )
        # Optional generative step. It never runs on its own: the caller has to ask
        # for it, and everything it produces is unverified prose.
        self.composer = Composer(self.config)
        # How many chunks one `embed` call is handed at a time. The models re-batch
        # internally at `embed_batch_size`, so this only sets how often indexing
        # progress is reported: measured warm, 64 and 256 cost the same, and 64 gives
        # four updates where 256 gives one on a small corpus.
        self.embed_batch_texts = embed_batch_texts
        self._index: Any | None = None
        self._lexical: Any | None = None
        # Optional persisted feature gate for answerability (see config). Loaded lazily:
        # a missing or damaged file must not stop the engine from answering, it only
        # falls back to Laya's own decision.
        self._answerability_gate: Any | None = None
        self._gate_loaded = False
        self._answerability_nli: Any | None = None
        self._nli_loaded = False

    # -- index state -----------------------------------------------------
    @property
    def index(self) -> Any:
        if self._index is None:
            if self.config.resolved_index_backend == "ondisk":
                from .ondisk import DENSE_DIRNAME, OnDiskVectorIndex

                directory = self.config.index_dir / DENSE_DIRNAME
                if not OnDiskVectorIndex.exists_on_disk(directory):
                    raise IndexNotBuiltError(
                        f"no on-disk index at {directory}. Index a directory first "
                        f"(laya-rag index <path> --rebuild)."
                    )
                self._index = OnDiskVectorIndex.open(directory)
            else:
                if not VectorIndex.exists_on_disk(self.config.index_dir):
                    raise IndexNotBuiltError(
                        f"no index at {self.config.index_dir}. Index a directory first "
                        f"(laya-rag index <path>)."
                    )
                self._index = VectorIndex.load(self.config.index_dir)
        return self._index

    def has_index(self) -> bool:
        if self._index is not None:
            return True
        if self.config.resolved_index_backend == "ondisk":
            from .ondisk import DENSE_DIRNAME, OnDiskVectorIndex

            return OnDiskVectorIndex.exists_on_disk(self.config.index_dir / DENSE_DIRNAME)
        return VectorIndex.exists_on_disk(self.config.index_dir)

    @property
    def lexical(self) -> Any | None:
        """The lexical recall channel, or None when it was never built."""
        if self._lexical is None:
            if self.config.resolved_index_backend == "ondisk":
                from .ondisk import BM25_DIRNAME, OnDiskBM25

                directory = self.config.index_dir / BM25_DIRNAME
                if OnDiskBM25.exists_on_disk(directory):
                    try:
                        self._lexical = OnDiskBM25.open(directory)
                    except (OSError, ValueError, KeyError):
                        self._lexical = None
            else:
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

    @property
    def answerability_gate(self) -> Any | None:
        """The optional trained answerability gate, or None when unset/unreadable."""
        if not self._gate_loaded:
            self._gate_loaded = True
            raw = (self.config.answerability_gate_path or "").strip()
            if raw:
                path = Path(raw).expanduser()
                if path.is_file():
                    from .answerability import LogisticGate

                    try:
                        self._answerability_gate = LogisticGate.load(path)
                    except (OSError, ValueError, KeyError):
                        # A stale or damaged gate degrades to Laya's own decision rather
                        # than breaking every query.
                        self._answerability_gate = None
        return self._answerability_gate

    @property
    def answerability_nli(self) -> Any | None:
        """The optional NLI answerability scorer, or None when not configured.

        Constructed lazily and *not* loaded here: weights are fetched on the first score,
        so a status call never triggers a model download. It has higher precedence than
        the feature gate and than Laya's own question.
        """
        if not self._nli_loaded:
            self._nli_loaded = True
            raw = (self.config.answerability_nli_model or "").strip()
            if raw:
                from .answerability import NLIAnswerability

                self._answerability_nli = NLIAnswerability(raw, device=self.config.device)
        return self._answerability_nli

    # -- indexing --------------------------------------------------------
    def index_paths(
        self,
        paths: Sequence[Path | str],
        rebuild: bool = False,
        extensions: Iterable[str] = DEFAULT_EXTENSIONS,
        max_file_bytes: int = MAX_FILE_BYTES,
        progress: ProgressFn | None = None,
        prune: bool = True,
    ) -> IndexReport:
        """Read, chunk and embed everything under ``paths``; then persist the index.

        ``progress`` is called with a counter event as the work happens — one per file
        read, one per document chunked, one per embedding batch — so a caller can show
        running numbers rather than an opaque wait. It is never required and never
        affects the result.

        ``prune`` (the default) drops the chunks of files that have disappeared from a
        directory being re-indexed; pass ``False`` to keep the old append-only
        behaviour where a deleted file's chunks survive until a rebuild.
        """
        started = time.perf_counter()
        report = IndexReport()

        if rebuild:
            self.reset()

        _progress(progress, "reading")
        resolved = [Path(p).expanduser() for p in paths]

        def reading(document: Any, documents: int, skipped: int) -> None:
            _progress(progress, "reading", documents=documents, skipped=skipped, path=document.path)

        loaded = load_documents(
            resolved,
            extensions=extensions,
            max_file_bytes=max_file_bytes,
            on_document=reading,
        )
        report.skipped = list(loaded.skipped)

        # The total to embed is only known once chunking has produced it.
        _progress(progress, "chunking", documents_total=len(loaded.documents))

        def chunking(done: int, total: int, chunks: int) -> None:
            _progress(progress, "chunking", documents=done, documents_total=total, chunks=chunks)

        chunks = chunk_documents(
            loaded.documents,
            chunk_chars=self.config.chunk_chars,
            overlap_chars=self.config.resolved_chunk_overlap,
            min_chunk_chars=self.config.min_chunk_chars,
            on_document=chunking,
        )
        report.documents = len(loaded.documents)
        report.chunks = len(chunks)

        if self.config.resolved_index_backend == "ondisk":
            self._build_ondisk(chunks, report, progress, rebuild)
            report.total_seconds = time.perf_counter() - started
            return report

        # A first run has nothing on disk; start from an empty index rather than
        # failing, which is what the `index` property would do.
        index = VectorIndex() if rebuild or not self.has_index() else self.index

        # A file that changed since the last run must not leave stale chunks behind.
        changed_paths = [doc.path for doc in loaded.documents]
        if changed_paths and len(index):
            report.removed_chunks = index.remove_paths(changed_paths)

        # A file that was deleted (or made unreadable) is never loaded, so it is not in
        # `changed_paths` and its chunks would otherwise stay retrievable forever. When a
        # directory is re-indexed, prune the paths under it that are gone or refused.
        if prune and len(index):
            stale = _stale_indexed_paths(
                index.paths(),
                resolved,
                [path for path, _reason in loaded.skipped],
            )
            if stale:
                report.removed_chunks += index.remove_paths(stale)

        if chunks:
            embed_started = time.perf_counter()
            # Preallocate the final matrix and fill it per batch. The previous version
            # kept every batch's array in a list and `np.vstack`ed at the end, which held
            # roughly two full copies of the corpus embeddings at peak — the difference
            # between an index that fits in RAM and one that does not at millions of
            # chunks. The embedder's dimension is only known from the first batch.
            matrix: np.ndarray | None = None
            embedded = 0
            _progress(progress, "embedding", chunks=len(chunks), embedded=0)
            for batch in _batched(chunks, self.embed_batch_texts):
                block = np.asarray(
                    self.embedder.embed([chunk.text for chunk in batch]), dtype=np.float32
                )
                if matrix is None:
                    matrix = np.empty((len(chunks), block.shape[1]), dtype=np.float32)
                matrix[embedded : embedded + len(batch)] = block
                embedded += len(batch)
                _progress(progress, "embedding", chunks=len(chunks), embedded=embedded)
            if matrix is None:
                matrix = np.zeros((0, 0), dtype=np.float32)
            report.embed_seconds = time.perf_counter() - embed_started
            index.add(chunks, matrix)
            if self.config.ann:
                index.ann_nprobe = self.config.ann_nprobe
                index.build_ann()

        _progress(progress, "saving", chunks=len(chunks), embedded=len(chunks))
        save_started = time.perf_counter()
        index.save(self.config.index_dir)
        # The lexical channel is rebuilt from the full chunk set so it always
        # matches the vector index exactly.
        lexical = BM25Index(stem=self.config.lexical_stem)
        lexical.build(index.chunks)
        lexical.save(self.config.index_dir / LEXICAL_FILE)
        self._lexical = lexical
        report.save_seconds = time.perf_counter() - save_started

        report.total_chunks = len(index)
        report.paths = len(index.paths())
        report.total_seconds = time.perf_counter() - started
        self._index = index
        return report

    def _build_ondisk(
        self,
        chunks: Sequence[Any],
        report: IndexReport,
        progress: ProgressFn | None,
        rebuild: bool,
    ) -> None:
        """Build the memory-mapped dense + packed-postings backends.

        The on-disk format is append-only, so an incremental pass over new paths would
        silently drop the existing chunks; the engine refuses it and the caller rebuilds.
        Embedding is streamed batch by batch, so the vector matrix is never resident.
        """
        from .ondisk import BM25_DIRNAME, DENSE_DIRNAME, OnDiskBM25, OnDiskVectorIndex

        if self.has_index() and not rebuild:
            raise ValueError(
                "the ondisk index backend is append-only and cannot re-index a path "
                "incrementally; call index_paths(..., rebuild=True)"
            )
        dense_dir = self.config.index_dir / DENSE_DIRNAME
        bm25_dir = self.config.index_dir / BM25_DIRNAME
        for directory in (dense_dir, bm25_dir):
            if directory.exists():
                shutil.rmtree(directory)

        dense = OnDiskVectorIndex(dense_dir, dim=0)
        embed_started = time.perf_counter()
        embedded = 0
        _progress(progress, "embedding", chunks=len(chunks), embedded=0)
        for batch in _batched(chunks, self.embed_batch_texts):
            vectors = np.asarray(
                self.embedder.embed([chunk.text for chunk in batch]), dtype=np.float32
            )
            dense.add(batch, vectors)
            embedded += len(batch)
            _progress(progress, "embedding", chunks=len(chunks), embedded=embedded)
        dense.finalize()
        report.embed_seconds = time.perf_counter() - embed_started
        # Re-open so the freshly built index is query-ready: the builder only wrote the
        # files, the memory maps and the deferred-text offset table come from `open`.
        dense = OnDiskVectorIndex.open(dense_dir)
        if self.config.ann:
            dense.nprobe = self.config.ann_nprobe
            dense.build_ann()

        save_started = time.perf_counter()
        _progress(progress, "saving", chunks=len(chunks), embedded=len(chunks))
        lexical = OnDiskBM25(bm25_dir, stem=self.config.lexical_stem)
        lexical.build((chunk.chunk_id, chunk.text) for chunk in chunks)
        report.save_seconds = time.perf_counter() - save_started

        self._index = dense
        self._lexical = lexical
        report.removed_chunks = 0
        report.total_chunks = len(chunks)
        report.paths = len({chunk.path for chunk in chunks})

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

        # Hybrid recall: dense similarity plus BM25, fused by rank. Which channels run
        # is a configuration question, not a constant: Laya's own encoder is not
        # retrieval-trained, and a dense weight of zero means the channel cannot affect
        # the ranking, so its query embedding — the expensive part — is skipped.
        #
        # If the lexical channel is missing, the dense channel has to carry the search
        # even when it is otherwise weighted to zero; leaving the weight at zero there
        # would fuse nothing and return an empty result.
        lexical = self.lexical
        dense_weight = self.config.resolved_dense_weight
        if lexical is None:
            dense_weight = 1.0
        # Diversification can only replace a duplicate with a deeper candidate if the
        # channels were searched deeper to begin with.
        recall_k = (
            candidate_k * 3
            if (self.config.recall_dedup_threshold > 0.0 or self.config.recall_coverage_lambda > 0.0)
            else candidate_k
        )
        if dense_weight == 0.0:
            dense_hits: list[Hit] = []
        else:
            dense_hits = index.search(self.embedder.embed_one(query), k=recall_k)
        lexical_hits = lexical.search(query, k=recall_k) if lexical else []
        candidates = self._fuse(
            index,
            dense_hits,
            lexical_hits,
            limit=candidate_k,
            dense_weight=dense_weight,
            lexical_weight=self.config.lexical_weight,
            dedup_threshold=self.config.recall_dedup_threshold,
            coverage_lambda=self.config.recall_coverage_lambda,
        )
        retrieval_seconds = time.perf_counter() - retrieve_started

        # An index written before the lexical channel existed, or one whose
        # lexical.json was lost, degrades to dense-only recall. Say so rather than
        # returning quietly worse results.
        degraded = (
            "lexical recall channel is missing, so this search used dense recall only "
            "and will miss exact-term matches; re-index to rebuild it"
        )

        rerank_started = time.perf_counter()
        # A configured cross-encoder replaces Laya's relevance scores for ordering and
        # filtering; `rerank` still runs the answerability gate on the kept passages.
        cross_scores = None
        if self.cross_encoder is not None:
            cross_scores = self.cross_encoder.score_pairs(
                [(query, hit.chunk.text) for hit in candidates]
            )
        nli = self.answerability_nli
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
            order=self.config.resolved_rerank_order,
            scores=cross_scores,
            blend=self.config.rerank_blend,
            gate=self.answerability_gate,
            answerability_fn=nli.score if nli is not None else None,
        )
        rerank_seconds = time.perf_counter() - rerank_started

        return QueryResult(
            query=query,
            answerable=reranked.answerable,
            answerability=reranked.answerability,
            answerability_measured=reranked.answerability_measured,
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
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
        dedup_threshold: float = 0.0,
        coverage_lambda: float = 0.0,
    ) -> list[Hit]:
        """Fuse dense and lexical rankings into one candidate list.

        The fused ``Hit.score`` is the RRF score, not a cosine similarity; the
        channels that produced each candidate travel with it so the evidence bundle
        can say where a passage came from.

        The weights are not cosmetic: equal-weight fusion of a near-chance dense channel
        with a strong lexical one lowers BM25's nDCG@10 by roughly 43% on BEIR. A weight
        of 0 removes that channel from the fusion without disabling it elsewhere.

        ``dedup_threshold`` > 0 walks the fused list and drops a candidate whose cosine to
        an already-kept chunk is at or above it, until ``limit`` are kept or the list ends.
        ``coverage_lambda`` > 0 instead runs the evidence-set coverage selection
        (:func:`_coverage_select`), which subsumes near-duplicate suppression and targets
        new facts. Both need a wider pool to replace duplicates with deeper candidates.
        """
        dense_ids = [hit.chunk.chunk_id for hit in dense_hits]
        lexical_ids = [hit.chunk_id for hit in lexical_hits]
        # Over-fetch when diversifying: the duplicates have to be *replaced* by deeper
        # candidates, so the fused pool is wider than the returned candidate list.
        pool = limit * 3 if (dedup_threshold > 0.0 or coverage_lambda > 0.0) else limit
        fused = reciprocal_rank_fusion(
            [dense_ids, lexical_ids], limit=pool, weights=[dense_weight, lexical_weight]
        )
        if coverage_lambda > 0.0:
            fused = _coverage_select(index, fused, limit, coverage_lambda)
        elif dedup_threshold > 0.0:
            fused = _suppress_near_duplicates(index, fused, limit, dedup_threshold)
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

        # 1. Recall per claim, over the whole corpus. As in `query`, a zero dense weight
        #    makes the claim embeddings unnecessary when a lexical channel exists, and a
        #    missing lexical channel forces the dense one back on.
        lexical = self.lexical
        dense_weight = self.config.resolved_dense_weight
        if lexical is None:
            dense_weight = 1.0
        use_dense = dense_weight != 0.0
        vectors = np.asarray(self.embedder.embed(cleaned), dtype=np.float32) if use_dense else None
        recall_k = (
            candidate_k * 3
            if (self.config.recall_dedup_threshold > 0.0 or self.config.recall_coverage_lambda > 0.0)
            else candidate_k
        )
        per_claim_hits: list[list[Hit]] = []
        for position, claim in enumerate(cleaned):
            dense = index.search(vectors[position], k=recall_k) if vectors is not None else []
            lexical_hits = lexical.search(claim, k=recall_k) if lexical else []
            per_claim_hits.append(
                self._fuse(
                    index,
                    dense,
                    lexical_hits,
                    limit=candidate_k,
                    dense_weight=dense_weight,
                    lexical_weight=self.config.lexical_weight,
                    dedup_threshold=self.config.recall_dedup_threshold,
                    coverage_lambda=self.config.recall_coverage_lambda,
                )
            )

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
            "embedder": self.embedder.info() if hasattr(self.embedder, "info") else {"embedder": type(self.embedder).__name__},
            "reranker": self.cross_encoder.info() if self.cross_encoder is not None else {"kind": "laya decisions"},
            "index_dir": str(self.config.index_dir),
            "index_backend": self.config.resolved_index_backend,
            "index_present": self.has_index(),
            "lexical_present": self.lexical is not None,
            "compose": self.composer.describe(),
        }
        if payload["index_present"]:
            try:
                payload["index"] = self.index.stats()
            except Exception as exc:  # noqa: BLE001 - reported, not raised, for a status endpoint
                payload["index_error"] = str(exc)
        else:
            payload["index"] = {"chunks": 0, "documents": 0, "paths": 0}
        payload["answerability_gate"] = {
            "configured": bool((self.config.answerability_gate_path or "").strip()),
            "loaded": self.answerability_gate is not None,
        }
        payload["answerability_nli"] = {
            "configured": bool((self.config.answerability_nli_model or "").strip()),
            "model": self.config.answerability_nli_model or None,
        }
        return payload
