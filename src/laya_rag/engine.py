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
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .audit import AuditLog
from .chunking import (
    DEFAULT_EXTENSIONS,
    MAX_FILE_BYTES,
    chunk_documents,
    derive_chunk_chars,
    load_documents,
)
from .compose import Composer
from .config import EngineConfig
from .index_store import Hit, VectorIndex, wipe
from .lexical import (
    EAGER_LEXICAL_FILE,
    LEXICAL_FILE,
    BM25Index,
    EagerBM25Index,
    convex_combination,
    reciprocal_rank_fusion,
)
from .model import BiEncoderModel, CrossEncoderModel, LayaModel
from .rerank import ANSWERABILITY_QUESTION, LayaReranker
from .trust import TrustManifest, filter_candidates


class IndexNotBuiltError(RuntimeError):
    """Raised when a query arrives before any index exists."""


class BuildInProgressError(RuntimeError):
    """Raised when a second build is requested while one is running.

    ``index_paths`` mutates the in-memory index and the persisted files; two overlapping
    builds, or a query reading the index mid-build, race. The API turns this into a 409.
    """


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
    sparse_seconds: float = 0.0
    derived_chunk_chars: int = 0
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
            "sparse_seconds": round(self.sparse_seconds, 3),
            "derived_chunk_chars": self.derived_chunk_chars,
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
    #: Candidates excluded because their document's trust tier was below the query's
    #: minimum. Zero unless a trust manifest and a minimum are both configured.
    trust_filtered: int = 0
    #: Candidates excluded by the manifest's per-tenant ACL. Zero unless a tenant block
    #: is configured.
    acl_filtered: int = 0
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
            "trust_filtered": self.trust_filtered,
            "acl_filtered": self.acl_filtered,
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
    *,
    use_redundancy: bool = True,
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

    ``use_redundancy=False`` drops the dense-gram penalty. On the full 2,255-query
    protocol at ``lam=0.5`` that measured MAP@10 0.2835 → 0.2850
    (+0.0015 [+0.0002, +0.0029], p=0.032) with Hits@4 0.7197 → 0.7326 and Hits@10
    0.8585 → 0.8723, so it is the shipped setting; at ``lam=0.7`` the penalty does
    help (+0.0036), which is why the flag is exposed rather than deleted.
    """
    relevance = [score for _chunk_id, score in fused]
    return _coverage_select_from(
        index, fused, relevance, limit, lam, use_redundancy=use_redundancy
    )


def _coverage_select_from(
    index: VectorIndex,
    fused: Sequence[tuple[str, float]],
    relevance: Sequence[float],
    limit: int,
    lam: float,
    *,
    use_redundancy: bool = True,
    docs: Sequence[str] | None = None,
    min_docs: int = 0,
) -> list[tuple[str, float]]:
    """Coverage selection with an explicit relevance signal.

    ``_coverage_select`` reads relevance from the fused RRF scores; this takes it as an
    argument so the same rule can run over a *cross-encoder* score vector after the
    second stage, which is the configuration measured to beat a full cross-encoder
    re-sort on MultiHop-RAG (MAP@10 0.2986 -> 0.3206 at lam 0.7). When the relevance
    signal already comes from a model that attends over the candidate, the dense-gram
    redundancy term is dropped: atom coverage already stops the same fact being selected
    twice, and the experiment that chose this rule did not use it.

    ``docs`` and ``min_docs`` add the document quota measured alongside it: the first
    ``min_docs`` selections are the best-relevance candidate of distinct documents (by
    ``docs[i]``), which stops one article occupying the top-k with overlapping chunks,
    and the remaining slots are filled by the coverage rule. The combined rule measured
    MAP@10 0.3280 against the cross-encoder's 0.2986 on the 400-query subset.
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

    relevance = np.asarray(list(relevance), dtype=np.float64)
    if relevance.size and float(relevance.max()) > float(relevance.min()):
        relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min())
    else:
        relevance = np.zeros_like(relevance)
    vectors = (
        [index.vector(chunk_id) for chunk_id, _score in fused] if use_redundancy else []
    )

    selected: list[int] = []
    covered: set[int] = set()
    candidates = list(range(len(fused)))
    if docs is not None and min_docs > 0:
        best_of_doc: dict[str, float] = {}
        for i, doc in enumerate(docs):
            best_of_doc[doc] = max(best_of_doc.get(doc, -1e18), float(relevance[i]))
        doc_order = sorted(best_of_doc, key=lambda doc: (-best_of_doc[doc], doc))
        represented: set[str] = set()
        for doc in doc_order:
            if len(selected) >= limit or len(represented) >= min_docs:
                break
            pool = [i for i in candidates if docs[i] == doc]
            if not pool:
                continue
            pick = max(pool, key=lambda i: (float(relevance[i]), -i))
            selected.append(pick)
            covered.update(candidate_atoms[pick])
            candidates.remove(pick)
            represented.add(doc)
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
            score = lam * float(relevance[i]) + (1.0 - lam) * marginal
            if use_redundancy:
                redundancy = 0.0
                if selected:
                    sims = [
                        float(np.dot(vectors[i], vectors[j]))
                        for j in selected
                        if vectors[i] is not None and vectors[j] is not None
                    ]
                    redundancy = max(sims) if sims else 0.0
                score -= (1.0 - lam) * redundancy
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
        # The learned-sparse channel (config.sparse_model). Loaded/encoded lazily so a
        # default engine never pays for it.
        self._sparse: Any | None = None
        self._sparse_encoder: Any | None = None
        self._sparse_query_cache: dict[str, Any] = {}
        # Optional hash-chained audit log of index mutations (config.audit_log).
        self._audit: AuditLog | None = None
        # Optional persisted feature gate for answerability (see config). Loaded lazily:
        # a missing or damaged file must not stop the engine from answering, it only
        # falls back to Laya's own decision.
        self._answerability_gate: Any | None = None
        self._gate_loaded = False
        self._answerability_nli: Any | None = None
        self._nli_loaded = False
        self._trust_manifest: TrustManifest | None = None
        self._trust_loaded = False
        # Bounded cache of query embeddings. A service sees the same question repeatedly
        # (retries, UI refreshes, shared links), and the query embedding is the one dense
        # cost on an otherwise lexical path. Keyed by the exact query string, so it can
        # never return a stale *result* — only a recomputed vector.
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        # One build at a time. The index is mutated in place, so a second build (or a
        # query reading it mid-build) would race; the API reports 409 to the loser.
        self._build_lock = threading.Lock()
        self._build_active = threading.Event()

    def _ensure_not_building(self) -> None:
        if self._build_active.is_set():
            raise BuildInProgressError(
                "an index build is in progress; retry when it finishes"
            )

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
        """Whether a *complete* index is on disk.

        The lexical artifact counts as part of the index. A build killed between saving the dense
        vectors and saving the lexical postings leaves a directory with the dense half only, and
        treating that as complete silently degrades every later query to dense-only recall — the
        engine warns at query time, but by then the measurement is already wrong. A partial index
        must be rebuilt, so this returns False for one.
        """
        if self._index is not None:
            return True
        if self.config.resolved_index_backend == "ondisk":
            from .ondisk import BM25_DIRNAME, DENSE_DIRNAME, OnDiskBM25, OnDiskVectorIndex

            if not OnDiskVectorIndex.exists_on_disk(self.config.index_dir / DENSE_DIRNAME):
                return False
            return OnDiskBM25.exists_on_disk(self.config.index_dir / BM25_DIRNAME)
        if not VectorIndex.exists_on_disk(self.config.index_dir):
            return False
        if self.config.resolved_lexical_backend == "eager":
            return (self.config.index_dir / EAGER_LEXICAL_FILE).is_file()
        return (self.config.index_dir / LEXICAL_FILE).is_file()

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
                if self.config.resolved_lexical_backend == "eager":
                    path = self.config.index_dir / EAGER_LEXICAL_FILE
                    loader = EagerBM25Index.load
                else:
                    path = self.config.index_dir / LEXICAL_FILE
                    loader = BM25Index.load
                if path.is_file():
                    try:
                        self._lexical = loader(path)
                    except (OSError, ValueError, KeyError):
                        # A stale or damaged lexical index must not break vector search;
                        # the engine degrades to dense-only recall.
                        self._lexical = None
        return self._lexical

    @property
    def sparse_encoder(self) -> Any | None:
        """The learned-sparse encoder, or None when no ``SPARSE_MODEL`` is configured."""
        if not self.config.sparse_enabled:
            return None
        if self._sparse_encoder is None:
            from .sparse import SparseEncoder

            self._sparse_encoder = SparseEncoder(
                self.config.sparse_model, top_k=self.config.sparse_top_k
            )
        return self._sparse_encoder

    @property
    def sparse(self) -> Any | None:
        """The persisted learned-sparse index, or None when it was never built."""
        if not self.config.sparse_enabled:
            return None
        if self._sparse is None:
            from .sparse import SPARSE_FILE, SparseIndex

            path = self.config.index_dir / SPARSE_FILE
            if path.is_file():
                try:
                    self._sparse = SparseIndex.load(path)
                except (OSError, ValueError, KeyError):
                    # A stale or damaged sparse index must not break retrieval; the engine
                    # simply falls back to dense + lexical.
                    self._sparse = None
        return self._sparse

    def _sparse_hits(self, query: str, k: int) -> list[tuple[str, float]]:
        """Rank chunks for ``query`` with the sparse channel, or ``[]`` when absent."""
        index = self.sparse
        encoder = self.sparse_encoder
        if index is None or encoder is None or not len(index):
            return []
        vector = self._sparse_query_cache.get(query)
        if vector is None:
            vector = encoder.embed_one(query)
            size = self.config.query_cache_size
            if size > 0:
                self._sparse_query_cache[query] = vector
                if len(self._sparse_query_cache) > size:
                    self._sparse_query_cache.pop(next(iter(self._sparse_query_cache)))
        return index.search(vector, k=k)

    def _build_sparse(self, chunks: Sequence[Any]) -> float:
        """Encode every chunk with the sparse model, persist the inverted index."""
        from .sparse import SPARSE_FILE, SparseIndex

        encoder = self.sparse_encoder
        if encoder is None:
            return 0.0
        started = time.perf_counter()
        vectors: list[Any] = []
        for batch in _batched(chunks, max(1, self.config.sparse_batch_texts)):
            vectors.extend(encoder.encode([chunk.index_text for chunk in batch]))
        index = SparseIndex()
        index.build([chunk.chunk_id for chunk in chunks], vectors)
        index.save(self.config.index_dir / SPARSE_FILE)
        self._sparse = index
        return time.perf_counter() - started

    def _record_audit(self, action: str, detail: dict[str, Any]) -> None:
        """Append a mutation to the optional hash-chained audit log.

        A logging failure must not undo a successful mutation: the index is already written,
        so a broken log path is swallowed here and surfaces through `audit` verification
        rather than as a failed request. `actor` is not known at the engine layer.
        """
        path = self.config.audit_log
        if path is None:
            return
        if self._audit is None:
            self._audit = AuditLog(path)
        try:
            self._audit.append(action, detail=detail)
        except OSError:
            return

    def reset(self) -> None:
        """Delete the on-disk index and drop the in-memory copies.

        Refused while a build is running, like a second build.
        """
        if not self._build_lock.acquire(blocking=False):
            raise BuildInProgressError("an index build is in progress; reset refused")
        self._build_active.set()
        try:
            self._reset_unlocked()
            self._record_audit("reset", {"index_dir": str(self.config.index_dir)})
        finally:
            self._build_active.clear()
            self._build_lock.release()

    def _reset_unlocked(self) -> None:
        """The body of :meth:`reset`, called with the build lock already held."""
        wipe(self.config.index_dir)
        self._index = None
        self._lexical = None
        self._sparse = None
        self._sparse_query_cache = {}

    def _rebuild_ondisk_lexical(self, chunks: Sequence[Any]) -> None:
        """Rewrite the packed on-disk BM25 channel from the surviving chunks."""
        from .ondisk import BM25_DIRNAME, OnDiskBM25

        bm25_dir = self.config.index_dir / BM25_DIRNAME
        if bm25_dir.exists():
            shutil.rmtree(bm25_dir)
        if not chunks:
            # Nothing survived: a packed postings file with no postings cannot be mmapped,
            # so drop the channel rather than writing an unusable one. The dense half is all
            # tombstones and `compact()` resets it.
            self._lexical = None
            return
        lexical = OnDiskBM25(
            bm25_dir,
            stem=self.config.lexical_stem,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )
        lexical.build((chunk.chunk_id, chunk.index_text) for chunk in chunks)
        self._lexical = lexical

    def compact(self) -> int:
        """Rewrite the on-disk index without tombstones, without re-embedding.

        A no-op for the in-memory backend and when nothing is tombstoned: it returns the live
        chunk count. On the on-disk backend the live chunk records and their stored vectors are
        streamed in batches into a staging directory, which is then swapped in, so the vector
        matrix is never fully resident and no embedding pass is paid. An index whose every chunk
        was removed is deleted outright. Returns the number of live chunks.
        """
        if not self.has_index():
            return 0
        if self.config.resolved_index_backend != "ondisk":
            return len(self.index)
        index = self.index
        if not getattr(index, "_deleted", None):
            return len(index)
        chunks = index.chunks
        if not chunks:
            self.reset()
            return 0
        from .ondisk import DENSE_DIRNAME, OnDiskVectorIndex

        dense_dir = self.config.index_dir / DENSE_DIRNAME
        staging = dense_dir.parent / f"{DENSE_DIRNAME}.compact"
        if staging.exists():
            shutil.rmtree(staging)
        fresh = OnDiskVectorIndex(staging, dim=index.dim)
        batch_size = 20000
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = np.asarray(
                [index.vector(chunk.chunk_id) for chunk in batch], dtype=np.float32
            )
            fresh.add(batch, vectors)
        fresh.finalize()
        if dense_dir.exists():
            shutil.rmtree(dense_dir)
        staging.rename(dense_dir)
        self._index = OnDiskVectorIndex.open(dense_dir)
        self._rebuild_ondisk_lexical(chunks)
        self._sparse = None
        self._sparse_encoder = None
        self._sparse_query_cache = {}
        self._record_audit("compact", {"live_chunks": len(chunks)})
        return len(chunks)

    def remove_documents(
        self,
        paths: Sequence[Path | str] = (),
        doc_ids: Sequence[str] = (),
    ) -> int:
        """Drop documents from the index and rebuild the lexical channel to match.

        The dense index is the source of truth for which chunks exist: chunks are removed
        there, then the lexical channel is rebuilt from the survivors so the two cannot
        disagree. On the in-memory backend removal is a real delete; on the append-only
        on-disk backend the chunk records are **tombstoned** (skipped by search, get and
        chunks) and the packed lexical channel is rebuilt from the survivors, so no saved
        byte offset is invalidated. Returns the number of chunks removed.

        Removing a path removes every chunk of the document(s) that path belongs to.
        """
        self._ensure_not_building()
        if not paths and not doc_ids:
            return 0
        if not self.has_index():
            return 0
        index = self.index
        removed = index.remove_paths([str(p) for p in paths]) if paths else 0
        if doc_ids:
            removed += index.remove_doc_ids(list(doc_ids))
        if not removed:
            return 0
        chunks = index.chunks
        if self.config.resolved_index_backend == "ondisk":
            self._rebuild_ondisk_lexical(chunks)
        else:
            index.save(self.config.index_dir)
            if self.config.resolved_lexical_backend == "eager":
                lexical = EagerBM25Index(
                    stem=self.config.lexical_stem, k1=self.config.bm25_k1, b=self.config.bm25_b
                )
                lexical.build(chunks)
                lexical.save(self.config.index_dir / EAGER_LEXICAL_FILE)
            else:
                lexical = BM25Index(
                    stem=self.config.lexical_stem, k1=self.config.bm25_k1, b=self.config.bm25_b
                )
                lexical.build(chunks)
                lexical.save(self.config.index_dir / LEXICAL_FILE)
            self._lexical = lexical
        self._sparse = None
        self._sparse_encoder = None
        self._sparse_query_cache = {}
        if self.config.sparse_enabled and chunks:
            self._build_sparse(chunks)
        self._record_audit(
            "delete",
            {
                "paths": [str(p) for p in paths],
                "doc_ids": list(doc_ids),
                "removed_chunks": removed,
                "total_chunks": len(chunks),
            },
        )
        return removed

    @property
    def trust_manifest(self) -> TrustManifest | None:
        """The optional path->tier manifest, or None when no path is configured.

        Loaded lazily and cached. A missing or malformed manifest degrades to no
        filtering: an unreadable policy must not silently refuse every query, but the
        caller can see the failure because ``stats`` reports the loaded state.
        """
        if not self._trust_loaded:
            self._trust_loaded = True
            raw = (self.config.trust_manifest or "").strip()
            if raw:
                path = Path(raw).expanduser()
                if path.is_file():
                    try:
                        self._trust_manifest = TrustManifest.from_path(
                            path, default_tier=self.config.trust_default_tier
                        )
                    except (OSError, ValueError, KeyError):
                        self._trust_manifest = None
        return self._trust_manifest

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

        Serialized against other builds: a second concurrent call raises
        :class:`BuildInProgressError` rather than racing on the shared index.
        """
        if not self._build_lock.acquire(blocking=False):
            raise BuildInProgressError("another index build is already running")
        self._build_active.set()
        try:
            report = self._index_paths_locked(
                paths,
                rebuild=rebuild,
                extensions=extensions,
                max_file_bytes=max_file_bytes,
                progress=progress,
                prune=prune,
            )
            self._record_audit(
                "index",
                {
                    "paths": [str(path) for path in paths],
                    "rebuild": bool(rebuild),
                    "prune": bool(prune),
                    "documents": report.documents,
                    "chunks": report.chunks,
                    "removed_chunks": report.removed_chunks,
                },
            )
            return report
        finally:
            self._build_active.clear()
            self._build_lock.release()

    def _index_paths_locked(
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

        if self.config.sparse_enabled and self.config.resolved_index_backend == "ondisk":
            raise ValueError(
                "SPARSE_MODEL is not supported with the ondisk index backend; index in "
                "memory or unset SPARSE_MODEL"
            )

        if rebuild:
            self._reset_unlocked()

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

        chunk_chars = self.config.chunk_chars
        if self.config.chunk_chars_auto:
            # Derive the size from this corpus rather than the fixed default; see
            # docs/findings-chunk-merge.md.
            chunk_chars = derive_chunk_chars(
                [len(document.text) for document in loaded.documents],
                maximum=self.config.chunk_chars_auto_max,
                percentile=self.config.chunk_chars_auto_percentile,
            )
            report.derived_chunk_chars = chunk_chars
        chunks = chunk_documents(
            loaded.documents,
            chunk_chars=chunk_chars,
            overlap_chars=self.config.resolved_chunk_overlap,
            min_chunk_chars=self.config.min_chunk_chars,
            on_document=chunking,
            context_prefix=self.config.context_prefix,
            merge_short=self.config.chunk_merge_short,
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
                    self.embedder.embed([chunk.index_text for chunk in batch]), dtype=np.float32
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
        if self.config.resolved_lexical_backend == "eager":
            lexical = EagerBM25Index(
                stem=self.config.lexical_stem, k1=self.config.bm25_k1, b=self.config.bm25_b
            )
            lexical.build(index.chunks)
            lexical.save(self.config.index_dir / EAGER_LEXICAL_FILE)
        else:
            lexical = BM25Index(
                stem=self.config.lexical_stem, k1=self.config.bm25_k1, b=self.config.bm25_b
            )
            lexical.build(index.chunks)
            lexical.save(self.config.index_dir / LEXICAL_FILE)
        self._lexical = lexical
        if self.config.sparse_enabled:
            report.sparse_seconds = self._build_sparse(index.chunks)
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
                self.embedder.embed([chunk.index_text for chunk in batch]), dtype=np.float32
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
        lexical = OnDiskBM25(
            bm25_dir, stem=self.config.lexical_stem,
            k1=self.config.bm25_k1, b=self.config.bm25_b,
        )
        lexical.build((chunk.chunk_id, chunk.index_text) for chunk in chunks)
        report.save_seconds = time.perf_counter() - save_started

        self._index = dense
        self._lexical = lexical
        report.removed_chunks = 0
        report.total_chunks = len(chunks)
        report.paths = len({chunk.path for chunk in chunks})

    # -- querying --------------------------------------------------------
    def _query_vector(self, query: str) -> np.ndarray:
        """Embed one query, reusing a bounded cache of recent query vectors.

        ``QUERY_CACHE_SIZE`` of 0 disables the cache. The cache holds vectors, not results,
        so an index change can never be hidden by it.
        """
        size = self.config.query_cache_size
        if size <= 0:
            return np.asarray(self.embedder.embed_one(query), dtype=np.float32)
        cached = self._query_cache.get(query)
        if cached is not None:
            self._query_cache.move_to_end(query)
            return cached
        vector = np.asarray(self.embedder.embed_one(query), dtype=np.float32)
        self._query_cache[query] = vector
        if len(self._query_cache) > size:
            self._query_cache.popitem(last=False)
        return vector

    def query(
        self,
        query: str,
        top_k: int | None = None,
        retrieve_k: int | None = None,
        relevance_threshold: float | None = None,
        answerability_threshold: float | None = None,
        max_passage_chars: int | None = 2000,
        tenant: str | None = None,
    ) -> QueryResult:
        """Retrieve, re-rank and gate evidence for ``query``.

        ``tenant`` selects a caller identity for the manifest's per-tenant ACL, when one is
        configured. With no tenant block in the manifest it is ignored, so single-tenant
        deployments are unaffected.
        """
        self._ensure_not_building()
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
        # channels were searched deeper to begin with. Trust and ACL filtering have the
        # same need: excluding a document must leave enough deeper candidates to refill.
        trust_manifest = self.trust_manifest
        min_trust = (self.config.query_min_trust or "").strip()
        acl_active = trust_manifest is not None and trust_manifest.acl_enabled
        trust_active = trust_manifest is not None and (bool(min_trust) or acl_active)
        effective_min_trust = min_trust or "untrusted"
        recall_k = (
            candidate_k * 3
            if (
                self.config.recall_dedup_threshold > 0.0
                or self.config.recall_coverage_lambda > 0.0
                or trust_active
            )
            else candidate_k
        )
        # The predicate is pushed into *every* channel before fusion: both vector backends
        # take `allow` for the dense scan, and the lexical/sparse hits are screened here so
        # a disallowed document cannot consume a fusion slot (or force a deeper over-fetch)
        # only to be dropped afterwards. The post-fusion filter stays as a safety net.
        allow_path: Any = None
        if trust_active:
            manifest, minimum, identity, acl = (
                trust_manifest, effective_min_trust, tenant, acl_active,
            )

            def allow_path(path: str | None) -> bool:
                if acl and not manifest.tenant_allowed(path, identity):
                    return False
                return manifest.allows(path, minimum, identity)

        def screen(hits: Sequence[Any], chunk_id_of: Any) -> tuple[list[Any], int, int]:
            """Drop disallowed hits early, returning ``(kept, trust_dropped, acl_dropped)``."""
            if allow_path is None:
                return list(hits), 0, 0
            kept: list[Any] = []
            dropped_trust = dropped_acl = 0
            for hit in hits:
                chunk = index.get(chunk_id_of(hit)) if index is not None else None
                path = chunk.path if chunk is not None else None
                if allow_path(path):
                    kept.append(hit)
                elif acl_active and not trust_manifest.tenant_allowed(path, tenant):
                    dropped_acl += 1
                else:
                    dropped_trust += 1
            return kept, dropped_trust, dropped_acl

        dense_hits: list[Hit] = []
        lexical_hits: list[Any] = []
        sparse_hits: list[Any] = []
        candidates: list[Hit] = []
        trust_filtered = 0
        acl_filtered = 0
        # A fixed over-fetch factor is a guess, and when it is wrong the result is *short*:
        # the top-`recall_k` window may hold fewer than `candidate_k` permitted candidates
        # even though the index has more. Deepen until enough survive (or the index is
        # exhausted), so the filter changes *which* documents are returned, never *how many*.
        limit = len(index) if index is not None else recall_k
        while True:
            if dense_weight == 0.0:
                dense_hits = []
            else:
                dense_hits = index.search(self._query_vector(query), k=recall_k, allow=allow_path)
            lexical_hits = lexical.search(query, k=recall_k) if lexical else []
            sparse_hits = self._sparse_hits(query, recall_k)
            lexical_hits, pre_trust, pre_acl = screen(lexical_hits, lambda hit: hit.chunk_id)
            sparse_hits, s_trust, s_acl = screen(sparse_hits, lambda pair: pair[0])
            pre_trust += s_trust
            pre_acl += s_acl
            candidates = self._fuse(
                index,
                dense_hits,
                lexical_hits,
                limit=candidate_k,
                dense_weight=dense_weight,
                lexical_weight=self.config.lexical_weight,
                dedup_threshold=self.config.recall_dedup_threshold,
                coverage_lambda=self.config.recall_coverage_lambda,
                coverage_redundancy=self.config.recall_coverage_redundancy,
                rrf_k=self.config.fusion_k,
                sparse_hits=sparse_hits,
                sparse_weight=self.config.sparse_weight,
                fusion_mode=self.config.fusion_mode,
            )
            trust_filtered = pre_trust
            acl_filtered = pre_acl
            if trust_active:
                pre_filter = list(candidates)
                if acl_active:
                    allowed = [
                        hit for hit in pre_filter
                        if trust_manifest.tenant_allowed(hit.chunk.path, tenant)
                    ]
                    acl_filtered += len(pre_filter) - len(allowed)
                    pre_filter = allowed
                remaining = len(pre_filter)
                pre_filter = filter_candidates(
                    pre_filter, trust_manifest, effective_min_trust, tenant
                )
                trust_filtered += remaining - len(pre_filter)
                candidates = pre_filter
            # Enough survived, nothing more to fetch, or the window already covers the index.
            if (not trust_active) or len(candidates) >= candidate_k or recall_k >= limit:
                break
            recall_k = min(limit, max(recall_k * 4, candidate_k))
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
        order = self.config.resolved_rerank_order
        if self.cross_encoder is not None:
            cross_scores = self.cross_encoder.score_pairs(
                [(query, hit.chunk.text) for hit in candidates]
            )
        # Optional coverage reordering over the cross-encoder-scored pool: spend the
        # recall budget on distinct facts *after* the second stage has scored, which
        # measured better than a full CE re-sort. It reorders the pool, so the reranker
        # must preserve that order rather than re-sort by the score again — and a blend
        # below 1 would override the order, so coverage forces it back to a pure
        # replacement of the ordering key.
        lambda_ = self.config.resolved_rerank_coverage_lambda
        blend = self.config.rerank_blend
        if cross_scores is not None and lambda_ > 0.0 and candidates:
            pairs = [(hit.chunk.chunk_id, float(score)) for hit, score in zip(candidates, cross_scores)]
            selected = _coverage_select_from(
                index, pairs, [score for _id, score in pairs], len(pairs), lambda_,
                use_redundancy=False,
                docs=[hit.chunk.path or hit.chunk.chunk_id for hit in candidates],
                min_docs=self.config.rerank_coverage_min_docs,
            )
            by_id = {hit.chunk.chunk_id: hit for hit in candidates}
            score_of = dict(pairs)
            candidates = [by_id[chunk_id] for chunk_id, _score in selected]
            cross_scores = [score_of[hit.chunk.chunk_id] for hit in candidates]
            order = "recall"
            # `blend < 1` reorders by a first-stage/CE mix and would discard the coverage
            # order entirely, so it is forced to a pure replacement here.
            blend = 1.0
        nli = self.answerability_nli
        # `RELEVANCE_THRESHOLD` is calibrated for Laya's probabilities. When a
        # cross-encoder supplies the scores they are a different model's sigmoid, and the
        # 0.5 cut is far too strict there: on the MultiHop CE pool only ~11% of candidates
        # clear it, so many queries would keep nothing. The experiments never thresholded
        # the second stage, so a substituted scorer gets `RERANK_THRESHOLD` (0.0 by
        # default) unless the caller passes an explicit relevance_threshold.
        if relevance_threshold is not None:
            effective_relevance = relevance_threshold
        elif cross_scores is not None:
            effective_relevance = self.config.rerank_threshold
        else:
            effective_relevance = self.config.relevance_threshold
        reranked = self.reranker.rerank(
            query,
            candidates,
            top_k=top_k or self.config.rerank_k,
            relevance_threshold=effective_relevance,
            answerability_threshold=(
                self.config.answerability_threshold
                if answerability_threshold is None
                else answerability_threshold
            ),
            order=order,
            scores=cross_scores,
            blend=blend,
            gate=self.answerability_gate,
            answerability_fn=nli.score if nli is not None else None,
            min_docs=self.config.rerank_min_docs,
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
            trust_filtered=trust_filtered,
            acl_filtered=acl_filtered,
            retrieval_seconds=retrieval_seconds,
            rerank_seconds=rerank_seconds,
            total_seconds=time.perf_counter() - started,
            usage=reranked.usage,
            checkpoint=self.config.checkpoint,
            note=None if lexical is not None or not len(index) else degraded,
        )

    def query_extractive(
        self,
        query: str,
        *,
        max_sentences: int = 5,
        min_score: float = 0.0,
        **query_kwargs: Any,
    ) -> dict[str, Any]:
        """Query, then emit only verbatim evidence sentences, each with its citation.

        Every emitted sentence is a substring of a retrieved passage, so the result cannot
        state something the corpus does not contain; with nothing to emit it abstains. No
        generator, no API. The structured citations are checkable with
        :func:`laya_rag.verify.verify_citations`.
        """
        from .extractive import compose_extractive

        result = self.query(query, **query_kwargs)
        composed = compose_extractive(
            result.passages, query=query, max_sentences=max_sentences, min_score=min_score
        )
        return {
            "query": query,
            "answerable": result.answerable,
            "answerability": result.answerability,
            **composed.as_dict(),
        }

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
        coverage_redundancy: bool = True,
        rrf_k: int = 60,
        sparse_hits: Sequence[tuple[str, float]] | None = None,
        sparse_weight: float = 1.0,
        fusion_mode: str = "rrf",
    ) -> list[Hit]:
        """Fuse dense, lexical and optional sparse rankings into one candidate list.

        ``fusion_mode`` selects the fusion rule: ``rrf`` (rank-based, the shipped default) or
        ``convex`` (:func:`convex_combination`, a weighted sum of min-max normalised scores).
        An unknown value raises rather than silently falling back.

        The fused ``Hit.score`` is the fusion score, not a cosine similarity; the
        channels that produced each candidate travel with it so the evidence bundle
        can say where a passage came from.

        The weights are not cosmetic: equal-weight fusion of a near-chance dense channel
        with a strong lexical one lowers BM25's nDCG@10 by roughly 43% on BEIR. A weight
        of 0 removes that channel from the fusion without disabling it elsewhere.

        ``sparse_hits`` is the learned-sparse channel's ``(chunk_id, score)`` list, which
        only its rank contributes to rank fusion — measured, it is the first additional
        channel that improves the fused ranking consistently (+0.004 to +0.023 nDCG@10).

        ``dedup_threshold`` > 0 walks the fused list and drops a candidate whose cosine to
        an already-kept chunk is at or above it, until ``limit`` are kept or the list ends.
        ``coverage_lambda`` > 0 instead runs the evidence-set coverage selection
        (:func:`_coverage_select`), which subsumes near-duplicate suppression and targets
        new facts. Both need a wider pool to replace duplicates with deeper candidates.
        """
        dense_ids = [hit.chunk.chunk_id for hit in dense_hits]
        lexical_ids = [hit.chunk_id for hit in lexical_hits]
        sparse_pairs = [(chunk_id, float(score)) for chunk_id, score in (sparse_hits or [])]
        sparse_ids = [chunk_id for chunk_id, _score in sparse_pairs]
        # Over-fetch when diversifying: the duplicates have to be *replaced* by deeper
        # candidates, so the fused pool is wider than the returned candidate list.
        pool = limit * 3 if (dedup_threshold > 0.0 or coverage_lambda > 0.0) else limit
        weights = [dense_weight, lexical_weight, sparse_weight]
        if fusion_mode == "convex":
            # Score-based, so this is the only branch that needs the channel scores.
            dense_pairs = [
                (hit.chunk.chunk_id, float(getattr(hit, "score", 0.0))) for hit in dense_hits
            ]
            lexical_pairs = [
                (hit.chunk_id, float(getattr(hit, "score", 0.0))) for hit in lexical_hits
            ]
            fused = convex_combination(
                [dense_pairs, lexical_pairs, sparse_pairs], limit=pool, weights=weights
            )
        elif fusion_mode == "rrf":
            fused = reciprocal_rank_fusion(
                [dense_ids, lexical_ids, sparse_ids], k=rrf_k, limit=pool, weights=weights
            )
        else:
            raise ValueError(f"unknown fusion_mode {fusion_mode!r}; expected 'rrf' or 'convex'")
        if coverage_lambda > 0.0:
            fused = _coverage_select(index, fused, limit, coverage_lambda,
                                     use_redundancy=coverage_redundancy)
        elif dedup_threshold > 0.0:
            fused = _suppress_near_duplicates(index, fused, limit, dedup_threshold)
        dense_set = set(dense_ids)
        lexical_set = set(lexical_ids)
        sparse_set = set(sparse_ids)
        candidates: list[Hit] = []
        for chunk_id, rrf_score in fused:
            chunk = index.get(chunk_id)
            if chunk is None:  # pragma: no cover - a channel naming an unknown chunk
                continue
            channels = tuple(
                name
                for name, present in (
                    ("dense", chunk_id in dense_set),
                    ("lexical", chunk_id in lexical_set),
                    ("sparse", chunk_id in sparse_set),
                )
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
        tenant: str | None = None,
    ) -> dict[str, Any]:
        """Ask Laya whether the corpus supports each claim.

        The second half of verification. The citation check decides whether an answer
        *cites* evidence; this decides whether the corpus actually *supports* what it
        says, by re-retrieving candidates for every claim across the whole index and
        adjudicating them with Laya's calibrated decisions.

        The trust manifest's tier minimum and per-tenant ACL apply here exactly as they do
        to a query: a claim can otherwise be adjudicated against a document the caller may
        not retrieve, which would make the second verification layer leak what the first
        one filters.

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
        self._ensure_not_building()
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
            "trust_filtered": 0,
            "acl_filtered": 0,
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
        # Reuse the query-embedding cache: claims repeat across adjudications (a composed
        # answer is re-checked when its evidence changes) and a claim identical to a prior
        # query reuses that vector too. Same vectors as `embed`, just not re-embedded.
        vectors = (
            np.asarray([self._query_vector(claim) for claim in cleaned], dtype=np.float32)
            if use_dense
            else None
        )
        trust_manifest = self.trust_manifest
        min_trust = (self.config.query_min_trust or "").strip()
        acl_active = trust_manifest is not None and trust_manifest.acl_enabled
        trust_active = trust_manifest is not None and (bool(min_trust) or acl_active)
        effective_min_trust = min_trust or "untrusted"
        recall_k = (
            candidate_k * 3
            if (
                self.config.recall_dedup_threshold > 0.0
                or self.config.recall_coverage_lambda > 0.0
                or trust_active
            )
            else candidate_k
        )
        per_claim_hits: list[list[Hit]] = []
        trust_filtered = 0
        acl_filtered = 0
        for position, claim in enumerate(cleaned):
            dense = index.search(vectors[position], k=recall_k) if vectors is not None else []
            lexical_hits = lexical.search(claim, k=recall_k) if lexical else []
            hits = self._fuse(
                index,
                dense,
                lexical_hits,
                limit=candidate_k,
                dense_weight=dense_weight,
                lexical_weight=self.config.lexical_weight,
                dedup_threshold=self.config.recall_dedup_threshold,
                coverage_lambda=self.config.recall_coverage_lambda,
                coverage_redundancy=self.config.recall_coverage_redundancy,
                rrf_k=self.config.fusion_k,
                sparse_hits=self._sparse_hits(claim, recall_k),
                sparse_weight=self.config.sparse_weight,
                fusion_mode=self.config.fusion_mode,
            )
            if trust_active:
                before = len(hits)
                if acl_active:
                    hits = [
                        hit for hit in hits
                        if trust_manifest.tenant_allowed(hit.chunk.path, tenant)
                    ]
                    acl_filtered += before - len(hits)
                remaining = len(hits)
                hits = filter_candidates(hits, trust_manifest, effective_min_trust, tenant)
                trust_filtered += remaining - len(hits)
            per_claim_hits.append(hits)

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
        contradicted = no_coverage = 0
        # A claim the relevance gate never reached is `not_measured` by default — but "the gate
        # never asked" is not the same as "the corpus says nothing". When a local NLI scorer is
        # configured it is asked anyway, over the candidates the gate rejected, which separates a
        # claim the corpus *refutes* from one it is simply *silent* about.
        nli = self.answerability_nli
        nli_threshold = float(self.config.claim_nli_threshold)
        for position, (claim, kept) in enumerate(zip(cleaned, kept_per_claim)):
            measured = position in by_position
            probability = float(by_position[position]) if measured else 0.0
            is_supported = bool(measured and probability >= gate)
            columns: dict[str, float] | None = None
            state = "supported" if is_supported else ("unsupported" if measured else "not_measured")
            if not measured and nli is not None:
                candidates = per_claim_hits[position]
                if candidates:
                    columns = nli.verdict(claim, candidates)
                    if columns.get("entailment", 0.0) >= nli_threshold:
                        state, is_supported = "supported", True
                    elif columns.get("contradiction", 0.0) >= nli_threshold:
                        state = "contradicted"
                    else:
                        state = "no_coverage"
            if state == "supported":
                supported += 1
            elif state == "unsupported":
                unsupported += 1
            elif state == "contradicted":
                contradicted += 1
            elif state == "no_coverage":
                no_coverage += 1
            else:
                not_measured += 1
            best = kept[0][1] if kept else 0.0
            adjudicated.append({
                "text": claim,
                "supported": is_supported,
                "measured": measured,
                "state": state,
                "support_probability": round(probability, 4),
                "best_relevance": round(best, 4),
                "nli": None if columns is None else {
                    "entailment": round(columns.get("entailment", 0.0), 4),
                    "contradiction": round(columns.get("contradiction", 0.0), 4),
                },
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
            "contradicted": contradicted,
            "no_coverage": no_coverage,
            "answerability_threshold": gate,
            "relevance_threshold": relevance_bar,
            "retrieve_k": candidate_k,
            "trust_filtered": trust_filtered,
            "acl_filtered": acl_filtered,
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
            "sparse": {
                "configured": self.config.sparse_enabled,
                "present": self.sparse is not None,
                "encoder": (
                    self._sparse_encoder.info()
                    if self._sparse_encoder is not None and hasattr(self._sparse_encoder, "info")
                    else None
                ),
            },
            "compose": self.composer.describe(),
        }
        if payload["index_present"]:
            try:
                payload["index"] = self.index.stats()
            except Exception as exc:  # noqa: BLE001 - reported, not raised, for a status endpoint
                payload["index_error"] = str(exc)
        else:
            payload["index"] = {"chunks": 0, "documents": 0, "paths": 0}
        # Freshness is a retrieval-quality property, not housekeeping: a deleted document
        # still indexed, or a passage indexed twice, changes what a query returns. Only the
        # linear detectors run here — the near-duplicate scan stays opt-in on
        # `laya-rag integrity` — and a failure is reported rather than raised, because a
        # status endpoint must not fail because a check did.
        if payload["index_present"]:
            try:
                from .integrity import audit

                report = audit(self.index.chunks, near_duplicates=False)
                payload["index"]["integrity"] = {
                    "stale_paths": report["stale_paths"]["count"],
                    "stale_chunks": report["stale_paths"]["stale_chunks"],
                    "duplicate_clusters": report["duplicate_content"]["clusters"],
                    "redundant_chunks": report["duplicate_content"]["redundant_chunks"],
                    "clean": report["clean"],
                    "near_duplicates_scanned": False,
                }
            except Exception as exc:  # noqa: BLE001 - see above
                payload["index"]["integrity_error"] = str(exc)
        payload["answerability_gate"] = {
            "configured": bool((self.config.answerability_gate_path or "").strip()),
            "loaded": self.answerability_gate is not None,
        }
        payload["answerability_nli"] = {
            "configured": bool((self.config.answerability_nli_model or "").strip()),
            "model": self.config.answerability_nli_model or None,
        }
        manifest = self.trust_manifest
        payload["trust"] = {
            "configured": bool((self.config.trust_manifest or "").strip()),
            "loaded": manifest is not None,
            "default_tier": self.config.trust_default_tier,
            "query_min_trust": self.config.query_min_trust or None,
            "rules": len(manifest.rules) if manifest is not None else 0,
            "tenants": sorted(manifest.tenants) if manifest is not None else [],
            "acl_enabled": bool(manifest.acl_enabled) if manifest is not None else False,
        }
        return payload
