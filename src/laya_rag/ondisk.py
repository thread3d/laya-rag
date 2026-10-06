"""On-disk retrieval backends for corpora that do not fit in RAM.

The default engine keeps every chunk's text and vector in memory: ``VectorIndex`` holds a
``(n, dim)`` array plus a list of ``Chunk`` records, and ``BM25Index`` holds a Python
``term -> postings`` dict. That is what puts BioASQ (14.9M documents), FEVER (5.4M) and NQ
(2.7M) out of reach — not the scoring math, the residency.

This module is the engine-facing form of the infrastructure first prototyped in
``tools/ondisk.py``:

* :class:`OnDiskBM25` spills term postings to sorted runs, merges them into one packed,
  memory-mapped ``postings.bin``, and searches through the mmap. The Python term dictionary
  is the only structure that scales with vocabulary, not with corpus size.
* :class:`OnDiskVectorIndex` appends embedding batches to a memory-mapped ``vectors.bin``
  and keeps chunk records in a newline sidecar whose byte offsets are indexed, so a chunk's
  text is read on demand rather than held for the whole corpus.

The on-disk backend is selected with ``LAYA_RAG_INDEX_BACKEND=ondisk``. It is append-only:
an incremental re-index over new paths requires the on-disk postings to be rebuilt, so the
engine performs a full rebuild instead of pretending the append is equivalent.
"""
from __future__ import annotations

import heapq
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .chunking import Chunk
from .index_store import Hit, _normalise
from .lexical import K1, B, STOPWORDS, LexicalHit, tokenize

#: File names for the dense half, kept beside the in-memory format's files under the same
#: index directory so the two backends cannot silently read each other's data.
DENSE_DIRNAME = "dense"
BM25_DIRNAME = "bm25"

_VECTORS = "vectors.bin"
_CHUNKS = "chunks.jsonl"
_CHUNK_OFFSETS = "chunk_offsets.npy"
_IDS = "ids.json"
_DENSE_META = "meta.json"

_POSTINGS = "postings.bin"
_OFFSETS = "offsets.npy"
_TERMS = "terms.json"
_LENGTHS = "lengths.npy"
_DF = "df.npy"
_BM25_META = "meta.json"
_ANN = "ann.faiss"


# ── lexical: packed postings, mmap search ────────────────────────────────

#: How many runs the merge may hold open at once. The final pass opens every remaining run, and a
#: small block cap multiplies runs — a 100k-token cap needs ~1,000 runs at a million documents and
#: ~10,000 at ten million — so without a bound the build fails on the file-descriptor limit as a
#: hard error. Collapsing in groups of this size first keeps the handle count flat.
_MERGE_FAN_IN = 64


def _merge_run_group(paths: Sequence[Path], target: Path) -> None:
    """Merge sorted ``(term, row, tf)`` runs into one, holding at most ``len(paths)`` open.

    Same heap loop as the final merge, writing triples rather than postings. The output is streamed
    through a memmap so a group larger than memory does not have to be materialised.
    """
    handles = [np.load(path, mmap_mode="r") for path in paths]
    total = int(sum(len(handle) for handle in handles))
    sink = np.lib.format.open_memmap(target, mode="w+", dtype=np.int32, shape=(total, 3))
    positions = [0] * len(handles)
    heads: list[tuple[int, int, int, int]] = []
    for r, handle in enumerate(handles):
        if len(handle):
            cell = handle[0]
            heapq.heappush(heads, (int(cell[0]), r, int(cell[1]), int(cell[2])))
            positions[r] = 1
    written = 0
    while heads:
        term, r, row, tf = heapq.heappop(heads)
        sink[written] = (term, row, tf)
        written += 1
        handle = handles[r]
        position = positions[r]
        if position < len(handle):
            nxt = handle[position]
            heapq.heappush(heads, (int(nxt[0]), r, int(nxt[1]), int(nxt[2])))
            positions[r] = position + 1
    sink.flush()
    del sink


class OnDiskBM25:
    """BM25 over a packed, memory-mapped postings file.

    Under ``directory``: ``postings.bin`` (int32 ``(row, tf)`` pairs, grouped by term),
    ``offsets.npy`` (int64, ``n_terms + 1``), ``terms.json`` (term -> id), ``df.npy``,
    ``lengths.npy`` (int32) and ``meta.json``.
    """

    def __init__(
        self,
        directory: Path,
        k1: float = K1,
        b: float = B,
        drop_stopwords: bool = True,
        stem: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.k1 = float(k1)
        self.b = float(b)
        self.drop_stopwords = bool(drop_stopwords)
        self.stem = bool(stem)
        self.terms: dict[str, int] = {}
        self.offsets: np.ndarray | None = None
        self.lengths: np.ndarray | None = None
        self.avg_len = 0.0
        self.n_docs = 0
        self._postings: np.ndarray | None = None
        self._df: np.ndarray | None = None

    # -- build -----------------------------------------------------------
    def build(
        self,
        documents: Iterable[tuple[str, str]],
        block_docs: int = 50000,
        block_tokens: int = 1_000_000,
    ) -> None:
        """Stream ``(doc_id, text)`` through sorted runs, then merge to packed form.

        A block is flushed when it reaches ``block_docs`` documents *or* ``block_tokens``
        tokens, whichever comes first. The token cap is what bounds build memory: a block
        of long documents otherwise holds a `term -> count` dict per document in RAM, and
        that one block — not the postings — was the 2.4 GB peak on TREC-COVID. More,
        smaller runs cost a little merge time and no ranking change. The merge collapses runs in
        bounded groups, so a small cap can no longer exhaust the file-descriptor limit — but the
        collapse rewrites the runs once, and at 400,000 documents a 250k cap measured **65% slower**
        (396 s against 240 s) for 11% less peak RSS, so the default stays at 1M: the bound is there
        for corpora large enough to need it, not as a licence to lower the cap.
        """
        if block_docs < 1:
            raise ValueError(f"block_docs must be positive, got {block_docs}")
        if block_tokens < 1:
            raise ValueError(f"block_tokens must be positive, got {block_tokens}")
        self.directory.mkdir(parents=True, exist_ok=True)
        run_dir = self.directory / "runs"
        run_dir.mkdir(exist_ok=True)
        term_to_id: dict[str, int] = {}
        ids: list[str] = []
        lengths: list[int] = []
        doc_freq: list[int] = []
        runs: list[Path] = []
        # The block holds *entries*, not one `term -> count` dict per document. Term ids are
        # assigned as the documents stream, in the same first-appearance order the deferred
        # loop produced, so the runs written below stay byte-identical — what changes is that
        # a document's counts dict is transient instead of one of ~10,000 alive at once. That
        # is where the build's peak actually is: profiling shows the persistent state
        # (vocabulary, document frequencies, ids, lengths) is a few MB against a peak
        # approaching a gigabyte.
        block_term_ids: list[int] = []
        block_rows: list[int] = []
        block_tfs: list[int] = []
        block_docs_seen = 0
        block_token_count = 0

        def flush() -> None:
            if not block_rows:
                return
            arr = np.stack(
                [np.asarray(block_term_ids, np.int32), np.asarray(block_rows, np.int32),
                 np.asarray(block_tfs, np.int32)],
                axis=1,
            )
            order = np.lexsort((arr[:, 1], arr[:, 0]))
            run_path = run_dir / f"run-{len(runs):04d}.npy"
            np.save(run_path, arr[order])
            runs.append(run_path)
            block_term_ids.clear()
            block_rows.clear()
            block_tfs.clear()

        for doc_id, text in documents:
            counts: dict[str, int] = {}
            for token in tokenize(text, stem=self.stem):
                counts[token] = counts.get(token, 0) + 1
            row = len(ids)
            ids.append(doc_id)
            lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                tid = term_to_id.get(term)
                if tid is None:
                    tid = len(term_to_id)
                    term_to_id[term] = tid
                    doc_freq.append(0)
                doc_freq[tid] += 1
                block_term_ids.append(tid)
                block_rows.append(row)
                block_tfs.append(tf)
            block_token_count += sum(counts.values())
            block_docs_seen += 1
            if block_docs_seen >= block_docs or block_token_count >= block_tokens:
                flush()
                block_docs_seen = 0
                block_token_count = 0
        flush()

        self.n_docs = len(ids)
        self.terms = term_to_id
        self.lengths = np.asarray(lengths, dtype=np.int32)
        self.avg_len = float(self.lengths.mean()) if len(self.lengths) else 0.0
        n_terms = len(term_to_id)
        df = np.asarray(doc_freq, dtype=np.int64)
        self._df = df
        offsets = np.zeros(n_terms + 1, dtype=np.int64)

        # K-way merge over the sorted runs. A heap holding one head row per run bounds
        # memory to one run's rows for the current term plus the heap; popping the
        # smallest run index first drains each run's block for a term before the next
        # run, which reproduces the previous per-run scan byte for byte — and therefore
        # the identical ranking.
        # Collapse the runs into at most `_MERGE_FAN_IN` before the final pass, so the number of
        # open handles does not grow with the number of blocks. The heap loop below is unchanged
        # and sees exactly the same (term, row, tf) order, so the ranking is untouched.
        level = list(runs)
        # A fan-in below 2 can never reduce the level, so it would spin forever; clamp rather than
        # trusting the constant to be sane.
        fan_in = max(2, int(_MERGE_FAN_IN))
        while len(level) > fan_in:
            collapsed: list[Path] = []
            for start in range(0, len(level), fan_in):
                group = level[start : start + fan_in]
                if len(group) == 1:
                    collapsed.append(group[0])
                    continue
                target = run_dir / f"merge-{len(level):05d}-{start:05d}.npy"
                _merge_run_group(group, target)
                collapsed.append(target)
            if len(collapsed) >= len(level):  # pragma: no cover - defensive
                raise RuntimeError(f"run merge made no progress at fan-in {fan_in}")
            level = collapsed
        runs = level
        handles = [np.load(path, mmap_mode="r") for path in runs]
        positions = [0] * len(handles)
        heads: list[tuple[int, int, int, int]] = []  # (term, run, row, tf)
        for r, handle in enumerate(handles):
            if len(handle):
                row0 = handle[0]
                heapq.heappush(heads, (int(row0[0]), r, int(row0[1]), int(row0[2])))
                positions[r] = 1
        written = 0
        closed_terms = 0
        postings_path = self.directory / _POSTINGS
        with open(postings_path, "wb") as out:
            while heads:
                term, r, row, tf = heapq.heappop(heads)
                out.write(np.asarray([row, tf], dtype=np.int32).tobytes())
                written += 1
                handle = handles[r]
                position = positions[r]
                if position < len(handle):
                    nxt = handle[position]
                    heapq.heappush(heads, (int(nxt[0]), r, int(nxt[1]), int(nxt[2])))
                    positions[r] = position + 1
                if not heads or heads[0][0] != term:
                    for earlier in range(closed_terms, term + 1):
                        offsets[earlier + 1] = written
                    closed_terms = term + 1
        np.save(self.directory / _DF, df)
        np.save(self.directory / _OFFSETS, offsets)
        self.offsets = offsets
        self._postings = np.memmap(postings_path, dtype=np.int32, mode="r").reshape(-1, 2)
        (self.directory / _TERMS).write_text(json.dumps(term_to_id), encoding="utf-8")
        np.save(self.directory / _LENGTHS, self.lengths)
        (self.directory / _IDS).write_text(json.dumps(ids), encoding="utf-8")
        (self.directory / _BM25_META).write_text(
            json.dumps({
                "version": 1,
                "n_docs": self.n_docs,
                "n_terms": n_terms,
                "n_postings": written,
                "avg_len": self.avg_len,
                "k1": self.k1,
                "b": self.b,
                "drop_stopwords": self.drop_stopwords,
                "stem": self.stem,
            }),
            encoding="utf-8",
        )
        # The run directory is scratch: beside the runs themselves the collapse may have left
        # intermediate `merge-*.npy` files, so clear the directory rather than only `runs`, or the
        # build fails at the last step having already produced a complete index.
        for path in run_dir.glob("*"):
            path.unlink()
        run_dir.rmdir()
        for handle in handles:
            del handle

    # -- open / search ---------------------------------------------------
    @classmethod
    def open(cls, directory: Path) -> "OnDiskBM25":
        directory = Path(directory)
        meta = json.loads((directory / _BM25_META).read_text(encoding="utf-8"))
        index = cls(
            directory,
            k1=meta["k1"],
            b=meta["b"],
            drop_stopwords=meta["drop_stopwords"],
            stem=meta["stem"],
        )
        index.n_docs = meta["n_docs"]
        index.avg_len = meta["avg_len"]
        index.terms = json.loads((directory / _TERMS).read_text(encoding="utf-8"))
        index.offsets = np.load(directory / _OFFSETS)
        index.lengths = np.load(directory / _LENGTHS, mmap_mode="r")
        index._df = np.load(directory / _DF)
        index._postings = np.memmap(directory / _POSTINGS, dtype=np.int32, mode="r").reshape(-1, 2)
        return index

    @staticmethod
    def exists_on_disk(directory: Path) -> bool:
        directory = Path(directory)
        return (directory / _BM25_META).is_file() and (directory / _POSTINGS).is_file()

    def _idf(self, term_id: int) -> float:
        df = int(self._df[term_id]) if self._df is not None else 0
        if not self.n_docs or not df:
            return 0.0
        return max(math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5)), 0.0)

    def __len__(self) -> int:
        return self.n_docs

    def search(self, query: str, k: int = 20) -> list[LexicalHit]:
        """Return the ``k`` highest-scoring chunks (id + score), mmap-backed."""
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")
        if not self.n_docs or self._postings is None or self.offsets is None:
            return []
        terms = tokenize(query, stem=self.stem)
        if self.drop_stopwords:
            terms = [t for t in terms if t not in STOPWORDS]
        if not terms:
            return []
        scores = np.zeros(self.n_docs, dtype=np.float64)
        for term in terms:
            term_id = self.terms.get(term)
            if term_id is None:
                continue
            idf = self._idf(term_id)
            if idf == 0.0:
                continue
            lo, hi = int(self.offsets[term_id]), int(self.offsets[term_id + 1])
            if hi <= lo:
                continue
            block = np.asarray(self._postings[lo:hi])
            rows = block[:, 0].astype(np.int64)
            tfs = block[:, 1].astype(np.float64)
            lengths = self.lengths[rows].astype(np.float64)
            denom = tfs + self.k1 * (1.0 - self.b + self.b * lengths / (self.avg_len or 1.0))
            scores[rows] += idf * (tfs * (self.k1 + 1.0)) / denom
        # Stable descending sort matches the in-memory implementation's tie-break, so the
        # two rankings are identical (verified in tools/ondisk.py).
        order = np.argsort(-scores, kind="stable")[:k]
        ids = self._ids
        return [LexicalHit(chunk_id=ids[int(i)], score=float(scores[int(i)])) for i in order if scores[int(i)] > 0.0]

    @property
    def _ids(self) -> list[str]:
        if not hasattr(self, "_id_list"):
            self._id_list = json.loads((self.directory / _IDS).read_text(encoding="utf-8"))
        return self._id_list

    def stats(self) -> dict[str, Any]:
        return {"documents": self.n_docs, "terms": len(self.terms), "avg_len": round(self.avg_len, 2)}

    def disk_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.directory.glob("*") if p.is_file())


# ── dense: streaming, memory-mapped vectors + deferred chunk text ────────

class OnDiskVectorIndex:
    """Append-only dense index exposing the engine's vector-index interface.

    Vectors live in a memory-mapped ``vectors.bin``; chunk records live in a newline
    ``chunks.jsonl`` whose byte offsets are saved, so :meth:`get` and :meth:`search` read
    one chunk's text on demand instead of keeping every chunk resident. ``vector`` is a
    zero-copy view into the mmap.
    """

    def __init__(self, directory: Path, dim: int = 0) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.dim = int(dim)
        self.count = 0
        self._ids: list[str] = []
        self._positions: dict[str, int] = {}
        self._vectors: np.ndarray | None = None
        self._offsets: np.ndarray | None = None
        self._fd: int | None = None
        self._ann: Any | None = None
        self.nprobe: int = 16

    # -- introspection ---------------------------------------------------
    def __len__(self) -> int:
        return self.count

    def get(self, chunk_id: str) -> Chunk | None:
        position = self._positions.get(chunk_id)
        if position is None:
            return None
        return self._chunk_at(position)

    def vector(self, chunk_id: str) -> np.ndarray | None:
        position = self._positions.get(chunk_id)
        if position is None or self._vectors is None:
            return None
        return self._vectors[position]

    @property
    def chunks(self) -> list[Chunk]:
        """Every chunk record materialised (introspection only; prefer ``get``)."""
        return [self._chunk_at(i) for i in range(self.count)]

    def document_count(self) -> int:
        return len({chunk.doc_id for chunk in self.chunks})

    def paths(self) -> list[str]:
        return sorted({chunk.path for chunk in self.chunks})

    def _chunk_at(self, position: int) -> Chunk:
        if self._offsets is None:
            raise IndexError_("on-disk chunk offsets are missing")
        start = int(self._offsets[position])
        end = int(self._offsets[position + 1]) if position + 1 < len(self._offsets) else os.fstat(self._chunk_fd())[1]
        # os.pread on a shared descriptor keeps this thread-safe: no seek cursor is
        # shared between concurrent requests.
        raw = os.pread(self._chunk_fd(), end - start, start)
        record = json.loads(raw.decode("utf-8"))
        return Chunk(
            chunk_id=record["chunk_id"],
            doc_id=record["doc_id"],
            path=record["path"],
            text=record["text"],
            index=record["index"],
            start_char=record["start_char"],
            end_char=record["end_char"],
        )

    def _chunk_fd(self) -> int:
        if self._fd is None:
            self._fd = os.open(self.directory / _CHUNKS, os.O_RDONLY)
        return self._fd

    # -- mutation --------------------------------------------------------
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        """Append chunks and vectors. Append-only: replacing an id is not supported."""
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return 0
        normalised = _normalise(vectors)
        if self.dim and normalised.shape[1] != self.dim:
            raise IndexError_(
                f"embedding dimension changed: index holds {self.dim}, new vectors are "
                f"{normalised.shape[1]}. Rebuild the index when the checkpoint changes."
            )
        if not self.dim:
            self.dim = int(normalised.shape[1])
        duplicates = [chunk.chunk_id for chunk in chunks if chunk.chunk_id in self._positions]
        if duplicates:
            raise IndexError_(
                f"the on-disk index is append-only; {len(duplicates)} chunk id(s) already "
                "exist. Rebuild the index instead of re-indexing a path incrementally."
            )
        path = self.directory / _VECTORS
        mode = "r+b" if path.exists() else "w+b"
        with open(path, mode) as handle:
            handle.seek(self.count * self.dim * 4)
            handle.write(np.asarray(normalised, dtype=np.float32).tobytes())
        # Append the chunk records and their start offsets. ``start`` is the current EOF
        # of chunks.jsonl, which equals the last saved offset, so the existing offset
        # table (whose final entry is that EOF) is spliced with the new run.
        chunks_path = self.directory / _CHUNKS
        offsets = self.directory / _CHUNK_OFFSETS
        start = chunks_path.stat().st_size if chunks_path.exists() else 0
        byte_offsets = [int(start)]
        with open(chunks_path, "ab") as handle:
            for chunk in chunks:
                payload = json.dumps({
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "path": chunk.path,
                    "text": chunk.text,
                    "index": chunk.index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }, ensure_ascii=False).encode("utf-8") + b"\n"
                handle.write(payload)
                byte_offsets.append(byte_offsets[-1] + len(payload))
        if offsets.exists():
            existing = np.load(offsets)
            merged = np.concatenate(
                [existing[:-1] if existing.size else existing, np.asarray(byte_offsets, dtype=np.int64)]
            )
        else:
            merged = np.asarray(byte_offsets, dtype=np.int64)
        np.save(offsets, merged)
        for position, chunk in enumerate(chunks):
            self._positions[chunk.chunk_id] = self.count + position
        self._ids.extend(chunk.chunk_id for chunk in chunks)
        self.count += len(chunks)
        return 0

    def finalize(self) -> None:
        (self.directory / _IDS).write_text(json.dumps(self._ids), encoding="utf-8")
        (self.directory / _DENSE_META).write_text(
            json.dumps({"version": 1, "dim": self.dim, "count": self.count}),
            encoding="utf-8",
        )

    def remove_paths(self, paths: Sequence[str]) -> int:
        raise IndexError_(
            "the on-disk backend is append-only and cannot drop paths; rebuild the index"
        )

    # -- open / search ---------------------------------------------------
    @classmethod
    def open(cls, directory: Path) -> "OnDiskVectorIndex":
        directory = Path(directory)
        meta = json.loads((directory / _DENSE_META).read_text(encoding="utf-8"))
        index = cls(directory, meta["dim"])
        index.count = int(meta["count"])
        index._ids = json.loads((directory / _IDS).read_text(encoding="utf-8"))
        index._positions = {chunk_id: row for row, chunk_id in enumerate(index._ids)}
        index._vectors = np.memmap(
            directory / _VECTORS, dtype=np.float32, mode="r", shape=(index.count, index.dim)
        )
        index._offsets = np.load(directory / _CHUNK_OFFSETS)
        ann_path = directory / _ANN
        if ann_path.is_file():
            try:
                import faiss

                index._ann = faiss.read_index(str(ann_path))
            except Exception:  # noqa: BLE001 - fall back to the exact mmap scan
                index._ann = None
        return index

    @staticmethod
    def exists_on_disk(directory: Path) -> bool:
        directory = Path(directory)
        return (directory / _DENSE_META).is_file() and (directory / _VECTORS).is_file()

    def build_ann(
        self,
        *,
        nlist: int | None = None,
        m: int = 48,
        nbits: int = 8,
        pq: bool = True,
        seed: int = 0,
    ) -> bool:
        """Build a quantized IVF index over the mmapped vectors. False without faiss.

        Product quantization is what makes the ANN small enough to keep in RAM at
        millions of vectors (384-d fp32 is 1.5 KB/vector; PQ at ``m=48`` is 48 B), while
        the full vectors stay mmapped for the exact fallback. ``pq=False`` builds a flat
        IVF (exact given enough probes) for tests and small corpora.
        """
        if self._vectors is None or not self.count:
            return False
        try:
            import faiss
        except ImportError:  # pragma: no cover - optional dependency
            return False
        nlist = nlist or max(1, int(np.sqrt(self.count)))
        nlist = max(1, min(nlist, self.count))
        quantiser = faiss.IndexFlatIP(self.dim)
        # PQ's sub-quantizers need at least ``2 ** nbits`` training points; below that
        # faiss cannot train them, so a small corpus gets a flat IVF (still exact with
        # enough probes) instead of an exception.
        if pq and self.count >= (1 << nbits):
            # PQ requires the sub-vector count to divide the dimension.
            sub = max(1, min(int(m), self.dim))
            while self.dim % sub:
                sub -= 1
            index = faiss.IndexIVFPQ(
                quantiser, self.dim, nlist, sub, nbits, faiss.METRIC_INNER_PRODUCT
            )
        else:
            index = faiss.IndexIVFFlat(quantiser, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        vectors = np.asarray(self._vectors, dtype=np.float32)
        train_count = max(nlist, min(self.count, max(10000, nlist * 40)))
        index.train(vectors[:train_count])
        index.add(vectors)
        index.nprobe = min(self.nprobe, nlist)
        self._ann = index
        faiss.write_index(index, str(self.directory / _ANN))
        return True

    @property
    def is_approximate(self) -> bool:
        return self._ann is not None

    def _rank(self, query: np.ndarray, count: int, block: int) -> list[tuple[float, int]]:
        """The top ``count`` ``(score, position)`` pairs, by ANN when available else exact."""
        if self._ann is not None:
            self._ann.nprobe = min(self.nprobe, self._ann.nlist)
            similarities, positions = self._ann.search(
                query.reshape(1, -1).astype(np.float32), min(count, self.count)
            )
            return [
                (float(similarity), int(position))
                for similarity, position in zip(similarities[0], positions[0])
                if position >= 0
            ]
        best: list[tuple[float, int]] = []
        for start in range(0, self.count, block):
            stop = min(start + block, self.count)
            scores = self._vectors[start:stop] @ query
            take = min(count, len(scores))
            local = np.argpartition(-scores, take - 1)[:take]
            best.extend((float(scores[int(i)]), start + int(i)) for i in local)
        best.sort(key=lambda item: -item[0])
        return best[:count]

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        block: int = 200000,
        *,
        allow: Callable[[str | None], bool] | None = None,
    ) -> list[Hit]:
        """Return the ``k`` nearest chunks, reading only the hits' text from disk.

        ``allow`` is the same path predicate `VectorIndex.search` takes, so the two
        backends are interchangeable: a trust-tier or tenant filter is applied **inside**
        the scan rather than after it, and a short filtered window deepens instead of
        returning fewer candidates than asked for. Only the candidate positions are read
        from disk to evaluate the predicate, not the whole corpus.
        """
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")
        if not self.count or self._vectors is None:
            return []
        query = _normalise(np.asarray(query_vector, dtype=np.float32))[0]
        if query.shape[0] != self.dim:
            raise IndexError_(
                f"query vector has {query.shape[0]} dimensions but the index holds {self.dim}"
            )
        if allow is None:
            return [
                Hit(chunk=self._chunk_at(position), score=score)
                for score, position in self._rank(query, min(k, self.count), block)
            ]
        # The predicate needs each candidate's path, which lives in its record. Rank a
        # window, read only those records, and deepen if too few survive.
        window = min(self.count, k)
        while True:
            kept: list[Hit] = []
            for score, position in self._rank(query, window, block):
                chunk = self._chunk_at(position)
                if allow(chunk.path):
                    kept.append(Hit(chunk=chunk, score=score))
                    if len(kept) >= k:
                        break
            if len(kept) >= k or window >= self.count:
                return kept[:k]
            window = min(self.count, max(window * 4, k))

    def stats(self) -> dict[str, Any]:
        return {
            "chunks": self.count,
            "documents": self.document_count(),
            "dim": self.dim,
            "paths": len(self.paths()),
        }

    def disk_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.directory.glob("*") if p.is_file())


class IndexError_(Exception):
    """Raised when an on-disk index cannot be used or updated."""
