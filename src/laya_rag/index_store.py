"""Persistent dense vector index.

Laya supplies the embedder and a cosine top-k helper, but no storage: its only
similarity code ranks choice labels in memory. This module owns the corpus side —
storing chunk vectors on disk and answering nearest-neighbour queries over them.

Vectors are L2-normalised once at write time, so a query is a single matrix
multiply. For the corpus sizes this engine targets (thousands to low millions of
chunks) an exhaustive scan is both exact and fast enough; an approximate index
would add a dependency without changing the answers.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .chunking import Chunk

INDEX_META = "index.json"
INDEX_VECTORS = "vectors.npy"
INDEX_ANN = "ann.faiss"
FORMAT_VERSION = 1


class IndexError_(Exception):
    """Raised when an index on disk cannot be used."""


@dataclass(frozen=True)
class Hit:
    """One retrieval result, with the recall channels that produced it."""

    chunk: Chunk
    score: float
    channels: tuple[str, ...] = ()


def _normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class VectorIndex:
    """An append-capable, normalised dense index over :class:`Chunk` records."""

    def __init__(self, dim: int | None = None) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray = np.zeros((0, dim or 0), dtype=np.float32)
        self._positions: dict[str, int] = {}
        # Optional approximate index. An exhaustive scan is exact and fast to a few
        # hundred thousand vectors; past that it is linear, and an IVF index (measured in
        # tools/ann_bench.py) trades a little recall for an order of magnitude.
        self._ann: Any | None = None
        self.ann_nprobe: int = 16

    # -- introspection ---------------------------------------------------
    @property
    def dim(self) -> int:
        return int(self._vectors.shape[1]) if self._vectors.size else 0

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def get(self, chunk_id: str) -> Chunk | None:
        """Look up one chunk by id."""
        position = self._positions.get(chunk_id)
        return self._chunks[position] if position is not None else None

    def vector(self, chunk_id: str) -> np.ndarray | None:
        """The normalised embedding for one chunk, or ``None`` if it is not indexed.

        Recall diversification needs this: the fused candidate list is ranked by channels
        whose scores are not comparable, so redundancy has to be measured in the embedding
        space instead.
        """
        position = self._positions.get(chunk_id)
        if position is None or not self._vectors.size:
            return None
        return self._vectors[position]

    def document_count(self) -> int:
        return len({chunk.doc_id for chunk in self._chunks})

    def paths(self) -> list[str]:
        return sorted({chunk.path for chunk in self._chunks})

    # -- mutation --------------------------------------------------------
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        """Insert or replace chunks. Returns how many already existed."""
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return 0
        normalised = _normalise(vectors)
        if self._vectors.size and normalised.shape[1] != self.dim:
            raise IndexError_(
                f"embedding dimension changed: index holds {self.dim}, new vectors are "
                f"{normalised.shape[1]}. Rebuild the index when the checkpoint changes."
            )
        replaced = 0
        fresh_chunks: list[Chunk] = []
        fresh_vectors: list[np.ndarray] = []
        for chunk, vector in zip(chunks, normalised):
            position = self._positions.get(chunk.chunk_id)
            if position is None:
                self._positions[chunk.chunk_id] = len(self._chunks) + len(fresh_chunks)
                fresh_chunks.append(chunk)
                fresh_vectors.append(vector)
            else:
                replaced += 1
                self._chunks[position] = chunk
                self._vectors[position] = vector
        if fresh_chunks:
            block = np.asarray(fresh_vectors, dtype=np.float32)
            self._chunks.extend(fresh_chunks)
            self._vectors = block if not self._vectors.size else np.vstack([self._vectors, block])
        return replaced

    def remove_doc_ids(self, doc_ids: Sequence[str]) -> int:
        """Drop every chunk belonging to ``doc_ids``. Returns chunks removed."""
        doomed = set(doc_ids)
        if not doomed:
            return 0
        keep = [i for i, chunk in enumerate(self._chunks) if chunk.doc_id not in doomed]
        removed = len(self._chunks) - len(keep)
        if not removed:
            return 0
        self._chunks = [self._chunks[i] for i in keep]
        self._vectors = self._vectors[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)
        self._reindex()
        return removed

    def remove_paths(self, paths: Sequence[str]) -> int:
        """Drop every chunk read from ``paths`` so a changed file can be re-indexed."""
        wanted = {str(Path(p)) for p in paths}
        doc_ids = {chunk.doc_id for chunk in self._chunks if chunk.path in wanted}
        return self.remove_doc_ids(sorted(doc_ids))

    def clear(self) -> None:
        self._chunks = []
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        self._positions = {}

    def _reindex(self) -> None:
        self._positions = {chunk.chunk_id: i for i, chunk in enumerate(self._chunks)}

    # -- approximate search ----------------------------------------------
    def build_ann(self, nlist: int | None = None) -> bool:
        """Build an IVF index over the current vectors. Returns False without faiss."""
        if not self._vectors.size:
            return False
        try:
            import faiss
        except ImportError:  # pragma: no cover - optional dependency
            return False
        nlist = nlist or max(1, int(np.sqrt(len(self._chunks))))
        quantiser = faiss.IndexFlatIP(self.dim)
        index = faiss.IndexIVFFlat(quantiser, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(self._vectors)
        index.add(self._vectors)
        index.nprobe = min(self.ann_nprobe, nlist)
        self._ann = index
        return True

    @property
    def is_approximate(self) -> bool:
        return self._ann is not None

    # -- query -----------------------------------------------------------
    def search(self, query_vector: np.ndarray, k: int = 10) -> list[Hit]:
        """Return the ``k`` nearest chunks by cosine similarity."""
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")
        if not len(self) or not self._vectors.size:
            return []
        query = _normalise(np.asarray(query_vector, dtype=np.float32))[0]
        if query.shape[0] != self.dim:
            raise IndexError_(
                f"query vector has {query.shape[0]} dimensions but the index holds {self.dim}"
            )
        if self._ann is not None:
            self._ann.nprobe = min(self.ann_nprobe, self._ann.nlist)
            similarities, positions = self._ann.search(
                query.reshape(1, -1).astype(np.float32), min(k, len(self._chunks))
            )
            return [
                Hit(chunk=self._chunks[int(position)], score=float(similarity))
                for similarity, position in zip(similarities[0], positions[0])
                if position >= 0
            ]

        scores = self._vectors @ query
        take = min(k, scores.shape[0])
        # argpartition selects the top-k unordered; sort those k for a stable ranking.
        top = np.argpartition(-scores, take - 1)[:take]
        ordered = top[np.argsort(-scores[top], kind="stable")]
        return [Hit(chunk=self._chunks[int(i)], score=float(scores[int(i)])) for i in ordered]

    # -- persistence -----------------------------------------------------
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "dim": self.dim,
            "count": len(self._chunks),
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "path": c.path,
                    "text": c.text,
                    "index": c.index,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                }
                for c in self._chunks
            ],
        }
        tmp_meta = directory / (INDEX_META + ".tmp")
        tmp_vec = directory / (INDEX_VECTORS + ".tmp")
        tmp_meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        # np.save appends ".npy" to a name that lacks it, so write through a handle.
        with open(tmp_vec, "wb") as handle:
            np.save(handle, self._vectors)
        # Replace-by-rename so a crash mid-save cannot leave a half-written index.
        tmp_meta.replace(directory / INDEX_META)
        tmp_vec.replace(directory / INDEX_VECTORS)
        if self._ann is not None:
            import faiss

            faiss.write_index(self._ann, str(directory / INDEX_ANN))

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        directory = Path(directory)
        meta_path = directory / INDEX_META
        vec_path = directory / INDEX_VECTORS
        if not meta_path.is_file():
            raise IndexError_(f"no index at {directory}: {INDEX_META} is missing")
        if not vec_path.is_file():
            raise IndexError_(f"no index at {directory}: {INDEX_VECTORS} is missing")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        version = meta.get("format_version")
        if version != FORMAT_VERSION:
            raise IndexError_(
                f"index format version {version!r} is not supported (expected {FORMAT_VERSION}); rebuild"
            )
        vectors = np.load(vec_path)
        index = cls(dim=meta.get("dim") or (vectors.shape[1] if vectors.ndim == 2 else 0))
        index._chunks = [
            Chunk(
                chunk_id=record["chunk_id"],
                doc_id=record["doc_id"],
                path=record["path"],
                text=record["text"],
                index=record["index"],
                start_char=record["start_char"],
                end_char=record["end_char"],
            )
            for record in meta.get("chunks", [])
        ]
        index._vectors = np.asarray(vectors, dtype=np.float32)
        index._reindex()
        ann_path = directory / INDEX_ANN
        if ann_path.is_file():
            try:
                import faiss

                index._ann = faiss.read_index(str(ann_path))
            except Exception:  # noqa: BLE001 - fall back to the exact scan
                index._ann = None
        return index

    @staticmethod
    def exists_on_disk(directory: Path) -> bool:
        directory = Path(directory)
        return (directory / INDEX_META).is_file() and (directory / INDEX_VECTORS).is_file()

    def stats(self) -> dict[str, Any]:
        return {
            "chunks": len(self._chunks),
            "documents": self.document_count(),
            "dim": self.dim,
            "paths": len(self.paths()),
        }


def wipe(directory: Path) -> None:
    """Remove an index directory entirely."""
    directory = Path(directory)
    if directory.exists():
        shutil.rmtree(directory)
