#!/usr/bin/env python
"""A learned-sparse (SPLADE-style) retrieval channel, for the BEIR harness.

The encoder and inverted index now live in ``laya_rag.sparse`` so the engine can use the
same implementation; this module keeps the harness-only pieces (caching encoded vectors and
producing rankings) and re-exports the classes under their historical names.

Why this checkpoint: OpenSearch's neural-sparse encoder ships ``safetensors``. The SPLADE
checkpoints on the Hub ship only ``pytorch_model.bin``, and transformers 4.57 refuses to
``torch.load`` them on torch < 2.6 (CVE-2025-32434); this project pins torch 2.2.2 because
macOS Intel has no newer wheel. A safetensors checkpoint is loadable through the same
interface without disabling that guard.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laya_rag.sparse import SPARSE_FILE, SparseEncoder, SparseIndex  # noqa: E402

#: Historical name used by ``tools/beir.py`` and the harness reports.
SpladeEncoder = SparseEncoder

__all__ = [
    "SPARSE_FILE",
    "SparseEncoder",
    "SparseIndex",
    "SpladeEncoder",
    "load_or_encode",
    "sparse_rankings",
]


def load_or_encode(
    encoder: SparseEncoder,
    texts: Sequence[str],
    cache_path: Path,
    use_cache: bool = True,
) -> list[dict[int, float]]:
    """Encode once and keep the sparse vectors, which are slow to recompute."""
    cache_path = Path(str(cache_path) + ".json")
    if use_cache and cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("count") == len(texts):
            print(f"  reusing cached sparse vectors ({cache_path.name})", flush=True)
            return [{int(term): float(weight) for term, weight in row.items()} for row in payload["vectors"]]
    started = time.perf_counter()
    vectors = encoder.encode(texts)
    print(f"  encoded {len(texts)} texts in {time.perf_counter() - started:.1f}s", flush=True)
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"count": len(texts), "vectors": vectors}), encoding="utf-8"
        )
    return vectors


def sparse_rankings(
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    encoder: SparseEncoder,
    top_k: int = 100,
    doc_vectors: Sequence[Mapping[int, float]] | None = None,
) -> tuple[dict[str, list[str]], float]:
    """Rank documents per query with the sparse channel."""
    doc_ids = list(corpus)
    started = time.perf_counter()
    if doc_vectors is None:
        doc_vectors = encoder.encode([corpus[doc_id] for doc_id in doc_ids])
    index = SparseIndex()
    index.build(doc_ids, doc_vectors)
    query_vectors = encoder.encode(list(queries.values()))
    rankings = {
        query_id: [doc_id for doc_id, _ in index.search(vector, k=top_k)]
        for (query_id, _), vector in zip(queries.items(), query_vectors)
    }
    return rankings, time.perf_counter() - started
