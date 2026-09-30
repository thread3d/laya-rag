#!/usr/bin/env python
"""A learned-sparse (SPLADE-style) retrieval channel, for the BEIR harness.

Lexical BM25 matches terms that appear; a learned-sparse model *expands* the query and the
document into weighted vocabulary terms, so it can match paraphrases while keeping an
inverted index. Published SPLADE scores 0.704 on SciFact and 0.521 on ArguAna — the latter
is the dataset where this project's dense channel is furthest behind, so it is the most
promising missing channel.

Implementation notes:

* The checkpoint is OpenSearch's neural-sparse encoder because it ships ``safetensors``.
  The SPLADE checkpoints on the Hub ship only ``pytorch_model.bin``, and transformers
  4.57 refuses to ``torch.load`` them on torch < 2.6 (CVE-2025-32434); this project pins
  torch 2.2.2 because macOS Intel has no newer wheel. A safetensors checkpoint is loadable
  through the same interface without disabling that guard.
* Encoding is the standard SPLADE activation: ``log(1 + relu(logits))`` max-pooled over
  the sequence, then the top ``k`` terms kept per text.
* Retrieval is a sparse dot product over an inverted index, so cost scales with matching
  terms rather than corpus size.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


class SpladeEncoder:
    """Encodes text into a sparse ``{term_id: weight}`` vector."""

    def __init__(
        self,
        model_id: str = "opensearch-project/opensearch-neural-sparse-encoding-v1",
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 256,
        top_k: int = 128,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.top_k = top_k
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._lock = threading.RLock()

    def ensure_loaded(self) -> "SpladeEncoder":
        with self._lock:
            if self._model is not None:
                return self
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            device = self.device
            if device in (None, "auto"):
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_id).to(device).eval()
            self._torch = torch
            self.device = device
            return self

    def encode(self, texts: Sequence[str]) -> list[dict[int, float]]:
        """Sparse vectors, one per text, keeping the top ``k`` active terms."""
        self.ensure_loaded()
        torch = self._torch
        rows = ["" if text is None else str(text) for text in texts]
        out: list[dict[int, float]] = []
        for start in range(0, len(rows), self.batch_size):
            batch = rows[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                output = self._model(**encoded)
            # Some checkpoints return a tuple rather than a ModelOutput.
            logits = output.logits if hasattr(output, "logits") else output[0]
            activations = torch.log1p(torch.relu(logits))
            mask = encoded["attention_mask"].unsqueeze(-1).to(activations.dtype)
            pooled = (activations * mask).max(dim=1).values
            values, indices = torch.topk(pooled, k=min(self.top_k, pooled.shape[1]), dim=1)
            for row_values, row_indices in zip(values.cpu(), indices.cpu()):
                out.append(
                    {
                        int(term): float(weight)
                        for term, weight in zip(row_indices.tolist(), row_values.tolist())
                        if weight > 0.0
                    }
                )
        return out


class SparseIndex:
    """Inverted index over sparse vectors; scores by sparse dot product."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._postings: dict[int, list[tuple[int, float]]] = {}

    def __len__(self) -> int:
        return len(self._ids)

    def build(self, ids: Sequence[str], vectors: Sequence[Mapping[int, float]]) -> None:
        self._ids = list(ids)
        postings: dict[int, list[tuple[int, float]]] = {}
        for row, vector in enumerate(vectors):
            for term, weight in vector.items():
                postings.setdefault(term, []).append((row, weight))
        self._postings = postings

    def search(self, vector: Mapping[int, float], k: int = 100) -> list[tuple[str, float]]:
        scores: dict[int, float] = {}
        for term, weight in vector.items():
            for row, doc_weight in self._postings.get(term, ()):  # sparse dot product
                scores[row] = scores.get(row, 0.0) + weight * doc_weight
        ordered = sorted(scores.items(), key=lambda item: -item[1])[:k]
        return [(self._ids[row], score) for row, score in ordered]


def load_or_encode(
    encoder: SpladeEncoder,
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
    encoder: SpladeEncoder,
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
