"""A learned-sparse (SPLADE-style) retrieval channel.

Lexical BM25 matches terms that appear; a learned-sparse model *expands* the query and the
document into weighted vocabulary terms, so it can match paraphrases while keeping an
inverted index. Measured on the BEIR harness it beats the stemmed lexical channel on every
dataset by +0.03 to +0.14 and is the first additional channel that improves the fusion
consistently (+0.004 to +0.023), which is why it belongs in the engine rather than only in
the benchmark tool.

The checkpoint is OpenSearch's neural-sparse encoder because it ships ``safetensors``: the
SPLADE checkpoints on the Hub ship only ``pytorch_model.bin``, and transformers 4.57 refuses
to ``torch.load`` them on torch < 2.6 (CVE-2025-32434), which this project cannot upgrade
past on macOS Intel.

* Encoding is the standard SPLADE activation ``log(1 + relu(logits))`` max-pooled over the
  sequence, with the top ``k`` terms kept per text.
* Retrieval is a sparse dot product over an inverted index, so cost scales with matching
  terms rather than corpus size.

Torch and transformers are imported only when the encoder loads, so the package's import
path stays light and the index half is pure Python + NumPy.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

SPARSE_FILE = "sparse.json"


class SparseEncoder:
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

    def ensure_loaded(self) -> "SparseEncoder":
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

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def info(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "top_k": self.top_k,
            "max_length": self.max_length,
            "loaded": self.loaded,
        }

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

    def embed_one(self, text: str) -> dict[int, float]:
        """A single text's sparse vector, for the query path."""
        return self.encode([text])[0]


class SparseIndex:
    """Inverted index over sparse vectors; scores by sparse dot product.

    The index is a plain ``term -> [(row, weight)]`` posting list, so a query touches only
    the terms it actually carries. It is small enough to persist as JSON next to the other
    index files, which keeps the no-database deployment story.
    """

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._postings: dict[int, list[tuple[int, float]]] = {}

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def ids(self) -> list[str]:
        return list(self._ids)

    @property
    def terms(self) -> int:
        return len(self._postings)

    def build(self, ids: Sequence[str], vectors: Sequence[Mapping[int, float]]) -> None:
        self._ids = list(ids)
        postings: dict[int, list[tuple[int, float]]] = {}
        for row, vector in enumerate(vectors):
            for term, weight in vector.items():
                postings.setdefault(int(term), []).append((row, float(weight)))
        self._postings = postings

    def search(self, vector: Mapping[int, float], k: int = 100) -> list[tuple[str, float]]:
        scores: dict[int, float] = {}
        for term, weight in vector.items():
            for row, doc_weight in self._postings.get(int(term), ()):  # sparse dot product
                scores[row] = scores.get(row, 0.0) + float(weight) * doc_weight
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [(self._ids[row], score) for row, score in ordered]

    # -- persistence -----------------------------------------------------
    def save(self, path: Any) -> None:
        """Write the posting lists as JSON, atomically."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "ids": self._ids,
            "postings": {
                str(term): [[row, weight] for row, weight in rows]
                for term, rows in self._postings.items()
            },
        }
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(target)

    @classmethod
    def load(cls, path: Any) -> "SparseIndex":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError(
                f"sparse index version {payload.get('version')!r} is unsupported; re-index"
            )
        index = cls()
        index._ids = list(payload.get("ids") or [])
        index._postings = {
            int(term): [(int(row), float(weight)) for row, weight in rows]
            for term, rows in (payload.get("postings") or {}).items()
        }
        return index

    def __repr__(self) -> str:
        return f"SparseIndex(documents={len(self._ids)}, terms={len(self._postings)})"
