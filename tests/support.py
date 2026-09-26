"""Shared test doubles.

The real Laya checkpoints are 2.3 GB of fp32 weights, so the unit suite runs against
a deterministic stand-in that speaks the same two APIs the engine uses: ``embed`` and
the ``predict`` result. That keeps these tests offline and fast while still
exercising the engine's real retrieval, fusion, ranking and persistence code.

Tests that need the actual checkpoint are marked ``real_weights`` and skip when it is
absent.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from laya_dsh_rag.chunking import Document
from laya_dsh_rag.config import EngineConfig
from laya_dsh_rag.engine import RagEngine
from laya_dsh_rag.model import HashingEmbedder

_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(str(text).lower()))


class StubModel:
    """Deterministic stand-in for :class:`laya_dsh_rag.model.LayaModel`.

    Embeddings are hashed bag-of-words; decisions are query/state token overlap. It
    is deliberately simple: the tests assert on plumbing and ordering, not on
    retrieval accuracy, which only the real checkpoint can establish.
    """

    def __init__(self, dim: int = 64) -> None:
        self._embedder = HashingEmbedder(dim=dim)
        self.predict_many_calls = 0
        self.predict_calls = 0

    @property
    def dim(self) -> int:
        return self._embedder.dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self._embedder.embed(texts)

    def embed_one(self, text: str) -> np.ndarray:
        return self._embedder.embed_one(text)

    def ensure_loaded(self) -> "StubModel":
        return self

    @staticmethod
    def _score(query: str, text: str) -> float:
        """Map query/state token overlap onto a probability-like score.

        Any shared term clears the default thresholds, so tests assert on retrieval
        and gating plumbing; a query sharing nothing scores zero and is gated out.
        """
        wanted = tokens(query)
        if not wanted:
            return 0.0
        overlap = len(wanted & tokens(text)) / len(wanted)
        if overlap == 0.0:
            return 0.0
        return round(min(1.0, 0.5 + overlap), 6)

    @staticmethod
    def _result(qid: str, value: float) -> dict[str, Any]:
        return {
            "model": "stub",
            "answers": {
                qid: {
                    "noul": value,
                    "probabilities": {"false": 1.0 - value, "true": value},
                    "confidence": value,
                    "legend": {"false": "B", "true": "A"},
                }
            },
            "usage": {"input_tokens": 10, "output_tokens": 0},
        }

    def predict_many(
        self,
        states: Sequence[Any],
        questions: Mapping[str, Any],
        batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        self.predict_many_calls += 1
        qid = next(iter(questions))
        out = []
        for state in states:
            state = state or {}
            out.append(self._result(qid, self._score(state.get("query", ""), state.get("passage", ""))))
        return out

    def predict(
        self,
        state: Any,
        questions: Mapping[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        self.predict_calls += 1
        qid = next(iter(questions))
        state = state or {}
        return self._result(qid, self._score(state.get("query", ""), state.get("evidence", "")))

    def info(self) -> dict[str, Any]:
        return {"model": "stub", "checkpoint": "stub", "embedding_dim": self.dim}


def make_config(tmp_path: Path, **overrides: Any) -> EngineConfig:
    """An engine config rooted in ``tmp_path`` so tests never touch the real index.

    ``dsh_settings`` points at a path that does not exist on purpose: the default is
    the developer's own ``~/.dsh/settings.yaml``, and a test that quietly picked up
    this machine's composition model would not be testing anything reproducible.
    """
    defaults: dict[str, Any] = {
        "index_dir": tmp_path / "index",
        "models_dir": tmp_path / "models",
        "chunk_chars": 400,
        "chunk_overlap_chars": 80,
        "min_chunk_chars": 40,
        "retrieve_k": 10,
        "rerank_k": 3,
        "relevance_threshold": 0.3,
        "answerability_threshold": 0.3,
        "dsh_settings": tmp_path / "dsh-settings-absent.yaml",
    }
    defaults.update(overrides)
    return EngineConfig(**defaults)


def write_corpus(root: Path) -> Path:
    """A small corpus with an unambiguous answer per file."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "refunds.md").write_text(
        "# Refund policy\n\n"
        "Refunds are processed within five business days to the original payment method. "
        "Contact billing support if the refund has not arrived after ten business days.\n\n"
        "Partial refunds are available for annual plans within thirty days of purchase.\n",
        encoding="utf-8",
    )
    (root / "shipping.md").write_text(
        "# Shipping\n\n"
        "Shipping is free on orders over fifty dollars in the continental United States. "
        "International shipping is calculated at checkout by destination and weight.\n",
        encoding="utf-8",
    )
    (root / "security.md").write_text(
        "# Account security\n\n"
        "Reset your password from Settings, then Security. Two-factor authentication "
        "supports authenticator apps and hardware security keys.\n",
        encoding="utf-8",
    )
    (root / "notes.bin").write_bytes(b"\x00\x01\x02binary\x00content")
    (root / "empty.txt").write_text("   \n\n  ", encoding="utf-8")
    return root


def stub_engine(tmp_path: Path, **overrides: Any) -> RagEngine:
    """An engine backed by :class:`StubModel` and a temp index."""
    return RagEngine(make_config(tmp_path, **overrides), model=StubModel())


def document(path: str, text: str, doc_id: str = "d1") -> Document:
    return Document(doc_id=doc_id, path=path, text=text)
