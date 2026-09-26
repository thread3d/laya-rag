"""The Laya model handle.

One loaded checkpoint serves both jobs the RAG pipeline needs from Laya:

* **embeddings** for dense retrieval, mean-pooled from the checkpoint encoder via
  ``laya.embed_fn_from_agent``;
* **decisions** for zero-shot relevance re-ranking and answerability gating via
  ``Agent.predict``.

Loading once and sharing the agent matters: the checkpoints are 322M-421M
parameters of fp32 weights, and a second load would double both memory and start
time. The wrapper is deliberately the only place that touches Laya's API, so the
rest of the engine can be tested against a stand-in.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# transformers consults these at import time. A stray TensorFlow install can
# deadlock Laya's loader, and telemetry is unwanted; the Laya checkout's own setup
# script sets the same defaults.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from .config import EngineConfig  # noqa: E402  (import after the env defaults)

CHECKPOINT_SUBFOLDERS: dict[str, str | None] = {
    "laya": None,
    "english": None,
    "multilingual": "multilingual",
    "typed-decisions": "typed-decisions",
}


class ModelUnavailableError(RuntimeError):
    """Raised when the Laya checkpoint cannot be loaded."""


class LayaModel:
    """Lazily-loaded, thread-safe handle on one Laya checkpoint."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self._agent: Any | None = None
        self._embed_fn: Any | None = None
        self._lock = threading.RLock()
        self._dim: int | None = None
        self._load_seconds: float | None = None

    # -- loading ---------------------------------------------------------
    def _resolve_spec(self) -> tuple[str, str | None]:
        """Return ``(model_id_or_path, subfolder)`` for the configured checkpoint."""
        name = self.config.checkpoint
        if name not in CHECKPOINT_SUBFOLDERS:
            known = ", ".join(sorted(CHECKPOINT_SUBFOLDERS))
            raise ModelUnavailableError(f"unknown checkpoint {name!r}; known checkpoints: {known}")
        subfolder = CHECKPOINT_SUBFOLDERS[name]

        bundle = self.config.models_dir / "laya"
        if bundle.is_dir():
            return str(bundle), subfolder

        # Fall back to the standalone Hub repo for this checkpoint.
        standalone = self.config.models_dir / f"laya-{name}"
        if standalone.is_dir():
            return str(standalone), None

        raise ModelUnavailableError(
            f"no Laya checkpoint found under {self.config.models_dir}. "
            "Run the Laya checkout's setup_laya.sh, or set LAYA_RAG_MODELS to a directory "
            "holding laya/ (plus laya-multilingual/ and laya-typed-decisions/)."
        )

    def ensure_loaded(self) -> Any:
        """Load the checkpoint on first use and return the agent."""
        with self._lock:
            if self._agent is not None:
                return self._agent
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ModelUnavailableError(
                    "torch is not installed. Laya needs the pinned CPU build; run the Laya "
                    "checkout's setup_laya.sh."
                ) from exc
            try:
                import laya
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ModelUnavailableError(
                    "the laya package is not installed. Run the Laya checkout's setup_laya.sh."
                ) from exc

            torch.set_num_threads(self.config.torch_threads)
            spec, subfolder = self._resolve_spec()
            # "auto" lets Laya pick CUDA -> MPS -> CPU itself.
            device = None if self.config.device in {"auto", ""} else self.config.device
            started = time.perf_counter()
            try:
                self._agent = laya.load(spec, device=device, subfolder=subfolder)
            except Exception as exc:  # noqa: BLE001 - surfaced with context below
                raise ModelUnavailableError(
                    f"failed to load Laya checkpoint {spec!r} (subfolder={subfolder!r}) "
                    f"on device {self.config.device!r}: {exc}"
                ) from exc
            self._load_seconds = time.perf_counter() - started
            self._embed_fn = laya.cached_embed_fn(
                laya.embed_fn_from_agent(
                    self._agent,
                    max_length=self.config.embed_max_length,
                    batch_size=self.config.embed_batch_size,
                )
            )
            return self._agent

    # -- embeddings ------------------------------------------------------
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts as an ``(n, dim)`` float32 array, mean-pooled by Laya."""
        self.ensure_loaded()
        assert self._embed_fn is not None
        rows = ["" if text is None else str(text) for text in texts]
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = np.asarray(self._embed_fn(rows), dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if self._dim is None:
            self._dim = int(vectors.shape[1])
        return vectors

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string as a 1-D vector."""
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.embed([""])
        return int(self._dim or 0)

    # -- decisions -------------------------------------------------------
    def predict(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        lang: str | None = None,
        max_len: int | None = None,
        head_max_len: int | None = None,
    ) -> dict[str, Any]:
        """Run one Laya forward pass over ``state`` for ``questions``."""
        agent = self.ensure_loaded()
        return agent.predict(
            state,
            dict(questions),
            lang=lang,
            max_len=max_len or self.config.max_len,
            head_max_len=head_max_len or self.config.head_max_len,
        )

    def predict_many(
        self,
        states: Sequence[Any],
        questions: Mapping[str, Any],
        batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        """Batched forward passes; Laya's throughput path for many candidate passages."""
        agent = self.ensure_loaded()
        return agent.predict_batch(
            list(states),
            dict(questions),
            batch_size=batch_size,
            max_len=self.config.max_len,
            head_max_len=self.config.head_max_len,
        )

    # -- introspection ---------------------------------------------------
    def info(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "checkpoint": self.config.checkpoint,
            "device": self.config.device,
            "loaded": self._agent is not None,
            "models_dir": str(self.config.models_dir),
            "max_len": self.config.max_len,
            "head_max_len": self.config.head_max_len,
        }
        if self._dim is not None:
            payload["embedding_dim"] = self._dim
        if self._load_seconds is not None:
            payload["load_seconds"] = round(self._load_seconds, 3)
        return payload


class HashingEmbedder:
    """A dependency-free stand-in embedder for tests.

    It is a hashed bag-of-words projection: deterministic, offline, and good enough
    to exercise retrieval plumbing without downloading 2.3 GB of weights. It is not
    a retrieval model and must never be presented as one.
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in str(text).lower().split():
                token = token.strip(".,;:!?()[]{}\"'`")
                if not token:
                    continue
                out[row, hash(token) % self._dim] += 1.0
        return out

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
