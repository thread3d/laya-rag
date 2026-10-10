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

import json
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

from .config import REPO_ROOT, EngineConfig  # noqa: E402  (import after the env defaults)

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
            # A checkpoint the engine trained itself rather than one shipped under models/:
            # a directory path, absolute or relative to the project root. This is what lets
            # a fine-tuned encoder be served through the Laya-native path (EMBED_MODEL="").
            candidate = Path(name).expanduser()
            if not candidate.is_absolute():
                candidate = REPO_ROOT / candidate
            if candidate.is_dir() and (candidate / "model.safetensors").is_file():
                return str(candidate), None
            known = ", ".join(sorted(CHECKPOINT_SUBFOLDERS))
            raise ModelUnavailableError(
                f"unknown checkpoint {name!r}: not one of {known} and not a checkpoint "
                f"directory (no {candidate / 'model.safetensors'})"
            )
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

    def weights_path(self) -> Path:
        """The directory holding the weights this handle serves (checkpoint root + subfolder).

        Embedding caches are named from a model's basename, which does not move when a
        retrained checkpoint is packaged at the same path; exposing the resolved directory lets
        a caller fingerprint its contents and notice the change.
        """
        spec, subfolder = self._resolve_spec()
        root = Path(spec)
        return root / subfolder if subfolder else root

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


def saved_pooling(model_id: str) -> str | None:
    """The pooling a saved SentenceTransformer directory was trained with, if it says.

    Returns ``"cls"`` or ``"mean"`` from ``1_Pooling/config.json``, or ``None`` for a Hub
    id (nothing local to read) or an unrecognised config. Exposed so the benchmark tools
    resolve pooling the same way the engine does.
    """
    if not model_id:
        return None
    path = Path(model_id)
    config = path / "1_Pooling" / "config.json"
    if not path.is_dir() or not config.is_file():
        return None
    try:
        data = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("pooling_mode_cls_token"):
        return "cls"
    if data.get("pooling_mode_mean_tokens"):
        return "mean"
    return None


class BiEncoderModel:
    """A retrieval-trained bi-encoder, used as the dense channel instead of Laya's encoder.

    Laya's encoder is mean-pooled and not retrieval-trained — its own README says a
    dedicated bi-encoder "will usually shortlist better". Measured on BEIR, that is an
    understatement: Laya's encoder scores nDCG@10 ~0.014 where BM25 scores ~0.65, and
    fusing it at equal weight costs ~43% of BM25's score. A retrieval-trained model makes
    the dense channel worth fusing again (SciFact 0.654 → 0.696, NFCorpus 0.304 → 0.352).

    This is only the *embedder*. Laya still owns the decisions — relevance re-ranking and
    answerability — so the two models are loaded side by side and ``RagEngine`` keeps
    them separate. Enable it with ``LAYA_RAG_EMBED_MODEL``; unset, the engine uses Laya's
    own encoder exactly as before.
    """

    #: bge-family models are trained with this instruction on the query side only.
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.model_id = self.config.embed_model
        self.pooling = self._resolve_pooling(self.config.embed_pooling)
        self.query_instruction = bool(self.config.embed_query_instruction)
        self._tokenizer: Any | None = None
        self._encoder: Any | None = None
        self._torch: Any | None = None
        self._device = self.config.device
        self._dim: int | None = None
        self._lock = threading.RLock()

    def _resolve_pooling(self, value: str) -> str:
        """``auto`` picks CLS for BGE-family models, mean pooling otherwise.

        BGE checkpoints are trained and served with the ``[CLS]`` token's hidden state
        (BAAI's model card uses ``model_output[0][:, 0]``), so mean pooling — the obvious
        transformer default — is a mismatch with how the model was trained. Measured on
        BEIR SciFact, correcting it is worth ~0.014 nDCG@10.

        A **local directory** is read from its own saved ``1_Pooling/config.json`` first:
        a fine-tuned or distilled model keeps its training pooling, and a name heuristic
        cannot see that a path like ``eval/distill-nfcorpus-model`` is BGE-derived. Getting
        this wrong is silent — it just embeds worse.
        """
        value = (value or "auto").strip().lower()
        if value == "auto":
            saved = saved_pooling(self.model_id)
            if saved:
                return saved
            return "cls" if "bge" in self.model_id.lower() else "mean"
        if value not in {"cls", "mean"}:
            raise ModelUnavailableError(
                f"LAYA_RAG_EMBED_POOLING must be auto, cls or mean, got {value!r}"
            )
        return value

    def ensure_loaded(self) -> "BiEncoderModel":
        with self._lock:
            if self._encoder is not None:
                return self
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ModelUnavailableError(
                    "torch is not installed; a bi-encoder needs the pinned CPU build."
                ) from exc
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ModelUnavailableError(
                    "transformers is not installed; a bi-encoder needs it (it ships with Laya)."
                ) from exc

            device = self.config.device
            if device in ("auto", ""):
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            torch.set_num_threads(self.config.torch_threads)
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._encoder = AutoModel.from_pretrained(self.model_id).to(device).eval()
            except Exception as exc:  # noqa: BLE001 - surfaced with the model name
                raise ModelUnavailableError(
                    f"failed to load embedding model {self.model_id!r} on device {device!r}: {exc}"
                ) from exc
            self._torch = torch
            self._device = device
            self._dim = int(self._encoder.config.hidden_size)
            return self

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        self.ensure_loaded()
        torch = self._torch
        assert torch is not None
        rows = ["" if text is None else str(text) for text in texts]
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        batch_size = max(1, self.config.embed_batch_size)
        blocks: list[np.ndarray] = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.embed_max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                hidden = self._encoder(**encoded).last_hidden_state
            if self.pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            blocks.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(blocks)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed documents. No query instruction on this side."""
        return self._encode(texts)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a query, with the retrieval instruction only if it is configured.

        bge-*-v1.5 barely degrades without the instruction and measured slightly better
        without it on BEIR SciFact, so the default is off and ``EMBED_QUERY_INSTRUCTION``
        turns it on.
        """
        query = str(text or "")
        if self.query_instruction:
            query = self.QUERY_PREFIX + query
        return self._encode([query])[0]

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.embed([""])
        return int(self._dim or 0)

    def info(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "embedder": self.model_id,
            "kind": "bi-encoder",
            "device": self._device,
            "loaded": self._encoder is not None,
            "pooling": self.pooling,
            "query_instruction": self.query_instruction,
        }
        if self._dim is not None:
            payload["embedding_dim"] = self._dim
        return payload


def build_embedder(config: EngineConfig) -> Any:
    """The dense-channel embedder for ``config``: a bi-encoder when one is configured."""
    if (config.embed_model or "").strip():
        return BiEncoderModel(config)
    return LayaModel(config)


class CrossEncoderModel:
    """A retrieval-trained cross-encoder that orders and filters candidate evidence.

    This is the second stage BAAI's model card recommends and that the engine's own
    design already describes: retrieve a wide candidate set cheaply, then score it more
    expensively. Laya's decision head was measured to be a stronger *filter* than
    *ranker*, so a cross-encoder is the natural replacement for the ordering. Measured
    on BEIR it lifts the engine's lexical recall from 0.654 to 0.680 (SciFact) and 0.304
    to 0.339 (NFCorpus). Laya still owns the answerability gate either way.

    ``score_pairs`` returns a sigmoid probability per pair, so the existing
    ``RELEVANCE_THRESHOLD`` keeps its meaning as a 0..1 cut.
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.model_id = self.config.rerank_model
        self._tokenizer: Any | None = None
        self._encoder: Any | None = None
        self._torch: Any | None = None
        self._device = self.config.device
        self._lock = threading.RLock()

    def ensure_loaded(self) -> "CrossEncoderModel":
        with self._lock:
            if self._encoder is not None:
                return self
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ModelUnavailableError("torch is not installed; a cross-encoder needs it.") from exc
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ModelUnavailableError(
                    "transformers is not installed; a cross-encoder needs it (it ships with Laya)."
                ) from exc

            device = self.config.device
            if device in ("auto", ""):
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            torch.set_num_threads(self.config.torch_threads)
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._encoder = (
                    AutoModelForSequenceClassification.from_pretrained(self.model_id)
                    .to(device)
                    .eval()
                )
            except Exception as exc:  # noqa: BLE001 - surfaced with the model name
                raise ModelUnavailableError(
                    f"failed to load rerank model {self.model_id!r} on device {device!r}: {exc}"
                ) from exc
            self._torch = torch
            self._device = device
            return self

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Relevance probability for each ``(query, passage)`` pair."""
        rows = list(pairs)
        if not rows:
            return []
        self.ensure_loaded()
        torch = self._torch
        assert torch is not None
        batch_size = max(1, self.config.embed_batch_size)
        out: list[float] = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = self._tokenizer(
                [query for query, _ in batch],
                [passage for _, passage in batch],
                padding=True,
                truncation=True,
                max_length=self.config.embed_max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                logits = self._encoder(**encoded, return_dict=True).logits.view(-1).float()
            out.extend(torch.sigmoid(logits).cpu().tolist())
        return out

    def info(self) -> dict[str, Any]:
        return {
            "reranker": self.model_id,
            "kind": "cross-encoder",
            "device": self._device,
            "loaded": self._encoder is not None,
        }


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
