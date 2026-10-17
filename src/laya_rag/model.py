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


#: How the encoders this repository can load are trained to pool, by family. The `auto` heuristic used
#: to be "CLS for names containing `bge`, mean otherwise", which is **wrong silently** for every modern
#: backbone: `Qwen/Qwen3-Embedding-0.6B` loads and runs today and would be embedded with mean pooling
#: when it is trained on the **last** token, producing a misleading negative rather than an error. Each
#: entry here was verified by loading the model on this host, not by reading a model card.
_FAMILY_POOLING: tuple[tuple[str, str], ...] = (
    ("bge", "cls"),                 # BAAI's cards use model_output[0][:, 0]
    ("mxbai", "cls"),               # mixedbread-ai: CLS, which `auto` got wrong by omission
    ("qwen3-embedding", "last"),    # Qwen3 Embedding is trained on the last token (EOS), plus an instruction
    ("e5", "mean"),                 # intfloat: mean, with `query: `/`passage: ` prefixes
)


#: Families trained with a prefix on **each** side. Applying one and not the other puts queries and
#: documents in different spaces, which is the same silent failure as the wrong pooling: it retrieves,
#: just worse. The e5 cards use exactly these two strings.
_FAMILY_PREFIXES: tuple[tuple[str, tuple[str, str]], ...] = (
    ("e5", ("query: ", "passage: ")),
)


#: The **query-side** line each family wants, which is not the same as its passage-side one. bge's is a
#: sentence, Qwen3 Embedding's is an `Instruct:`/`Query:` pair, e5's is the bare `query: ` token — and
#: applying one family's to another is the third silent mismatch on P3's list, of exactly the same shape
#: as the pooling and the passage prefix.
_FAMILY_QUERY_PREFIX: tuple[tuple[str, str], ...] = (
    ("e5", "query: "),
    ("qwen3-embedding",
     "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "),
)


def _family_query_prefix(model_id: str) -> str:
    """The query-side prefix for a known family, or ``""`` — never another family's."""
    lowered = (model_id or "").lower()
    for needle, prefix in _FAMILY_QUERY_PREFIX:
        if needle in lowered:
            return prefix
    return ""


def _family_prefixes(model_id: str) -> tuple[str, str]:
    """``(query_prefix, passage_prefix)`` for a known family, or two empty strings."""
    lowered = (model_id or "").lower()
    for needle, prefixes in _FAMILY_PREFIXES:
        if needle in lowered:
            return prefixes
    return ("", "")


def _family_pooling(model_id: str) -> str:
    """The pooling a known family is trained with, or ``mean`` for a model nobody has verified.

    The fallback is deliberate rather than clever: a name this table does not know may pool any way at
    all, and choosing one silently is the failure mode the table exists to remove. `mean` is the
    historical default, and a caller who cares passes `--pooling` (or `EMBED_POOLING`) explicitly.
    """
    lowered = (model_id or "").lower()
    for needle, pooling in _FAMILY_POOLING:
        if needle in lowered:
            return pooling
    return "mean"


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
        self.passage_prefix = str(getattr(self.config, "embed_passage_prefix", "") or "")
        self.truncate_dim = int(getattr(self.config, "embed_truncate_dim", 0) or 0)
        wanted_query, wanted_passage = _family_prefixes(self.model_id)
        if wanted_query and not (self.query_instruction and self.passage_prefix):
            # Loud rather than silent: this model is trained with a prefix on each side, and without
            # both it retrieves in a mismatched space. A warning, not an error, because a caller may
            # have measured that it does not matter for their corpus — but they will have been told.
            print(
                f"warning: {self.model_id} is trained with prefixes ({wanted_query!r} / "
                f"{wanted_passage!r}); configure EMBED_QUERY_INSTRUCTION=1 and EMBED_PASSAGE_PREFIX="
                f"{wanted_passage!r} or the two sides embed into different spaces",
                flush=True,
            )
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
            return _family_pooling(self.model_id)
        if value not in {"cls", "mean", "last"}:
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
            elif self.pooling == "last":
                # The **last real** token, not the last column: under left padding those differ, and the
                # same trap was found in the generative reranker's batching an hour ago. `argmax` over
                # `mask * position` gives the highest index whose mask is set, which is correct either way.
                positions = torch.arange(hidden.size(1), device=hidden.device).unsqueeze(0)
                last_index = (
                    encoded["attention_mask"].to(torch.long) * positions
                ).max(dim=1).values
                pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_index]
            else:
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            # Truncate **before** normalising: an MRL prefix is a usable embedding, a truncated unit
            # vector is not a unit vector, and normalising first would leave every similarity scaled by
            # the discarded mass.
            if self.truncate_dim:
                if self.truncate_dim > pooled.size(1):
                    raise ModelUnavailableError(
                        f"EMBED_TRUNCATE_DIM={self.truncate_dim} exceeds the pooled width "
                        f"{pooled.size(1)}"
                    )
                pooled = pooled[:, : self.truncate_dim]
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            blocks.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(blocks)

    def embed_queries(self, texts: Sequence[str], fixed_length: bool = False) -> np.ndarray:
        """The harness's dense-path interface: embed **queries**, instruction included.

        `tools/beir.py`'s `dense_rankings` and `cached_embed` call `embed`/`embed_queries` on their embedder,
        which is `bi_encoder.BiEncoder` there and this class in the engine — and the two had drifted apart:
        this one offered `embed_one` and no batch query method, so it could not be passed to the harness's
        dense path at all. Two classes for one job with different method names is how a *comparison* between
        the engine's encoder and the harness's silently stops being one, which is the same shape as the three
        silent mismatches fixed in this file last round.
        """
        return np.vstack([self.embed_one(text) for text in texts]) if texts else np.zeros((0, 1), np.float32)

    @property
    def name(self) -> str:
        """A cache key that carries the **pooling** as well as the model.

        `bi_encoder.name` in the harness does the same, and for the same reason: two pooling modes are two
        different embedders, so an embedding cache keyed by the model id alone would hand one mode's vectors
        to the other — which is how a silent mismatch of exactly the kind this file spent three rounds
        fixing would reappear through the cache. It was missing here, and found by passing
        `BiEncoderModel` to the harness's `cached_embed`.
        """
        return f"{self.model_id}.{self.pooling}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed documents, with the passage prefix when one is configured."""
        rows = [str(text or "") for text in texts]
        if self.passage_prefix:
            rows = [self.passage_prefix + row for row in rows]
        return self._encode(rows)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a query, with the retrieval instruction only if it is configured.

        bge-*-v1.5 barely degrades without the instruction and measured slightly better
        without it on BEIR SciFact, so the default is off and ``EMBED_QUERY_INSTRUCTION``
        turns it on.
        """
        return self._encode([self._query_text(text)])[0]

    def _query_text(self, text: str) -> str:
        """The string this model wants for a query — prefixed only if the caller asked, and with the
        **family's own** line rather than bge's sentence, which is the difference between an instruction
        a model was trained with and one it was not. Split out so it can be asserted without a forward."""
        query = str(text or "")
        if self.query_instruction:
            query = (_family_query_prefix(self.model_id) or self.QUERY_PREFIX) + query
        return query

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


#: The instruction Qwen3-Reranker's card uses when a caller supplies none. It is the model's own
#: default, kept verbatim because the prompt *is* the interface for a generative reranker.
QWEN3_RERANK_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def qwen3_rerank_prompt(query: str, document: str, *, instruction: str = "") -> str:
    """The chat-template prompt a Qwen3 reranker is scored on.

    A generative reranker has no classification head and no ``[CLS]``: it is asked a question and its
    answer's *probability* is the score. That makes the prompt part of the model, not a formatting
    detail — a different template is a different retriever — so it lives here, spelled out, rather
    than inside the batching loop.
    """
    instruct = instruction or QWEN3_RERANK_INSTRUCTION
    return (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the Instruct "
        'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
        "<|im_start|>user\n"
        f"<Instruct>: {instruct}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def answer_positions(attention_mask: Any) -> list[int]:
    """Each row's final **real** token — where a decoder-only model's answer is read from.

    Batching is not a detail here: scoring one pair at a time cost ~1,100 s for 200 queries, and a batch
    is the same arithmetic for a fraction of the forwards. The trap is padding. Right-padded rows end in
    pad tokens and left-padded rows end in the answer either way, so the index that is correct for both
    is the **count of real tokens minus one** rather than the last column — which is why this is a
    function with a test rather than an inline expression.
    """
    positions: list[int] = []
    for mask in attention_mask:
        # **The last real token's index, not the count of real tokens minus one.** Those differ under left
        # padding, and Qwen tokenizers commonly left-pad for generation: with a left-padded row the answer
        # sits in the final column, so "count - 1" would read the second-to-last *pad* token and produce a
        # plausible but wrong score for every row in the batch. This was a real latent bug — the batching
        # parity check passed only because this tokenizer happens to right-pad — and it was caught by
        # writing this function's test rather than by the measurement.
        values = [int(value) for value in mask]
        positions.append(max((i for i, value in enumerate(values) if value), default=len(values) - 1))
    return positions


def yes_no_probability(yes_logit: float, no_logit: float) -> float:
    """``P("yes")`` from the two logits, so the score keeps the meaning of a 0..1 relevance cut.

    Softmax over *just* those two, not over the vocabulary: the model's other tokens are not answers
    to the question and letting them contribute would make the number depend on the corpus.
    """
    import math

    gap = float(yes_logit) - float(no_logit)
    # Numerically stable logistic on the difference — softmax(yes, no)[0].
    if gap >= 0:
        return 1.0 / (1.0 + math.exp(-gap))
    exponential = math.exp(gap)
    return exponential / (1.0 + exponential)


def is_generative_reranker(model_id: str) -> bool:
    """Whether an id names a model that is asked a question rather than asked to classify.

    Kept as a named predicate so the engine's choice is inspectable: the two kinds need different
    classes, different prompts and different scoring, and a caller who points ``RERANK_MODEL`` at one
    while the engine loads the other would be measuring the wrong thing silently.
    """
    return "qwen3-reranker" in (model_id or "").lower()


def build_reranker(config: EngineConfig) -> Any | None:
    """The second stage for ``config``: no reranker, a cross-encoder, or a generative one.

    Kept next to :func:`build_embedder` for the same reason — the choice is a property of the
    configuration, and an engine that picked the class itself could point ``RERANK_MODEL`` at a
    generative model and silently load a classification head that does not exist.
    """
    model_id = (config.rerank_model or "").strip()
    if not model_id:
        return None
    if is_generative_reranker(model_id):
        return GenerativeRerankerModel(config)
    return CrossEncoderModel(config)


class GenerativeRerankerModel:
    """A listwise-capable reranker scored by the probability of the answer "yes".

    §P4's open question is whether a *listwise* ranker beats a pointwise cross-encoder, and the
    repository's recorded rejection was specifically "a pointwise MS MARCO cross-encoder over the
    fused top-100". Qwen3-Reranker-0.6B is Apache-2.0 and needs no classification head; what it needs
    is this path, because ``AutoModelForSequenceClassification`` does not load it.

    ``score_pairs`` returns ``P("yes")`` per pair, so ``RELEVANCE_THRESHOLD`` keeps its meaning and
    the two kinds of reranker can be compared at the same cut.
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.model_id = self.config.rerank_model
        self._tokenizer: Any | None = None
        self._encoder: Any | None = None
        self._torch: Any | None = None
        self._device = self.config.device
        self._yes_id: int | None = None
        self._no_id: int | None = None
        self._lock = threading.RLock()

    def ensure_loaded(self) -> "GenerativeRerankerModel":
        with self._lock:
            if self._encoder is not None:
                return self
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ModelUnavailableError("torch is not installed; a reranker needs it.") from exc
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ModelUnavailableError(
                    "transformers is not installed; a generative reranker needs it."
                ) from exc

            device = self.config.device
            if device in ("auto", ""):
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            torch.set_num_threads(self.config.torch_threads)
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._encoder = (
                    AutoModelForCausalLM.from_pretrained(self.model_id).to(device).eval()
                )
                self._yes_id = int(self._tokenizer.convert_tokens_to_ids("yes"))
                self._no_id = int(self._tokenizer.convert_tokens_to_ids("no"))
            except Exception as exc:  # noqa: BLE001 - surfaced with the model name
                raise ModelUnavailableError(
                    f"failed to load generative rerank model {self.model_id!r} on device "
                    f"{device!r}: {exc}"
                ) from exc
            self._torch = torch
            self._device = device
            return self

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """``P("yes")`` for each ``(query, passage)`` pair."""
        rows = list(pairs)
        if not rows:
            return []
        self.ensure_loaded()
        torch = self._torch
        assert torch is not None
        instruction = getattr(self.config, "rerank_instruction", "") or ""
        out: list[float] = []
        # **One pair per forward, deliberately.** A causal LM returns logits over `(batch, seq, vocab)`, and this
        # checkpoint's vocabulary is 151,936 — so batch 8 x ~1000 positions is **4.9 GB**, which is the
        # `RuntimeError: Invalid buffer size: 4.63 GB` this path died with three times. The obvious fix is to ask
        # for the final position only (`num_logits_to_keep=1`), but **this transformers build's Qwen3 model does not
        # implement that argument** — measured, not assumed: grep finds it nowhere in `modeling_qwen3.py`, and
        # passing it changes nothing. So the batch is pinned to 1, which is the configuration that demonstrably
        # works (round 121 ran 200 queries this way), at the cost of speed: ~18 minutes per 200-query arm.
        batch_size = 1
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            prompts = [
                qwen3_rerank_prompt(query, passage, instruction=instruction)
                for query, passage in chunk
            ]
            # **Left padding and one kept position, or the forward allocates the whole vocabulary.**
            # A causal LM's logits are (batch, seq, vocab) — for Qwen3-Reranker that is
            # batch x ~1000 x 151,936 x 4 bytes, i.e. **4.9 GB at batch 8**, which is the
            # `RuntimeError: Invalid buffer size: 4.63 GB` this path died with. The single-pair path
            # before batching existed fit; batching is what broke it. `num_logits_to_keep=1` keeps only
            # the final position, and **left padding is what makes that position the answer** rather
            # than a pad token — the same padding-side trap as `answer_positions`, in the direction
            # that costs 4 GB instead of correctness.
            previous_side = self._tokenizer.padding_side
            self._tokenizer.padding_side = "left"
            try:
                encoded = self._tokenizer(
                    prompts, padding=True, truncation=True,
                    max_length=self.config.embed_max_length, return_tensors="pt",
                ).to(self._device)
            finally:
                self._tokenizer.padding_side = previous_side
            with torch.no_grad():
                logits = self._encoder(
                    **encoded, return_dict=True, num_logits_to_keep=1
                ).logits
            for row in range(len(chunk)):
                last = logits[row, -1]           # the only kept position: the row's final real token
                out.append(
                    yes_no_probability(float(last[self._yes_id]), float(last[self._no_id]))
                )
        return out

    def info(self) -> dict[str, Any]:
        return {
            "reranker": self.model_id,
            "kind": "generative-reranker",
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
