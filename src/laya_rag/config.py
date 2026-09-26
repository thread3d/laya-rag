"""Configuration for the Laya RAG engine.

Every value here is deployment-varying and therefore overridable through the
environment, matching the "no hardcoded tunables" rule the harness applies to its
own plugins. Defaults target this repository's local checkout: the Laya submodule
under ``laya/`` with its checkpoints in ``laya/models``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

# <repo>/src/laya_rag/config.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]

ENV_PREFIX = "LAYA_RAG_"


def _env(name: str) -> str | None:
    value = os.environ.get(ENV_PREFIX + name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{ENV_PREFIX}{name} must be a boolean, got {raw!r}")


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_PREFIX}{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_PREFIX}{name} must be a number, got {raw!r}") from exc


@dataclass
class EngineConfig:
    """Resolved engine configuration.

    ``checkpoint`` names one of the bundled Laya checkpoints. ``multilingual`` is
    the default because it is the smallest (322M) and the only one that reads long
    documents up to 8192 tokens, which is what chunking feeds it.
    """

    models_dir: Path = field(
        default_factory=lambda: Path(_env("MODELS") or REPO_ROOT / "models")
    )
    index_dir: Path = field(default_factory=lambda: Path(_env("INDEX_DIR") or REPO_ROOT / "indexes" / "default"))
    checkpoint: str = field(default_factory=lambda: _env("CHECKPOINT") or "multilingual")
    # "auto" defers to Laya's CUDA -> MPS -> CPU selection. On this host that picks
    # MPS (Metal), which measured 1.6-3x faster than CPU in the Laya checkout.
    device: str = field(default_factory=lambda: _env("DEVICE") or "auto")

    # Encoder limits. max_len governs the state (document) budget; head_max_len the
    # option/instructions budget. Laya defaults to 512/192.
    max_len: int = field(default_factory=lambda: _env_int("MAX_LEN", 2048))
    head_max_len: int = field(default_factory=lambda: _env_int("HEAD_MAX_LEN", 192))
    embed_max_length: int = field(default_factory=lambda: _env_int("EMBED_MAX_LENGTH", 512))
    embed_batch_size: int = field(default_factory=lambda: _env_int("EMBED_BATCH_SIZE", 16))

    # torch thread oversubscription hurts on this host (56 logical cores); 16 is the
    # measured sweet spot recorded in the Laya checkout's LOCAL_SETUP.md.
    torch_threads: int = field(default_factory=lambda: _env_int("TORCH_THREADS", 16))

    # Chunking. Character-based with paragraph preference; sized to sit well inside
    # max_len so a rerank call sees the whole chunk plus the query.
    chunk_chars: int = field(default_factory=lambda: _env_int("CHUNK_CHARS", 1200))
    chunk_overlap_chars: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP_CHARS", 200))
    min_chunk_chars: int = field(default_factory=lambda: _env_int("MIN_CHUNK_CHARS", 80))

    # Retrieval funnel: dense candidates -> Laya re-ranked -> evidence passages.
    retrieve_k: int = field(default_factory=lambda: _env_int("RETRIEVE_K", 20))
    rerank_k: int = field(default_factory=lambda: _env_int("RERANK_K", 5))

    # Laya emits calibrated probabilities, so these are real thresholds, not scores.
    relevance_threshold: float = field(default_factory=lambda: _env_float("RELEVANCE_THRESHOLD", 0.5))
    answerability_threshold: float = field(default_factory=lambda: _env_float("ANSWERABILITY_THRESHOLD", 0.5))

    # Service
    host: str = field(default_factory=lambda: _env("HOST") or "127.0.0.1")
    port: int = field(default_factory=lambda: _env_int("PORT", 8077))
    api_key: str | None = field(default_factory=lambda: _env("API_KEY"))

    warmup: bool = field(default_factory=lambda: _env_bool("WARMUP", True))

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = str(value) if isinstance(value, Path) else value
        return out

    def ensure_dirs(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
