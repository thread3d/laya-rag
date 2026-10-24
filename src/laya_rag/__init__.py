"""Laya RAG: retrieval-augmented generation over the Laya decision engine.

Laya supplies embeddings and calibrated decisions; this package adds the corpus
side it does not ship (loading, chunking, a persistent vector index) and exposes the
result to callers over HTTP.
"""
from .config import EngineConfig
from .engine import (
    BuildInProgressError,
    IndexNotBuiltError,
    IndexReport,
    QueryResult,
    RagEngine,
)
from .model import LayaModel, ModelUnavailableError

__all__ = [
    "BuildInProgressError",
    "EngineConfig",
    "IndexNotBuiltError",
    "IndexReport",
    "LayaModel",
    "ModelUnavailableError",
    "QueryResult",
    "RagEngine",
]

__version__ = "1.2.1"
