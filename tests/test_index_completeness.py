"""Offline tests for index completeness.

A build killed between the dense save and the lexical save leaves a directory with half an index.
`has_index()` used to accept it, so a later run reused the partial index and every query silently
degraded to dense-only recall — which is how a full 2,255-query measurement came out below the
same engine measured on a 400-query subset of it. A partial index must read as *absent*.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laya_rag.config import EngineConfig  # noqa: E402
from laya_rag.engine import RagEngine  # noqa: E402
from laya_rag.index_store import INDEX_META, INDEX_VECTORS  # noqa: E402
from laya_rag.lexical import EAGER_LEXICAL_FILE, LEXICAL_FILE  # noqa: E402


def _engine(tmp_path: Path, **overrides) -> RagEngine:
    config = EngineConfig(index_dir=tmp_path, **overrides)
    return RagEngine(config)


def _dense_only(tmp_path: Path) -> None:
    """Exactly what a build killed before the lexical save leaves behind."""
    (tmp_path / INDEX_META).write_text("{}", encoding="utf-8")
    (tmp_path / INDEX_VECTORS).write_bytes(b"")


def test_a_dense_only_directory_is_not_an_index(tmp_path):
    _dense_only(tmp_path)
    assert _engine(tmp_path).has_index() is False


def test_adding_the_lexical_artifact_makes_it_complete(tmp_path):
    _dense_only(tmp_path)
    (tmp_path / LEXICAL_FILE).write_text("{}", encoding="utf-8")
    assert _engine(tmp_path).has_index() is True


def test_removing_the_lexical_artifact_makes_it_incomplete_again(tmp_path):
    _dense_only(tmp_path)
    (tmp_path / LEXICAL_FILE).write_text("{}", encoding="utf-8")
    engine = _engine(tmp_path)
    assert engine.has_index() is True
    (tmp_path / LEXICAL_FILE).unlink()
    assert engine.has_index() is False


def test_the_eager_backend_requires_its_own_artifact(tmp_path):
    _dense_only(tmp_path)
    (tmp_path / LEXICAL_FILE).write_text("{}", encoding="utf-8")
    eager = _engine(tmp_path, lexical_backend="eager")
    assert eager.has_index() is False
    (tmp_path / EAGER_LEXICAL_FILE).write_bytes(b"")
    assert eager.has_index() is True


def test_an_empty_directory_is_not_an_index(tmp_path):
    assert _engine(tmp_path).has_index() is False


def test_a_missing_lexical_artifact_degrades_the_channel_and_the_index_is_rebuilt(
    tmp_path, monkeypatch
):
    """The channel really is absent when the artifact is, which is why this matters."""
    _dense_only(tmp_path)
    engine = _engine(tmp_path)
    assert engine.lexical is None
    assert engine.has_index() is False
