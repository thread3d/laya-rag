"""Tests for removing documents without rebuilding the whole index.

`VectorIndex.remove_doc_ids`/`remove_paths` existed but no engine method, route or CLI reached
them, so a document could only be dropped by rebuilding. Removal must keep the two channels
consistent: the dense index is authoritative and the lexical channel is rebuilt from the
survivors, otherwise a deleted document keeps surfacing through BM25.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from laya_rag import api
from laya_rag.engine import RagEngine

from .support import StubModel, make_config


def _corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir(parents=True, exist_ok=True)
    (root / "refunds.md").write_text(
        "# Refunds\n\nRefunds are processed within five business days.\n", encoding="utf-8"
    )
    (root / "shipping.md").write_text(
        "# Shipping\n\nShipping is free on orders over fifty dollars.\n", encoding="utf-8"
    )
    return root


def _engine(tmp_path, **overrides):
    config = make_config(
        tmp_path, relevance_threshold=0.0, answerability_threshold=0.0, **overrides
    )
    return RagEngine(config, model=StubModel())


def test_remove_documents_drops_chunks_and_rebuilds_the_lexical_channel(tmp_path):
    root = _corpus(tmp_path)
    engine = _engine(tmp_path)
    engine.index_paths([root], rebuild=True)
    before = engine.stats()["index"]["chunks"]
    assert any("shipping" in p["path"] for p in engine.query("shipping cost").passages)

    removed = engine.remove_documents([root / "shipping.md"])
    assert removed > 0
    assert engine.stats()["index"]["chunks"] == before - removed

    # The dense channel dropped it *and* the lexical channel was rebuilt, so it cannot
    # come back through BM25 either — here or from a freshly opened engine.
    assert all("shipping" not in p["path"] for p in engine.query("shipping cost").passages)
    reopened = _engine(tmp_path)
    assert all("shipping" not in p["path"] for p in reopened.query("shipping cost").passages)


def test_remove_documents_by_doc_id(tmp_path):
    engine = _engine(tmp_path)
    engine.index_paths([_corpus(tmp_path)], rebuild=True)
    doc_id = engine.list_documents()[0]["doc_id"]
    assert engine.remove_documents(doc_ids=[doc_id]) > 0
    assert engine.stats()["index"]["documents"] == 1


def test_removing_nothing_removes_nothing(tmp_path):
    engine = _engine(tmp_path)
    engine.index_paths([_corpus(tmp_path)], rebuild=True)
    before = engine.stats()["index"]["chunks"]
    assert engine.remove_documents([]) == 0
    assert engine.remove_documents([tmp_path / "corpus" / "absent.md"]) == 0
    assert engine.stats()["index"]["chunks"] == before


def test_the_ondisk_backend_tombstones_removed_documents(tmp_path):
    """The append-only backend cannot rewrite offsets, so it tombstones and rebuilds lexical."""
    engine = _engine(tmp_path, index_backend="ondisk")
    engine.index_paths([_corpus(tmp_path)], rebuild=True)
    before = engine.stats()["index"]["chunks"]
    removed = engine.remove_documents([tmp_path / "corpus" / "shipping.md"])
    assert removed > 0
    assert engine.stats()["index"]["chunks"] == before - removed
    assert all("shipping" not in p["path"] for p in engine.query("shipping cost").passages)

    # The tombstone is persisted, so a reopened engine agrees.
    reopened = _engine(tmp_path, index_backend="ondisk")
    assert all("shipping" not in p["path"] for p in reopened.query("shipping cost").passages)


def test_delete_documents_route_removes_and_reports(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    monkeypatch.setattr(api, "RagEngine", lambda _config: engine)
    app = api.create_app(make_config(tmp_path, warmup=False))
    client = TestClient(app)

    corpus = _corpus(tmp_path)
    indexed = client.post("/v1/index", json={"paths": [str(corpus)]})
    assert indexed.status_code == 200
    path = engine.list_documents()[0]["path"]

    response = client.request("DELETE", "/v1/documents", json={"paths": [path]})
    assert response.status_code == 200
    body = response.json()
    assert body["removed_chunks"] > 0
    assert body["total_chunks"] == engine.stats()["index"]["chunks"]


def test_compact_reclaims_tombstoned_on_disk_documents(tmp_path):
    from laya_rag.ondisk import DENSE_DIRNAME

    engine = _engine(tmp_path, index_backend="ondisk")
    engine.index_paths([_corpus(tmp_path)], rebuild=True)
    engine.remove_documents([tmp_path / "corpus" / "shipping.md"])
    live = engine.stats()["index"]["chunks"]

    assert engine.compact() == live
    assert engine.stats()["index"]["chunks"] == live
    deleted_path = engine.config.index_dir / DENSE_DIRNAME / "deleted.json"
    assert not deleted_path.exists() or deleted_path.read_text(encoding="utf-8").strip() in ("[]", "")
    assert all("shipping" not in p["path"] for p in engine.query("shipping cost").passages)

    # The compacted index reopens cleanly and still excludes the removed document.
    reopened = _engine(tmp_path, index_backend="ondisk")
    assert reopened.stats()["index"]["chunks"] == live
    assert all("shipping" not in p["path"] for p in reopened.query("shipping cost").passages)


def test_compact_on_the_memory_backend_is_a_no_op(tmp_path):
    engine = _engine(tmp_path)
    engine.index_paths([_corpus(tmp_path)], rebuild=True)
    before = engine.stats()["index"]["chunks"]
    assert engine.compact() == before
    assert engine.stats()["index"]["chunks"] == before


def test_compacting_an_entirely_deleted_index_resets_it(tmp_path):
    engine = _engine(tmp_path, index_backend="ondisk")
    root = _corpus(tmp_path)
    engine.index_paths([root], rebuild=True)
    engine.remove_documents([root / "refunds.md", root / "shipping.md"])
    assert engine.compact() == 0
    assert engine.has_index() is False
