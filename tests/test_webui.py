"""Tests for the bundled web GUI and the read-only endpoints that back it.

The GUI is static files served by the same FastAPI app as the JSON API, so these
tests assert the two things that can silently break: that the assets are actually
shipped and reachable, and that mounting a catch-all at ``/`` does not shadow the
API routes registered before it.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from laya_dsh_rag import api

from .support import make_config, stub_engine, write_corpus


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    engine = stub_engine(tmp_path)
    monkeypatch.setattr(api, "RagEngine", lambda _config: engine)
    app = api.create_app(make_config(tmp_path, warmup=False))
    return TestClient(app), engine, write_corpus(tmp_path / "corpus")


@pytest.fixture
def indexed(ctx):
    client, engine, corpus = ctx
    assert client.post("/v1/index", json={"paths": [str(corpus)]}).status_code == 200
    return client, engine, corpus


# -- assets ------------------------------------------------------------------

def test_web_assets_exist_in_the_package():
    web_dir = Path(api.__file__).resolve().parent / "web"
    assert (web_dir / "index.html").is_file()
    assert (web_dir / "app.css").is_file()
    assert (web_dir / "app.js").is_file()


def test_root_serves_the_gui(ctx):
    client, _engine, _corpus = ctx
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Laya DSH RAG" in response.text


def test_stylesheet_and_script_are_served(ctx):
    client, _engine, _corpus = ctx

    css = client.get("/app.css")
    assert css.status_code == 200
    assert "css" in css.headers["content-type"]

    js = client.get("/app.js")
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]


def test_index_html_references_its_assets(ctx):
    """Renaming an asset without updating the markup would ship a broken page."""
    client, _engine, _corpus = ctx
    html = client.get("/").text
    assert "app.css" in html
    assert "app.js" in html


def test_the_mount_does_not_shadow_the_api(ctx):
    client, _engine, _corpus = ctx
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/v1/stats").status_code == 200
    # FastAPI's generated docs are registered before the mount, so they survive too.
    assert client.get("/docs").status_code == 200


def test_only_the_web_directory_is_exposed(ctx):
    """A catch-all mount must not turn the repository into a file server."""
    client, _engine, _corpus = ctx
    for probe in ("/pyproject.toml", "/../pyproject.toml", "/web/../app.py"):
        assert client.get(probe).status_code == 404


def test_unknown_static_path_is_a_404(ctx):
    client, _engine, _corpus = ctx
    assert client.get("/no-such-asset.js").status_code == 404


def test_the_page_loads_without_a_token_so_it_can_ask_for_one(tmp_path, monkeypatch):
    """The shell must be reachable unauthenticated, or the token could never be entered."""
    engine = stub_engine(tmp_path)
    monkeypatch.setattr(api, "RagEngine", lambda _config: engine)
    app = api.create_app(make_config(tmp_path, warmup=False, api_key="secret"))
    client = TestClient(app)

    assert client.get("/").status_code == 200
    assert client.get("/app.js").status_code == 200
    assert client.get("/health").status_code == 401


# -- corpus endpoints --------------------------------------------------------

def test_documents_is_empty_before_indexing(ctx):
    client, _engine, _corpus = ctx
    body = client.get("/v1/documents").json()
    assert body["total"] == 0
    assert body["documents"] == []


def test_chunks_is_empty_before_indexing(ctx):
    client, _engine, _corpus = ctx
    body = client.get("/v1/chunks").json()
    assert body["total"] == 0
    assert body["chunks"] == []


def test_documents_lists_each_indexed_file(indexed):
    client, _engine, _corpus = indexed
    body = client.get("/v1/documents").json()

    assert body["total"] == 3
    names = sorted(Path(doc["path"]).name for doc in body["documents"])
    assert names == ["refunds.md", "security.md", "shipping.md"]
    for doc in body["documents"]:
        assert doc["chunks"] >= 1
        assert doc["chars"] > 0
        assert doc["doc_id"]


def test_chunks_returns_the_whole_corpus(indexed):
    client, _engine, _corpus = indexed
    body = client.get("/v1/chunks").json()
    assert body["total"] >= 3
    assert body["path"] is None
    for chunk in body["chunks"]:
        assert chunk["text"].strip()
        assert chunk["chunk_id"]


def test_chunks_can_be_narrowed_to_one_path(indexed):
    client, engine, _corpus = indexed
    path = next(doc["path"] for doc in engine.list_documents() if doc["path"].endswith("refunds.md"))

    body = client.get("/v1/chunks", params={"path": path}).json()
    assert body["total"] >= 1
    assert body["path"] == path
    assert all(chunk["path"] == path for chunk in body["chunks"])
    assert "five business days" in " ".join(chunk["text"] for chunk in body["chunks"])


def test_chunks_for_an_unknown_path_is_empty_not_an_error(indexed):
    client, _engine, _corpus = indexed
    body = client.get("/v1/chunks", params={"path": "/nope/missing.md"}).json()
    assert body["total"] == 0
    assert body["chunks"] == []


def test_chunks_are_ordered_by_path_then_position(indexed):
    client, _engine, _corpus = indexed
    keys = [(chunk["path"], chunk["index"]) for chunk in client.get("/v1/chunks").json()["chunks"]]
    assert keys == sorted(keys)


def test_documents_require_the_token_when_configured(tmp_path, monkeypatch):
    engine = stub_engine(tmp_path)
    monkeypatch.setattr(api, "RagEngine", lambda _config: engine)
    app = api.create_app(make_config(tmp_path, warmup=False, api_key="secret"))
    client = TestClient(app)

    assert client.get("/v1/documents").status_code == 401
    assert client.get("/v1/chunks").status_code == 401
    headers = {"authorization": "Bearer secret"}
    assert client.get("/v1/documents", headers=headers).status_code == 200
    assert client.get("/v1/chunks", headers=headers).status_code == 200


def test_documents_survive_a_reset(ctx):
    client, _engine, corpus = ctx
    client.post("/v1/index", json={"paths": [str(corpus)]})
    assert client.get("/v1/documents").json()["total"] == 3

    client.post("/v1/reset")
    assert client.get("/v1/documents").json()["total"] == 0


# -- engine-level introspection ---------------------------------------------

def test_list_chunks_is_index_optional(tmp_path):
    """Callers poll these on every refresh, so an unbuilt index must not raise."""
    engine = stub_engine(tmp_path)
    assert engine.list_documents() == []
    assert engine.list_chunks() == []


def test_list_documents_counts_every_chunk(indexed):
    _client, engine, _corpus = indexed
    documents = engine.list_documents()
    chunks = engine.list_chunks()
    assert sum(doc["chunks"] for doc in documents) == len(chunks)
    assert sum(doc["chars"] for doc in documents) == sum(chunk["chars"] for chunk in chunks)
