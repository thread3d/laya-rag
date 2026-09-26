"""HTTP service tests, driven through the real FastAPI app with a stub engine."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from laya_rag import api

from .support import make_config, stub_engine, write_corpus


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    """A TestClient whose engine is the deterministic stub."""
    engine = stub_engine(tmp_path)
    monkeypatch.setattr(api, "RagEngine", lambda _config: engine)
    app = api.create_app(make_config(tmp_path, warmup=False))
    return TestClient(app), engine, write_corpus(tmp_path / "corpus")


def test_health_reports_a_ready_engine(ctx):
    client, _engine, _corpus = ctx
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["checkpoint"] == "multilingual"


def test_stats_are_available_before_indexing(ctx):
    client, _engine, _corpus = ctx
    body = client.get("/v1/stats").json()
    assert body["index_present"] is False
    assert body["index"]["chunks"] == 0


def test_query_before_indexing_is_a_conflict(ctx):
    client, _engine, _corpus = ctx
    response = client.post("/v1/query", json={"query": "anything"})
    assert response.status_code == 409
    assert "no index at" in response.json()["detail"]


def test_index_then_query_round_trip(ctx):
    client, _engine, corpus = ctx

    indexed = client.post("/v1/index", json={"paths": [str(corpus)]})
    assert indexed.status_code == 200
    assert indexed.json()["documents"] == 3

    response = client.post("/v1/query", json={"query": "How long does a refund take?"})
    assert response.status_code == 200
    body = response.json()
    assert body["passages"]
    assert "refunds.md" in body["passages"][0]["path"]
    assert body["timings"]["total_seconds"] >= 0


def test_index_reports_nothing_found_for_unreadable_paths(ctx, tmp_path):
    client, _engine, _corpus = ctx
    empty = tmp_path / "nothing"
    empty.mkdir()
    (empty / "blob.bin").write_bytes(b"\x00\x01\x02")

    body = client.post("/v1/index", json={"paths": [str(empty)]}).json()
    assert body["documents"] == 0
    assert "no readable text documents" in body["note"]


def test_index_requires_at_least_one_path(ctx):
    client, _engine, _corpus = ctx
    assert client.post("/v1/index", json={"paths": []}).status_code == 422


def test_query_rejects_a_blank_query(ctx):
    client, _engine, _corpus = ctx
    assert client.post("/v1/query", json={"query": ""}).status_code == 422


def test_query_rejects_out_of_range_top_k(ctx):
    client, _engine, _corpus = ctx
    assert client.post("/v1/query", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/v1/query", json={"query": "x", "top_k": 999}).status_code == 422


def test_reset_clears_the_index(ctx):
    client, _engine, corpus = ctx
    client.post("/v1/index", json={"paths": [str(corpus)]})
    assert client.get("/v1/stats").json()["index_present"] is True

    assert client.post("/v1/reset").json()["status"] == "ok"
    assert client.get("/v1/stats").json()["index_present"] is False


def test_api_key_is_enforced_when_configured(tmp_path, monkeypatch):
    engine = stub_engine(tmp_path)
    monkeypatch.setattr(api, "RagEngine", lambda _config: engine)
    app = api.create_app(make_config(tmp_path, warmup=False, api_key="secret"))
    client = TestClient(app)

    assert client.get("/health").status_code == 401
    assert client.get("/health", headers={"authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/health", headers={"authorization": "Bearer secret"}).status_code == 200
