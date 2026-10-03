"""HTTP service tests, driven through the real FastAPI app with a stub engine."""
from __future__ import annotations

import json

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


def test_index_stream_reports_progress_then_the_report(ctx):
    """The streaming variant exists so an interface can show running counts; it has to
    emit them as they happen and finish with the same report the plain route returns."""
    client, _engine, corpus = ctx
    response = client.post("/v1/index/stream", json={"paths": [str(corpus)]})
    assert response.status_code == 200
    assert "ndjson" in response.headers["content-type"]

    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert events[-1]["type"] == "done"
    progress = [e for e in events if e["type"] == "progress"]
    assert progress, "the stream emitted no progress at all"
    assert progress[0]["stage"] == "reading"
    assert progress[-1]["stage"] == "saving"
    # Every progress event carries the full counter shape, not a stage-specific subset.
    for event in progress:
        assert {"documents", "documents_total", "skipped", "chunks", "embedded", "path"} <= set(event)

    report = events[-1]["report"]
    assert report["documents"] == 3
    assert report["total_chunks"] > 0

    plain = client.post("/v1/index", json={"paths": [str(corpus)]}).json()
    assert report["documents"] == plain["documents"]
    assert report["total_chunks"] == plain["total_chunks"]


def test_index_stream_reports_a_failure_as_an_event(ctx, monkeypatch):
    """An error after the stream has started cannot be an HTTP status, so it has to
    arrive as an event the caller can read."""
    client, engine, corpus = ctx

    def boom(*_args, **_kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(engine, "index_paths", boom)
    response = client.post("/v1/index/stream", json={"paths": [str(corpus)]})
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert events[-1]["type"] == "error"
    assert "disk on fire" in events[-1]["message"]


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


def test_stats_never_disclose_the_api_key(tmp_path, monkeypatch):
    """A configured secret must be reported as present, never echoed back."""
    engine = stub_engine(tmp_path, api_key="secret")
    monkeypatch.setattr(api, "RagEngine", lambda _config: engine)
    app = api.create_app(make_config(tmp_path, warmup=False, api_key="secret"))
    client = TestClient(app)
    headers = {"authorization": "Bearer secret"}

    config = client.get("/v1/stats", headers=headers).json()["config"]
    assert config["api_key"] == "<redacted>"
    assert "secret" not in str(config)


def test_stats_report_an_absent_api_key_as_null(tmp_path, monkeypatch):
    engine = stub_engine(tmp_path)
    monkeypatch.setattr(api, "RagEngine", lambda _config: engine)
    app = api.create_app(make_config(tmp_path, warmup=False))
    client = TestClient(app)

    assert client.get("/v1/stats").json()["config"]["api_key"] is None


# ── index integrity ──────────────────────────────────────────────────────

def test_integrity_is_clean_before_indexing(ctx):
    client, _engine, _corpus = ctx
    body = client.get("/v1/integrity").json()
    assert body["index_present"] is False
    assert body["clean"] is True


def test_integrity_reports_a_duplicate_after_indexing(ctx):
    """Two files with identical text leave a redundant chunk the report must surface."""
    client, _engine, corpus = ctx
    (corpus / "refunds-copy.md").write_text(
        (corpus / "refunds.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    client.post("/v1/index", json={"paths": [str(corpus)]})
    body = client.get("/v1/integrity").json()
    assert body["index_present"] is True
    assert body["chunks"] > 0
    assert body["duplicate_content"]["clusters"] >= 1
    assert body["duplicate_content"]["redundant_chunks"] >= 1
    assert body["clean"] is False


def test_integrity_can_scan_for_near_duplicates(ctx):
    client, _engine, corpus = ctx
    client.post("/v1/index", json={"paths": [str(corpus)]})
    body = client.get("/v1/integrity", params={"near_duplicates": True, "threshold": 0.8}).json()
    assert body["near_duplicate_content"]["threshold"] == 0.8
    assert "pairs" in body["near_duplicate_content"]


def test_stats_surface_index_drift(ctx):
    """Freshness belongs in stats: a stale or duplicated index silently changes results."""
    client, _engine, corpus = ctx
    (corpus / "refunds-copy.md").write_text(
        (corpus / "refunds.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    client.post("/v1/index", json={"paths": [str(corpus)]})
    body = client.get("/v1/stats").json()
    integrity = body["index"]["integrity"]
    assert integrity["stale_paths"] == 0
    assert integrity["duplicate_clusters"] >= 1
    assert integrity["redundant_chunks"] >= 1
    assert integrity["clean"] is False
    # The non-linear scan stays opt-in; stats must say so rather than imply it ran.
    assert integrity["near_duplicates_scanned"] is False


def test_stats_are_clean_before_indexing(ctx):
    client, _engine, _corpus = ctx
    body = client.get("/v1/stats").json()
    assert body["index"]["chunks"] == 0
    assert "integrity" not in body["index"] or body["index"]["integrity"]["clean"] is True
