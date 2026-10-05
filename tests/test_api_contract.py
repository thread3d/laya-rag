"""The HTTP contract: typed responses, RFC 9457 problems, correlation, pagination.

These assert the things other software is meant to depend on, so they check the
*published* schema as well as the wire: a response model that is not in the OpenAPI
document is not a contract.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from laya_rag import api
from laya_rag.problems import PROBLEM_MEDIA_TYPE

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


# -- OpenAPI -----------------------------------------------------------------

def _operations(schema):
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            if method in {"get", "post", "put", "patch", "delete"}:
                yield path, method, operation


def test_openapi_types_every_success_response(ctx):
    """A generated client must get a named schema, not `Any`, for every 200."""
    client, _engine, _corpus = ctx
    schema = client.app.openapi()
    for path, method, operation in _operations(schema):
        response = operation["responses"]["200"]
        content = response.get("content")
        assert content, f"{method.upper()} {path} documents no 200 content"
        for media_type, media in content.items():
            assert "$ref" in media.get("schema", {}) or media.get("schema"), (
                f"{method.upper()} {path} {media_type} has no schema"
            )


def test_openapi_streaming_routes_document_both_encodings(ctx):
    client, _engine, _corpus = ctx
    schema = client.app.openapi()
    for path in ("/v1/index/stream", "/v1/compose"):
        content = schema["paths"][path]["post"]["responses"]["200"]["content"]
        assert "application/x-ndjson" in content
        assert "text/event-stream" in content


def test_openapi_declares_bearer_security_and_public_probes(ctx):
    client, _engine, _corpus = ctx
    schema = client.app.openapi()
    schemes = schema["components"]["securitySchemes"]
    assert schemes["bearerAuth"]["scheme"] == "bearer"
    assert schema["paths"]["/v1/query"]["post"]["security"] == [{"bearerAuth": []}]
    assert schema["paths"]["/healthz"]["get"]["security"] == []
    assert schema["paths"]["/readyz"]["get"]["security"] == []


def test_openapi_error_responses_use_problem_json(ctx):
    client, _engine, _corpus = ctx
    schema = client.app.openapi()
    # /readyz answers 503 with the readiness report, which a probe can read; every other
    # error is an RFC 9457 problem.
    exempt = {("/readyz", "503")}
    for path, method, operation in _operations(schema):
        for status, response in operation["responses"].items():
            if (status.isdigit() and int(status) >= 400) or status == "default":
                if (path, status) in exempt:
                    continue
                assert PROBLEM_MEDIA_TYPE in response["content"], (
                    f"{method.upper()} {path} documents {status} as the wrong media type"
                )


def test_openapi_references_resolve(ctx):
    """Every ``$ref`` in the document must point at a defined component."""
    client, _engine, _corpus = ctx
    schema = client.app.openapi()
    defined = set(schema.get("components", {}).get("schemas", {}))

    def walk(node):
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
                yield ref.rsplit("/", 1)[-1]
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)

    missing = sorted({name for name in walk(schema) if name not in defined})
    assert missing == []


def test_openapi_operations_have_stable_ids(ctx):
    client, _engine, _corpus = ctx
    schema = client.app.openapi()
    ids = [operation["operationId"] for _p, _m, operation in _operations(schema)]
    assert len(ids) == len(set(ids))
    assert "create_query" in ids
    assert "list_documents" in ids
    assert "stream_index" in ids


# -- problem errors ----------------------------------------------------------

def test_validation_error_is_a_problem_with_field_pointers(ctx):
    client, _engine, _corpus = ctx
    response = client.post("/v1/query", json={"query": ""})
    assert response.status_code == 422
    assert response.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    body = response.json()
    assert body["status"] == 422
    assert body["code"] == "validation_error"
    assert body["type"].endswith("/validation_error")
    assert isinstance(body["detail"], str)
    assert body["errors"][0]["pointer"] == "#/query"


def test_missing_index_is_a_stable_conflict_code(ctx):
    client, _engine, _corpus = ctx
    response = client.post("/v1/query", json={"query": "anything"})
    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "index_not_built"
    assert "no index at" in body["detail"]


def test_unauthorised_problem_carries_a_challenge(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "RagEngine", lambda _config: stub_engine(tmp_path))
    app = api.create_app(make_config(tmp_path, warmup=False, api_key="secret"))
    response = TestClient(app).get("/v1/stats")
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer")
    assert response.json()["code"] == "unauthorized"


def test_unhandled_error_is_a_problem_not_a_traceback(tmp_path, monkeypatch):
    engine = stub_engine(tmp_path)

    def explode() -> dict:
        raise RuntimeError("secret internal detail")

    engine.stats = explode  # type: ignore[method-assign]
    monkeypatch.setattr(api, "RagEngine", lambda _config: engine)
    app = api.create_app(make_config(tmp_path, warmup=False))

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/v1/stats")
    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "internal_error"
    assert "secret internal detail" not in response.text
    assert body["request_id"]


# -- correlation -------------------------------------------------------------

def test_request_id_is_generated_and_echoed(ctx):
    client, _engine, _corpus = ctx
    generated = client.get("/healthz")
    assert generated.headers["x-request-id"].startswith("req_")

    echoed = client.get("/healthz", headers={"X-Request-ID": "trace-me-123"})
    assert echoed.headers["x-request-id"] == "trace-me-123"


def test_traceparent_is_validated_and_reused(ctx):
    client, _engine, _corpus = ctx
    trace = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"
    response = client.post(
        "/v1/query", json={"query": "x"}, headers={"traceparent": trace}
    )
    assert response.headers["x-trace-id"] == "a" * 32
    assert response.json()["trace_id"] == "a" * 32

    # An all-zero trace id is invalid per W3C and must be ignored, not echoed.
    bad = "00-" + "0" * 32 + "-" + "0" * 16 + "-01"
    ignored = client.get("/healthz", headers={"traceparent": bad})
    assert "x-trace-id" not in ignored.headers


# -- probes ------------------------------------------------------------------

def test_liveness_does_not_require_an_index_or_model(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "RagEngine", lambda _config: stub_engine(tmp_path))
    client = TestClient(api.create_app(make_config(tmp_path, warmup=False)))
    assert client.get("/healthz").json() == {"status": "ok"}


def test_readiness_is_503_before_indexing_and_200_after(indexed):
    client, _engine, corpus = indexed
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["ready"] is True


def test_readiness_reports_not_ready_without_an_index(ctx):
    client, _engine, _corpus = ctx
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["ready"] is False
    assert response.headers["retry-after"] == "30"


def test_health_alias_is_deprecated_but_unchanged(ctx):
    client, _engine, _corpus = ctx
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["deprecation"].startswith("@")
    assert "successor-version" in response.headers["link"]


# -- pagination --------------------------------------------------------------

def _make_many(tmp_path, count=7):
    corpus = tmp_path / "many"
    corpus.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (corpus / f"doc{index:02d}.md").write_text(
            f"Document number {index} about topic {index}.", encoding="utf-8"
        )
    return corpus


def test_documents_default_to_the_full_listing(tmp_path, monkeypatch):
    engine = stub_engine(tmp_path)
    monkeypatch.setattr(api, "RagEngine", lambda _config: engine)
    client = TestClient(api.create_app(make_config(tmp_path, warmup=False)))
    corpus = _make_many(tmp_path)
    client.post("/v1/index", json={"paths": [str(corpus)]})

    body = client.get("/v1/documents").json()
    assert body["total"] == 7
    assert len(body["documents"]) == 7
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_document_cursor_walks_every_record_once(tmp_path, monkeypatch):
    engine = stub_engine(tmp_path)
    monkeypatch.setattr(api, "RagEngine", lambda _config: engine)
    client = TestClient(api.create_app(make_config(tmp_path, warmup=False)))
    corpus = _make_many(tmp_path)
    client.post("/v1/index", json={"paths": [str(corpus)]})

    seen: list[str] = []
    cursor = None
    for _ in range(20):
        params = {"limit": 3}
        if cursor:
            params["cursor"] = cursor
        body = client.get("/v1/documents", params=params).json()
        seen.extend(record["path"] for record in body["documents"])
        if not body["has_more"]:
            assert body["next_cursor"] is None
            break
        cursor = body["next_cursor"]
    assert len(seen) == len(set(seen)) == 7


def test_chunk_cursor_walks_every_record_once(tmp_path, monkeypatch):
    engine = stub_engine(tmp_path)
    monkeypatch.setattr(api, "RagEngine", lambda _config: engine)
    client = TestClient(api.create_app(make_config(tmp_path, warmup=False)))
    corpus = _make_many(tmp_path)
    client.post("/v1/index", json={"paths": [str(corpus)]})

    seen: list[str] = []
    cursor = None
    total = None
    for _ in range(20):
        params = {"limit": 2}
        if cursor:
            params["cursor"] = cursor
        body = client.get("/v1/chunks", params=params).json()
        total = body["total"]
        seen.extend(record["chunk_id"] for record in body["chunks"])
        if not body.get("has_more"):
            break
        cursor = body["next_cursor"]
    assert total is not None
    assert len(seen) == len(set(seen)) == total


def test_invalid_cursor_is_a_400_problem(ctx):
    client, _engine, _corpus = ctx
    response = client.get("/v1/documents", params={"cursor": "not-base64!!"})
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_cursor"
