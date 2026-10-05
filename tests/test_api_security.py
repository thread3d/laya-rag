"""Authorization, tenancy, index path scope, idempotency and rate limiting.

These cover the P0 items from ``docs/api-mcp-design.md``: a token's tenant is the only
tenant it can reach, write routes need a write scope, client paths are confined to
configured roots, and a retried mutating request does not run twice.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from laya_rag import api
from laya_rag.auth import parse_api_keys
from laya_rag.idempotency import IdempotencyStore
from laya_rag.problems import APIProblem

from .support import make_config, stub_engine, write_corpus


def _client(tmp_path, monkeypatch, engine, **overrides):
    monkeypatch.setattr(api, "RagEngine", lambda _config: engine)
    return TestClient(api.create_app(make_config(tmp_path, warmup=False, **overrides)))


@pytest.fixture
def corpus(tmp_path):
    return write_corpus(tmp_path / "corpus")


# -- api_keys parsing --------------------------------------------------------

def test_parse_api_keys_supports_shorthand_and_objects():
    principals = parse_api_keys(
        json.dumps(
            {
                "acme": "acme",
                "reader": {"tenant": "acme", "scopes": ["read"], "name": "reader"},
                "admin": {"scopes": ["read", "write"]},
            }
        )
    )
    assert principals["acme"].tenant == "acme"
    assert principals["acme"].scopes == {"read", "write"}
    assert principals["reader"].scopes == {"read"}
    assert principals["admin"].tenant is None


def test_parse_api_keys_rejects_an_unknown_scope():
    with pytest.raises(ValueError):
        parse_api_keys(json.dumps({"tok": {"scopes": ["admin"]}}))


# -- tenancy -----------------------------------------------------------------

def test_bound_token_rejects_a_conflicting_body_tenant(tmp_path, monkeypatch, corpus):
    engine = stub_engine(tmp_path)
    seen: list[str | None] = []
    original = engine.query

    def spy_query(query, **kwargs):
        seen.append(kwargs.get("tenant"))
        return original(query, **kwargs)

    engine.query = spy_query  # type: ignore[method-assign]
    token = {"tok": {"tenant": "acme", "scopes": ["read", "write"]}}
    client = _client(tmp_path, monkeypatch, engine, api_keys=json.dumps(token))
    client.post("/v1/index", json={"paths": [str(corpus)]}, headers={"authorization": "Bearer tok"})

    # No body tenant: the token's tenant is used.
    ok = client.post(
        "/v1/query", json={"query": "refunds"}, headers={"authorization": "Bearer tok"}
    )
    assert ok.status_code == 200
    assert seen[-1] == "acme"

    # A different body tenant is refused, so the ACL cannot be spoofed.
    conflict = client.post(
        "/v1/query",
        json={"query": "refunds", "tenant": "other"},
        headers={"authorization": "Bearer tok"},
    )
    assert conflict.status_code == 403
    assert conflict.json()["code"] == "tenant_mismatch"


def test_legacy_api_key_is_an_admin_with_no_tenant_binding(tmp_path, monkeypatch, corpus):
    engine = stub_engine(tmp_path)
    client = _client(tmp_path, monkeypatch, engine, api_key="secret")
    headers = {"authorization": "Bearer secret"}
    client.post("/v1/index", json={"paths": [str(corpus)]}, headers=headers)
    response = client.post(
        "/v1/query", json={"query": "refunds", "tenant": "anyone"}, headers=headers
    )
    assert response.status_code == 200


# -- scopes ------------------------------------------------------------------

def test_read_only_token_cannot_index(tmp_path, monkeypatch, corpus):
    engine = stub_engine(tmp_path)
    token = {"tok": {"tenant": "acme", "scopes": ["read"]}}
    client = _client(tmp_path, monkeypatch, engine, api_keys=json.dumps(token))
    response = client.post(
        "/v1/index",
        json={"paths": [str(corpus)]},
        headers={"authorization": "Bearer tok"},
    )
    assert response.status_code == 403
    assert response.json()["code"] == "insufficient_scope"
    assert "insufficient_scope" in response.headers["www-authenticate"]


def test_write_only_token_cannot_query(tmp_path, monkeypatch, corpus):
    engine = stub_engine(tmp_path)
    token = {"tok": {"tenant": "acme", "scopes": ["write"]}}
    client = _client(tmp_path, monkeypatch, engine, api_keys=json.dumps(token))
    headers = {"authorization": "Bearer tok"}
    assert client.post("/v1/index", json={"paths": [str(corpus)]}, headers=headers).status_code == 200
    assert client.post("/v1/query", json={"query": "refunds"}, headers=headers).status_code == 403


# -- index path scope --------------------------------------------------------

def test_paths_outside_configured_roots_are_refused(tmp_path, monkeypatch, corpus):
    engine = stub_engine(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("secret", encoding="utf-8")
    client = _client(
        tmp_path, monkeypatch, engine, index_roots=str(corpus), api_key="secret"
    )
    headers = {"authorization": "Bearer secret"}

    denied = client.post("/v1/index", json={"paths": [str(outside)]}, headers=headers)
    assert denied.status_code == 403
    assert denied.json()["code"] == "path_not_allowed"

    allowed = client.post("/v1/index", json={"paths": [str(corpus)]}, headers=headers)
    assert allowed.status_code == 200


def test_loopback_without_roots_keeps_the_historical_behaviour(tmp_path, monkeypatch, corpus):
    engine = stub_engine(tmp_path)
    client = _client(tmp_path, monkeypatch, engine)
    assert client.post("/v1/index", json={"paths": [str(corpus)]}).status_code == 200


def test_reachable_service_without_roots_refuses_arbitrary_paths(tmp_path, monkeypatch, corpus):
    engine = stub_engine(tmp_path)
    client = _client(tmp_path, monkeypatch, engine, host="0.0.0.0")
    response = client.post("/v1/index", json={"paths": [str(corpus)]})
    assert response.status_code == 403
    assert response.json()["code"] == "path_not_allowed"
    assert "LAYA_RAG_INDEX_ROOTS" in response.json()["detail"]


# -- idempotency -------------------------------------------------------------

def test_idempotent_index_replays_without_reindexing(tmp_path, monkeypatch, corpus):
    engine = stub_engine(tmp_path)
    calls: list[int] = []
    original = engine.index_paths

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    engine.index_paths = counted  # type: ignore[method-assign]
    client = _client(tmp_path, monkeypatch, engine)
    headers = {"Idempotency-Key": "abc-123"}
    body = {"paths": [str(corpus)]}

    first = client.post("/v1/index", json=body, headers=headers)
    second = client.post("/v1/index", json=body, headers=headers)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert second.headers["idempotency-replayed"] == "true"
    assert len(calls) == 1


def test_idempotency_key_reused_with_a_different_payload_is_422(tmp_path, monkeypatch, corpus):
    engine = stub_engine(tmp_path)
    client = _client(tmp_path, monkeypatch, engine)
    headers = {"Idempotency-Key": "abc-123"}
    client.post("/v1/index", json={"paths": [str(corpus)]}, headers=headers)
    changed = client.post(
        "/v1/index", json={"paths": [str(corpus)], "rebuild": True}, headers=headers
    )
    assert changed.status_code == 422
    assert changed.json()["code"] == "idempotency_mismatch"


def test_in_flight_idempotency_key_is_a_conflict():
    store = IdempotencyStore(ttl_s=60)
    assert store.begin("scope", "key", "fingerprint") is None
    with pytest.raises(APIProblem) as excinfo:
        store.begin("scope", "key", "fingerprint")
    assert excinfo.value.status_code == 409
    assert excinfo.value.code == "idempotency_conflict"
    store.abandon("scope", "key")
    assert store.begin("scope", "key", "fingerprint") is None


# -- rate limiting -----------------------------------------------------------

def test_rate_limit_is_off_by_default(tmp_path, monkeypatch):
    engine = stub_engine(tmp_path)
    client = _client(tmp_path, monkeypatch, engine)
    assert "ratelimit" not in {name.lower() for name in client.get("/healthz").headers}


def test_rate_limit_blocks_after_the_window_quota(tmp_path, monkeypatch):
    engine = stub_engine(tmp_path)
    token = {"tok": {"tenant": "acme", "scopes": ["read"]}}
    client = _client(
        tmp_path, monkeypatch, engine, api_keys=json.dumps(token), rate_limit=2
    )
    headers = {"authorization": "Bearer tok"}

    first = client.get("/v1/stats", headers=headers)
    second = client.get("/v1/stats", headers=headers)
    assert first.status_code == second.status_code == 200
    assert "ratelimit-policy" in {name.lower() for name in first.headers}
    assert "ratelimit" in {name.lower() for name in first.headers}

    third = client.get("/v1/stats", headers=headers)
    assert third.status_code == 429
    assert third.headers["retry-after"]
    assert third.json()["code"] == "rate_limited"
