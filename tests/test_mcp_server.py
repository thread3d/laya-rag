"""The MCP server: tool set, the honest-answer contract, gating and HTTP auth.

The MCP SDK is an optional extra, so the whole module skips when it is not installed.
The substance is tested against the same deterministic stub engine as the rest of the
suite, over the SDK's in-memory client — no network, no checkpoint.
"""
from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from laya_rag import mcp_server  # noqa: E402
from laya_rag.auth import Authenticator  # noqa: E402

from .support import make_config, stub_engine, write_corpus  # noqa: E402


@pytest.fixture
def ctx(tmp_path):
    corpus = write_corpus(tmp_path / "corpus")
    config = make_config(tmp_path, warmup=False)
    engine = stub_engine(tmp_path)
    engine.index_paths([str(corpus)])
    backend = mcp_server.LocalBackend(config, engine=engine)
    return config, engine, backend, corpus


def _server(ctx, writes=True):
    config, _engine, backend, _corpus = ctx
    return mcp_server.build_server(config, backend=backend, writes=writes)


def _call(server, name, arguments=None):
    async def run():
        async with Client(server) as client:
            return await client.call_tool(name, arguments or {})

    return asyncio.run(run())


def _tools(server):
    async def run():
        async with Client(server) as client:
            return [tool.name for tool in (await client.list_tools()).tools]

    return asyncio.run(run())


# -- tool set ----------------------------------------------------------------

def test_read_tools_are_registered(ctx):
    names = _tools(_server(ctx, writes=False))
    for expected in (
        "search_evidence",
        "list_documents",
        "read_document",
        "corpus_stats",
        "check_index_integrity",
        "extract_answer",
        "verify_answer",
        "adjudicate_answer",
    ):
        assert expected in names


def test_write_tools_are_gated(ctx):
    assert "index_documents" not in _tools(_server(ctx, writes=False))
    assert "reset_index" not in _tools(_server(ctx, writes=False))
    names = _tools(_server(ctx, writes=True))
    assert "index_documents" in names
    assert "reset_index" in names


def test_compose_tool_is_absent_when_composition_is_disabled(tmp_path):
    config = make_config(tmp_path, warmup=False, compose_model="none")
    backend = mcp_server.LocalBackend(config, engine=stub_engine(tmp_path))
    names = _tools(mcp_server.build_server(config, backend=backend, writes=False))
    assert "compose_answer" not in names


# -- the honest-answer contract ----------------------------------------------

def test_search_evidence_returns_calibrated_evidence(ctx):
    result = _call(_server(ctx), "search_evidence", {"query": "How long does a refund take?"})
    assert result.is_error is False
    body = result.structured_content
    assert body["answerable"] is True
    assert body["answerability"] > 0
    assert body["passages"]
    assert "refunds.md" in body["passages"][0]["path"]
    assert body["passages"][0]["relevance"] > 0


def test_search_evidence_abstains_when_the_corpus_cannot_answer(ctx):
    """The tool's whole value: nothing clears the gate, and `answerable` says so."""
    result = _call(_server(ctx), "search_evidence", {"query": "zzzzz unrelated gibberish"})
    body = result.structured_content
    assert body["answerable"] is False
    assert body["passages"] == []


def test_search_evidence_rejects_an_out_of_range_top_k(ctx):
    result = _call(_server(ctx), "search_evidence", {"query": "refunds", "top_k": 999})
    assert result.is_error is True


# -- corpus inspection -------------------------------------------------------

def test_list_and_read_documents(ctx):
    listing = _call(_server(ctx), "list_documents", {"limit": 2}).structured_content
    assert listing["total"] == 3
    assert len(listing["documents"]) == 2
    assert listing["has_more"] is True

    path = listing["documents"][0]["path"]
    document = _call(_server(ctx), "read_document", {"path": path}).structured_content
    assert document["path"] == path
    assert document["total"] >= 1
    assert document["chunks"][0]["text"]


def test_corpus_stats_and_integrity(ctx):
    stats = _call(_server(ctx), "corpus_stats").structured_content
    assert stats["index_present"] is True
    assert stats["chunks"] >= 3
    assert stats["documents"] == 3

    integrity = _call(_server(ctx), "check_index_integrity").structured_content
    assert integrity["clean"] is True
    assert integrity["chunks"] >= 3


def test_verify_answer_reports_citations(ctx):
    result = _call(
        _server(ctx),
        "verify_answer",
        {
            "answer": "Refunds take five business days [1].",
            "passages": [{"text": "Refunds are processed within five business days."}],
        },
    )
    assert result.is_error is False
    assert result.structured_content["clean"] is True


# -- the generator-free answer path ------------------------------------------

def test_extract_answer_quotes_the_corpus_and_carries_its_evidence(ctx):
    result = _call(_server(ctx), "extract_answer", {"query": "How long does a refund take?"})
    assert result.is_error is False
    body = result.structured_content
    assert body["abstained"] is False
    assert body["answerable"] is True

    texts = [passage["text"] for passage in body["passages"]]
    assert texts
    for sentence in body["sentences"]:
        assert any(sentence["text"] in text for text in texts), sentence["text"]
        assert f"[{sentence['citation']}]" in body["answer"]
        assert sentence["path"]


def test_extract_answer_round_trips_through_verify_answer(ctx):
    """The tool's citation contract has to hold against the tool that checks it."""
    server = _server(ctx)
    extracted = _call(server, "extract_answer", {"query": "How long does a refund take?"})
    body = extracted.structured_content
    verified = _call(
        server,
        "verify_answer",
        {
            "answer": body["answer"],
            "passages": [
                {"text": passage["text"], "path": passage["path"]}
                for passage in body["passages"]
            ],
            "query": body["query"],
        },
    )
    assert verified.is_error is False
    assert verified.structured_content["counts"]["uncited"] == 0
    assert verified.structured_content["clean"] is True


def test_extract_answer_abstains_when_nothing_in_the_corpus_qualifies(ctx):
    result = _call(_server(ctx), "extract_answer", {"query": "What is the airspeed of a swallow?"})
    assert result.is_error is False
    assert result.structured_content["abstained"] is True
    assert result.structured_content["sentences"] == []


def test_extract_answer_rejects_bad_budgets(ctx):
    server = _server(ctx)
    assert _call(server, "extract_answer", {"query": "x", "top_k": 0}).is_error is True
    assert _call(server, "extract_answer", {"query": "x", "max_sentences": 99}).is_error is True


# -- write tools -------------------------------------------------------------

def test_reset_index_requires_confirmation(ctx):
    result = _call(_server(ctx, writes=True), "reset_index", {"confirm": "please"})
    assert result.is_error is True
    assert "confirm" in result.content[0].text


def test_reset_index_deletes_the_index(ctx):
    server = _server(ctx, writes=True)
    result = _call(server, "reset_index", {"confirm": "reset"})
    assert result.is_error is False
    assert _call(server, "corpus_stats").structured_content["index_present"] is False


# -- resources and prompts ---------------------------------------------------

def test_resources_and_prompts_are_exposed(ctx):
    server = _server(ctx)

    async def run():
        async with Client(server) as client:
            resources = [str(r.uri) for r in (await client.list_resources()).resources]
            prompts = [p.name for p in (await client.list_prompts()).prompts]
            stats = await client.read_resource("laya-rag://stats")
            return resources, prompts, stats

    resources, prompts, stats = asyncio.run(run())
    assert "laya-rag://stats" in resources
    assert "laya-rag://documents" in resources
    assert "answer_from_corpus" in prompts
    assert "audit_answer" in prompts
    payload = json.loads(stats.contents[0].text)
    assert payload["index_present"] is True


# -- transports and configuration --------------------------------------------

def test_backend_defaults_to_local(ctx, monkeypatch):
    monkeypatch.delenv("LAYA_RAG_MCP_BACKEND", raising=False)
    config, _engine, _backend, _corpus = ctx
    assert isinstance(mcp_server.make_backend(config), mcp_server.LocalBackend)


def test_http_backend_is_selected_by_environment(ctx, monkeypatch):
    config, _engine, _backend, _corpus = ctx
    monkeypatch.setenv("LAYA_RAG_MCP_BACKEND", "http")
    monkeypatch.setenv("LAYA_RAG_MCP_URL", "http://example.test:9000")
    backend = mcp_server.make_backend(config)
    assert isinstance(backend, mcp_server.HttpBackend)
    assert backend.base_url == "http://example.test:9000"


def test_http_transport_refuses_non_loopback_without_a_token(ctx):
    server = _server(ctx)
    config, _engine, _backend, _corpus = ctx
    with pytest.raises(SystemExit):
        mcp_server.run_http(server, config, "0.0.0.0", 8078)


def test_bearer_middleware_challenges_and_admits(tmp_path):
    authenticator = Authenticator(make_config(tmp_path, api_key="secret"))

    async def exercise(headers):
        sent: list[dict] = []

        async def inner(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        middleware = mcp_server._bearer_middleware(authenticator)(inner)

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        await middleware(
            {"type": "http", "method": "POST", "path": "/mcp", "headers": headers},
            receive,
            send,
        )
        return sent

    challenged = asyncio.run(exercise([]))
    assert challenged[0]["status"] == 401
    admitted = asyncio.run(exercise([(b"authorization", b"Bearer secret")]))
    assert admitted[0]["status"] == 200


def test_write_tools_flag_reads_the_environment(monkeypatch):
    monkeypatch.setenv("LAYA_RAG_MCP_ALLOW_WRITES", "1")
    assert mcp_server.allow_writes() is True
    monkeypatch.setenv("LAYA_RAG_MCP_ALLOW_WRITES", "no")
    assert mcp_server.allow_writes() is False
