"""The streaming contract: NDJSON by default, SSE on request, and index build locking."""
from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from laya_rag import api
from laya_rag.engine import BuildInProgressError
from laya_rag.streaming import _drain, ndjson, sse_frame, wants_sse

from .support import make_config, stub_engine, write_corpus


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    engine = stub_engine(tmp_path)
    monkeypatch.setattr(api, "RagEngine", lambda _config: engine)
    app = api.create_app(make_config(tmp_path, warmup=False))
    return TestClient(app), engine, write_corpus(tmp_path / "corpus")


def _sse_events(text: str) -> list[dict]:
    """Parse the JSON payload of each SSE ``data:`` frame (comments are skipped)."""
    events = []
    for frame in text.split("\n\n"):
        if not frame.strip() or frame.lstrip().startswith(":"):
            continue
        data = "\n".join(
            line[len("data:") :].lstrip()
            for line in frame.splitlines()
            if line.startswith("data:")
        )
        if data:
            events.append(json.loads(data))
    return events


# -- negotiation -------------------------------------------------------------

def test_wants_sse_checks_the_accept_header():
    assert wants_sse("text/event-stream")
    assert wants_sse("application/json, text/event-stream;q=0.9")
    assert not wants_sse(None)
    assert not wants_sse("application/x-ndjson")


def test_ndjson_is_the_default_encoding(ctx):
    client, _engine, corpus = ctx
    response = client.post("/v1/index/stream", json={"paths": [str(corpus)]})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert events[-1]["type"] == "done"


def test_sse_is_used_when_requested_and_carries_the_same_events(ctx):
    client, _engine, corpus = ctx
    response = client.post(
        "/v1/index/stream",
        json={"paths": [str(corpus)]},
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    # The wire uses the standard framing, and every frame carries an event name + id.
    frames = [f for f in response.text.split("\n\n") if f.strip()]
    assert all("event:" in frame for frame in frames)
    assert [int(line.split(":", 1)[1]) for line in response.text.splitlines() if line.startswith("id:")] == list(
        range(1, len(frames) + 1)
    )

    events = _sse_events(response.text)
    assert events[0]["type"] == "progress"
    assert events[-1]["type"] == "done"
    assert events[-1]["report"]["documents"] == 3

    # NDJSON and SSE carry the same JSON objects.
    plain = client.post("/v1/index/stream", json={"paths": [str(corpus)]})
    plain_events = [json.loads(line) for line in plain.text.splitlines() if line.strip()]
    assert [e["type"] for e in plain_events] == [e["type"] for e in events]


def test_sse_reports_a_failure_as_an_event(ctx, monkeypatch):
    client, engine, corpus = ctx

    def boom(*_args, **_kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(engine, "index_paths", boom)
    response = client.post(
        "/v1/index/stream",
        json={"paths": [str(corpus)]},
        headers={"Accept": "text/event-stream"},
    )
    events = _sse_events(response.text)
    assert events[-1]["type"] == "error"
    assert "disk on fire" in events[-1]["message"]


def test_compose_streams_sse_from_the_composer(ctx):
    client, engine, _corpus = ctx

    class FakeComposer:
        config = engine.config

        def stream(self, query, passages, max_tokens=None):
            yield {"type": "delta", "text": "Refunds "}
            yield {"type": "delta", "text": "take five days."}
            yield {"type": "done", "model": "fake", "chars": 24, "finish_reason": "stop"}

    engine.composer = FakeComposer()
    response = client.post(
        "/v1/compose",
        json={"query": "refunds", "passages": [{"text": "Refunds take five days."}]},
        headers={"Accept": "text/event-stream"},
    )
    events = _sse_events(response.text)
    assert [event["type"] for event in events] == ["delta", "delta", "done"]
    assert "".join(e.get("text", "") for e in events) == "Refunds take five days."


# -- framing and keepalive ---------------------------------------------------

def test_sse_frame_is_wire_valid():
    frame = sse_frame({"type": "delta", "text": "hi"}, 3).decode()
    assert "event: delta" in frame
    assert 'data: {"type": "delta", "text": "hi"}' in frame
    assert "id: 3" in frame
    assert frame.endswith("\n\n")


def test_sse_emits_a_heartbeat_while_the_producer_is_idle():
    def slow():
        time.sleep(0.05)
        yield {"type": "done"}

    async def collect():
        return [chunk async for chunk in _drain(slow(), heartbeat_s=0.01)]

    chunks = asyncio.run(collect())
    assert any(chunk.startswith(b": ping") for chunk in chunks)
    assert chunks[-1].startswith(b"event: done") or b"event: done" in chunks[-1]


def test_ndjson_iterates_events():
    assert list(ndjson([{"type": "a"}, {"type": "b"}])) == [
        '{"type": "a"}\n',
        '{"type": "b"}\n',
    ]


# -- build locking -----------------------------------------------------------

def test_concurrent_build_is_refused(tmp_path):
    engine = stub_engine(tmp_path)
    corpus = write_corpus(tmp_path / "corpus")
    held = threading.Event()
    release = threading.Event()

    def hold():
        with engine._build_lock:
            held.set()
            release.wait(timeout=5)

    thread = threading.Thread(target=hold)
    thread.start()
    assert held.wait(timeout=5)
    try:
        with pytest.raises(BuildInProgressError):
            engine.index_paths([str(corpus)])
        with pytest.raises(BuildInProgressError):
            engine.reset()
    finally:
        release.set()
        thread.join(timeout=5)

    # Once the lock is free the build proceeds.
    assert engine.index_paths([str(corpus)]).documents == 3


def test_query_is_refused_while_a_build_is_active(tmp_path):
    engine = stub_engine(tmp_path)
    corpus = write_corpus(tmp_path / "corpus")
    engine.index_paths([str(corpus)])
    engine._build_active.set()
    try:
        with pytest.raises(BuildInProgressError):
            engine.query("refunds")
        with pytest.raises(BuildInProgressError):
            engine.adjudicate_claims(["refunds take five days"])
    finally:
        engine._build_active.clear()


def test_api_maps_a_build_conflict_to_409(tmp_path, monkeypatch):
    engine = stub_engine(tmp_path)
    monkeypatch.setattr(api, "RagEngine", lambda _config: engine)
    app = api.create_app(make_config(tmp_path, warmup=False))
    client = TestClient(app)
    corpus = write_corpus(tmp_path / "corpus")

    # Hold the build lock as a concurrent build would, so the second request loses.
    engine._build_lock.acquire()
    try:
        response = client.post("/v1/index", json={"paths": [str(corpus)]})
    finally:
        engine._build_lock.release()
    assert response.status_code == 409
    assert response.json()["code"] == "build_in_progress"

    # With the lock free the same request succeeds.
    assert client.post("/v1/index", json={"paths": [str(corpus)]}).status_code == 200


def test_build_lock_releases_after_a_failed_build(tmp_path):
    """A failed build must not leave the lock held, or the service is wedged."""
    engine = stub_engine(tmp_path)

    def boom(*_args, **_kwargs):
        raise RuntimeError("build exploded")

    engine._index_paths_locked = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        engine.index_paths([str(tmp_path)])
    assert engine._build_lock.acquire(blocking=False)
    engine._build_lock.release()
    assert not engine._build_active.is_set()
