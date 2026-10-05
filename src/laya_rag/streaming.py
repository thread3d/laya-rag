"""Streaming encodings for the index and compose routes.

The service has always streamed newline-delimited JSON. That stays the default, so the
bundled GUI is unchanged. A client that sends ``Accept: text/event-stream`` gets the same
events as SSE instead — the framing the AI ecosystem expects, with ``event:`` names,
monotonic ``id:`` values for reconnection, and a comment heartbeat while the producer is
idle.

Both encodings carry the *same* JSON object per event, so a consumer can switch on
``Accept`` without changing its parser.
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
from typing import Any, AsyncIterator, Iterable, Iterator

from fastapi.sse import format_sse_event

NDJSON_MEDIA_TYPE = "application/x-ndjson"
SSE_MEDIA_TYPE = "text/event-stream"

#: Seconds of silence before a comment heartbeat is emitted.
HEARTBEAT_S = 15.0

STREAM_HEADERS = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}


def wants_sse(accept: str | None) -> bool:
    """Whether the caller asked for SSE explicitly."""
    return bool(accept) and "text/event-stream" in accept.lower()


def ndjson(events: Iterable[dict[str, Any]]) -> Iterator[str]:
    for event in events:
        yield json.dumps(event, ensure_ascii=False) + "\n"


def sse_frame(event: dict[str, Any], sequence: int) -> bytes:
    """One SSE frame carrying the same JSON the NDJSON encoding would send."""
    etype = event.get("type")
    return format_sse_event(
        data_str=json.dumps(event, ensure_ascii=False),
        event=str(etype) if etype else None,
        id=str(sequence),
    )


async def _drain(
    events: Iterable[dict[str, Any]], heartbeat_s: float
) -> AsyncIterator[bytes]:
    """Run a blocking event iterator in a thread, heartbeat while it is idle.

    Indexing is CPU-bound and synchronous, so its iterator blocks; ``queue.get`` with a
    timeout is what lets an idle stream emit a keep-alive comment instead of holding the
    connection silently.
    """
    channel: "queue.Queue[Any]" = queue.Queue()
    sentinel = object()

    def produce() -> None:
        try:
            for event in events:
                channel.put(event)
        finally:
            channel.put(sentinel)

    worker = threading.Thread(target=produce, name="laya-stream", daemon=True)
    worker.start()

    sequence = 0
    while True:
        try:
            item = await asyncio.to_thread(channel.get, True, heartbeat_s)
        except queue.Empty:
            yield format_sse_event(comment="ping")
            continue
        if item is sentinel:
            break
        sequence += 1
        yield sse_frame(item, sequence)


def sse(events: Iterable[dict[str, Any]], heartbeat_s: float = HEARTBEAT_S) -> AsyncIterator[bytes]:
    """The SSE encoding of ``events``."""
    return _drain(events, heartbeat_s)
