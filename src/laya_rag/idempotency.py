"""Idempotency keys for the mutating routes.

``POST /v1/index`` can take minutes and ``POST /v1/reset`` destroys state, so a client
that retries after a dropped connection must not run the work twice. A caller that sends
``Idempotency-Key`` gets the first response replayed verbatim for ``IDEMPOTENCY_TTL_S``.

Follows the IETF idempotency draft's mapping: a key reused with a different payload is a
422, a duplicate while the first is still running is a 409, and a completed duplicate
replays the stored response. The store is bounded and in-process; it is a correctness
guard for retries, not a distributed lock.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .problems import (
    CODE_IDEMPOTENCY_CONFLICT,
    CODE_IDEMPOTENCY_MISMATCH,
    APIProblem,
)


@dataclass
class StoredResponse:
    """A recorded response (or an in-flight marker) for one idempotency key."""

    status: int
    body: dict[str, Any]
    fingerprint: str
    completed: bool


def fingerprint(payload: Any) -> str:
    """A stable hash of a request payload, so a key reuse with new arguments is caught."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class IdempotencyStore:
    """A bounded, TTL'd map from ``(scope, key)`` to a recorded response."""

    def __init__(self, ttl_s: int = 86400, max_entries: int = 1024) -> None:
        self.ttl_s = max(1, int(ttl_s))
        self.max_entries = max(1, int(max_entries))
        self._lock = threading.Lock()
        self._entries: "OrderedDict[str, tuple[float, StoredResponse]]" = OrderedDict()

    def _prune(self, now: float) -> None:
        expired = [key for key, (expiry, _record) in self._entries.items() if expiry <= now]
        for key in expired:
            self._entries.pop(key, None)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def begin(self, scope: str, key: str, request_fingerprint: str) -> StoredResponse | None:
        """Claim ``key``, or return the replay.

        Raises 409 while the original is in flight and 422 when the key was first used
        with a different payload.
        """
        composite = f"{scope}\x00{key}"
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            existing = self._entries.get(composite)
            if existing is not None:
                expiry, record = existing
                if record.fingerprint != request_fingerprint:
                    raise APIProblem(
                        422,
                        CODE_IDEMPOTENCY_MISMATCH,
                        "this Idempotency-Key was already used with a different request",
                    )
                if not record.completed:
                    raise APIProblem(
                        409,
                        CODE_IDEMPOTENCY_CONFLICT,
                        "a request with this Idempotency-Key is still in progress",
                    )
                self._entries.move_to_end(composite)
                return record
            self._entries[composite] = (
                now + self.ttl_s,
                StoredResponse(0, {}, request_fingerprint, completed=False),
            )
            return None

    def complete(
        self, scope: str, key: str, request_fingerprint: str, status: int, body: dict[str, Any]
    ) -> None:
        composite = f"{scope}\x00{key}"
        with self._lock:
            if composite in self._entries:
                self._entries[composite] = (
                    time.monotonic() + self.ttl_s,
                    StoredResponse(status, body, request_fingerprint, completed=True),
                )
                self._entries.move_to_end(composite)

    def abandon(self, scope: str, key: str) -> None:
        """Drop an in-flight marker after a failure, so the client may retry."""
        composite = f"{scope}\x00{key}"
        with self._lock:
            existing = self._entries.get(composite)
            if existing is not None and not existing[1].completed:
                self._entries.pop(composite, None)
