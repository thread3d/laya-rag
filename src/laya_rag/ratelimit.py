"""An optional per-principal rate limiter.

Off unless ``LAYA_RAG_RATE_LIMIT`` is a positive number of requests per minute. When on,
every response carries the current draft's two structured fields —
``RateLimit-Policy: "default";q=N;w=60`` and ``RateLimit: "default";r=remaining;t=reset`` —
and an exhausted bucket gets a 429 with ``Retry-After``.

The limiter is a fixed window per identity (the bearer token, or the client address when
no token is configured). It is deliberately small: this is a local service, and the goal
is to stop a runaway client, not to meter a fleet.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .problems import APIProblem

WINDOW_S = 60


@dataclass
class LimitState:
    limit: int
    remaining: int
    reset: int

    def headers(self) -> dict[str, str]:
        return {
            "RateLimit-Policy": f'"default";q={self.limit};w={WINDOW_S}',
            "RateLimit": f'"default";r={self.remaining};t={self.reset}',
        }


class RateLimiter:
    """A fixed-window counter per identity, bounded to the identities seen."""

    def __init__(self, limit: int, max_identities: int = 4096) -> None:
        self.limit = max(1, int(limit))
        self.max_identities = max(1, int(max_identities))
        self._lock = threading.Lock()
        self._windows: dict[str, tuple[float, int]] = {}

    def check(self, identity: str, *, now: float | None = None) -> LimitState:
        """Account one request, or raise 429 when the window is exhausted."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if len(self._windows) > self.max_identities:
                # Drop the oldest half rather than grow without bound.
                for key in list(self._windows)[: self.max_identities // 2]:
                    self._windows.pop(key, None)
            start, count = self._windows.get(identity, (now, 0))
            if now - start >= WINDOW_S:
                start, count = now, 0
            reset = max(1, int(WINDOW_S - (now - start) + 0.999))
            if count >= self.limit:
                self._windows[identity] = (start, count)
                raise APIProblem(
                    429,
                    "rate_limited",
                    f"rate limit of {self.limit} requests per minute exceeded",
                    headers={"Retry-After": str(reset)},
                )
            count += 1
            self._windows[identity] = (start, count)
            return LimitState(limit=self.limit, remaining=max(0, self.limit - count), reset=reset)
