"""Opaque cursor pagination for the corpus listings.

Offset pagination shifts under an index that is being rebuilt, so these cursors are
keyset: they name the last item's sort key and the next page starts strictly after it.
The key is base64url JSON — opaque to callers, which is what lets the sort change later
without a wire change.

The service keeps its historical list keys (``documents``, ``chunks``) and adds
``next_cursor``/``has_more`` around them, so a caller that ignores pagination still gets
today's full listing and the bundled GUI is unaffected. ``next_cursor`` is null, not
absent, on the last page.
"""
from __future__ import annotations

import base64
import bisect
import json
from typing import Any, Callable, Iterable, Sequence

from .problems import APIProblem

MAX_LIMIT = 1000


def encode_cursor(key: Sequence[Any]) -> str:
    raw = json.dumps(list(key), separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> list[Any]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except Exception as exc:  # noqa: BLE001 - any decode failure is the same 400
        raise APIProblem(400, "invalid_cursor", "the cursor is not valid") from exc
    if not isinstance(value, list):
        raise APIProblem(400, "invalid_cursor", "the cursor is not valid")
    return value


def paginate(
    items: Sequence[Any],
    key_of: Callable[[Any], Sequence[Any]],
    *,
    limit: int | None = None,
    cursor: str | None = None,
) -> tuple[list[Any], str | None, bool, int]:
    """Return ``(page, next_cursor, has_more, total)``.

    ``items`` must already be sorted by ``key_of``. With no ``limit`` the whole listing
    is returned and pagination is inert, which is the pre-pagination behaviour.
    """
    keys = [tuple(key_of(item)) for item in items]
    start = 0
    if cursor is not None:
        target = tuple(decode_cursor(cursor))
        start = bisect.bisect_right(keys, target)
    if limit is None:
        return list(items[start:]), None, False, len(items)
    if limit < 1 or limit > MAX_LIMIT:
        raise APIProblem(
            422,
            "validation_error",
            f"limit must be between 1 and {MAX_LIMIT}",
        )
    end = start + limit
    page = list(items[start:end])
    has_more = end < len(items)
    next_cursor = encode_cursor(key_of(page[-1])) if has_more and page else None
    return page, next_cursor, has_more, len(items)
