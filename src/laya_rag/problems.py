"""RFC 9457 problem responses and request correlation.

FastAPI's default error body is ``{"detail": ...}`` and ``detail`` changes type between
status codes — a string for an ``HTTPException``, a list for a validation failure. This
module installs one error contract instead:

* ``application/problem+json`` with the five RFC 9457 members (``type``, ``title``,
  ``status``, ``detail``, ``instance``), plus two widely-used extensions: ``code`` (a
  stable machine identifier) and ``request_id``.
* ``detail`` stays a **string**, so the bundled GUI's ``detailOf`` reader is unaffected;
  field-level validation errors travel in the ``errors`` extension array.
* Every response carries an ``X-Request-ID``; an inbound one is echoed, otherwise one is
  generated. An inbound W3C ``traceparent`` is validated and its trace id reused.

There is no released FastAPI flag for this (PR #15951 was closed unmerged), so it is
implemented with exception handlers.
"""
from __future__ import annotations

import logging
import re
import secrets
from typing import Any, Iterable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("laya_rag.problems")

PROBLEM_MEDIA_TYPE = "application/problem+json"

#: Problem-type URIs double as the documentation links for a machine-readable error.
PROBLEM_TYPE_BASE = "https://laya-rag.dev/problems/"

#: Stable codes per status, used when a caller raises a bare HTTPException.
DEFAULT_CODES: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    406: "not_acceptable",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    502: "decision_failed",
    503: "service_unavailable",
    504: "gateway_timeout",
}

#: The project's own codes, so every raise site names the same string.
CODE_INDEX_NOT_BUILT = "index_not_built"
CODE_MODEL_UNAVAILABLE = "model_unavailable"
CODE_BUILD_IN_PROGRESS = "build_in_progress"
CODE_INDEX_NOT_UPDATABLE = "index_not_updatable"
CODE_TENANT_MISMATCH = "tenant_mismatch"
CODE_PATH_NOT_ALLOWED = "path_not_allowed"
CODE_SCOPE_REQUIRED = "insufficient_scope"
CODE_COMPOSE_DISABLED = "compose_unavailable"
CODE_IDEMPOTENCY_CONFLICT = "idempotency_conflict"
CODE_IDEMPOTENCY_MISMATCH = "idempotency_mismatch"

_REQUEST_ID = re.compile(r"^[\x20-\x7e]{1,128}$")
_TRACEPARENT = re.compile(r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")


def new_request_id() -> str:
    """A short, opaque correlation id."""
    return f"req_{secrets.token_hex(12)}"


def parse_traceparent(value: str | None) -> str | None:
    """The trace id from a valid W3C ``traceparent``, else ``None``.

    All-zero trace/span ids are invalid per the W3C recommendation, and a malformed
    header is ignored rather than rejected: correlation must never fail a request.
    """
    if not value:
        return None
    match = _TRACEPARENT.match(value.strip())
    if not match:
        return None
    _version, trace_id, span_id, _flags = match.groups()
    if trace_id == "0" * 32 or span_id == "0" * 16:
        return None
    return trace_id


class APIProblem(Exception):
    """An error rendered as RFC 9457 problem+json.

    ``detail`` is always a string. Field-level problems go in ``errors`` as
    ``{"detail", "pointer"}`` objects.
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        detail: str = "",
        *,
        title: str | None = None,
        type_uri: str | None = None,
        headers: dict[str, str] | None = None,
        errors: Iterable[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(detail or code)
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.title = title or http_phrase(status_code)
        self.type_uri = type_uri or f"{PROBLEM_TYPE_BASE}{code}"
        self.headers = dict(headers or {})
        self.errors = list(errors) if errors else None

    def payload(
        self, request_id: str | None, instance: str | None, trace_id: str | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "type": self.type_uri,
            "title": self.title,
            "status": self.status_code,
            "detail": self.detail,
            "instance": instance,
            "code": self.code,
        }
        if request_id:
            body["request_id"] = request_id
        if trace_id:
            body["trace_id"] = trace_id
        if self.errors:
            body["errors"] = self.errors
        return body


_PHRASES = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    406: "Not Acceptable",
    409: "Conflict",
    413: "Content Too Large",
    415: "Unsupported Media Type",
    422: "Unprocessable Content",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}


def http_phrase(status: int) -> str:
    return _PHRASES.get(status, "Error")


def _request_id(request: Request) -> str | None:
    rid = getattr(request.state, "request_id", None)
    if isinstance(rid, str) and rid:
        return rid
    return None


def render_problem(problem: APIProblem, request: Request) -> JSONResponse:
    """Render a problem as the wire response.

    Public so middleware (rate limiting, body size) can answer before the app's
    exception handlers are reached.
    """
    trace_id = getattr(request.state, "trace_id", None)
    return JSONResponse(
        problem.payload(_request_id(request), request.url.path, trace_id),
        status_code=problem.status_code,
        media_type=PROBLEM_MEDIA_TYPE,
        headers=problem.headers,
    )


_render = render_problem


def _from_http_exception(exc: StarletteHTTPException, request: Request) -> JSONResponse:
    """Render FastAPI's ``HTTPException`` as a problem, keeping ``detail`` readable."""
    status = exc.status_code
    detail = exc.detail
    errors: list[dict[str, Any]] = []
    if isinstance(detail, list):
        errors = [
            {
                "detail": str(item.get("msg", item)) if isinstance(item, dict) else str(item),
                "pointer": _pointer(item) if isinstance(item, dict) else None,
            }
            for item in detail
        ]
        detail = "request validation failed"
    elif not isinstance(detail, str):
        detail = str(detail)
    headers = dict(getattr(exc, "headers", None) or {})
    problem = APIProblem(
        status,
        DEFAULT_CODES.get(status, "error"),
        detail,
        headers=headers,
        errors=errors or None,
    )
    return _render(problem, request)


def _pointer(item: dict[str, Any]) -> str | None:
    location = list(item.get("loc") or ())
    # The first element is the location ("body", "query", ...); everything after it is
    # the field path. Stripping every element would also strip a field literally named
    # "query", which is exactly the field one of these routes has.
    if location and str(location[0]) in {"body", "query", "path", "header", "cookie"}:
        location = location[1:]
    if not location:
        return None
    return "#/" + "/".join(str(part) for part in location)


def _from_validation_error(exc: RequestValidationError, request: Request) -> JSONResponse:
    errors = [
        {"detail": str(item.get("msg", "invalid value")), "pointer": _pointer(item)}
        for item in exc.errors()
    ]
    problem = APIProblem(
        422,
        DEFAULT_CODES[422],
        "the request failed validation; see errors",
        errors=errors,
    )
    return _render(problem, request)


def install(app: FastAPI) -> None:
    """Register the problem handlers on ``app``."""

    @app.exception_handler(APIProblem)
    async def _api_problem(request: Request, exc: APIProblem) -> JSONResponse:
        return _render(exc, request)

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _from_http_exception(exc, request)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _from_validation_error(exc, request)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # The traceback goes to the log; the client gets a correlation id, not internals.
        log.exception("unhandled error on %s", request.url.path)
        problem = APIProblem(
            500, DEFAULT_CODES[500], "an internal error occurred; quote request_id when reporting it"
        )
        return _render(problem, request)


async def request_context_middleware(request: Request, call_next: Any) -> Any:
    """Stamp every request with a correlation id and reuse a valid ``traceparent``."""
    inbound = request.headers.get("x-request-id")
    request_id = inbound if inbound and _REQUEST_ID.match(inbound) else new_request_id()
    request.state.request_id = request_id
    trace_id = parse_traceparent(request.headers.get("traceparent"))
    request.state.trace_id = trace_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    if trace_id:
        response.headers["X-Trace-ID"] = trace_id
    return response
