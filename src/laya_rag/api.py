"""HTTP service exposing the RAG engine.

The route names mirror what the engine actually does —
``/v1/index`` and ``/v1/query`` — rather than pretending to be an OpenAI-compatible
endpoint, because Laya is not a completion model and a shim would misrepresent it.

The service is designed as a contract other software can depend on: every route has a
typed response model, errors are RFC 9457 ``application/problem+json`` with a stable
``code`` and a correlation id, the corpus listings paginate, the mutating routes accept
an ``Idempotency-Key``, and the streaming routes content-negotiate between NDJSON (the
historical default, used by the bundled GUI) and SSE. See
``docs/api-mcp-design.md`` for the design and what each choice conforms to.

Handlers are synchronous on purpose: Laya's forward passes are CPU-bound, and
FastAPI runs ``def`` handlers in its worker threadpool, which keeps the event loop
free without any hand-rolled thread management.
"""
from __future__ import annotations

import logging
import queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from fastapi import Depends, FastAPI, Header, Query, Request, Response
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import DEFAULT_EXCLUDED_CONTENT_TYPES, GZipMiddleware

from . import __version__
from .auth import SCOPE_READ, SCOPE_WRITE, Authenticator, Principal, check_index_paths
from .compose import resolve_target
from .config import EngineConfig
from .engine import BuildInProgressError, IndexNotBuiltError, RagEngine
from .evidence import group_passages_by_document
from .idempotency import IdempotencyStore, fingerprint
from .integrity import audit
from .model import ModelUnavailableError
from .paging import MAX_LIMIT, paginate
from .problems import (
    CODE_BUILD_IN_PROGRESS,
    CODE_COMPOSE_DISABLED,
    CODE_INDEX_NOT_UPDATABLE,
    CODE_MODEL_UNAVAILABLE,
    PROBLEM_MEDIA_TYPE,
    APIProblem,
    http_phrase,
    install as install_problem_handlers,
    render_problem,
    request_context_middleware,
)
from .ratelimit import RateLimiter
from .rerank import DecisionError
from .schemas import (
    AdjudicateResponse,
    ChunkListResponse,
    DeleteDocumentsRequest,
    DeleteDocumentsResponse,
    DocumentListResponse,
    HealthResponse,
    IndexReportResponse,
    IntegrityResponse,
    LivenessResponse,
    Problem,
    QueryResponse,
    ReadinessResponse,
    ResetResponse,
    StatsResponse,
    VerifyResponse,
)
from .streaming import (
    NDJSON_MEDIA_TYPE,
    SSE_MEDIA_TYPE,
    STREAM_HEADERS,
    ndjson,
    sse,
    wants_sse,
)
from .verify import split_claims, verify_citations

log = logging.getLogger("laya_rag.api")

WEB_DIR = Path(__file__).resolve().parent / "web"

#: When ``/health`` was superseded by ``/healthz`` + ``/readyz`` (RFC 9745 `@` seconds).
#: No `Sunset` is emitted: the alias has no scheduled removal.
HEALTH_DEPRECATED_AT = 1759622400

_PROBLEM_DESCRIPTIONS = {
    400: "Malformed request",
    401: "Missing or invalid bearer token",
    403: "The token is not allowed to perform this action",
    404: "The requested path or document does not exist",
    409: "The request conflicts with current server state (no index, or a build is running)",
    413: "Request body too large",
    422: "Request validation failed; field errors are in `errors`",
    429: "Rate limit exceeded",
    502: "The decision engine returned an unusable result",
    503: "The model or composer is unavailable",
}


def _problems(*codes: int) -> dict[int, dict[str, Any]]:
    """Document error responses against the shared :class:`Problem` schema.

    The media type is spelled out because RFC 9457 requires
    ``application/problem+json``; letting FastAPI infer it would document
    ``application/json`` and misdescribe the wire.
    """
    return {
        code: {
            "description": _PROBLEM_DESCRIPTIONS.get(code, http_phrase(code)),
            "content": {
                PROBLEM_MEDIA_TYPE: {"schema": {"$ref": "#/components/schemas/Problem"}}
            },
        }
        for code in codes
    }


_READ_ERRORS = _problems(400, 401, 403, 409, 422, 429, 503)
_WRITE_ERRORS = _problems(400, 401, 403, 409, 413, 422, 429, 503)
_STREAM_RESPONSES = {
    200: {
        "description": "Event stream; the encoding follows the Accept header",
        "content": {
            NDJSON_MEDIA_TYPE: {"schema": {"type": "string"}},
            SSE_MEDIA_TYPE: {"schema": {"type": "string"}},
        },
    },
    **_problems(401, 403, 409, 413, 422, 429, 503),
}

OPENAPI_TAGS = [
    {"name": "health", "description": "Liveness and readiness probes."},
    {"name": "corpus", "description": "Read-only corpus introspection."},
    {"name": "retrieval", "description": "Retrieve calibrated evidence for a query."},
    {"name": "verification", "description": "Check an answer against evidence or the corpus."},
    {"name": "indexing", "description": "Build, stream and delete the index."},
]


def _web_dir() -> Path | None:
    """The bundled GUI directory, or ``None`` when its assets are not installed."""
    return WEB_DIR if (WEB_DIR / "index.html").is_file() else None


def _index_html() -> str:
    """The GUI document with its asset URLs stamped by modification time.

    There is no build step and no content hash in the filenames, so a browser left to
    heuristically cache ``app.js`` can pair an old script with new markup — the page
    then keeps its previous layout and a shipped change looks like it never happened.
    Appending the file's mtime gives each change a URL the browser has never seen, so
    the new code is fetched instead of reused, while an unchanged file still answers a
    conditional request from its ETag.
    """
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    for asset in ("app.css", "app.js"):
        path = WEB_DIR / asset
        stamp = int(path.stat().st_mtime) if path.is_file() else 0
        html = html.replace(f'"{asset}"', f'"{asset}?v={stamp}"')
    return html


class _GuiFiles(StaticFiles):
    """Static GUI assets, served so a stale page cannot outlive the code.

    The interface is plain HTML, CSS and JavaScript with no build step and no content
    hash in its filenames, so a browser left to heuristically cache ``app.js`` can keep
    running an old UI against new markup — which is how a change that has shipped looks
    like it never happened. ``no-cache`` means "revalidate before reuse", not "do not
    store": the ETag still makes an unchanged asset a cheap 304, while a changed one is
    picked up by an ordinary refresh rather than a hard reload.
    """

    async def get_response(self, path: str, scope: Any) -> Any:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


class IndexRequest(BaseModel):
    """A request to index one or more files or directories."""

    paths: list[str] = Field(..., min_length=1, description="Files or directories to index")
    rebuild: bool = Field(False, description="Discard the existing index first")
    extensions: list[str] | None = Field(None, description="Override the indexable file extensions")
    prune: bool = Field(
        True,
        description="Drop chunks of files deleted since the last run when re-indexing a directory",
    )


class QueryRequest(BaseModel):
    """A request for evidence supporting a query."""

    query: str = Field(..., min_length=1, description="The question to find evidence for")
    top_k: int | None = Field(None, ge=1, le=50, description="Passages to return after re-ranking")
    retrieve_k: int | None = Field(None, ge=1, le=500, description="Dense candidates to consider")
    relevance_threshold: float | None = Field(None, ge=0.0, le=1.0)
    answerability_threshold: float | None = Field(None, ge=0.0, le=1.0)
    max_passage_chars: int | None = Field(2000, ge=100, le=20000)
    tenant: str | None = Field(
        None,
        description="Caller identity for the trust manifest's per-tenant ACL. When the "
                    "bearer token is bound to a tenant, this must match it or be omitted",
    )


class ComposePassage(BaseModel):
    """One piece of evidence to compose from."""

    text: str = Field(..., min_length=1)
    path: str | None = Field(None, description="Source path, cited in the answer")


#: Characters that occupy a string without being visible. A query pasted from a rich
#: text source can be nothing but these, which passes `min_length=1`, reaches the model
#: as "Question: " and produces an answer complaining the question was empty.
_INVISIBLE = "\u200b\u200c\u200d\u200e\u200f\u2060\u2028\u2029\ufeff"


def _require_visible(value: str, field: str) -> str:
    """Reject a string that has no visible content, and trim the rest."""
    stripped = value.strip()
    visible = stripped.strip(_INVISIBLE).strip()
    if not visible:
        raise ValueError(f"{field} must contain visible text, not just whitespace")
    return stripped


class ComposeRequest(BaseModel):
    """A request to write prose from evidence the caller already holds.

    The evidence travels with the request rather than being re-retrieved, so a
    composed answer is written from exactly the passages the caller was shown.
    """

    query: str = Field(..., min_length=1, description="The question to answer")
    passages: list[ComposePassage] = Field(..., min_length=1, max_length=50)
    max_tokens: int | None = Field(None, ge=64, le=4000)

    @field_validator("query")
    @classmethod
    def _query_has_content(cls, value: str) -> str:
        return _require_visible(value, "query")


class VerifyRequest(BaseModel):
    """A request to check an answer's citations against the evidence it was given."""

    answer: str = Field(..., min_length=1, description="The composed answer to check")
    passages: list[ComposePassage] = Field(..., min_length=1, max_length=50)
    query: str | None = Field(
        None,
        description="The question the answer was written from, so that quoting it back "
                    "is not mistaken for a fabricated quote",
    )

    @field_validator("answer")
    @classmethod
    def _answer_has_content(cls, value: str) -> str:
        return _require_visible(value, "answer")


class AdjudicateRequest(BaseModel):
    """A request to ask Laya whether the corpus supports each claim in an answer."""

    answer: str = Field(..., min_length=1, description="The composed answer to adjudicate")
    retrieve_k: int | None = Field(None, ge=1, le=100, description="Candidates per claim")
    relevance_threshold: float | None = Field(None, ge=0.0, le=1.0)
    answerability_threshold: float | None = Field(None, ge=0.0, le=1.0)
    tenant: str | None = Field(
        None, description="Caller identity for the trust manifest's per-tenant ACL"
    )

    @field_validator("answer")
    @classmethod
    def _answer_has_content(cls, value: str) -> str:
        return _require_visible(value, "answer")


def _engine_holder(config: EngineConfig) -> tuple[dict[str, RagEngine], Any]:
    holder: dict[str, RagEngine] = {}

    def get_engine() -> RagEngine:
        engine = holder.get("engine")
        if engine is None:
            engine = RagEngine(config)
            holder["engine"] = engine
        return engine

    return holder, get_engine


def create_app(config: EngineConfig | None = None) -> FastAPI:
    """Build the FastAPI application for ``config``."""
    config = config or EngineConfig()
    holder, get_engine = _engine_holder(config)
    authenticator = Authenticator(config)
    idempotency = IdempotencyStore(ttl_s=config.idempotency_ttl_s)
    limiter = RateLimiter(config.rate_limit) if config.rate_limit > 0 else None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if config.warmup:
            try:
                get_engine().model.ensure_loaded()
                log.info("Laya checkpoint %r warmed up", config.checkpoint)
            except ModelUnavailableError as exc:
                # A missing checkpoint must not stop the service: /readyz reports it
                # and every real request fails with the same actionable message.
                log.warning("warmup skipped: %s", exc)
        yield
        holder.clear()

    openapi_url = "/openapi.json" if config.schema_public else None
    app = FastAPI(
        title="Laya RAG engine",
        description=(
            "Retrieval-augmented generation over Laya embeddings and decisions. The engine "
            "returns calibrated evidence, not prose: `answerable` is Laya's verdict on whether "
            "the returned passages support an answer, and callers should not synthesise one when "
            "it is false. Errors are RFC 9457 `application/problem+json`."
        ),
        version=__version__,
        lifespan=lifespan,
        openapi_tags=OPENAPI_TAGS,
        openapi_url=openapi_url,
        docs_url="/docs" if config.schema_public else None,
        redoc_url="/redoc" if config.schema_public else None,
    )

    install_problem_handlers(app)

    def authorise(authorization: str | None = Header(default=None)) -> Principal:
        return authenticator.authenticate(authorization)

    def read_access(principal: Principal = Depends(authorise)) -> Principal:
        """Authenticate and require the read scope."""
        authenticator.check_scope(principal, SCOPE_READ)
        return principal

    def tenant_for(principal: Principal, requested: str | None) -> str | None:
        return authenticator.resolve_tenant(principal, requested)

    def require_write(principal: Principal) -> None:
        authenticator.check_scope(principal, SCOPE_WRITE)

    def idempotency_scope(request: Request, principal: Principal) -> str:
        return principal.token or (request.client.host if request.client else "anonymous")

    # -- middleware ------------------------------------------------------
    @app.middleware("http")
    async def body_size_limit(request: Request, call_next: Any) -> Any:
        """Refuse an oversized body before the handler reads it."""
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > config.max_request_bytes:
            problem = APIProblem(
                413,
                "payload_too_large",
                f"request body exceeds {config.max_request_bytes} bytes",
            )
            return render_problem(problem, request)
        return await call_next(request)

    @app.middleware("http")
    async def rate_limit(request: Request, call_next: Any) -> Any:
        if limiter is None:
            return await call_next(request)
        identity = request.headers.get("authorization") or (
            request.client.host if request.client else "anonymous"
        )
        try:
            state = limiter.check(identity)
        except APIProblem as problem:
            return render_problem(problem, request)
        response = await call_next(request)
        for name, value in state.headers().items():
            response.headers[name] = value
        return response

    # Added before gzip/CORS so it sits outside the rate-limit and body-size
    # middlewares, which render problems and need the correlation id already set.
    app.middleware("http")(request_context_middleware)

    if config.gzip:
        app.add_middleware(
            GZipMiddleware,
            minimum_size=1024,
            # A progress stream must never be buffered to compress it.
            exclude_content_types=(*DEFAULT_EXCLUDED_CONTENT_TYPES, NDJSON_MEDIA_TYPE),
        )
    if config.resolved_cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.resolved_cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[
                "X-Request-ID",
                "Idempotency-Replayed",
                "RateLimit",
                "RateLimit-Policy",
                "Retry-After",
            ],
            allow_credentials=False,
        )

    # -- health ----------------------------------------------------------
    @app.get(
        "/healthz",
        tags=["health"],
        operation_id="get_liveness",
        response_model=LivenessResponse,
        summary="Liveness probe",
    )
    def liveness() -> dict[str, Any]:
        """Cheap liveness: the process serves HTTP. Never touches the model."""
        return {"status": "ok"}

    def readiness_payload() -> dict[str, Any]:
        engine = get_engine()
        model_loaded = bool(getattr(engine.model, "_agent", None) is not None)
        error: str | None = None
        if not model_loaded:
            try:
                engine.model.ensure_loaded()
                model_loaded = True
            except ModelUnavailableError as exc:
                error = str(exc)
        index_present = engine.has_index()
        ready = model_loaded and index_present
        return {
            "status": "ready" if ready else "degraded",
            "ready": ready,
            "checkpoint": config.checkpoint,
            "model_loaded": model_loaded,
            "index_present": index_present,
            "error": error,
            "retry_after": None if ready else 30,
        }

    @app.get(
        "/readyz",
        tags=["health"],
        operation_id="get_readiness",
        response_model=ReadinessResponse,
        summary="Readiness probe",
        responses={503: {"model": ReadinessResponse, "description": "Not ready to serve queries"}},
    )
    def readiness(response: Response) -> dict[str, Any]:
        """Whether the model can load and an index exists; 503 when it cannot."""
        payload = readiness_payload()
        if not payload["ready"]:
            response.status_code = 503
            response.headers["Retry-After"] = str(payload["retry_after"])
        return payload

    @app.get(
        "/health",
        tags=["health"],
        operation_id="get_health",
        response_model=HealthResponse,
        summary="Legacy combined health (deprecated)",
        responses=_problems(401, 422, 429),
    )
    def health(response: Response, _: Principal = Depends(authorise)) -> dict[str, Any]:
        """Liveness plus whether the checkpoint and index are actually usable.

        Deprecated in favour of ``/healthz`` and ``/readyz``, which separate the two.
        """
        engine = get_engine()
        model_loaded = bool(getattr(engine.model, "_agent", None) is not None)
        payload: dict[str, Any] = {
            "status": "ok",
            "checkpoint": config.checkpoint,
            "model_loaded": model_loaded,
            "index_present": engine.has_index(),
        }
        if not model_loaded:
            try:
                engine.model.ensure_loaded()
                payload["model_loaded"] = True
            except ModelUnavailableError as exc:
                payload["status"] = "degraded"
                payload["error"] = str(exc)
        response.headers["Deprecation"] = f"@{HEALTH_DEPRECATED_AT}"
        response.headers["Link"] = '</healthz>; rel="successor-version"'
        return payload

    # -- corpus introspection --------------------------------------------
    @app.get(
        "/v1/stats",
        tags=["corpus"],
        operation_id="get_stats",
        response_model=StatsResponse,
        summary="Configuration, model and corpus statistics",
        responses=_READ_ERRORS,
    )
    def stats(_: Principal = Depends(read_access)) -> dict[str, Any]:
        return get_engine().stats()

    @app.get(
        "/v1/documents",
        tags=["corpus"],
        operation_id="list_documents",
        response_model=DocumentListResponse,
        summary="Indexed documents, page by page",
        responses=_READ_ERRORS,
    )
    def documents(
        limit: int | None = Query(None, ge=1, le=MAX_LIMIT, description="Documents per page"),
        cursor: str | None = Query(None, description="Opaque cursor from a previous page"),
        _: Principal = Depends(read_access),
    ) -> dict[str, Any]:
        """Every indexed document, with its chunk and character counts.

        Without ``limit`` the whole listing is returned, which is the historical
        behaviour; with it, ``next_cursor`` walks the corpus. Returns an empty listing
        rather than an error when no index exists.
        """
        listing = get_engine().list_documents()
        page, next_cursor, has_more, total = paginate(
            listing, lambda record: (record["path"],), limit=limit, cursor=cursor
        )
        return {
            "total": total,
            "documents": page,
            "has_more": has_more,
            "next_cursor": next_cursor,
        }

    @app.get(
        "/v1/integrity",
        tags=["corpus"],
        operation_id="get_integrity",
        response_model=IntegrityResponse,
        summary="Index drift: deleted, duplicate and near-duplicate chunks",
        responses=_READ_ERRORS,
    )
    def integrity(
        near_duplicates: bool = False,
        threshold: float = 0.85,
        max_chunks: int = 20000,
        _: Principal = Depends(read_access),
    ) -> dict[str, Any]:
        """Index drift: deleted documents, duplicate and near-duplicate chunks.

        Reports only — nothing here mutates the index. The near-duplicate scan is opt-in
        because it is the only part that is not linear in the corpus.
        """
        engine = get_engine()
        if not engine.has_index():
            return {"chunks": 0, "paths": 0, "clean": True, "index_present": False}
        report = audit(
            engine.index.chunks,
            near_duplicates=near_duplicates,
            threshold=threshold,
            max_chunks=max_chunks,
        )
        report["index_present"] = True
        return report

    @app.get(
        "/v1/chunks",
        tags=["corpus"],
        operation_id="list_chunks",
        response_model=ChunkListResponse,
        summary="Indexed chunks in document order, page by page",
        responses=_READ_ERRORS,
    )
    def chunks(
        path: str | None = None,
        limit: int | None = Query(None, ge=1, le=MAX_LIMIT, description="Chunks per page"),
        cursor: str | None = Query(None, description="Opaque cursor from a previous page"),
        _: Principal = Depends(read_access),
    ) -> dict[str, Any]:
        """Indexed chunks in document order, optionally narrowed to one path."""
        records = get_engine().list_chunks(path)
        page, next_cursor, has_more, total = paginate(
            records,
            lambda record: (record["path"], record["index"]),
            limit=limit,
            cursor=cursor,
        )
        return {
            "total": total,
            "path": path,
            "chunks": page,
            "has_more": has_more,
            "next_cursor": next_cursor,
        }

    # -- indexing --------------------------------------------------------
    @app.post(
        "/v1/index",
        tags=["indexing"],
        operation_id="create_index",
        response_model=IndexReportResponse,
        summary="Index files or directories",
        responses=_WRITE_ERRORS,
    )
    def index_documents(
        request: IndexRequest,
        http_request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        principal: Principal = Depends(authorise),
    ) -> Any:
        """Index the requested paths and return what was read, skipped and embedded.

        Safe to retry with an ``Idempotency-Key``: a replay returns the first report
        instead of indexing twice.
        """
        require_write(principal)
        check_index_paths(config, request.paths)
        engine = get_engine()
        extensions: Iterable[str] | None = request.extensions

        scope: str | None = None
        digest: str | None = None
        if idempotency_key:
            scope = idempotency_scope(http_request, principal)
            digest = fingerprint(
                {
                    "paths": list(request.paths),
                    "rebuild": request.rebuild,
                    "prune": request.prune,
                    "extensions": list(extensions) if extensions else None,
                }
            )
            stored = idempotency.begin(scope, idempotency_key, digest)
            if stored is not None:
                return JSONResponse(
                    stored.body,
                    status_code=stored.status,
                    headers={"Idempotency-Replayed": "true"},
                )

        def abandoned() -> None:
            if idempotency_key and scope is not None:
                idempotency.abandon(scope, idempotency_key)

        try:
            report = engine.index_paths(
                request.paths,
                rebuild=request.rebuild,
                prune=request.prune,
                **({"extensions": extensions} if extensions else {}),
            )
        except BuildInProgressError as exc:
            abandoned()
            raise APIProblem(409, CODE_BUILD_IN_PROGRESS, str(exc)) from exc
        except ModelUnavailableError as exc:
            abandoned()
            raise APIProblem(503, CODE_MODEL_UNAVAILABLE, str(exc)) from exc
        except FileNotFoundError as exc:
            abandoned()
            raise APIProblem(404, "not_found", str(exc)) from exc
        payload = report.as_dict()
        # Present the same shape the response model serialises, so an idempotent replay
        # is byte-identical to the original response.
        payload.setdefault("note", None)
        if report.documents == 0:
            payload["note"] = "no readable text documents were found under the given paths"
        if idempotency_key and scope is not None and digest is not None:
            idempotency.complete(scope, idempotency_key, digest, 200, payload)
        return payload

    @app.post(
        "/v1/index/stream",
        tags=["indexing"],
        operation_id="stream_index",
        summary="Index files or directories, streaming progress",
        responses=_STREAM_RESPONSES,
    )
    def index_documents_stream(
        request: IndexRequest,
        accept: str | None = Header(default=None),
        principal: Principal = Depends(authorise),
    ) -> StreamingResponse:
        """Index the requested paths, streaming progress as NDJSON or SSE.

        The same work as ``POST /v1/index``, reported while it happens: a ``progress``
        event per file read, per document chunked and per embedding batch, then one
        terminal ``done`` carrying the report ``/v1/index`` returns, or an ``error``.
        Because indexing is synchronous and CPU-bound it runs on a worker thread, and
        the stream reads its counter events off a queue. Send
        ``Accept: text/event-stream`` for SSE framing; NDJSON is the default.
        """
        require_write(principal)
        check_index_paths(config, request.paths)
        engine = get_engine()
        extensions: Iterable[str] | None = request.extensions

        def events() -> Iterator[dict[str, Any]]:
            counters: queue.Queue[dict[str, Any] | None] = queue.Queue()
            outcome: dict[str, Any] = {}

            def work() -> None:
                try:
                    outcome["report"] = engine.index_paths(
                        request.paths,
                        rebuild=request.rebuild,
                        prune=request.prune,
                        progress=counters.put,
                        **({"extensions": extensions} if extensions else {}),
                    )
                except BuildInProgressError as exc:
                    outcome["error"] = (409, str(exc))
                except ModelUnavailableError as exc:
                    outcome["error"] = (503, str(exc))
                except FileNotFoundError as exc:
                    outcome["error"] = (404, str(exc))
                except Exception as exc:  # noqa: BLE001 - reported as an event, never raised
                    outcome["error"] = (500, str(exc))
                finally:
                    counters.put(None)

            worker = threading.Thread(target=work, name="laya-index", daemon=True)
            worker.start()

            while True:
                item = counters.get()
                if item is None:
                    break
                yield {"type": "progress", **item}

            report = outcome.get("report")
            if report is not None:
                payload = report.as_dict()
                if report.documents == 0:
                    payload["note"] = "no readable text documents were found under the given paths"
                yield {"type": "done", "report": payload}
                return
            status, message = outcome.get("error", (500, "indexing failed"))
            yield {"type": "error", "status": status, "message": message}

        if wants_sse(accept):
            return StreamingResponse(
                sse(events()), media_type=SSE_MEDIA_TYPE, headers=STREAM_HEADERS
            )
        return StreamingResponse(
            ndjson(events()), media_type=NDJSON_MEDIA_TYPE, headers=STREAM_HEADERS
        )

    def reset_index(http_request: Request, idempotency_key: str | None, principal: Principal) -> Any:
        require_write(principal)
        engine = get_engine()
        body = {"status": "ok", "index_dir": str(config.index_dir)}
        scope: str | None = None
        digest: str | None = None
        if idempotency_key:
            scope = idempotency_scope(http_request, principal)
            digest = fingerprint({})
            stored = idempotency.begin(scope, idempotency_key, digest)
            if stored is not None:
                return JSONResponse(
                    stored.body,
                    status_code=stored.status,
                    headers={"Idempotency-Replayed": "true"},
                )
        try:
            engine.reset()
        except BuildInProgressError as exc:
            if idempotency_key and scope is not None:
                idempotency.abandon(scope, idempotency_key)
            raise APIProblem(409, CODE_BUILD_IN_PROGRESS, str(exc)) from exc
        if idempotency_key and scope is not None and digest is not None:
            idempotency.complete(scope, idempotency_key, digest, 200, body)
        return body

    @app.delete(
        "/v1/index",
        tags=["indexing"],
        operation_id="delete_index",
        response_model=ResetResponse,
        summary="Delete the index",
        responses=_WRITE_ERRORS,
    )
    def delete_index(
        http_request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        principal: Principal = Depends(authorise),
    ) -> Any:
        """Delete the index. The standard verb for a destructive operation."""
        return reset_index(http_request, idempotency_key, principal)

    @app.post(
        "/v1/reset",
        tags=["indexing"],
        operation_id="reset_index",
        response_model=ResetResponse,
        summary="Delete the index (deprecated: use DELETE /v1/index)",
        responses=_WRITE_ERRORS,
    )
    def reset(
        response: Response,
        http_request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        principal: Principal = Depends(authorise),
    ) -> Any:
        """Delete the index. Deprecated in favour of ``DELETE /v1/index``."""
        response.headers["Deprecation"] = f"@{HEALTH_DEPRECATED_AT}"
        response.headers["Link"] = '</v1/index>; rel="successor-version"'
        return reset_index(http_request, idempotency_key, principal)

    @app.delete(
        "/v1/documents",
        tags=["indexing"],
        operation_id="delete_documents",
        response_model=DeleteDocumentsResponse,
        summary="Remove documents from the index",
        responses=_WRITE_ERRORS,
    )
    def delete_documents(
        request: DeleteDocumentsRequest,
        principal: Principal = Depends(authorise),
    ) -> Any:
        """Remove documents by path or doc id, without rebuilding the whole index.

        Removal rebuilds the lexical channel from the surviving chunks, so the two
        channels cannot disagree. The on-disk backend is append-only and answers
        ``409 index_not_updatable``; rebuild it instead. A request naming nothing
        removes nothing.
        """
        require_write(principal)
        engine = get_engine()
        try:
            removed = engine.remove_documents(request.paths, request.doc_ids)
        except ValueError as exc:
            raise APIProblem(409, CODE_INDEX_NOT_UPDATABLE, str(exc)) from exc
        return {
            "removed_chunks": removed,
            "total_chunks": engine.stats()["index"]["chunks"],
        }

    # -- retrieval -------------------------------------------------------
    @app.post(
        "/v1/query",
        tags=["retrieval"],
        operation_id="create_query",
        response_model=QueryResponse,
        summary="Return calibrated evidence for a query",
        responses=_READ_ERRORS,
    )
    def query(request: QueryRequest, principal: Principal = Depends(read_access)) -> dict[str, Any]:
        """Return Laya-ranked, Laya-gated evidence for a query.

        ``answerable`` is Laya's calibrated verdict on whether the returned passages
        actually support an answer. Callers should not synthesise an answer when it
        is false.
        """
        engine = get_engine()
        try:
            result = engine.query(
                request.query,
                top_k=request.top_k,
                retrieve_k=request.retrieve_k,
                relevance_threshold=request.relevance_threshold,
                answerability_threshold=request.answerability_threshold,
                max_passage_chars=request.max_passage_chars,
                tenant=tenant_for(principal, request.tenant),
            )
        except BuildInProgressError as exc:
            raise APIProblem(409, CODE_BUILD_IN_PROGRESS, str(exc)) from exc
        except IndexNotBuiltError as exc:
            raise APIProblem(409, "index_not_built", str(exc)) from exc
        except ModelUnavailableError as exc:
            raise APIProblem(503, CODE_MODEL_UNAVAILABLE, str(exc)) from exc
        except ValueError as exc:
            raise APIProblem(400, "invalid_request", str(exc)) from exc
        return result.as_dict()

    @app.post(
        "/v1/compose",
        tags=["retrieval"],
        operation_id="create_compose",
        summary="Write prose from evidence, streaming",
        responses=_STREAM_RESPONSES,
    )
    def compose(
        request: ComposeRequest,
        accept: str | None = Header(default=None),
        _: Principal = Depends(read_access),
    ) -> StreamingResponse:
        """Write prose from evidence, streaming NDJSON or SSE events.

        The only route that returns generated text, and therefore the only one whose
        output a model can have invented. Events are ``delta`` while the model writes,
        then one terminal ``done`` or ``error`` — and an ``error`` may arrive after
        some ``delta`` events if the connection drops mid-answer, which is the
        caller's signal that the text it holds is incomplete. Send
        ``Accept: text/event-stream`` for SSE framing; NDJSON is the default.
        """
        engine = get_engine()
        target, reason = resolve_target(engine.config)
        if target is None:
            raise APIProblem(503, CODE_COMPOSE_DISABLED, reason)

        pairs = [(passage.path or "", passage.text) for passage in request.passages]

        def events() -> Iterator[dict[str, Any]]:
            yield from engine.composer.stream(request.query, pairs, max_tokens=request.max_tokens)

        if wants_sse(accept):
            return StreamingResponse(
                sse(events()), media_type=SSE_MEDIA_TYPE, headers=STREAM_HEADERS
            )
        return StreamingResponse(
            ndjson(events()), media_type=NDJSON_MEDIA_TYPE, headers=STREAM_HEADERS
        )

    # -- verification ----------------------------------------------------
    @app.post(
        "/v1/verify",
        tags=["verification"],
        operation_id="create_verify",
        response_model=VerifyResponse,
        summary="Check an answer's citations against the evidence",
        responses=_READ_ERRORS,
    )
    def verify(request: VerifyRequest, _: Principal = Depends(read_access)) -> dict[str, Any]:
        """Check an answer's citations and quoted spans against the evidence.

        Deterministic and model-free: it reports whether each claim cites in-range
        evidence and whether each quoted span actually appears in the passages. It
        makes no claim about whether the answer is *true*, and nothing consuming this
        should present it as if it did.

        A separate ``content`` block adds *advisory* checks — numbers, dates and units
        that do not appear in the cited evidence, and a possible negation flip. Those
        never change ``clean``; they are flagged for a reader to weigh.

        The passages are grouped by document with the same rule the composer uses, so
        the ``[n]`` in a composed answer means the same evidence here as it does there.
        """
        grouped = group_passages_by_document(
            (passage.path or "", passage.text) for passage in request.passages
        )
        return verify_citations(
            request.answer,
            [text for _path, text in grouped],
            query=request.query or "",
        )

    @app.post(
        "/v1/adjudicate",
        tags=["verification"],
        operation_id="create_adjudicate",
        response_model=AdjudicateResponse,
        summary="Ask Laya whether the corpus supports each claim",
        responses=_READ_ERRORS,
    )
    def adjudicate(
        request: AdjudicateRequest,
        principal: Principal = Depends(read_access),
    ) -> dict[str, Any]:
        """Ask Laya whether the corpus supports each claim in an answer.

        The counterpart to ``/v1/verify``: that endpoint checks citations
        deterministically, this one asks a calibrated decision engine whether the
        claims hold up against the whole corpus. It reuses the already-measured
        answerability question rather than a new claim-support wording, and reports
        which question it used.

        Claims are split server-side from the same answer text ``/v1/verify`` takes, so
        the two reports describe exactly the same sentences.
        """
        engine = get_engine()
        try:
            report = engine.adjudicate_claims(
                split_claims(request.answer),
                retrieve_k=request.retrieve_k,
                relevance_threshold=request.relevance_threshold,
                answerability_threshold=request.answerability_threshold,
                tenant=tenant_for(principal, request.tenant),
            )
        except BuildInProgressError as exc:
            raise APIProblem(409, CODE_BUILD_IN_PROGRESS, str(exc)) from exc
        except IndexNotBuiltError as exc:
            raise APIProblem(409, "index_not_built", str(exc)) from exc
        except ModelUnavailableError as exc:
            raise APIProblem(503, CODE_MODEL_UNAVAILABLE, str(exc)) from exc
        except DecisionError as exc:
            raise APIProblem(502, "decision_failed", str(exc)) from exc
        return report

    # -- OpenAPI ---------------------------------------------------------
    def _servers() -> list[dict[str, str]]:
        display = "127.0.0.1" if config.host in {"0.0.0.0", "::"} else config.host or "127.0.0.1"
        return [
            {
                "url": f"http://{display}:{config.port}",
                "description": "This service",
            }
        ]

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            tags=OPENAPI_TAGS,
            servers=_servers(),
        )
        components = schema.setdefault("components", {})
        components.setdefault("securitySchemes", {})["bearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "description": (
                "Static bearer token. Set LAYA_RAG_API_KEYS for per-token tenants and scopes, "
                "or LAYA_RAG_API_KEY for a single admin token."
            ),
        }
        # Error responses reference this schema by raw $ref (their media type is
        # application/problem+json, which FastAPI's `model=` shortcut cannot express),
        # so make sure it is present.
        schemas = components.setdefault("schemas", {})
        if "Problem" not in schemas:
            problem_schema = Problem.model_json_schema(
                ref_template="#/components/schemas/{model}"
            )
            schemas.update(problem_schema.pop("$defs", {}))
            schemas["Problem"] = problem_schema
        public = {"/healthz", "/readyz"}
        for path, operations in schema.get("paths", {}).items():
            for method, operation in operations.items():
                if method not in {"get", "post", "put", "patch", "delete"}:
                    continue
                if path in public:
                    operation["security"] = []
                else:
                    operation.setdefault("security", [{"bearerAuth": []}])
                # A stream route has no response_model, so FastAPI leaves an untyped
                # application/json 200 alongside the media types we declared. Drop any
                # content entry with no schema: an untyped response is not a contract.
                for response in operation.get("responses", {}).values():
                    content = response.get("content")
                    if not content:
                        continue
                    for media_type in [name for name, spec in content.items() if not spec.get("schema")]:
                        content.pop(media_type)
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi  # type: ignore[method-assign]

    # The GUI ships as plain static files — no build step, no CDN, no bundler — so the
    # service stays self-contained and works offline. Mounted last on purpose: /health,
    # /v1/* and FastAPI's generated /docs are registered first and therefore win.
    web_dir = _web_dir()
    if web_dir is not None:
        # Served through a route rather than the mount so the asset URLs can be stamped
        # with their mtimes; that is what stops an old app.js being paired with new
        # markup, which otherwise shows a previous layout indefinitely.
        @app.get("/", include_in_schema=False)
        def gui_index() -> HTMLResponse:
            return HTMLResponse(_index_html(), headers={"Cache-Control": "no-cache"})

        app.mount("/", _GuiFiles(directory=web_dir, html=True), name="web")
    else:  # pragma: no cover - only if installed without package data
        log.warning("GUI assets are not installed; serving the JSON API only")

    return app


def main() -> None:
    """Console entry point: serve the engine and its GUI with uvicorn."""
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = EngineConfig()
    config.ensure_dirs()
    app = create_app(config)
    host = "127.0.0.1" if config.host in {"0.0.0.0", "::"} else config.host
    if _web_dir() is not None:
        log.info("GUI at http://%s:%d/", host, config.port)
    log.info("API docs at http://%s:%d/docs", host, config.port)
    log.info("probes at http://%s:%d/healthz and /readyz", host, config.port)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
