"""HTTP service exposing the RAG engine.

DeepSeek Harness plugins run in Node and this engine runs in Python, so the two
halves meet over HTTP. The route names mirror what the engine actually does —
``/v1/index`` and ``/v1/query`` — rather than pretending to be an OpenAI-compatible
endpoint, because Laya is not a completion model and a shim would misrepresent it.

Handlers are synchronous on purpose: Laya's forward passes are CPU-bound, and
FastAPI runs ``def`` handlers in its worker threadpool, which keeps the event loop
free without any hand-rolled thread management.
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .compose import resolve_target
from .config import EngineConfig
from .engine import IndexNotBuiltError, RagEngine
from .model import ModelUnavailableError
from .rerank import DecisionError
from .verify import split_claims, verify_citations

log = logging.getLogger("laya_dsh_rag.api")

WEB_DIR = Path(__file__).resolve().parent / "web"


def _web_dir() -> Path | None:
    """The bundled GUI directory, or ``None`` when its assets are not installed."""
    return WEB_DIR if (WEB_DIR / "index.html").is_file() else None


class IndexRequest(BaseModel):
    """A request to index one or more files or directories."""

    paths: list[str] = Field(..., min_length=1, description="Files or directories to index")
    rebuild: bool = Field(False, description="Discard the existing index first")
    extensions: list[str] | None = Field(None, description="Override the indexable file extensions")


class QueryRequest(BaseModel):
    """A request for evidence supporting a query."""

    query: str = Field(..., min_length=1, description="The question to find evidence for")
    top_k: int | None = Field(None, ge=1, le=50, description="Passages to return after re-ranking")
    retrieve_k: int | None = Field(None, ge=1, le=500, description="Dense candidates to consider")
    relevance_threshold: float | None = Field(None, ge=0.0, le=1.0)
    answerability_threshold: float | None = Field(None, ge=0.0, le=1.0)
    max_passage_chars: int | None = Field(2000, ge=100, le=20000)


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

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if config.warmup:
            try:
                get_engine().model.ensure_loaded()
                log.info("Laya checkpoint %r warmed up", config.checkpoint)
            except ModelUnavailableError as exc:
                # A missing checkpoint must not stop the service: /health reports it
                # and every real request fails with the same actionable message.
                log.warning("warmup skipped: %s", exc)
        yield
        holder.clear()

    app = FastAPI(
        title="Laya DSH RAG engine",
        description="Retrieval-augmented generation over Laya embeddings and decisions.",
        version="0.1.0",
        lifespan=lifespan,
    )

    def authorise(authorization: str | None = Header(default=None)) -> None:
        if not config.api_key:
            return
        expected = f"Bearer {config.api_key}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    @app.get("/health")
    def health(_: None = Depends(authorise)) -> dict[str, Any]:
        """Liveness plus whether the checkpoint and index are actually usable."""
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
        return payload

    @app.get("/v1/stats")
    def stats(_: None = Depends(authorise)) -> dict[str, Any]:
        """Configuration, model and corpus statistics."""
        return get_engine().stats()

    @app.get("/v1/documents")
    def documents(_: None = Depends(authorise)) -> dict[str, Any]:
        """Every indexed document, with its chunk and character counts.

        Backs the corpus browser. Returns an empty listing rather than an error when
        no index exists, so a dashboard can render its empty state from one call.
        """
        listing = get_engine().list_documents()
        return {"total": len(listing), "documents": listing}

    @app.get("/v1/chunks")
    def chunks(
        path: str | None = None,
        _: None = Depends(authorise),
    ) -> dict[str, Any]:
        """Indexed chunks in document order, optionally narrowed to one path."""
        records = get_engine().list_chunks(path)
        return {"total": len(records), "path": path, "chunks": records}

    @app.post("/v1/index")
    def index_documents(request: IndexRequest, _: None = Depends(authorise)) -> dict[str, Any]:
        """Index the requested paths and return what was read, skipped and embedded."""
        engine = get_engine()
        extensions: Iterable[str] | None = request.extensions
        try:
            report = engine.index_paths(
                request.paths,
                rebuild=request.rebuild,
                **({"extensions": extensions} if extensions else {}),
            )
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        payload = report.as_dict()
        if report.documents == 0:
            payload["note"] = "no readable text documents were found under the given paths"
        return payload

    @app.post("/v1/query")
    def query(request: QueryRequest, _: None = Depends(authorise)) -> dict[str, Any]:
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
            )
        except IndexNotBuiltError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.as_dict()

    @app.post("/v1/compose")
    def compose(request: ComposeRequest, _: None = Depends(authorise)) -> StreamingResponse:
        """Write prose from evidence, streaming newline-delimited JSON events.

        The only route that returns generated text, and therefore the only one whose
        output a model can have invented. Events are ``delta`` while the model writes,
        then one terminal ``done`` or ``error`` — and an ``error`` may arrive after
        some ``delta`` events if the connection drops mid-answer, which is the
        caller's signal that the text it holds is incomplete.
        """
        engine = get_engine()
        target, reason = resolve_target(engine.config)
        if target is None:
            raise HTTPException(status_code=503, detail=reason)

        pairs = [(passage.path or "", passage.text) for passage in request.passages]

        def events() -> Iterator[str]:
            for event in engine.composer.stream(request.query, pairs, max_tokens=request.max_tokens):
                yield json.dumps(event, ensure_ascii=False) + "\n"

        return StreamingResponse(
            events(),
            media_type="application/x-ndjson",
            # Proxies and browsers must not buffer a stream that can take a minute.
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/verify")
    def verify(request: VerifyRequest, _: None = Depends(authorise)) -> dict[str, Any]:
        """Check an answer's citations and quoted spans against the evidence.

        Deterministic and model-free: it reports whether each claim cites in-range
        evidence and whether each quoted span actually appears in the passages. It
        makes no claim about whether the answer is *true*, and nothing consuming this
        should present it as if it did.
        """
        return verify_citations(
            request.answer,
            [passage.text for passage in request.passages],
            query=request.query or "",
        )

    @app.post("/v1/adjudicate")
    def adjudicate(request: AdjudicateRequest, _: None = Depends(authorise)) -> dict[str, Any]:
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
            )
        except IndexNotBuiltError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except DecisionError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return report

    @app.post("/v1/reset")
    def reset(_: None = Depends(authorise)) -> dict[str, Any]:
        """Delete the index."""
        engine = get_engine()
        engine.reset()
        return {"status": "ok", "index_dir": str(config.index_dir)}

    # The GUI ships as plain static files — no build step, no CDN, no bundler — so the
    # service stays self-contained and works offline. Mounted last on purpose: /health,
    # /v1/* and FastAPI's generated /docs are registered first and therefore win.
    web_dir = _web_dir()
    if web_dir is not None:
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
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
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
