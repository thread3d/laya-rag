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

import logging
from contextlib import asynccontextmanager
from typing import Any, Iterable

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .config import EngineConfig
from .engine import IndexNotBuiltError, RagEngine
from .model import ModelUnavailableError

log = logging.getLogger("laya_dsh_rag.api")


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
        title="Laya RAG engine",
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

    @app.post("/v1/reset")
    def reset(_: None = Depends(authorise)) -> dict[str, Any]:
        """Delete the index."""
        engine = get_engine()
        engine.reset()
        return {"status": "ok", "index_dir": str(config.index_dir)}

    return app


def main() -> None:
    """Console entry point: serve the engine with uvicorn."""
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = EngineConfig()
    config.ensure_dirs()
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
