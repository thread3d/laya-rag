"""An MCP server over the Laya RAG engine.

Exposes the engine's calibrated evidence as tools an agent can call, and nothing it
cannot honestly promise. The important constraint is in `search_evidence`'s description:
when `answerable` is false the retrieved passages do not support an answer, and the
agent should abstain rather than write one anyway. `compose_answer` is the only tool that
returns prose, and it is labelled `unverified` because it is the one step that is not
Laya's.

Two transports:

* ``stdio`` (the default) for a desktop/agent host that launches the process directly.
* ``streamable-http`` for a client that connects over the network. Beyond loopback the
  server refuses to start without a bearer token, because it reads the corpus.

Two backends:

* ``local`` (the default) constructs the engine in-process, exactly as the CLI does, so
  no running service is needed and it works offline.
* ``http`` proxies to a running ``laya-rag-serve`` instead, so a second 2.3 GB checkpoint
  is not loaded into memory.

The MCP SDK is an optional extra (``pip install -e '.[mcp]'``); importing this module
without it is what the actionable error below reports.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Sequence
from urllib import error as urlerror, request as urlrequest
from urllib.parse import urlencode

from pydantic import BaseModel, Field

from .config import EngineConfig
from .engine import BuildInProgressError, IndexNotBuiltError, RagEngine
from .evidence import group_passages_by_document
from .integrity import audit
from .model import ModelUnavailableError
from .rerank import DecisionError
from .schemas import (
    AdjudicateResponse,
    ChunkRecord,
    DeleteDocumentsResponse,
    DocumentRecord,
    IndexReportResponse,
    ResetResponse,
    VerifyResponse,
)

log = logging.getLogger("laya_rag.mcp")

try:  # pragma: no cover - exercised by the import guard, not by the suite
    from mcp.server import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.server.transport_security import TransportSecuritySettings
    from mcp.types import ToolAnnotations

    MCP_AVAILABLE = True
except ImportError as exc:  # pragma: no cover
    MCPServer = None  # type: ignore[assignment]
    TransportSecuritySettings = None  # type: ignore[assignment]
    ToolAnnotations = None  # type: ignore[assignment]
    MCP_AVAILABLE = False
    MCP_IMPORT_ERROR: Exception | None = exc
else:
    MCP_IMPORT_ERROR = None


SERVER_NAME = "laya-rag"
SERVER_INSTRUCTIONS = (
    "Retrieval over a local document corpus, powered by the Laya decision engine. "
    "`search_evidence` returns calibrated evidence, not an answer: it reports whether the "
    "corpus supports answering the question and returns the passages that do. If "
    "`answerable` is false, do not answer from those passages. `compose_answer` and "
    "`index_documents`/`reset_index` are optional and off unless enabled."
)


# -- MCP-shaped result models ------------------------------------------------

class EvidencePassage(BaseModel):
    """One ranked piece of evidence."""

    path: str = Field(..., description="Source document path")
    text: str = Field(..., description="Passage text")
    relevance: float = Field(..., description="Laya's calibrated relevance probability")
    channels: list[str] = Field(default_factory=list, description="Recall channels that found it")
    chunk_id: str


class EvidenceBundle(BaseModel):
    """The result of a corpus search."""

    query: str
    answerable: bool = Field(
        ..., description="Laya's verdict on whether these passages support an answer"
    )
    answerability: float | None = Field(None, description="Calibrated P(answerable)")
    answerability_measured: bool = Field(
        ..., description="False when the relevance gate emptied the set before the decision"
    )
    passages: list[EvidencePassage]
    considered: int
    kept: int
    note: str | None = None


class DocumentPage(BaseModel):
    total: int
    documents: list[DocumentRecord]
    has_more: bool = False
    next_cursor: str | None = None


class DocumentText(BaseModel):
    path: str
    offset: int
    limit: int
    total: int
    chunks: list[ChunkRecord]


class CorpusStats(BaseModel):
    index_present: bool
    chunks: int
    documents: int
    paths: int
    checkpoint: str
    index_backend: str
    lexical_present: bool
    compose_enabled: bool


class IntegritySummary(BaseModel):
    chunks: int
    paths: int
    clean: bool
    index_present: bool
    stale_paths: int
    stale_chunks: int
    duplicate_clusters: int
    redundant_chunks: int


class ComposedAnswer(BaseModel):
    answer: str = Field(..., description="Generated prose; not verified by Laya")
    unverified: bool = True
    model: str | None = None
    finish_reason: str | None = None
    note: str | None = None


class PassageInput(BaseModel):
    text: str = Field(..., description="Passage text")
    path: str | None = Field(None, description="Source path, cited in an answer")


# -- backends ----------------------------------------------------------------

class ToolFailure(RuntimeError):
    """A failure a caller can act on; surfaced to the agent as a tool error."""


class LocalBackend:
    """Runs the engine in this process, like the CLI does."""

    def __init__(self, config: EngineConfig, engine: RagEngine | None = None) -> None:
        self.config = config
        self._engine = engine

    def engine(self) -> RagEngine:
        if self._engine is None:
            self._engine = RagEngine(self.config)
        return self._engine

    def search(self, query: str, top_k: int, tenant: str | None) -> dict[str, Any]:
        return self.engine().query(query, top_k=top_k, tenant=tenant).as_dict()

    def documents(self, limit: int, cursor: str | None) -> dict[str, Any]:
        from .paging import paginate

        listing = self.engine().list_documents()
        page, next_cursor, has_more, total = paginate(
            listing, lambda record: (record["path"],), limit=limit, cursor=cursor
        )
        return {
            "total": total,
            "documents": page,
            "has_more": has_more,
            "next_cursor": next_cursor,
        }

    def read_document(self, path: str, offset: int, limit: int) -> dict[str, Any]:
        records = self.engine().list_chunks(path)
        return {
            "path": path,
            "offset": offset,
            "limit": limit,
            "total": len(records),
            "chunks": records[offset : offset + limit],
        }

    def stats(self) -> dict[str, Any]:
        return self.engine().stats()

    def integrity(self, near_duplicates: bool) -> dict[str, Any]:
        engine = self.engine()
        if not engine.has_index():
            return {
                "chunks": 0,
                "paths": 0,
                "clean": True,
                "index_present": False,
                "stale_paths": {"count": 0, "stale_chunks": 0},
                "duplicate_content": {"clusters": 0, "redundant_chunks": 0},
            }
        report = audit(engine.index.chunks, near_duplicates=near_duplicates)
        report["index_present"] = True
        return report

    def verify(
        self, answer: str, passages: Sequence[PassageInput], query: str | None
    ) -> dict[str, Any]:
        from .verify import verify_citations

        grouped = group_passages_by_document(
            (passage.path or "", passage.text) for passage in passages
        )
        return verify_citations(answer, [text for _path, text in grouped], query=query or "")

    def adjudicate(self, answer: str, tenant: str | None) -> dict[str, Any]:
        from .verify import split_claims

        return self.engine().adjudicate_claims(split_claims(answer), tenant=tenant)

    def compose(
        self, query: str, passages: Sequence[PassageInput], max_tokens: int | None
    ) -> dict[str, Any]:
        pairs = [(passage.path or "", passage.text) for passage in passages]
        chunks: list[str] = []
        model: str | None = None
        finish_reason: str | None = None
        error: str | None = None
        for event in self.engine().composer.stream(query, pairs, max_tokens=max_tokens):
            kind = event.get("type")
            if kind == "delta":
                chunks.append(str(event.get("text", "")))
            elif kind == "done":
                model = event.get("model")
                finish_reason = event.get("finish_reason")
            elif kind == "error":
                error = str(event.get("message", "composition failed"))
        return {
            "answer": "".join(chunks),
            "model": model,
            "finish_reason": finish_reason,
            "note": error,
        }

    def index(self, paths: Sequence[str], rebuild: bool) -> dict[str, Any]:
        report = self.engine().index_paths(list(paths), rebuild=rebuild)
        payload = report.as_dict()
        if report.documents == 0:
            payload["note"] = "no readable text documents were found under the given paths"
        return payload

    def reset(self) -> dict[str, Any]:
        engine = self.engine()
        engine.reset()
        return {"status": "ok", "index_dir": str(self.config.index_dir)}

    def delete(self, paths: Sequence[str], doc_ids: Sequence[str]) -> dict[str, Any]:
        engine = self.engine()
        removed = engine.remove_documents(list(paths), list(doc_ids))
        return {"removed_chunks": removed, "total_chunks": engine.stats()["index"]["chunks"]}


class HttpBackend:
    """Proxies to a running ``laya-rag-serve`` over HTTP."""

    def __init__(self, base_url: str, api_key: str | None, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self.base_url + path
        if params:
            url += "?" + urlencode({k: v for k, v in params.items() if v is not None})
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers: dict[str, str] = {}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urlrequest.Request(url, data=data, headers=headers, method=method)
        try:
            with urlrequest.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            detail = ""
            try:
                body = json.loads(exc.read().decode("utf-8", "replace"))
                detail = str(body.get("detail") or body.get("title") or "")
            except Exception:  # noqa: BLE001 - the body is best-effort context
                detail = ""
            raise ToolFailure(
                f"the service returned HTTP {exc.code}" + (f": {detail}" if detail else "")
            ) from exc
        except urlerror.URLError as exc:
            raise ToolFailure(f"could not reach the service at {self.base_url}: {exc}") from exc

    def search(self, query: str, top_k: int, tenant: str | None) -> dict[str, Any]:
        return self._request(
            "POST", "/v1/query", {"query": query, "top_k": top_k, "tenant": tenant}
        )

    def documents(self, limit: int, cursor: str | None) -> dict[str, Any]:
        return self._request("GET", "/v1/documents", params={"limit": limit, "cursor": cursor})

    def read_document(self, path: str, offset: int, limit: int) -> dict[str, Any]:
        body = self._request("GET", "/v1/chunks", params={"path": path})
        records = body.get("chunks", [])
        return {
            "path": path,
            "offset": offset,
            "limit": limit,
            "total": len(records),
            "chunks": records[offset : offset + limit],
        }

    def stats(self) -> dict[str, Any]:
        return self._request("GET", "/v1/stats")

    def integrity(self, near_duplicates: bool) -> dict[str, Any]:
        return self._request(
            "GET", "/v1/integrity", params={"near_duplicates": str(near_duplicates).lower()}
        )

    def verify(
        self, answer: str, passages: Sequence[PassageInput], query: str | None
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/verify",
            {
                "answer": answer,
                "passages": [passage.model_dump() for passage in passages],
                "query": query,
            },
        )

    def adjudicate(self, answer: str, tenant: str | None) -> dict[str, Any]:
        return self._request("POST", "/v1/adjudicate", {"answer": answer, "tenant": tenant})

    def compose(
        self, query: str, passages: Sequence[PassageInput], max_tokens: int | None
    ) -> dict[str, Any]:
        # The compose route streams NDJSON; aggregate the deltas here.
        url = self.base_url + "/v1/compose"
        payload = {
            "query": query,
            "passages": [passage.model_dump() for passage in passages],
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json", "Accept": "application/x-ndjson"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urlrequest.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        chunks: list[str] = []
        model = finish_reason = None
        error = None
        try:
            with urlrequest.urlopen(request, timeout=self.timeout) as response:
                for line in response:
                    if not line.strip():
                        continue
                    event = json.loads(line.decode("utf-8"))
                    if event.get("type") == "delta":
                        chunks.append(str(event.get("text", "")))
                    elif event.get("type") == "done":
                        model = event.get("model")
                        finish_reason = event.get("finish_reason")
                    elif event.get("type") == "error":
                        error = str(event.get("message", "composition failed"))
        except urlerror.HTTPError as exc:
            raise ToolFailure(f"the service returned HTTP {exc.code}") from exc
        except urlerror.URLError as exc:
            raise ToolFailure(f"could not reach the service at {self.base_url}: {exc}") from exc
        return {
            "answer": "".join(chunks),
            "model": model,
            "finish_reason": finish_reason,
            "note": error,
        }

    def index(self, paths: Sequence[str], rebuild: bool) -> dict[str, Any]:
        return self._request("POST", "/v1/index", {"paths": list(paths), "rebuild": rebuild})

    def reset(self) -> dict[str, Any]:
        return self._request("DELETE", "/v1/index")

    def delete(self, paths: Sequence[str], doc_ids: Sequence[str]) -> dict[str, Any]:
        return self._request(
            "DELETE", "/v1/documents", {"paths": list(paths), "doc_ids": list(doc_ids)}
        )


def make_backend(config: EngineConfig) -> LocalBackend | HttpBackend:
    """The backend named by ``LAYA_RAG_MCP_BACKEND`` (default ``local``)."""
    kind = (os.environ.get("LAYA_RAG_MCP_BACKEND") or "local").strip().lower()
    if kind == "http":
        url = (os.environ.get("LAYA_RAG_MCP_URL") or "http://127.0.0.1:8077").strip()
        key = (
            os.environ.get("LAYA_RAG_MCP_API_KEY")
            or os.environ.get("LAYA_RAG_API_KEY")
            or config.api_key
        )
        return HttpBackend(url, key)
    if kind != "local":
        raise ValueError(f"LAYA_RAG_MCP_BACKEND must be 'local' or 'http', got {kind!r}")
    return LocalBackend(config)


# -- helpers -----------------------------------------------------------------

_RECOVERABLE = (
    IndexNotBuiltError,
    BuildInProgressError,
    ModelUnavailableError,
    DecisionError,
    ToolFailure,
    ValueError,
)


def _call(operation: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a backend call, turning an actionable failure into a tool error."""
    try:
        return fn(*args, **kwargs)
    except _RECOVERABLE as exc:
        raise ToolError(f"{operation}: {exc}") from exc


def _trim_stats(stats: dict[str, Any]) -> CorpusStats:
    index = stats.get("index") or {}
    compose = stats.get("compose") or {}
    return CorpusStats(
        index_present=bool(stats.get("index_present")),
        chunks=int(index.get("chunks", 0) or 0),
        documents=int(index.get("documents", 0) or 0),
        paths=int(index.get("paths", 0) or 0),
        checkpoint=str(stats.get("config", {}).get("checkpoint") or ""),
        index_backend=str(stats.get("index_backend") or ""),
        lexical_present=bool(stats.get("lexical_present")),
        compose_enabled=bool(compose.get("available", compose.get("configured", False))),
    )


def _trim_integrity(report: dict[str, Any]) -> IntegritySummary:
    stale = report.get("stale_paths") or {}
    duplicates = report.get("duplicate_content") or {}
    return IntegritySummary(
        chunks=int(report.get("chunks", 0) or 0),
        paths=int(report.get("paths", 0) or 0),
        clean=bool(report.get("clean", True)),
        index_present=bool(report.get("index_present", True)),
        stale_paths=int(stale.get("count", 0) or 0),
        stale_chunks=int(stale.get("stale_chunks", 0) or 0),
        duplicate_clusters=int(duplicates.get("clusters", 0) or 0),
        redundant_chunks=int(duplicates.get("redundant_chunks", 0) or 0),
    )


def _evidence(body: dict[str, Any]) -> EvidenceBundle:
    return EvidenceBundle(
        query=str(body.get("query", "")),
        answerable=bool(body.get("answerable")),
        answerability=body.get("answerability"),
        answerability_measured=bool(body.get("answerability_measured")),
        passages=[
            EvidencePassage(
                path=str(passage.get("path", "")),
                text=str(passage.get("text", "")),
                relevance=float(passage.get("relevance", 0.0)),
                channels=list(passage.get("channels") or []),
                chunk_id=str(passage.get("chunk_id", "")),
            )
            for passage in body.get("passages", [])
        ],
        considered=int(body.get("considered", 0) or 0),
        kept=int(body.get("kept", 0) or 0),
        note=body.get("note"),
    )


# -- server ------------------------------------------------------------------

def allow_writes() -> bool:
    """Whether the mutating MCP tools are registered (``LAYA_RAG_MCP_ALLOW_WRITES``)."""
    return (os.environ.get("LAYA_RAG_MCP_ALLOW_WRITES") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def build_server(
    config: EngineConfig | None = None,
    backend: LocalBackend | HttpBackend | None = None,
    *,
    writes: bool | None = None,
) -> Any:
    """Build the MCP server: tools, resources and prompts.

    The write tools (``index_documents``, ``reset_index``) are omitted unless
    ``writes`` is true or ``LAYA_RAG_MCP_ALLOW_WRITES`` is set.
    """
    if not MCP_AVAILABLE:
        raise RuntimeError(
            "the MCP SDK is not installed; run `pip install -e '.[mcp]'` to enable the "
            "MCP server"
        )
    config = config or EngineConfig()
    backend = backend or make_backend(config)
    enable_writes = allow_writes() if writes is None else writes

    server = MCPServer(
        name=SERVER_NAME,
        title="Laya RAG",
        description="Calibrated evidence retrieval over a local corpus.",
        instructions=SERVER_INSTRUCTIONS,
    )

    read_only = ToolAnnotations(read_only_hint=True, open_world_hint=False)

    @server.tool(title="Search corpus", annotations=read_only)
    def search_evidence(query: str, top_k: int = 5, tenant: str | None = None) -> EvidenceBundle:
        """Search the indexed corpus for passages that support a question.

        Returns Laya's calibrated `answerable` verdict and the ranked passages with their
        source paths and relevance. This tool never writes prose. If `answerable` is
        false, the passages do not support an answer: say the corpus does not cover the
        question instead of answering from them.
        """
        if top_k < 1 or top_k > 50:
            raise ToolError("search_evidence: top_k must be between 1 and 50")
        body = _call("search_evidence", backend.search, query, top_k, tenant)
        return _evidence(body)

    @server.tool(title="List documents", annotations=read_only)
    def list_documents(limit: int = 20, cursor: str | None = None) -> DocumentPage:
        """List the documents in the index, one page at a time.

        Returns each document's path, id, chunk count and indexed character count.
        Pass `next_cursor` from the previous page to continue; a null cursor is the end.
        """
        if limit < 1 or limit > 100:
            raise ToolError("list_documents: limit must be between 1 and 100")
        body = _call("list_documents", backend.documents, limit, cursor)
        return DocumentPage(
            total=int(body.get("total", 0) or 0),
            documents=[DocumentRecord(**record) for record in body.get("documents", [])],
            has_more=bool(body.get("has_more")),
            next_cursor=body.get("next_cursor"),
        )

    @server.tool(title="Read document", annotations=read_only)
    def read_document(path: str, offset: int = 0, limit: int = 20) -> DocumentText:
        """Read the indexed chunks of one document, in order.

        Use this to inspect what the engine actually indexed for a path returned by
        `list_documents` or `search_evidence`. Paginate with `offset`/`limit`.
        """
        if offset < 0:
            raise ToolError("read_document: offset must be >= 0")
        if limit < 1 or limit > 100:
            raise ToolError("read_document: limit must be between 1 and 100")
        body = _call("read_document", backend.read_document, path, offset, limit)
        return DocumentText(
            path=str(body.get("path", path)),
            offset=int(body.get("offset", offset)),
            limit=int(body.get("limit", limit)),
            total=int(body.get("total", 0) or 0),
            chunks=[ChunkRecord(**record) for record in body.get("chunks", [])],
        )

    @server.tool(title="Corpus stats", annotations=read_only)
    def corpus_stats() -> CorpusStats:
        """Report what the index contains and which channels are active.

        A compact status: whether an index exists, how many chunks and documents it
        holds, the checkpoint and backend, and whether composition is enabled.
        """
        stats = _call("corpus_stats", backend.stats)
        return _trim_stats(stats)

    @server.tool(title="Check index integrity", annotations=read_only)
    def check_index_integrity(near_duplicates: bool = False) -> IntegritySummary:
        """Report index drift: deleted documents and duplicate chunks.

        `clean` is false when the index holds chunks for files that no longer exist, or
        duplicate content. The near-duplicate scan is opt-in because it is not linear in
        the corpus.
        """
        report = _call("check_index_integrity", backend.integrity, near_duplicates)
        return _trim_integrity(report)

    @server.tool(title="Verify citations", annotations=read_only)
    def verify_answer(
        answer: str, passages: list[PassageInput], query: str | None = None
    ) -> VerifyResponse:
        """Check an answer's citations and quoted spans against the evidence given.

        Deterministic and model-free: it reports whether each claim cites in-range
        evidence and whether each quoted span actually appears in the passages. It says
        nothing about whether the answer is true; `clean` is not a truth verdict.
        """
        report = _call("verify_answer", backend.verify, answer, passages, query)
        return VerifyResponse(**report)

    @server.tool(title="Adjudicate claims", annotations=read_only)
    def adjudicate_answer(answer: str, tenant: str | None = None) -> AdjudicateResponse:
        """Ask Laya whether the whole corpus supports each claim in an answer.

        Re-retrieves for every claim and reuses the measured answerability question, so
        a claim that is true of the corpus but was not in the shown passages is still
        supported. Read the per-claim `state`; `not_measured` means the corpus was silent.
        """
        report = _call("adjudicate_answer", backend.adjudicate, answer, tenant)
        return AdjudicateResponse(**report)

    compose_target = None
    try:
        from .compose import resolve_target

        compose_target, _reason = resolve_target(config)
    except Exception:  # noqa: BLE001 - a status check must not fail the server
        compose_target = None

    if compose_target is not None:

        @server.tool(
            title="Compose answer",
            annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True),
        )
        def compose_answer(
            query: str,
            passages: list[PassageInput],
            max_tokens: int | None = None,
        ) -> ComposedAnswer:
            """Write prose from evidence the caller supplies.

            The only tool that generates text, and the only output Laya does not verify:
            treat it as `unverified` and prefer the passages returned by
            `search_evidence`. Nothing is re-retrieved here; the answer is written from
            exactly the passages passed in.
            """
            body = _call("compose_answer", backend.compose, query, passages, max_tokens)
            return ComposedAnswer(
                answer=str(body.get("answer", "")),
                model=body.get("model"),
                finish_reason=body.get("finish_reason"),
                note=body.get("note"),
            )

    if enable_writes:

        @server.tool(
            title="Index documents",
            annotations=ToolAnnotations(
                read_only_hint=False, destructive_hint=True, idempotent_hint=False
            ),
        )
        def index_documents(paths: list[str], rebuild: bool = False) -> IndexReportResponse:
            """Index files or directories so they can be searched.

            Expensive and stateful. `rebuild` discards the existing index first. The
            service may refuse paths outside its configured roots.
            """
            if not paths:
                raise ToolError("index_documents: at least one path is required")
            body = _call("index_documents", backend.index, paths, rebuild)
            return IndexReportResponse(**body)

        @server.tool(
            title="Reset index",
            annotations=ToolAnnotations(
                read_only_hint=False, destructive_hint=True, idempotent_hint=True
            ),
        )
        def reset_index(confirm: str) -> ResetResponse:
            """Delete the entire index. Pass confirm="reset" to proceed.

            Destructive and irreversible: every indexed document is removed.
            """
            if confirm != "reset":
                raise ToolError('reset_index: pass confirm="reset" to delete the index')
            body = _call("reset_index", backend.reset)
            return ResetResponse(**body)

        @server.tool(
            title="Delete documents",
            annotations=ToolAnnotations(
                read_only_hint=False, destructive_hint=True, idempotent_hint=True
            ),
        )
        def delete_documents(
            paths: list[str] | None = None, doc_ids: list[str] | None = None
        ) -> DeleteDocumentsResponse:
            """Remove documents from the index by path or document id.

            Cheaper than a rebuild: the lexical channel is rebuilt from the surviving chunks.
            The append-only on-disk backend refuses, and the tool reports that as an error.
            Removing an already-absent document removes nothing.
            """
            if not paths and not doc_ids:
                raise ToolError("delete_documents: pass paths, doc_ids, or both")
            try:
                body = _call("delete_documents", backend.delete, paths or [], doc_ids or [])
            except Exception as exc:  # noqa: BLE001 - surfaced as a tool error
                raise ToolError(f"delete_documents: {exc}") from exc
            return DeleteDocumentsResponse(**body)

    # -- resources -------------------------------------------------------
    @server.resource(
        "laya-rag://stats",
        mime_type="application/json",
        description="A compact JSON view of the corpus and engine status.",
    )
    def stats_resource() -> str:
        stats = _trim_stats(_call("corpus_stats", backend.stats))
        return stats.model_dump_json()

    @server.resource(
        "laya-rag://documents",
        mime_type="application/json",
        description="The first page of indexed documents as JSON.",
    )
    def documents_resource() -> str:
        body = _call("list_documents", backend.documents, 20, None)
        return json.dumps(body, ensure_ascii=False)

    # -- prompts ---------------------------------------------------------
    @server.prompt(
        name="answer_from_corpus",
        description="Answer a question from the corpus, abstaining when unsupported.",
    )
    def answer_from_corpus(query: str) -> str:
        return (
            f"Answer the question using only the indexed corpus.\n\n"
            f"Question: {query}\n\n"
            "Call search_evidence first. If answerable is false, say the corpus does not "
            "cover the question instead of answering. Otherwise answer from the returned "
            "passages only, cite each claim with its [n] source, and report the "
            "answerability probability alongside the answer."
        )

    @server.prompt(
        name="audit_answer",
        description="Check an answer's citations, then whether the corpus supports it.",
    )
    def audit_answer(answer: str) -> str:
        return (
            "Audit the following answer against the corpus.\n\n"
            f"Answer:\n{answer}\n\n"
            "Call verify_answer for the deterministic citation and quote check, then "
            "adjudicate_answer to see whether the corpus supports each claim. Report both, "
            "and treat verify_answer's `clean` as a citation check, not a truth verdict."
        )

    return server


# -- HTTP transport ----------------------------------------------------------

def _bearer_middleware(authenticator: Any) -> Any:
    """A minimal ASGI bearer guard for the streamable-HTTP transport.

    The MCP spec's OAuth resource-server machinery is for a real hosted service; this is
    a static-token local deployment, so a token check is the honest thing to implement.
    """

    class BearerAuthMiddleware:
        def __init__(self, app: Any) -> None:
            self.app = app

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            if scope["type"] != "http":
                await self.app(scope, receive, send)
                return
            from starlette.datastructures import Headers

            headers = Headers(scope=scope)
            try:
                authenticator.authenticate(headers.get("authorization"))
            except Exception as exc:  # APIProblem
                body = json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "error": {"code": -32001, "message": str(getattr(exc, "detail", exc))},
                        "id": None,
                    }
                ).encode("utf-8")
                await send(
                    {
                        "type": "http.response.start",
                        "status": getattr(exc, "status_code", 401),
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"www-authenticate", b"Bearer"),
                            (b"content-length", str(len(body)).encode()),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
            await self.app(scope, receive, send)

    return BearerAuthMiddleware


def _transport_security(host: str, port: int) -> Any:
    """DNS-rebinding protection; beyond loopback an explicit host allowlist is required."""
    if not TransportSecuritySettings:
        return None
    extra = [
        part.strip()
        for part in (os.environ.get("LAYA_RAG_MCP_ALLOWED_HOSTS") or "").split(",")
        if part.strip()
    ]
    allowed_hosts = [f"{host}:{port}", "127.0.0.1:*", "localhost:*", *extra]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=[f"http://{host}:{port}", "http://127.0.0.1:*", "http://localhost:*"],
    )


def run_http(server: Any, config: EngineConfig, host: str, port: int) -> None:
    """Serve the streamable-HTTP transport, requiring auth beyond loopback."""
    import uvicorn

    from .auth import Authenticator

    loopback = host in {"127.0.0.1", "localhost", "::1"}
    authenticator = Authenticator(config)
    if not loopback and not authenticator.required:
        raise SystemExit(
            "refusing to serve MCP beyond loopback without a token: set "
            "LAYA_RAG_API_KEYS (or LAYA_RAG_API_KEY), or bind 127.0.0.1"
        )
    # The helper app's default transport security allows only localhost; a remote bind
    # needs an explicit host allowlist or every request is refused.
    app = server.streamable_http_app(
        host=host,
        transport_security=None if loopback else _transport_security(host, port),
    )
    if authenticator.required:
        app.add_middleware(_bearer_middleware(authenticator))
    uvicorn.run(app, host=host, port=port, log_level="info")


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point: ``laya-rag-mcp``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not MCP_AVAILABLE:
        print(
            "error: the MCP SDK is not installed. Run `pip install -e '.[mcp]'`.\n"
            f"       ({MCP_IMPORT_ERROR})",
            file=sys.stderr,
        )
        return 2

    parser = argparse.ArgumentParser(
        prog="laya-rag-mcp",
        description="MCP server over the Laya RAG engine (stdio by default).",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default=os.environ.get("LAYA_RAG_MCP_TRANSPORT", "stdio"),
        help="stdio (default) or streamable-http",
    )
    parser.add_argument(
        "--host", default=os.environ.get("LAYA_RAG_MCP_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("LAYA_RAG_MCP_PORT", "8078"))
    )
    args = parser.parse_args(argv)

    config = EngineConfig()
    server = build_server(config)
    if args.transport == "streamable-http":
        run_http(server, config, args.host, args.port)
    else:
        # stdout is the MCP wire; logging already goes to stderr.
        server.run(transport="stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
