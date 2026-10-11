"""Typed response bodies for the HTTP API.

The engine returns plain dicts so the CLI and the GUI can read them without importing
Pydantic models. These models describe the *same* JSON at the boundary: the engine stays
dict-based, and FastAPI validates the dict against the model on the way out. Two rules
keep the contract honest:

* A model must not silently drop a field the engine returns. The rich reports (stats,
  integrity, verify, adjudicate) therefore allow extra keys, so they stay extensible
  without the schema lying about the fields it does name.
* A model must not change a value's shape. ``extra="allow"`` is not a licence to
  re-type; every declared field matches the dict the handler returns.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Extensible(BaseModel):
    """Base for reports whose nested structure is deliberately open-ended."""

    model_config = ConfigDict(extra="allow")


# -- errors ------------------------------------------------------------------

class Problem(BaseModel):
    """RFC 9457 problem details, plus the two extensions every major API carries.

    ``detail`` stays a string so the bundled GUI's existing error reader keeps working;
    field-level errors travel in ``errors``, exactly as RFC 9457's own 422 example does.
    """

    type: str = Field("about:blank", description="URI identifying the problem type")
    title: str = Field(..., description="Short, human-readable summary of the problem type")
    status: int = Field(..., description="HTTP status code for this occurrence")
    detail: str = Field("", description="Explanation specific to this occurrence")
    instance: str | None = Field(None, description="Path that produced the problem")
    code: str = Field(..., description="Stable machine-readable code, e.g. index_not_built")
    request_id: str | None = Field(None, description="Correlation id; also sent as X-Request-ID")
    errors: list["ProblemField"] | None = Field(
        None, description="Field-level errors for a validation failure"
    )


class ProblemField(BaseModel):
    """One field-level problem inside ``Problem.errors``."""

    detail: str = Field(..., description="What is wrong with the field")
    pointer: str | None = Field(None, description="JSON Pointer (RFC 6901) to the field")


# -- health ------------------------------------------------------------------

class LivenessResponse(BaseModel):
    """Cheap liveness: the process can serve HTTP. Never touches the model."""

    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    """Whether the engine can actually serve a query right now."""

    status: Literal["ready", "degraded"]
    ready: bool
    checkpoint: str
    model_loaded: bool
    index_present: bool
    error: str | None = None
    retry_after: int | None = None


class HealthResponse(_Extensible):
    """The legacy combined health report, kept for the bundled GUI."""

    status: str
    checkpoint: str
    model_loaded: bool
    index_present: bool
    error: str | None = None


# -- retrieval ---------------------------------------------------------------

class Passage(BaseModel):
    """One piece of evidence: Laya-ranked and Laya-gated."""

    text: str
    path: str
    chunk_id: str
    recall_score: float
    relevance: float
    channels: list[str] = Field(default_factory=list)
    start_char: int
    end_char: int


class Timings(BaseModel):
    retrieval_seconds: float
    rerank_seconds: float
    total_seconds: float


class QueryResponse(BaseModel):
    """The evidence bundle for one query."""

    query: str
    answerable: bool = Field(..., description="Laya's calibrated verdict on the evidence")
    answerability_measured: bool = Field(
        ..., description="False when the relevance gate emptied the set before the decision"
    )
    answerability: float | None = None
    passages: list[Passage]
    considered: int
    kept: int
    dropped_below_threshold: int
    trust_filtered: int
    acl_filtered: int
    checkpoint: str
    timings: Timings
    usage: dict[str, Any]
    note: str | None = None
    pool: dict[str, Any] = Field(default_factory=dict)


# -- corpus listings ---------------------------------------------------------

class DocumentRecord(BaseModel):
    """One indexed document."""

    doc_id: str
    path: str
    chunks: int
    chars: int


class DocumentListResponse(BaseModel):
    """A page of indexed documents.

    ``documents`` is the historical key, retained so the GUI keeps reading it; the
    pagination fields are added around it. ``next_cursor`` is null on the last page.
    """

    total: int = Field(..., description="Documents matching the request")
    documents: list[DocumentRecord]
    has_more: bool = False
    next_cursor: str | None = None


class ChunkRecord(BaseModel):
    """One indexed chunk, in document order."""

    chunk_id: str
    doc_id: str
    path: str
    index: int
    start_char: int
    end_char: int
    chars: int
    text: str


class ChunkListResponse(BaseModel):
    """A page of indexed chunks, optionally narrowed to one path."""

    total: int
    path: str | None = None
    chunks: list[ChunkRecord]
    has_more: bool = False
    next_cursor: str | None = None


# -- indexing ----------------------------------------------------------------

class SkippedPath(BaseModel):
    path: str
    reason: str


class IndexReportResponse(_Extensible):
    """Outcome of one indexing run, as ``POST /v1/index`` reports it."""

    documents: int
    chunks: int
    removed_chunks: int
    total_chunks: int
    paths: int
    embed_seconds: float
    save_seconds: float
    sparse_seconds: float
    derived_chunk_chars: int
    total_seconds: float
    skipped: list[SkippedPath] = Field(default_factory=list)
    note: str | None = None


# -- complex reports ---------------------------------------------------------

class StatsResponse(_Extensible):
    """Configuration, model and corpus statistics."""

    config: dict[str, Any]
    model: dict[str, Any]
    embedder: dict[str, Any]
    reranker: dict[str, Any]
    index_dir: str
    index_backend: str
    index_present: bool
    lexical_present: bool
    sparse: dict[str, Any]
    compose: dict[str, Any]
    index: dict[str, Any]


class IntegrityResponse(_Extensible):
    """Index drift: deleted documents, duplicate and near-duplicate chunks."""

    chunks: int
    paths: int
    clean: bool
    index_present: bool
    stale_paths: dict[str, Any] | None = None
    duplicate_content: dict[str, Any] | None = None
    near_duplicate_max_chunks: int | None = None
    near_duplicate_content: dict[str, Any] | None = None


class VerifyResponse(_Extensible):
    """Deterministic citation and quote checks over an answer."""

    claims: list[dict[str, Any]]
    quotes: list[dict[str, Any]]
    invalid_citations: list[int]
    counts: dict[str, Any]
    content: dict[str, Any]
    injection: dict[str, Any]
    passages: int
    clean: bool
    summary: str


class AdjudicateResponse(_Extensible):
    """Laya's verdict on whether the corpus supports each claim in an answer."""

    claims: list[dict[str, Any]]
    total: int
    supported: int
    unsupported: int
    not_measured: int
    answerability_threshold: float
    relevance_threshold: float
    retrieve_k: int
    trust_filtered: int
    acl_filtered: int
    question: str
    checkpoint: str
    note: str | None = None
    timings: dict[str, Any]
    usage: dict[str, Any]


class ExtractedSentence(BaseModel):
    """One emitted sentence and the citation index it must satisfy."""

    text: str = Field(..., description="The sentence, verbatim from the passage below")
    citation: int = Field(..., description="The `[n]` this sentence carries in `answer`")
    path: str | None = Field(None, description="Source document the span was copied from")
    chunk_id: str | None = Field(None, description="Chunk the span was copied from")
    score: float = Field(..., description="The document's relevance — not a per-sentence score")


class ExtractResponse(QueryResponse):
    """`/v1/query`'s evidence, plus an answer assembled only from verbatim spans of it.

    Every sentence in ``sentences`` is a substring of one of ``passages``, so the answer
    cannot state anything the corpus does not contain, and with nothing to emit it abstains.
    No generator is involved, so an end-to-end score over this output is the engine's alone.
    The evidence travels with the answer because ``[n]`` is only checkable against it —
    ``/v1/verify`` takes exactly these passages.
    """

    answer: str
    abstained: bool
    sentences: list[ExtractedSentence]
    sentences_considered: int = Field(
        ..., description="Candidate sentences considered before the max_sentences cut"
    )


class ResetResponse(BaseModel):
    """The index was deleted."""

    status: str
    index_dir: str


class DeleteDocumentsRequest(BaseModel):
    """A request to remove documents from the index without rebuilding it."""

    paths: list[str] = Field(default_factory=list)
    doc_ids: list[str] = Field(default_factory=list)


class DeleteDocumentsResponse(BaseModel):
    """Outcome of a delete: how many chunks were removed and what remains."""

    removed_chunks: int
    total_chunks: int


Problem.model_rebuild()
