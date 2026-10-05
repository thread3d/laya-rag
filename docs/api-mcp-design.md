# An industry-standard API and an MCP server for `laya-rag`

Investigation and implementation record. Companion to [`PLAN.md`](../PLAN.md) and
[`ROADMAP.md`](../ROADMAP.md).

**Status: implemented (phases A–G).** The proposal below is kept as the design record;
[§9 As built](#9-as-built) maps each phase to the files and tests that shipped and lists
the deliberate deviations. Every change is additive: the wire behaviour the GUI, CLI and
benchmarks depend on is unchanged, and the full offline suite is green (1166 → 1229
tests).

Standards verified against primary sources in October 2026: RFC 9457, RFC 6750, RFC 9745,
RFC 8594, RFC 9110, the IETF idempotency draft, and the Model Context Protocol
`2026-07-28` revision with the Python SDK v2. Where a choice follows a vendor convention
rather than a standard, the text says so.

---

## 1. Why this is worth doing

The engine already has the hard part: a calibration-honest retrieval pipeline with an
HTTP service, a GUI and a CLI. What it does **not** have is a contract other software can
depend on:

- The OpenAPI document at `/openapi.json` describes the *request* bodies but none of the
  responses. Every `200` is an untyped `application/json` object, so a generated client
  gets `Any` back and can promise nothing to its caller.
- Errors are FastAPI's default `{"detail": ...}`. The `detail` field changes type
  (string for a 409, list for a 422) and no endpoint publishes the codes it can return.
- There is no MCP surface at all, so an agent that already speaks MCP cannot ask this
  engine a question without a bespoke integration.

For a project whose whole claim is *"the evidence bundle is calibrated and honest"*, the
integration layer is where that honesty is lost: a caller cannot tell from the schema
what `answerable` means, cannot branch on a stable error code, and cannot discover the
tools.

---

## 2. Current surface (inventory)

Routes registered by `src/laya_rag/api.py`, all behind the optional static bearer token
(`LAYA_RAG_API_KEY`), none declaring a security scheme in the OpenAPI document:

| Route | Request typed | Response typed | Notes |
| --- | --- | --- | --- |
| `GET /health` | — | no | loads the model; returns `200 degraded`, never `503` |
| `GET /v1/stats` | — | no | full config + index + trust state |
| `GET /v1/documents` | — | no | returns **every** document, unpaginated |
| `GET /v1/chunks?path=` | — | no | returns **every** chunk (full text), unpaginated |
| `GET /v1/integrity` | query only | no | linear by default, near-dup scan opt-in |
| `POST /v1/index` | yes | no | synchronous, CPU-bound, no lock |
| `POST /v1/index/stream` | yes | no | NDJSON (`application/x-ndjson`) |
| `POST /v1/query` | yes | no | the core evidence call |
| `POST /v1/compose` | yes | no | NDJSON; the only generated text |
| `POST /v1/verify` | yes | no | deterministic, model-free |
| `POST /v1/adjudicate` | yes | no | corpus-wide claim support |
| `POST /v1/reset` | — | no | destructive |

Measured from the running app (`create_app`):

- OpenAPI **3.1.0**, no `servers`, no `tags`, no `securitySchemes`, auto-generated
  `operationId`s (`query_v1_query_post`).
- Every operation documents only `200` and `422`. The real codes — `401`, `404`, `409`,
  `502`, `503` — are absent.
- `/v1/index/stream` and `/v1/compose` document `200 application/json` while actually
  sending `application/x-ndjson`.
- A 409 is `{"detail": "<string>"}`; a 422 is `{"detail": [<validation objects>]}`;
  both are `application/json`, not `application/problem+json`.
- `OPTIONS` returns `405`; there is no CORS configuration.
- `/docs`, `/redoc` and `/openapi.json` are **not** behind `authorise`.

---

## 3. Gap analysis against industry practice

Severity: **P0** = correctness or security, **P1** = interoperability, **P2** = polish.

| # | Concern | Current | Standard practice | Severity |
| --- | --- | --- | --- | --- |
| 1 | Response schemas | untyped `Any` | named Pydantic models per response, referenced from OpenAPI | P1 |
| 2 | Error contract | `{"detail": ...}`, mixed types | RFC 9457 `application/problem+json` with `type`/`title`/`status`/`detail`/`instance` + `code`/`request_id` | P1 |
| 3 | Liveness vs readiness | one `/health`, loads the model | cheap `/healthz`, readiness `/readyz` returning `503` when unusable | P1 |
| 4 | Request correlation | none | echo/generate `X-Request-ID`, propagate W3C `traceparent` | P1 |
| 5 | Listing pagination | full dump | cursor pagination `{items, next_cursor, has_more, total}` | P1 |
| 6 | Idempotency | none on `POST` | `Idempotency-Key` on index/reset | P2 |
| 7 | Rate limiting | none | `RateLimit` + `RateLimit-Policy` structured fields (IETF draft-11); `429` + `Retry-After` | P2 |
| 8 | Streaming | NDJSON only, mis-declared | SSE `text/event-stream`, correct media types in the spec | P1 |
| 9 | Write authorization | one all-powerful token | scopes (read vs write), tenant from the principal | **P0** |
| 10 | Tenant trust | `tenant` asserted in the request body | derive from the authenticated principal; never trust client input | **P0** |
| 11 | Index path scope | client passes any filesystem path | constrain to configured index roots | **P0** |
| 12 | Build concurrency | no lock around index mutation | serialize builds; reject/queue a concurrent build | **P0** |
| 13 | CORS | none (`405` on preflight) | configurable origin allowlist, off by default | P2 |
| 14 | Async jobs | synchronous / NDJSON only | `202` + `Location` + job status, for long builds | P2 |
| 15 | Deprecation | none | `Deprecation` (`@`-Unix-seconds, RFC 9745) + `Link rel="successor-version"`; `Sunset` (HTTP-date, RFC 8594) must not precede `Deprecation` | P2 |
| 16 | Transport plumbing | none | gzip, request size limits, timeouts, `ETag` on read routes | P2 |
| 17 | Client surface | none | a generated/typed client and runnable examples | P2 |
| 18 | Schema exposure | `/docs`, `/openapi.json` open | gate the schema when a key is configured (or make it explicit) | P2 |

### The P0 items are not "standards" — they are live hazards

Three of them are security defects independent of any standard:

- **#9/#10.** `tenant` is a plain body field (`QueryRequest.tenant`), so any holder of the
  one shared token can read any tenant's documents by asserting that tenant's name. The
  trust manifest *appears* to enforce a per-tenant ACL, but the identity it enforces
  against is caller-supplied.
- **#11.** `POST /v1/index` takes `paths` and hands them to `load_documents`, which reads
  an arbitrary absolute path. On the default `127.0.0.1` bind that is a local-only
  concern; the moment `LAYA_RAG_HOST=0.0.0.0` is set it becomes "index and then retrieve
  any file on the host".
- **#12.** `RagEngine.index_paths` mutates `self._index`, `self._lexical`, the on-disk
  index and the query cache with no lock. Two overlapping `POST /v1/index` requests, or a
  query during a rebuild, race.

They belong in the same programme because the fix for #9/#10 (an API-key → principal map)
is also what makes the API "standard" (declared `securitySchemes`, scoped authorization).

---

## 4. MCP server design

### 4.1 What it is for

MCP is how an agent discovers and calls tools. This server's job is to expose the
engine's *existing* honesty constraints as tools an agent can use safely — above all that
`search_evidence` returns calibrated evidence and an answerability verdict, and never
prose that a model could mistake for verified fact.

The mapping is deliberately **not one tool per route**. MCP guidance favours few,
task-shaped tools over a thin HTTP mirror, because each tool's description and schema is
context the model must carry.

### 4.2 Tools

| Tool | Input | Returns | Notes |
| --- | --- | --- | --- |
| `search_evidence` | `query`, `top_k=5`, `tenant?` | answerability verdict + ranked passages (`path`, `text`, `relevance`, `channels`) | the core tool; description states "if `answerable` is false, do not answer from these passages" |
| `list_documents` | `limit?`, `cursor?` | paginated document list | |
| `read_document` | `path`, `offset=0`, `limit=20` | paginated chunk text for one document | avoids dumping a whole corpus into context |
| `corpus_stats` | — | trimmed stats (index, model, channels, integrity) | token-efficient projection, not the raw config dump |
| `check_index_integrity` | `near_duplicates=false` | drift/duplicate report | |
| `compose_answer` | `query`, `passages` | prose, `unverified: true` | only registered when composition is configured |
| `verify_answer` | `answer`, `passages`, `query?` | citation/quote report | deterministic, model-free |
| `adjudicate_answer` | `answer`, `tenant?` | per-claim corpus verdict | |
| `index_documents` | `paths`, `rebuild=false` | index report | **write tool, off unless `LAYA_RAG_MCP_ALLOW_WRITES=1`**; paths constrained to configured roots |
| `reset_index` | `confirm` | status | **destructive, off unless writes enabled and `confirm == "reset"`** |

### 4.3 Resources and prompts

- Resources (read-only, addressable): `laya-rag://stats`, `laya-rag://documents`, and a
  template `laya-rag://document/{path}`. Resources are the right container for "this
  exists and the client may attach it", as opposed to a tool call that spends a turn.
- Prompts: `answer_from_corpus(query)` — instructs the model to call `search_evidence`,
  abstain when `answerable` is false, and cite `[n]`; `audit_answer(answer)` — run
  `verify_answer` then `adjudicate_answer` and reconcile.

### 4.4 Transports and backend

- **stdio** is the default (`laya-rag-mcp`), which is what Claude Desktop / Cursor /
  VS Code register in a config file. It is also the safest: no port, no network.
- **Streamable HTTP** (`--transport streamable-http --host 127.0.0.1 --port 8078`,
  endpoint `/mcp`) is for remote clients. It must require the same bearer token as the
  HTTP API; local single-user use stays unauthenticated if no key is set.
- Backend selection: **local in-process `RagEngine`** by default — exactly what the CLI
  already does, so no running service is needed and it works offline. An optional
  `LAYA_RAG_MCP_BACKEND=http` mode proxies to a running service instead, which avoids
  loading a second 2.3 GB checkpoint into memory.

### 4.5 Packaging and testing

- Optional extra: `pip install -e '.[mcp]'` adds the MCP Python SDK (`mcp>=2.3`). The
  server imports it lazily and prints an actionable message if it is missing, so the
  base install and every existing test are unaffected.
- SDK surface: the v2 high-level server is `mcp.server.MCPServer` (the v1 `FastMCP` name
  and import path were removed, not deprecated), tools are typed functions decorated with
  `@server.tool()`, and the server serves both the current `2026-07-28` revision and the
  2025-era clients. Transport options live on `run()`/`streamable_http_app()`, not the
  constructor.
- Console script: `laya-rag-mcp`.
- Tests: tool bodies are thin wrappers over `RagEngine`, so the substance is tested
  offline against the existing `StubModel` double. A small integration test drives
  `tools/list`, `tools/call`, resources and prompts over the SDK's in-memory client. No
  network, no checkpoint.

---

## 5. Implementation plan

Ordered so each phase is independently shippable and testable. Every phase ships offline
tests, per the project's rule.

### Phase A — complete the OpenAPI contract (no behaviour change)

- `src/laya_rag/schemas.py`: response models for every route — `Health`, `Readiness`,
  `Stats`, `DocumentList`, `ChunkList`, `IntegrityReport`, `IndexReport`, `QueryResponse`,
  `VerifyResponse`, `AdjudicateResponse`, `Error`.
- `api.py`: `response_model=` on every route, `tags`, explicit `operationId`, examples,
  `servers`, bearer `securitySchemes`, and a documented error-response set per operation.
- Keep the returned JSON identical (Pydantic `model_dump` of the same shape).
- Test: a spec test asserts every operation has a `2xx` schema, an `operationId`, a tag,
  and that the declared media type matches the actual `Content-Type`.

### Phase B — RFC 9457 errors and request correlation

- Problem+json exception handlers for `HTTPException`, validation errors and unhandled
  exceptions, emitting `type`/`title`/`status`/`detail`/`instance` plus `code` and
  `request_id`. `detail` stays a **string**, so the GUI's `detailOf` keeps working.
- Middleware: accept or generate `X-Request-ID`, echo it, put it on every response and
  in every problem body; propagate W3C `traceparent`.
- Map the existing raises to stable codes: `index_not_built` (409), `model_unavailable`
  (503), `invalid_request` (400/422), `decision_failed` (502), `path_not_found` (404).
- Test: each documented status is produced with the right media type and a stable `code`;
  a supplied `X-Request-ID` is echoed; a generated one is present.

### Phase C — operational endpoints and transport plumbing

- `/healthz` (liveness: process is up, no model touch), `/readyz` (readiness: model can
  load and index present → 200, else 503 with `Retry-After`). `/health` stays as a
  deprecated alias with `Deprecation` + `Sunset` headers (RFC 9745 / 8594).
- Optional CORS allowlist (`LAYA_RAG_CORS_ORIGINS`), off by default; `GZipMiddleware`;
  a request-body size cap.
- Test: readiness is 503 before indexing and 200 after; CORS off by default and correct
  when configured; liveness never loads the model.

### Phase D — pagination, idempotency, rate limiting

- Cursor pagination on `/v1/documents` and `/v1/chunks` (`limit`, `cursor`). The
  historical list keys (`documents`, `chunks`) are retained and `next_cursor`/`has_more`
  added around them, so the bundled GUI is untouched; `next_cursor` is null on the last
  page and `total` is the exact count. With no `limit`, the current full listing is
  returned — additive.
- `Idempotency-Key` on `POST /v1/index` and `POST /v1/reset` (and `DELETE /v1/index`),
  backed by a bounded TTL store; a replay returns the stored response with
  `Idempotency-Replayed: true`.
- Optional fixed-window limiter (`LAYA_RAG_RATE_LIMIT`, requests/minute), off by default,
  emitting the draft-11 `RateLimit`/`RateLimit-Policy` fields and `429`/`Retry-After`.
- Test: page traversal reaches every record exactly once; a replayed key does not
  re-index; the limiter is inert unless configured.

### Phase E — streaming contract

- Content-negotiate `Accept: text/event-stream` on `/v1/index/stream` and `/v1/compose`,
  emitting SSE (`event:`/`data:`/`id:`, comment heartbeat while idle). NDJSON remains the
  default so the GUI is untouched. The stream is not an OpenAI chat stream, so it ends
  with the project's own terminal `done` (or `error`) event rather than `[DONE]`; both
  encodings carry the same JSON object per event.
- Correct the declared media types in the OpenAPI document.
- Test: both encodings parse and carry the same events; SSE frames use the standard
  framing and a heartbeat.

### Phase F — authorization, tenancy and path scope (P0)

- `LAYA_RAG_API_KEYS`: a JSON map from token to `{tenant, scopes}`. A token's tenant is
  the *only* tenant it may query; a body `tenant` that disagrees is rejected.
- Scopes: `read` (query/verify/adjudicate/list) and `write` (index/reset). Declared in
  the OpenAPI security scheme.
- `LAYA_RAG_INDEX_ROOTS`: every path passed to `/v1/index` must resolve inside a
  configured root. Default remains the current behaviour only when unset *and* the bind
  is loopback; otherwise refuse.
- Engine build lock so concurrent builds serialize or return `409 build_in_progress`.
- Test: a tenant token cannot read another tenant; a read-scoped token is refused on
  index/reset; a path outside the roots is refused; a second concurrent build is refused.

### Phase G — MCP server

- `src/laya_rag/mcp_server.py`, optional `[mcp]` extra, `laya-rag-mcp` console script.
- Tools, resources and prompts as designed in §4; stdio and streamable-HTTP transports.
- `pyproject.toml`: extra + script; `make mcp` / `make test-mcp` targets.
- Test: `tools/list` shape, each tool's happy path and its "do not answer when
  `answerable` is false" contract, write tools disabled by default, HTTP auth rejection.

### Phase H — documentation and examples

- README: an "Integrate" section — OpenAPI URL, error contract, an MCP client config
  snippet, and one curl + one Python example.
- This document updated from proposal to as-built, with the standards it conforms to.

---

## 6. Backward compatibility

| Change | Compatibility |
| --- | --- |
| Response models | wire-identical JSON |
| problem+json | `detail` remains a string; the GUI reads only `detail` |
| `/health` deprecated alias | unchanged behaviour |
| pagination | only active when `limit` is supplied |
| SSE | only when the client sends `Accept: text/event-stream` |
| idempotency / rate limit | inert unless configured |
| tenant-from-token | **breaking only for a caller that currently asserts a body tenant with the shared token** — which is the bug being fixed; document the migration |
| path roots | unset + loopback keeps today's behaviour |
| MCP | purely additive |

---

## 7. What I would not do

- **No OpenAI-compatible `/v1/chat/completions`.** Laya is not a chat model; a shim would
  misrepresent it, which is the exact thing the current API header comment warns against.
- **No auto-generated "answer" endpoint.** The engine returns evidence; composing stays an
  explicit, separately-labelled call.
- **No silent re-interpretation of measured defaults.** New behaviour is opt-in, matching
  the project's rule that existing benchmark numbers stay comparable.

An optional **`/v1/embeddings`** (OpenAI-shaped) is the one compatibility surface worth
considering, because it lets the engine's embedder drop into existing tooling; it exposes
only the encoder, not a decision, so it does not misrepresent Laya. Proposed as opt-in,
not part of the core phases.

---

## 8. Decisions taken

1. **Scope:** phases A–G (full API hardening plus the MCP server).
2. **Tenant from the token:** a bound token's tenant is the only tenant it may reach; a
   conflicting body `tenant` is a 403. The single legacy `LAYA_RAG_API_KEY` stays a
   tenant-unbound admin so existing deployments keep working.
3. **Path roots:** enforced absolutely when configured; with no allowlist, indexing is
   allowed only on a loopback bind, so a reachable service must opt in explicitly.
4. **MCP write tools:** registered only when `LAYA_RAG_MCP_ALLOW_WRITES` is set;
   `reset_index` additionally requires `confirm="reset"`.

---

## 9. As built

| Phase | What shipped | Files | Tests |
| --- | --- | --- | --- |
| A | Response models for every route, `operationId`s, tags, servers, bearer scheme, accurate streaming media types, problem-typed error responses | `schemas.py`, `api.py` | `test_api_contract.py` |
| B | RFC 9457 `application/problem+json`, stable `code`, `errors[]` pointers, unhandled-error handler, `X-Request-ID`, W3C `traceparent` | `problems.py` | `test_api_contract.py` |
| C | `/healthz` + `/readyz` (503 when unusable), deprecated `/health` alias, optional CORS, gzip (stream media types excluded), body-size cap, schema gating | `api.py`, `config.py` | `test_api_contract.py` |
| D | Keyset cursor pagination, idempotency store, fixed-window rate limiter | `paging.py`, `idempotency.py`, `ratelimit.py` | `test_api_contract.py`, `test_api_security.py` |
| E | SSE content negotiation with heartbeats; NDJSON unchanged | `streaming.py` | `test_streaming.py` |
| F | API-key principals, read/write scopes, tenant binding, index path roots, engine build lock | `auth.py`, `engine.py`, `api.py` | `test_api_security.py`, `test_streaming.py` |
| G | MCP server: 10 tools, 2 resources, 2 prompts, stdio + streamable HTTP, local/HTTP backends | `mcp_server.py` | `test_mcp_server.py` |

### Deliberate deviations from the proposal

- **Listing envelopes.** The proposal called for `{items, ...}`. The shipped responses
  keep the historical `documents`/`chunks` keys and add `has_more`/`next_cursor` around
  them, because the bundled GUI reads those keys and the project's rule is that existing
  behaviour is not silently reinterpreted.
- **Readiness body.** `/readyz` answers 503 with the readiness report rather than a
  problem, so a probe client can read *why* it is not ready. Every other error is a
  problem.
- **Probes are public.** `/healthz` and `/readyz` are not behind the bearer token, so
  Kubernetes-style probes work. `/health` stays authenticated for compatibility.
- **`/docs` and `/openapi.json`.** Left public by default via `LAYA_RAG_SCHEMA_PUBLIC`;
  set it false to disable them on an authenticated service.
- **No `RateLimit-*` triple.** The shipped headers are the draft-11 `RateLimit` and
  `RateLimit-Policy` fields, with `Retry-After` on 429.
- **`Deprecation` is `@`-Unix-seconds** (RFC 9745) with a `Link rel="successor-version"`;
  no `Sunset` is emitted because no removal is scheduled.
- **No `202`/job queue.** Indexing still streams progress; the async job resource was
  left out (it is a convention, not an RFC requirement, and the streaming path already
  reports progress).
- **No `/v1/embeddings`.** The one optional compatibility surface in §7 was not built;
  it would expose the encoder alone, which is a different contract from the calibrated
  evidence the rest of the API promises. It remains available as a future opt-in.

### Known limits

- The idempotency store and rate limiter are in-process and bounded. They guard a single
  instance against retries and runaway clients; they are not a distributed lock or a
  fleet-wide quota.
- Read endpoints other than query/adjudicate are not isolated from a concurrent build:
  the build lock serializes builds and refuses queries during one, but `stats` and the
  listings read the index without taking it.
