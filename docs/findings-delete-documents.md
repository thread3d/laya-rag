# Removing documents without rebuilding the index

The audit's Bet 8 found that `VectorIndex.remove_doc_ids` / `remove_paths` existed but **no HTTP,
MCP or CLI route reached them**: a document could only be dropped by rebuilding the whole index.
That surface is now reachable.

## The engine path

`RagEngine.remove_documents(paths=(), doc_ids=())` removes chunks from the dense index and then
**rebuilds the lexical channel from the survivors**, so the two channels cannot disagree about
which documents exist. The dense index is authoritative; a stale BM25 posting would otherwise keep
surfacing a deleted document (the fusion resolves every candidate through `index.get`, so a removed
chunk id resolves to `None` — safe, but wasteful and confusing). The learned-sparse channel is
rebuilt too when configured, and the query cache is cleared.

Removing a path removes every chunk of the document(s) that path belongs to. A request naming
nothing removes nothing and returns `0`.

**The on-disk backend refuses** with a clear error: it is append-only, and dropping chunks would
invalidate its saved byte offsets. `--rebuild` remains the way to remove a document there.

## The surfaces

- **HTTP:** `DELETE /v1/documents` with `{"paths": [...], "doc_ids": [...]}` → `200`
  `{"removed_chunks": n, "total_chunks": n}`; requires the write scope; the on-disk backend answers
  **`409 index_not_updatable`** (a new problem code).
- **CLI:** `laya-rag delete PATH... [--doc-id ID] [--json]`; exits `2` with the message when the
  backend cannot remove.
- **MCP:** a write-gated `delete_documents(paths, doc_ids)` tool (destructive, idempotent), backed
  by either the local engine or the HTTP transport.

## Verification

Eight tests: removal drops the chunks and the lexical channel agrees both live and from a freshly
reopened engine; removal by doc id; an absent path removes nothing; the on-disk backend refuses;
the HTTP route removes and reports; the CLI parser collects paths and ids, forwards them, and
reports the append-only refusal. `make test`: **1304 passed, 5 deselected**.

```sh
.venv/bin/python -m pytest tests/test_remove_documents.py tests/test_cli.py -q
```

## What this does not show

- **The on-disk backend still cannot remove**, which is the incremental-lifecycle half of Bet 8.
- Removal is not transactional across the two files: a crash between `index.save` and the lexical
  save would leave a dense index missing documents the lexical channel still holds. The engine
  tolerates that (unknown chunk ids resolve to `None`), but a rebuild is the recovery.
- The MCP tool is covered by the existing import-guard tests only; the suite skips it when the
  `[mcp]` extra is absent, as it does for every MCP tool.
