# A tamper-evident audit log for index mutations

The security brief asks for an audit log (`docs/research/05` §7) and the repository had none: index
builds, deletes and resets left no durable record, so "who changed the corpus, when" was
unanswerable. `src/laya_rag/audit.py` is a hash-chained append-only JSONL log, and the engine
writes to it when `LAYA_RAG_AUDIT_LOG` names a path.

## The chain

Each record carries `prev` (the previous record's hash) and `hash = sha256(prev ‖ canonical(record))`
over the record without its own `hash`. Editing, reordering, removing or inserting a line breaks the
chain at that point, and `AuditLog.verify()` returns `{ok, entries, head}` or the index and reason
of the **first bad record**. `make audit-verify` runs it and exits `1` on a break.

```sh
LAYA_RAG_AUDIT_LOG=eval/audit.jsonl laya-rag index docs/
make audit-verify AUDIT_LOG=eval/audit.jsonl
```

Appended actions:

| action | written by | detail |
| --- | --- | --- |
| `index` | `RagEngine.index_paths` on success | paths, rebuild/prune flags, documents, chunks, removed_chunks |
| `delete` | `RagEngine.remove_documents` on success | paths, doc_ids, removed_chunks, total_chunks |
| `reset` | `RagEngine.reset` on success | index_dir |

A logging failure (`OSError`) is swallowed rather than failing the request: the index is already
written, so the mutation stands and the gap surfaces through verification.

## What this does not show

- **Attribution is surface-supplied.** The engine has no identity context, so `actor` is a
  parameter of the mutation methods: the HTTP layer passes the authenticated principal's name (or
  tenant, `None` when anonymous), and the CLI and MCP backends pass the fixed labels `cli` / `mcp`.
  A caller that omits it records `null`, which is visibly unattributed rather than silently
  misattributed.
- **Not a security boundary on its own.** Anyone who can rewrite the file can recompute the whole
  chain. What it catches is edit-after-the-fact by someone who did not; anchoring the head hash
  elsewhere is what would make forgery harder.
- **The log is off by default**, a durable side effect a deployment opts into.
- **Indexing writes one record per call, not per file**, so a bulk re-index is one entry. A
  per-document granularity would need the chunker's per-file results.

## Verification

Ten tests: an intact chain verifies from genesis; an edit, a removal and a reorder each break it at
the expected record; a record needs an action; the engine writes `index`, `delete` and `reset` in
order and the chain verifies; no configured path means no file; and the `audit_verify` tool returns
`0` intact / `1` broken / `0` for a missing log. `make test`: **1325 passed, 5 deselected**.
