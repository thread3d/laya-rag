# Retrieval-time predicate pushdown: on-disk parity, and the engine now uses it

The audit's Bet 8 names a specific asymmetry: `VectorIndex.search` takes an
`allow: Callable[[str | None], bool]` path predicate, "which no caller ever passes", and
`OnDiskVectorIndex.search` "has no such parameter at all". Every trust-tier and tenant-ACL
filter was therefore applied only to the *fused* list after retrieval — a recall and a cost
problem at scale, because an ACL window can exclude most of a top-k and leave the result short.

Two things changed.

## The on-disk backend takes the predicate

`OnDiskVectorIndex.search(query_vector, k, block, *, allow=None)` now accepts the same predicate
as the in-memory `VectorIndex.search`, so the two backends are interchangeable for a filtered
query. The implementation does not keep a path table in memory: it ranks a window with the
existing `_rank` helper (ANN when present, exact mmap scan otherwise), reads only the candidate
records to evaluate `allow`, and **deepens** the window when too few candidates survive — the same
"change *which*, never *how many*" contract the in-memory path states. `allow=None` is the
previous code path, unchanged.

## The engine passes it

`RagEngine.query` builds one predicate from the trust manifest — tenant ACL first, then the tier
minimum — and passes it to the dense search. The post-fusion filter stays, because the lexical and
sparse channels have no predicate yet; it is now the authority for those channels only.

## What this fixes, and what it does not

- **Fixed:** the two vector backends have predicate parity, and the engine no longer ignores the
  parameter — a dense search under an ACL selects from the whole index rather than a fixed
  over-fetch window.
- **Not yet:** the lexical and sparse channels still filter after fusion (they expose no
  predicate), so the cost win is partial, and the on-disk backend still refuses incremental
  updates. Delete-by-id is now reachable by HTTP and CLI
  ([`findings-delete-documents.md`](findings-delete-documents.md)); MCP is not yet wired.

## Verification

Four tests: the predicate excludes a nearer blocked document inside the scan and returns the next
allowed one; the window deepens to reach an allowed document that the unfiltered top-k does not
contain; `allow=None` is unchanged; and an end-to-end on-disk engine with a tenant ACL still
isolates tenants and returns nothing to an anonymous caller. `make test`: **1296 passed, 5
deselected**.

```sh
.venv/bin/python -m pytest tests/test_ondisk_engine.py tests/test_trust.py -q
```
