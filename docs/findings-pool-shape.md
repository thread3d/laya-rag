# The engine reports its retrieval pool's shape

The audit's measurement-gap list says pool **completeness** is instrumented only in
`tools/multihop_diagnostics.py`. Gold completeness genuinely needs qrels, so the engine cannot
report it — but the engine can and now does report the **shape** of the pool it built, which is what
a caller needs to see whether a filter, a deepening loop or a channel change altered the search.

## What shipped

Every `RagEngine.query` result carries `pool`:

| field | meaning |
| --- | --- |
| `requested` | `retrieve_k`, the candidate count the caller/config asked for |
| `searched` | the window actually scanned — larger than `requested` when the deepening loop ran |
| `deepened` | whether `searched > requested` (a trust/ACL filter or dedup forced a deeper fetch) |
| `dense`, `lexical`, `sparse` | candidates each channel returned for that window |
| `candidates` | the fused pool the second stage received |

It surfaces in `QueryResult.as_dict()`, the `QueryResponse` schema (`/v1/query`) and the CLI's
`--json` query output. An early-return result (no index) carries no `pool` and `as_dict` omits it.

This pairs with the generator-free-iteration finding: that experiment reports a pool growing
100 → 327 under probe expansion, and this is the field that makes such a change visible on a live
query rather than only inside a harness.

## What this does not show

- **Shape, not completeness.** It does not say whether the pool contains the gold facts — that needs
  qrels and stays in `tools/multihop_diagnostics.py`.
- **Per-channel counts are raw.** `dense`/`lexical`/`sparse` are what each channel returned before
  fusion, not what survived it; `candidates` is the fused count.
- **No score distribution.** The pool's relevance/recall spread is not summarised here.

## Verification

Two tests: a query reports the pool with every field and `searched >= requested >= 1` and
`candidates >= len(passages)`, and its `as_dict` carries the same dict; a default `QueryResult` has
an empty pool and `as_dict` omits the key. `make test`: **1399 passed, 5 deselected** (plus these).
