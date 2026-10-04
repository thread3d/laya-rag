# Findings: the cost axis

ROADMAP 3.4. Quality without cost is half a claim, and cost is the axis this project can actually
win: a CPU-only, 33M-encoder system will not take an nDCG@10 column from an API-scale model, but
"best retrieval per unit of compute" is measurable. This measures per-query latency, throughput,
peak resident memory and index bytes per query against an **existing** index — nothing is rebuilt,
so the system measured is the one the quality numbers describe.

Reproduce:

```sh
make cost-report                      # the shipped default, over eval/.../index/engine-default
make cost-report COST_ARGS="--embed-model BAAI/bge-small-en-v1.5 \
    --index-dir eval/benchmarks/multihop-rag/index/engine-hybrid --label hybrid"
```

## Result — MultiHop-RAG, 4,592 chunks, 60 timed queries after 3 warmup, shipped defaults

| reading | **shipped default** (lexical, dense weight 0) | hybrid (`bge-small` dense on) |
| --- | --- | --- |
| mean latency | **1.66 s** | 1.88 s |
| median | 1.63 s | 1.85 s |
| p95 | **2.14 s** | 2.41 s |
| throughput | **0.60 q/s** | 0.53 q/s |
| peak RSS | **7,038 MiB** | 7,831 MiB |
| index on disk | 31.2 MiB | 24.5 MiB |
| bytes per chunk | 7,132 | 5,596 |

Three readings:

1. **The shipped default wins on cost as well as quality.** It is ~13% faster per query and
   ~800 MiB lighter than the hybrid, and from
   [findings-engine-multihop.md](findings-engine-multihop.md) it also scores higher MAP@10
   (**0.3128** against 0.3009) on this corpus. On MultiHop-RAG there is no trade to make: the
   dense channel costs time, memory *and* quality.
2. **The cost is the decision head, not the retriever.** Peak RSS is ~7 GiB for a 31 MiB index and
   a 33M encoder. That is Laya's 322M decision model, and it dominates everything the retrieval
   stack contributes — which is the honest way to read "cheap": the *index* is cheap, the *model*
   is not.
3. **The hybrid's index is smaller than the default's.** 24.5 MiB against 31.2 MiB, because the
   default's lexical artifact is larger than the dense vectors it replaces. Index bytes per query
   is a real axis, but at this corpus size it is 5.6–7.1 KiB/query against a 1.7-second latency —
   disk is not the constraint, compute is.

## What this does not show

- **Not a cross-system comparison.** Latency and RSS here are for *this* host (Intel Mac, MPS) on
  *this* corpus. A published q/s figure belongs to its own hardware and protocol; without matching
  both, the numbers are not comparable, and no such comparison is made.
- **Not peak RSS attribution.** `ru_maxrss` is the process high-water mark, so it covers the model
  *and* the index in one number rather than splitting memory by component. That is deliberate — an
  attributed figure would need per-component instrumentation this does not have.
- **Not indexing cost.** The measurement is query-time only, against an index already built. Build
  cost (bytes written, wall time, peak RSS during `index_paths`) is a separate axis and is not
  measured here.
- **One corpus, one chunking.** 4,592 chunks at the corpus-aware 2,000-character size. A 10M-chunk
  index would move the memory and bytes-per-query columns, which is what the on-disk backend work
  in Phase 4 exists for.

## Method notes

- **A p95 that is really the maximum flatters the tail.** The percentile interpolates between
  closest ranks, the definition numpy uses, and a test pins it: with 99 values at 1.0 and one at
  900.0 the p95 is 1.0, not 900.
- **`ru_maxrss` is not the same unit everywhere.** It is bytes on macOS and kilobytes on Linux, so
  the report normalises to bytes and a test asserts the value is plausible for a Python process —
  a silent unit error would misreport memory by 1024×.
