# Findings: an eager sparse BM25 backend — identical ranking, ~11x faster, ~5x smaller

This records the scale work from [`RESEARCH.md`](../RESEARCH.md) Tier 2.6, implemented
without a new dependency (NumPy only). The historical `BM25Index` stores a `Counter` per
document and scores by looping over the postings of every query term in Python, so both
its resident size and its per-query cost scale with the Python object graph.

## Design

`EagerBM25Index` precomputes each posting's final BM25 weight once — the BM25S "eager
scoring" idea — and stores the postings as NumPy arrays (`term -> (rows int32, weights
float64)`). A query is then a handful of slice-adds:

```python
scores[rows] += weights        # once per query term
```

The weights come from the same formula as `BM25Index` and are accumulated in the same
query-term order, and ties break to the lower document index exactly as the stable sort
there does, so the **ranking is identical** — pinned by tests and below.

## Measured (deterministic; `make bench-lexical`)

**Synthetic corpus**, 20,000 documents x 120 tokens, vocab 5,000, 200 queries, top-100:

| backend | build (s) | index (MB) | ms/query |
| --- | --- | --- | --- |
| bm25 | 1.66 | 242.7 | 15.53 |
| **eager** | 3.54 | **43.1** | **1.16** |

Parity: **200/200 queries identical**. Speedup **13.3x**, size ratio **5.6x**.

**Real MultiHop-RAG corpus** (17,259 chunks, 300 queries, top-100):

| backend | build (s) | index (MB) | ms/query |
| --- | --- | --- | --- |
| bm25 | 3.12 | 389.7 | 17.53 |
| **eager** | 6.73 | **81.1** | **1.57** |

Parity: **300/300 queries identical**. Speedup **11.1x**, size ratio **4.8x**.

Size is a deterministic object-graph estimate, not process RSS: RSS cannot attribute
memory to one structure on a 56-core host shared with the interpreter.

## How to enable

`LAYA_RAG_LEXICAL_BACKEND=eager` (default `bm25`). It changes the on-disk format
(`lexical-eager.npz` + `.meta.json`), so switching backends means **re-indexing**; the
engine loads whichever file matches the configured backend and degrades to dense-only
recall if it is missing or damaged, exactly as before.

## Honest limits

- **Build is about 2x slower** (weight precomputation), and it writes a larger transient
  during the build; the win is in query time and resident size, which is where a deployed
  index spends its life.
- Parity is verified on the synthetic corpus and the real MultiHop corpus at top-100, plus
  the unit corpus; the BEIR path uses the same `BM25Index` class directly and is unchanged,
  so it is not re-measured here.
- It is a **performance and memory** change, not a quality change — the ranking is
  identical by construction and by measurement.
