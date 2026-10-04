# Findings: quantized ANN at 1M — the rescore is not the bottleneck

ROADMAP Phase 4.5. The engine's dense channel is an exhaustive scan: `VectorIndex.search` multiplies
the query against every stored vector, which is exact and fine to a few hundred thousand chunks and
linear after that. The scale claim rests on a quantized index shrinking enough to stay in RAM while
the fp32 vectors — mmapped, so not resident — restore the exact order. This verifies both halves at
**one million vectors**, on synthetic clustered vectors, against the exhaustive scan.

Reproduce:

```sh
make ann-bench ANNBENCH_ARGS="--sizes 10000,100000,1000000"           # the trend
make ann-bench ANNBENCH_ARGS="--sizes 1000000 --pq-nprobe 32 \
    --pq-candidates 100,500,1000 --out eval/ann-bench-sweep"          # candidate-depth sweep
```

## Result — 1M vectors, 384 dimensions, 100 queries, k=10

Queries are drawn from the **same** mixture as the documents; vectors drawn from a fresh set of
centroids collapse recall for reasons that have nothing to do with the index.

| index | index size | recall@10 | ms/query | vs exhaustive |
| --- | --- | --- | --- | --- |
| exhaustive (flat fp32) | 1,536 MB | 1.000 (by definition) | 10.45 | 1× |
| IVF-flat, `nprobe=32` | 1,536 MB | 1.000 | 0.697 | 15× |
| HNSW-32 | 1,536 MB | 0.814 | **0.015** | 683× |
| **IVF-PQ (m=48) + fp32 rescore, `nprobe=32`, 500 candidates** | **49.5 MB** | **0.998** | 0.510 | 20× |
| IVF-PQ + rescore, `nprobe=32`, 1000 candidates | 49.5 MB | **1.000** | 0.838 | 12× |

**The compression is real: 1,536 MB → 49.5 MB, 48 bytes per vector instead of 1,536, a 31×
reduction.** And with a deep enough candidate list the recall is retained — 0.998 at 500 candidates,
1.000 at 1,000 — at 0.51 ms against the exhaustive scan's 10.45 ms.

## The lesson: candidate-set recall, not the rescore, is the constraint

The first run at `nprobe=8` told a worse story, and it is worth recording because it is the mistake
the design invites:

| candidates rescored | recall@10 (nprobe 8) | recall@10 (nprobe 32) |
| --- | --- | --- |
| 10 | 0.066 | — |
| 50 | 0.226 | — |
| 100 | 0.370 | 0.406 |
| 500 | — | **0.998** |
| 1000 | — | **1.000** |

A rescore can only reorder what it is given. At `nprobe=8` and 100 candidates the candidate list
simply does not contain the true neighbours — the PQ codes ranked them out — so no amount of exact
rescoring recovers them. Raising the probe count and the candidate depth fixes it almost entirely.
**The tuning knob is the candidate set, not the reranker**, and at 100 candidates the configuration
looks like a recall failure when it is a selection failure.

The trend across sizes makes the same point: the same `nprobe=8`, 100-candidate setting scores
0.646 at 10k, 1.000 at 100k and 0.370 at 1M, because `nlist` grows with √n while `nprobe` stays
fixed, so a shrinking fraction of cells is probed. A fixed `nprobe` is not a scale-invariant
setting.

## What this does not show

- **Not real embeddings.** Synthetic clustered vectors, not a corpus. Real embeddings are not
  isotropic and their intrinsic dimensionality differs, which changes how much PQ loses; the
  synthetic set establishes the mechanics and the memory arithmetic, not a recall figure for a
  specific corpus.
- **Not a trained PQ.** `m=48`, `nbits=8` off-the-shelf. Optimised product quantizers or a
  rerank-refine index (`IndexIVFPQR`) would trade differently.
- **Not a query-mix claim.** 100 queries, k=10, one seed. The recall figures carry no interval.
- **Not RSS.** "Index size" is bytes on disk/allocated, not measured resident memory; the rescore
  needs the fp32 vectors available, which is the mmap's job and is not measured here.

## Method notes

- **A rescore's recall is bounded by its candidate set.** Measure both, or a selection failure reads
  as a quantization failure — which is exactly what the first run looked like.
- **Fixed `nprobe` is not scale-invariant.** `nlist` grows with the corpus, so a probe budget that
  is generous at 100k is starved at 1M.
