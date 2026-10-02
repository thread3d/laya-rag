# Findings: the cross-encoder + coverage default at the engine level

Round 11 made `RERANK_COVERAGE_LAMBDA` resolve to **0.7 automatically whenever a
`RERANK_MODEL` is configured**, so a caller who sets up a second stage gets the configuration
that measured best. That evidence came from MultiHop-RAG, where a cross-encoder *helps* — but
on BEIR a pointwise cross-encoder is known to hurt (ArguAna 0.568 → 0.250 under a full
re-sort), so the open question was whether the auto default damages BEIR. This runs the real
engine on SciFact to answer it.

```sh
make engine-beir ENGINE_BEIR_ARGS="--dataset scifact --tag ceauto \
    --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2"
make engine-beir ENGINE_BEIR_ARGS="--dataset scifact --tag ceplain \
    --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 --rerank-coverage-lambda 0.0"
```

## Results (SciFact, 300 queries, `bge-small`, top-10)

| configuration | nDCG@10 | wall time |
| --- | --- | --- |
| base — Laya decisions, no cross-encoder | 0.7264 | 616 s |
| **cross-encoder + coverage auto (0.7)** | **0.7274** | **204 s** |
| cross-encoder, pure relevance order | 0.7091 | 198 s |

Paired against the base: coverage-auto **+0.0010 [−0.0045, +0.0068]**, p=0.81; pure CE order
**−0.0173 [−0.0412, +0.0079]**, p=0.17; coverage-auto minus pure order **+0.0183 [−0.0056,
+0.0417]**, p=0.13. None is individually significant on 300 queries, so the honest claim is
about direction and safety rather than a new win:

1. **The auto default does not damage BEIR.** It lands at or just above the no-cross-encoder
   baseline, while the same cross-encoder ordered purely by its own score drifts below it.
   The coverage pass is doing exactly the job it was added for — bounding a second stage that
   is worse than the first on this corpus — and the `auto` resolution is therefore validated
   on the dataset where the risk was.
2. **A cross-encoder plus coverage is ~3× faster at equal quality.** 204 s against 616 s for
   the same 300 queries and the same nDCG. The mechanism is not the cross-encoder being fast:
   it is that the 322M decision model no longer scores every candidate for *ordering*, and is
   used only to gate the handful of kept passages. Configuring a 33M second stage therefore
   buys a large latency reduction as well as a ranking control.

## Honest limits

- One dataset, one cross-encoder (`ms-marco-MiniLM-L-6-v2`), 300 queries, and every paired
  delta is individually non-significant. The safety claim ("does not damage") is supported;
  the improvement claim is not, and is not made.
- Thresholds are 0 to isolate retrieval, so answerability gating is not exercised; the timing
  includes one model load per process, which is a real part of a single-shot CLI run and a
  negligible part of a long-running server.
- Wall times are this CPU-only host and should be read as a ratio, not a benchmark.
