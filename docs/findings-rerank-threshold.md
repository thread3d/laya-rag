# Findings: a Laya-calibrated threshold was cutting the cross-encoder's scores

The engine applied `RELEVANCE_THRESHOLD` (0.5) to whatever scores the second stage supplied.
That threshold is documented as meaningful **because Laya emits calibrated probabilities** —
but when a cross-encoder substitutes for Laya, the 0.5 cut lands on a *different model's*
sigmoid, and it is far too strict. The experiments that measured the second stage never
thresholded it at all, so the shipped CE path could not have reproduced its own numbers.

## The measurement

The cached cross-encoder scores are raw logits, so `--score-threshold 0.0` models the
engine's `sigmoid >= 0.5` cut exactly and omitting it is what the experiments did. At the
engine's depth (`RETRIEVE_K=20`), coverage-0.7, 400 queries:

| | MAP@10 | Hits@10 | Hits@4 |
| --- | --- | --- | --- |
| no cut (the experiments) | **0.3161** | **0.8750** | **0.7750** |
| with the 0.5 cut (the old engine) | 0.2359 | 0.5725 | 0.6150 |

Paired, removing the cut is worth **+0.0801 MAP@10 [+0.0674, +0.0936], p=0.0002**. It is not
a subtle effect: Hits@10 falls by **0.30**, because with the cut the engine returns little or
nothing for many queries.

Why it is so severe: over the top-100 CE pool only **11.5%** of candidates clear the cut;
within the engine's top-20 pool the median is **4** positive candidates and **83 of 400
queries have zero**. Those queries short-circuit to "nothing cleared relevance", which the
engine honestly reports as unanswerable — but the evidence was there and the cross-encoder
had ranked it.

## The fix

`RERANK_THRESHOLD` (default `0.0`) is used whenever a **substituted** scorer supplies the
scores; Laya's own path keeps `RELEVANCE_THRESHOLD`, and an explicit
`query(relevance_threshold=...)` still overrides both. Keeping the whole pool and letting the
ordering and `top_k` decide is exactly what the measured configuration did.

## Honest limits

- The cost is **simulated** from the cached logits (monotone with the engine's sigmoid) at
  depth 20 on the 400-query subset, not measured by running the engine with a real
  cross-encoder. The simulation is faithful to the cut but is not itself the engine's row.
- It says nothing about the *right* threshold for a given cross-encoder; `0.0` (keep the
  pool) is the measured configuration here, not a universal calibration. A deployment that
  wants a cut should tune `RERANK_THRESHOLD` on its own data.
