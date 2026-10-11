# The fusion ablation: RRF against a convex combination

The audit asks, before any fusion gain is claimed, to "ablate `FUSION_K` ∈ {5, 10, 60} against a
**tuned convex combination**". `FUSION_MODE=convex` made the convex arm runnable in the engine; this
runs the comparison in the BEIR replica, where the cached SciFact embeddings make it cheap and the
per-query scores are persisted for a paired test.

## What was run

SciFact, 300 queries, `bge-small` cached embeddings, `top_k=100`, the replica's RRF at its default
`--rrf-k 60`, max-normalised CombSUM (`--combsum`) and min-max normalised convex (`--convex`).
Differences are paired bootstraps over the per-query nDCG@10 the report persists (10,000 resamples).

```sh
.venv/bin/python tools/beir.py --datasets scifact \
    --retrievers biencoder,bm25-engine,hybrid-biencoder --combsum --convex \
    --out eval/beir/convex-scifact
```

## The numbers

| arm | nDCG@10 |
| --- | --- |
| `bm25-engine` | 0.6762 |
| `biencoder` (dense) | 0.7200 |
| `hybrid-combsum` (max-normalised) | 0.7149 |
| `hybrid-biencoder` (RRF, k=60) | 0.7320 |
| **`hybrid-convex` (min-max)** | **0.7412** |

Paired, convex minus each baseline:

| comparison | delta | 95% CI | p |
| --- | --- | --- | --- |
| convex − RRF (k=60) | **+0.0092** | [−0.0029, +0.0217] | 0.139 |
| convex − CombSUM (max) | **+0.0263** | [+0.0118, +0.0415] | 0.0000 |
| convex − dense | +0.0212 | [−0.0024, +0.0447] | 0.078 |

## Reading

The convex combination is the **best single arm** here, and it is **significantly better than
max-normalised CombSUM** (+0.0263) — so the fusion rule is not neutral, which is the part of the
audit's caution that survives. But its lead over RRF is **+0.0092 and not significant** (p=0.139),
so this is not evidence that min-max convex should replace RRF. The honest summary is: CombSUM's
max-normalisation is the wrong normalisation, and RRF is not clearly beaten by the convex rule.

Two things make this a weak confirmation rather than a verdict. The RRF arm here is the replica's
`k=60`, while the engine ships `FUSION_K=5`; the repository's own `FUSION_K` finding is that `5`
beats `60` on BEIR by +0.004–0.009, so the engine's RRF is probably closer to convex than this
table shows. And the convex `dense_weight`/`lexical_weight` are **not tuned** — a tuned convex
combination is exactly what the audit asked for and is not what this run used.

## Superseded, in part, by the tuned sweep

Every "what this does not show" below has since been closed or narrowed — see
[`docs/findings-fusion-tuning.md`](findings-fusion-tuning.md), which tunes the weights on a seeded
half and reads them on the other half, against the **`k=5` the engine ships**, on six corpora:

- ArguAna **+0.0266** [+0.0133, +0.0403], FiQA **+0.0296** [+0.0177, +0.0413] and Quora **+0.0215**
  [+0.0186, +0.0245] over 10,000 queries are significant wins over RRF `k=5`;
- SciFact (+0.0004) and SciDocs (+0.0056) are ties, and NFCorpus is a −0.0038 near-negatives;
- a single **global** dense weight of 2.0 beats the shipped RRF by **+0.0127** on the held-out
  halves.

So the +0.0092 against `k=60` above was part `k` and part fusion rule. The table on this page
stands as the record of the untuned comparison; the tuned numbers are the ones to quote.

## What this does not show

- **One corpus.** SciFact only; no NFCorpus, no MultiHop. The audit's MultiHop `FUSION_K` arms are
  separately recorded (`eval/beir/rrf-*.json`), but the convex arm is not there yet.
- **Untuned weights**, as above.
- **RRF at k=60**, not the engine default k=5.
- **The engine's own path is not what was measured.** This is the `tools/beir.py` replica; the
  engine-level convex number would need `tools/engine_beir.py` with `FUSION_MODE=convex`.
