# Findings: coverage selection over the cross-encoder pool beats a CE re-sort

This records the Tier 1.2 experiment from [`RESEARCH.md`](../RESEARCH.md): combining the
cross-encoder's relevance with the sentence-atom coverage objective. It is the first
retrieval change here measured to beat the previously best MultiHop configuration.

Reproduce (the first step is the expensive cross-encoder run, ~10 minutes on this host;
the second is pure NumPy over its cache):

```sh
make multihop-ce-cache          # 400 queries x top-100, bge-reranker-base
make multihop-rerank-experiment
```

## Setup

The deterministic 400-query subset `tools/multihop_official.py --limit-queries 400
--sample-seed 17` uses, at the official `overlap=200` chunking. Every strategy ranks the
**same** fused top-100 per query, so the only difference is the selection rule. The
experiment reproduces the two known rows exactly — fused MAP@10 0.2522 and full CE re-sort
0.2986 — which is the control that says the harness is measuring what it claims.

| strategy | Hits@10 | Hits@4 | MAP@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| fused (first stage) | 0.7975 | 0.6625 | 0.2522 | 0.5342 |
| ce (full re-sort) | 0.8600 | 0.7225 | 0.2986 | 0.6010 |
| blend-0.5 | 0.8425 | 0.7175 | 0.2928 | 0.5986 |
| blend-0.75 | 0.8625 | 0.7375 | 0.3078 | 0.6188 |
| guardrail-1 (pin rank 1) | 0.8600 | 0.7375 | 0.2893 | 0.5684 |
| coverage-0.3 | 0.8500 | 0.7425 | 0.2999 | 0.5842 |
| coverage-0.5 | 0.8725 | 0.7450 | 0.3201 | 0.6134 |
| **coverage-0.7** | **0.8850** | **0.7600** | **0.3206** | 0.6125 |

`coverage-l` is greedy sentence-atom max-coverage using the cross-encoder score as the
relevance term: at each step a candidate scores `l * normalized_ce + (1 - l) *
marginal_new_atom_weight`. It targets the metric's definition — MAP@10 credits *distinct
facts found early*, so spending a top-10 slot on a second near-duplicate of a kept
passage is waste.

## Significance (paired, over the same 400 queries)

Against the full CE re-sort — the previous best:

| comparison | delta | 95% CI | permutation p |
| --- | --- | --- | --- |
| coverage-0.7 MAP@10 | **+0.0220** | [+0.0165, +0.0286] | 0.0010 |
| coverage-0.7 Hits@4 | **+0.0375** | [+0.0125, +0.0650] | 0.0040 |
| coverage-0.5 MAP@10 | +0.0215 | [+0.0126, +0.0314] | 0.0010 |
| coverage-0.3 MAP@10 | +0.0012 | [−0.0107, +0.0137] | 0.84 (ns) |
| blend-0.75 MAP@10 | +0.0091 | [−0.0025, +0.0204] | 0.12 (ns) |

Against the fused first stage, coverage-0.7 is MAP@10 **+0.0684** [+0.0488, +0.0854] and
Hits@4 **+0.0975** [+0.0550, +0.1401], both p=0.001.

So the win is not "the cross-encoder is good" — that was already known and measured. It is
that **an atom-coverage selection over the CE-scored pool is significantly better than
trusting the CE order**, on MAP@10 and Hits@4 at once. The rank-1 guardrail did *not* beat
the plain CE order on MAP@10 (0.2893, ns), and the blends only matched it.

## What shipped

`RERANK_COVERAGE_LAMBDA` (default `0.0`) in `src/laya_rag/config.py`, wired into
`RagEngine.query`: when a cross-encoder is configured and the value is positive, the
candidate pool is reordered by `_coverage_select_from` using the cross-encoder score as
relevance, and the reranker preserves that order (`order="recall"`) instead of re-sorting
by the score. It is off by default, so nothing changes unless it is set; the measured
setting is **0.7**.

## Honest limits

- This is the **400-query subset**, not the full 2,255-query protocol, and lambda was
  chosen on it. Coverage at 0.3/0.5/0.7 is positive-to-neutral against CE, so the choice
  is not knife-edge, but a held-out confirmation on the full set is still owed.
- It **requires a cross-encoder**: there is no CE score to select on otherwise, so it
  does not help the shipped lexical-only default.
- It cannot reach the diagnostic's 0.73 oracle ceiling; coverage-0.7 closes roughly a
  third of the gap from CE to that ceiling, not all of it.
