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

`RERANK_COVERAGE_LAMBDA` (**default `auto`**) in `src/laya_rag/config.py`, wired into
`RagEngine.query`: when a cross-encoder is configured and the value resolves positive, the
candidate pool is reordered by `_coverage_select_from` using the cross-encoder score as
relevance, and the reranker preserves that order (`order="recall"`) instead of re-sorting
by the score. `auto` resolves to the measured **0.7 when a `RERANK_MODEL` is set** and to
`0.0` otherwise — the same shape `DENSE_WEIGHT` and `RERANK_ORDER` already use — so a caller
who configures a second stage gets the measured-best configuration without setting anything
else; an explicit `0.0` still disables it.

## Document quota, and the combined rule

The same cached pool also tests the "one article dominates the top-10" failure mode
directly. `docquota-n` keeps the CE order but forces the first `n` slots onto distinct
articles; `covquota-3-l` combines that with atom coverage (three article seeds, then
coverage fills).

| strategy | Hits@10 | Hits@4 | MAP@10 |
| --- | --- | --- | --- |
| ce | 0.8600 | 0.7225 | 0.2986 |
| docquota-3 | 0.8725 | 0.7375 | 0.3187 |
| coverage-0.7 | 0.8850 | 0.7600 | 0.3206 |
| **covquota-3-0.7** | **0.8850** | 0.7550 | **0.3280** |
| covquota-3-0.5 | 0.8750 | 0.7575 | 0.3277 |

`docquota-3` is +0.0201 [+0.0142, +0.0262] over CE (p=0.001). The combined rule is best,
and it is **significantly better than coverage alone**: MAP@10 +0.0074 [+0.0006, +0.0140],
p=0.026, with a small Hits@4 change (−0.0050, ns). Against CE it is +0.0293
[+0.0235, +0.0358], p=0.001, Hits@4 +0.0325 [+0.0050, +0.0650].

Shipped as `RERANK_COVERAGE_MIN_DOCS` (default `0`), used with
`RERANK_COVERAGE_LAMBDA=0.7`; `0` keeps the plain coverage rule, so existing behaviour is
unchanged unless it is set.

**This is the item with the most overfitting risk**: ~20 strategies were compared on the
same 400 queries, and +0.0074 is inside the noise band of a 400-query draw. It is the
reason the full 2,255-query confirmation is the next step rather than an afterthought —
if `covquota-3-0.7` does not hold against both `ce` and `coverage-0.7` there, the honest
shipped setting is `RERANK_COVERAGE_MIN_DOCS=0`.

## A marginal negative: query-aware atom weighting

Plain coverage weights an atom by its length, so a long generic sentence counts as much as
a short sentence that names the thing asked about. Weighting each atom by
`1 + qboost * (query-term overlap)` and re-running the same greedy rule gives, on the
400-query subset:

| strategy | Hits@10 | Hits@4 | MAP@10 |
| --- | --- | --- | --- |
| coverage-0.7 | 0.8850 | 0.7600 | 0.3206 |
| coverage-q-1.0 | 0.8875 | 0.7650 | 0.3202 |
| coverage-q-3.0 | 0.8925 | 0.7650 | 0.3238 |

`coverage-q-3.0` is +0.0033 MAP@10 [−0.0015, +0.0080] over coverage-0.7 — **not
significant** (p=0.17), and Hits@4 is flat. It is recorded as a marginal change rather than
shipped; the honest reading is that the query's own terms are already what the cross-encoder
scored, so re-weighting atoms by them adds little. The tooling stays in the experiment as
`coverage-q-*`.

## A negative: the lexical bridge-linked rescue

The diagnostic (§3.1 of `RESEARCH.md`) predicted that the second hop is what the top-10
discards, and the research proposed a "reasoning-preserving" rescue: keep relevance seeds,
then readmit low-relevance chunks lexically linked to a seed through the bridge entity.
The same cache makes that measurable, and on MultiHop-RAG **it does not work**:

| strategy | Hits@10 | Hits@4 | MAP@10 |
| --- | --- | --- | --- |
| ce | 0.8600 | 0.7225 | 0.2986 |
| rescue-0.3 | 0.8025 | 0.7025 | 0.2786 |
| rescue-0.5 | 0.7725 | 0.6975 | 0.2714 |
| rescue-0.7 | 0.7550 | 0.6925 | 0.2682 |

Every weight is **significantly worse than the plain cross-encoder** on MAP@10
(rescue-0.3 −0.0200 [−0.0231, −0.0168], p=0.001; rescue-0.5 and 0.7 larger), and well
below coverage-0.7 (0.3206). The mechanism is plausible — a candidate sharing a rare term
with a selected seed is a proxy for the bridge — but on news text the proxy is too noisy:
shared rare terms are usually topic words, so the rescue spends slots on off-topic
candidates and drops real evidence. redhop's own caveat (lexical linkage misses paraphrase
bridges, and retention is not answer quality) is the right reading; a rescue that depends
on a validated grounding signal cannot be ported as raw term overlap.

`rescue_order` and its three strategies stay in the experiment tool as a recorded negative,
not in the engine.

## Full-protocol confirmation (2,255 queries)

The subset results were re-run over the **full** MultiHop-RAG answerable set by caching
cross-encoder scores for all 2,255 queries (`--limit-queries 2255`, ~53 minutes on this
host). The control reproduces the repo's known fused baseline exactly (MAP@10 0.2546).

| strategy | Hits@10 | Hits@4 | MAP@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| fused | 0.8120 | 0.6825 | 0.2546 | 0.5482 |
| ce | 0.8576 | 0.7348 | 0.2948 | 0.6106 |
| coverage-0.5 | 0.8914 | 0.7619 | 0.3138 | 0.6193 |
| **coverage-0.7** | **0.8971** | **0.7694** | 0.3165 | 0.6255 |
| docquota-3 | 0.8639 | 0.7472 | 0.3099 | 0.6127 |
| covquota-3-0.7 | 0.8958 | 0.7588 | **0.3192** | 0.6193 |

Paired over 2,255 queries:

| comparison | delta | 95% CI | permutation p |
| --- | --- | --- | --- |
| coverage-0.7 vs ce MAP@10 | **+0.0217** | [+0.0190, +0.0242] | 0.0005 |
| coverage-0.7 vs ce Hits@4 | **+0.0346** | [+0.0235, +0.0457] | 0.0005 |
| covquota-3-0.7 vs ce MAP@10 | **+0.0244** | [+0.0219, +0.0270] | 0.0005 |
| covquota-3-0.7 vs ce Hits@4 | **+0.0239** | [+0.0124, +0.0355] | 0.0005 |
| covquota-3-0.7 vs coverage-0.7 MAP@10 | +0.0027 | [+0.0002, +0.0052] | 0.0370 |
| covquota-3-0.7 vs coverage-0.7 Hits@4 | −0.0106 | [−0.0195, −0.0018] | 0.0265 |

The coverage win is **robust**: on the full protocol it is a significant MAP@10 and Hits@4
gain over the cross-encoder re-sort, at both lambda 0.5 and 0.7. The document quota is
real but small, and it is a genuine trade — **+0.0027 MAP@10 for −0.0106 Hits@4** against
coverage alone.

**Choose-on-half, report-on-half** (seed 17, 1,127 / 1,128 queries): selection on half A
picks `covquota-3-0.7`; on the untouched half B it scores MAP@10 **0.3302**, against `ce`
0.3046 and `coverage-0.7` 0.3262. So the combined rule's held-out MAP beats both
baselines, and the small MAP edge over coverage alone survives out of sample.

**Recommended shipped setting:** `RERANK_COVERAGE_LAMBDA=0.7` (robust MAP and Hits@4).
`RERANK_COVERAGE_MIN_DOCS=3` is the MAP-priority option — it adds a small significant
MAP@10 and costs Hits@4, so it is left off by default while the project's headline
strength is its Hits@4/Hits@10.

## Depth: the engine's `RETRIEVE_K=20` is enough

The experiments above score a depth-**100** pool, but the engine's default first stage is
`RETRIEVE_K=20`, so the shipped path might have been leaving the measured gain on the table
— at five times the cross-encoder scoring cost. `--depth-limit` truncates the cached pool
and re-scores without re-running the cross-encoder, which tests exactly that.

Depth 100 minus depth 20, MAP@10 (paired, n=400):

| strategy | delta | 95% CI | p |
| --- | --- | --- | --- |
| ce | −0.0010 | [−0.0124, +0.0106] | 0.87 |
| coverage-0.5 | −0.0005 | [−0.0135, +0.0125] | 0.93 |
| coverage-0.7 | −0.0045 | [−0.0166, +0.0074] | 0.46 |
| covquota-3-0.7 | −0.0063 | [−0.0187, +0.0058] | 0.32 |

**None is significant**, so the default `RETRIEVE_K=20` is kept: a cross-encoder plus
coverage does not need a five-times wider scoring pass. Hit@4 point values are actually
*mixed* (coverage-0.5 scores 0.7825 at depth 20 against 0.7450 at depth 100), but per-query
Hits@4 is not stored in the report, so that observation is untested and not claimed.

This is the useful shape of a negative: a plausible five-fold cost increase was measured
before being shipped, and the measurement said no.

## Honest limits

- The subset experiment was confirmed on the **full 2,255-query protocol** (above), and the
  document-quota's small edge also survives the held-out half. The lambda value itself was
  chosen on the 400-query subset, but coverage at 0.5 and 0.7 is positive on the full set,
  so the choice is not knife-edge.
- It **requires a cross-encoder**: there is no CE score to select on otherwise, so it
  does not help the shipped lexical-only default.
- It cannot reach the diagnostic's 0.73 oracle ceiling; coverage-0.7 closes roughly a
  third of the gap from CE to that ceiling, not all of it.
