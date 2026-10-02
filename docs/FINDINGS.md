# Findings index

Every measurement the improvement programme produced, with the headline result and the
document that carries the detail. Each document states what was measured, how to reproduce
it, and what it does **not** show.

Two kinds of entry are recorded: **shipped** changes that improved the system, and
**negatives** that close a direction off. The negatives matter as much as the wins — they
are what stops the next pass from re-trying them.

## Shipped improvements

| area | headline result | document |
| --- | --- | --- |
| Multi-hop selection | The MAP@10 gap is **selection, not recall**: the fused top-100 oracle ceiling is 0.7307 while recall reaches 1.000 by depth 500; hop-2 recall is 0.24 at depth 10 | [findings-multihop-diagnostics.md](findings-multihop-diagnostics.md) |
| Fact recall (partial credit) | The shipped coverage config covers **55.2%** of gold facts in its top-10 vs 46.4% for the baseline (+0.0880 [+0.0797, +0.0964], p<0.0001) | [findings-multihop-diagnostics.md](findings-multihop-diagnostics.md) |
| Cross-encoder × coverage | Coverage over the CE-scored pool beats a full CE re-sort on the full protocol: MAP@10 **0.2948 → 0.3165**, Hits@4 0.7348 → 0.7694; `auto` enables 0.7 when a CE is configured | [findings-rerank-selection.md](findings-rerank-selection.md) |
| Fused coverage objective | Dropping the dense-gram redundancy term at λ=0.5: MAP@10 0.2835 → **0.2850**, Hits@4 0.7197 → 0.7326 | [findings-rerank-selection.md](findings-rerank-selection.md) |
| Fusion constant | The engine fused at `k=60` while every documented MultiHop number used `k=5`; `FUSION_K` now defaults to 5 (MAP@10 +0.0040 [+0.0012, +0.0069], p=0.008; BEIR +0.004–0.009) | [findings-fusion-k.md](findings-fusion-k.md) |
| Second-stage threshold | The engine cut a substituted cross-encoder at Laya's 0.5; `RERANK_THRESHOLD` now keeps the pool, worth **+0.0801 MAP@10** [+0.0674, +0.0936] (p=0.0002) and +0.30 Hits@10 | [findings-rerank-threshold.md](findings-rerank-threshold.md) |
| Semantic cache | Bounded LRU of query embeddings (`QUERY_CACHE_SIZE`); repeated questions skip their one dense cost, and it caches vectors, never results | [findings-lexical-backend.md](findings-lexical-backend.md) |
| Lexical backend | Eager BM25 has **identical ranking** (200/200 and 300/300 parity) at **11–13×** the query speed and **~5×** smaller index | [findings-lexical-backend.md](findings-lexical-backend.md) |
| Evaluation rigor | Paired permutation + BH-FDR (16/32 cells survive) and a selective-prediction suite: the gate's false-accept rate is 40.7% at 80% coverage vs the scalar's 51.0% | [findings-evaluation.md](findings-evaluation.md) |
| Answerability transfer | The 0.815 AUC is **within-question-type**; leave-one-type-out gives **0.7666** [0.6712, 0.8267], worst on comparison questions (0.6640) | [findings-evaluation.md](findings-evaluation.md) |
| Trust tiers + tenant ACL | Deterministic path tiers and a per-tenant allow-list enforce isolation fail-closed; `adjudicate_claims` applies the same filter as `query` | [findings-trust.md](findings-trust.md) |
| Adversarial audit | Nine deterministic checks (poison exclusion, gold retention, negative rejection, injection flags, tenant isolation) run in `make adversarial-audit --check` | [findings-adversarial-audit.md](findings-adversarial-audit.md) |
| Structural context prefix | Indexing a breadcrumb lifts MRR **0.6429 → 0.8333** on the vendored docs, with no query worse | [findings-context-prefix.md](findings-context-prefix.md) |
| BRIGHT retrieval | Six subsets, macro mean nDCG@10: hybrid **0.1436** > BM25 0.1324 > dense 0.0991; dense alone is significantly worse than BM25 on pony and biology | [findings-bright.md](findings-bright.md) |

## Recorded negatives

| idea | result | document |
| --- | --- | --- |
| Bridge-linked second-hop rescue | Worse than the cross-encoder on every weight (MAP@10 0.2786/0.2714/0.2682 vs 0.2986) | [findings-rerank-selection.md](findings-rerank-selection.md) |
| Query-aware atom weighting | +0.0033 MAP@10, p=0.17 — not significant; the CE already scored the query's terms | [findings-rerank-selection.md](findings-rerank-selection.md) |
| Engineered gate features | Nine derived columns move every metric the wrong way (AUC 0.8146 → 0.8106) | [findings-answerability-features.md](findings-answerability-features.md) |
| Atomic lexical channel on BEIR | Lowers nDCG@10 on all five datasets, significantly on three (fiqa −0.0528) | [findings-atom-channel-beir.md](findings-atom-channel-beir.md) |
| Atom-level lexical retrieval (MultiHop alone) | Worse than chunk BM25 on every metric; only useful fused, and only for Hits@4 | [findings-atom-index.md](findings-atom-index.md) |
| Widening `RETRIEVE_K` for the CE path | A depth-100 pool is not significantly better than depth 20 for CE or coverage (all MAP deltas p>0.3), so the 5× scoring cost was not taken | [findings-rerank-selection.md](findings-rerank-selection.md) |
| Coverage as a default | The coverage objective is a **multi-fact instrument**: it costs 0.02–0.08 nDCG@10 on every BEIR dataset measured (ArguAna 0.560 → 0.087 at λ=0.5, all significant), so it stays off by default | [findings-coverage-scope.md](findings-coverage-scope.md) |
| RGB negative rejection | Not adapted: no clean no-answer label in the English files and a judge-based official metric | [../BENCHMARKS.md](../BENCHMARKS.md) |

## Method notes worth carrying forward

- **Pin the chunking constant.** Up to +0.067 Hits@10 comes from an undeclared overlap.
- **Report the pool-reorder ceiling and fact recall** beside any reranking result, so a
  selection gain is never confused with a recall gain.
- **Use grouped splits** for transfer claims; a random half reuses the same question types.
- **Label every generator-attributed number** with its model id.
- **A metric-contract bug is silent.** `official_metrics` takes a list of per-query ranked
  lists; a flat list scored everything zero including the control, which is why the metric
  path is now a single tested helper.
