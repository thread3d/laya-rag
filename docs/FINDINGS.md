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
| Multi-hop selection | The MAP@10 gap is **recall *and* selection**: the corrected fused top-100 ceiling is **0.6088** and **rises** to 0.6592 at depth 500, while **35.4%** of depth-100 pools (799 of 2,255) are missing a gold fact and recall still reaches only **0.9811** at depth 500; hop-2 recall is 0.2407 at depth 10 | [findings-multihop-diagnostics.md](findings-multihop-diagnostics.md) |
| Fact recall (partial credit) | The shipped coverage config covers **55.2%** of gold facts in its top-10 vs 46.4% for the baseline (+0.0880 [+0.0797, +0.0964], p<0.0001) | [findings-multihop-diagnostics.md](findings-multihop-diagnostics.md) |
| Cross-encoder × coverage | Coverage over the CE-scored pool beats a full CE re-sort on the full protocol: MAP@10 **0.2948 → 0.3165**, Hits@4 0.7348 → 0.7694; `auto` enables 0.7 when a CE is configured | [findings-rerank-selection.md](findings-rerank-selection.md) |
| Fused coverage objective | Dropping the dense-gram redundancy term at λ=0.5: MAP@10 0.2835 → **0.2850**, Hits@4 0.7197 → 0.7326 | [findings-rerank-selection.md](findings-rerank-selection.md) |
| Fusion constant | The engine fused at `k=60` while every documented MultiHop number used `k=5`; `FUSION_K` now defaults to 5 (MAP@10 +0.0040 [+0.0012, +0.0069], p=0.008; BEIR +0.004–0.009) | [findings-fusion-k.md](findings-fusion-k.md) |
| Learned-sparse channel | SPLADE-style sparse now ships as a third engine channel: measured +0.03–0.14 over BM25 alone and +0.004–0.023 over two-channel fusion on BEIR (SciFact 0.741 → **0.745**, NFCorpus 0.367 → **0.372**, ArguAna 0.568 → **0.591**); opt-in via `SPARSE_MODEL` | [findings-sparse-channel.md](findings-sparse-channel.md) |
| Engine-level BEIR | The engine itself scores SciFact nDCG@10 **0.7264** and **0.7443** with sparse (**+0.0179 [+0.0028, +0.0337]**, p=0.025) — the channel's gain transfers and is larger than the replica's, while the `tools/beir.py` replica **overstates the engine by 0.0144** | [findings-engine-parity.md](findings-engine-parity.md) |
| Engine-level MultiHop | The 256-token harness **understates** the engine, the opposite sign to BEIR: on the seed-17 400-query subset the real engine is **+0.0535 MAP@10 [+0.0387, +0.0680]** over the harness hybrid first stage and level with its reranked row (**+0.0071, ns**), on corpus-aware 2,000-character chunking and no cross-encoder; it remains below the subset's coverage-plus-quota bar of **0.3280** | [findings-engine-multihop.md](findings-engine-multihop.md) |
| Engine-level MuSiQue | **The MultiHop win does not travel.** On the first 200 validation queries the engine is **−0.0097 MAP@10 [−0.0190, −0.0007]** against the harness's fused first stage, and raising its pool from the default 20 to the harness's depth 100 moves MAP@10 by only **+0.0019** — so pool depth is not the cause. It still beats the harness's `ce` (0.4634) and `covquota` (0.4529) on the same subset | [findings-engine-musique.md](findings-engine-musique.md) |
| Cross-encoder + coverage at the engine | The auto coverage default is **safe on BEIR** — re-verified under the corpus-aware chunking default: 0.7417 with and without the CE (p=0.99) — and **4.3× faster** (167 s against 709 s), because the 322M decision model then only gates the kept passages | [findings-engine-cross-encoder.md](findings-engine-cross-encoder.md) |
| Corpus-aware chunk size | The engine's chunking gap was the **chunked representation**: deriving the size from the corpus (p90, clamped to 2,000) gives **+0.0153 nDCG@10** (0.7264 → 0.7417, p=0.023, 40% fewer chunks) on SciFact and MRR **0.6429 → 0.7143** on the vendored docs; `CHUNK_CHARS_AUTO` now defaults on | [findings-chunk-merge.md](findings-chunk-merge.md) |
| Second-stage threshold | The engine cut a substituted cross-encoder at Laya's 0.5; `RERANK_THRESHOLD` now keeps the pool, worth **+0.0801 MAP@10** [+0.0674, +0.0936] (p=0.0002) and +0.30 Hits@10 | [findings-rerank-threshold.md](findings-rerank-threshold.md) |
| Semantic cache | Bounded LRU of query embeddings (`QUERY_CACHE_SIZE`); repeated questions skip their one dense cost, and it caches vectors, never results | [findings-lexical-backend.md](findings-lexical-backend.md) |
| Lexical backend | Eager BM25 has **identical ranking** (200/200 and 300/300 parity) at **11–13×** the query speed and **~5×** smaller index | [findings-lexical-backend.md](findings-lexical-backend.md) |
| Evaluation rigor | Paired permutation + BH-FDR (16/32 cells survive), a minimum-detectable-effect column (MAP@10 MDE ≈ 0.008–0.011 on 2,255 queries), and a selective-prediction suite: the gate's false-accept rate is 40.7% at 80% coverage vs the scalar's 51.0% | [findings-evaluation.md](findings-evaluation.md) |
| Answerability transfer | The 0.815 AUC is **within-question-type**; leave-one-type-out gives **0.7666** [0.6712, 0.8267], worst on comparison questions (0.6640) | [findings-evaluation.md](findings-evaluation.md) |
| Evidence-shape gate features | Three appended raw signals (max single-passage coverage, coverage spread, pairwise redundancy) improve the gate significantly on five of six metrics: AUC 0.8146 → **0.8290**, risk@80 0.4069 → **0.4009**; old nine-feature artefacts still load by name. They also **carry the gate across the chunking-default change**: the nine-feature gate falls to 0.7862 AUC while the twelve hold at 0.8323 | [findings-gate-shape-features.md](findings-gate-shape-features.md) |
| Trust tiers + tenant ACL | Deterministic path tiers and a per-tenant allow-list enforce isolation fail-closed; `adjudicate_claims` applies the same filter as `query` | [findings-trust.md](findings-trust.md) |
| Adversarial audit | Nine deterministic checks (poison exclusion, gold retention, negative rejection, injection flags, tenant isolation) run in `make adversarial-audit --check` | [findings-adversarial-audit.md](findings-adversarial-audit.md) |
| Structural context prefix | Indexing a breadcrumb lifts MRR **0.6429 → 0.8333** on the vendored docs, with no query worse | [findings-context-prefix.md](findings-context-prefix.md) |
| BRIGHT retrieval | Six subsets, macro mean nDCG@10: hybrid **0.1436** > BM25 0.1324 > dense 0.0991; dense alone is significantly worse than BM25 on pony and biology | [findings-bright.md](findings-bright.md) |
| Encoder distillation | The 3-model ensemble (478M) distils into one 33M `bge-small`: NFCorpus **0.3385 → 0.3505** (+0.0120 [+0.0022, +0.0216], significant), recovering ~29% of the ensemble's +0.0413 — adapted, not zero-shot | [findings-distillation.md](findings-distillation.md) |

## Recorded negatives

| idea | result | document |
| --- | --- | --- |
| Bridge-linked second-hop rescue | Worse than the cross-encoder on every weight (MAP@10 0.2786/0.2714/0.2682 vs 0.2986) | [findings-rerank-selection.md](findings-rerank-selection.md) |
| Query-aware atom weighting | +0.0033 MAP@10, p=0.17 — not significant; the CE already scored the query's terms | [findings-rerank-selection.md](findings-rerank-selection.md) |
| Engineered gate features | Nine derived columns move every metric the wrong way (AUC 0.8146 → 0.8106) | [findings-answerability-features.md](findings-answerability-features.md) |
| Atomic lexical channel on BEIR | Lowers nDCG@10 on all five datasets, significantly on three (fiqa −0.0528) | [findings-atom-channel-beir.md](findings-atom-channel-beir.md) |
| Atom-level lexical retrieval (MultiHop alone) | Worse than chunk BM25 on every metric; only useful fused, and only for Hits@4 | [findings-atom-index.md](findings-atom-index.md) |
| Widening `RETRIEVE_K` for the CE path | A depth-100 pool is not significantly better than depth 20 for CE or coverage (all MAP deltas p>0.3), so the 5× scoring cost was not taken | [findings-rerank-selection.md](findings-rerank-selection.md) |
| Coverage and dedup as defaults | Both fused-pool rewriters are **corpus-shape instruments**: coverage costs 0.02–0.08 nDCG@10 and dedup 0.002–0.51 on every BEIR dataset measured (all significant; ArguAna 0.560 → 0.087), so both stay off by default | [findings-coverage-scope.md](findings-coverage-scope.md) |
| Gate threshold and calibration | A threshold learned per split averages 0.470 but buys **+0.0004** balanced accuracy (the optimum is flat), and temperature scaling makes ECE/Brier/NLL worse — so `0.5` and the raw probability stay | [findings-gate-calibration.md](findings-gate-calibration.md) |
| Document quota as a ranking gain | `RERANK_MIN_DOCS` is **exactly neutral** on document-level nDCG (identical per-query series over 300 SciFact queries) — it guarantees prefix diversity, not a better ranking; it stays off by default | [findings-document-quota.md](findings-document-quota.md) |
| BM25 parameters | The engine ran Lucene `k1=1.2, b=0.75` while every BEIR row used Anserini `0.9/0.4`, and could not be configured; at the engine level the two are **indistinguishable** (0.7264 vs 0.7238, p=0.45), so the default stays and `BM25_K1`/`BM25_B` now exist to reproduce the baseline exactly | [findings-bm25-parameters.md](findings-bm25-parameters.md) |
| Chunking-gap hypotheses | Three targeted fixes for the engine's 0.013 chunking gap all fail — merging documents ≤ `chunk_chars+overlap` (0.7236 vs 0.7264), doubling `RETRIEVE_K` (0.7273), indexing a context breadcrumb (0.7243) — while whole-document chunking reaches 0.7396, so the loss is the **chunked representation**, not its selection | [findings-chunk-merge.md](findings-chunk-merge.md) |
| RGB negative rejection | Not adapted: no clean no-answer label in the English files and a judge-based official metric | [../BENCHMARKS.md](../BENCHMARKS.md) |
| Late-interaction ordering head (`answerai-colbert-small-v1` MaxSim) | Worse than the unordered first stage: **−0.0430 MAP@10 [−0.0632, −0.0229]** against `fused`, −0.0894 against the CE order, −0.1016 against covquota. Reordering the CE's *own* top-10 by MaxSim — recall fixed, Hits@10 unchanged — still costs **−0.0257 [−0.0434, −0.0079]**, so the +0.1254 ordering headroom is not reachable by this instrument | [findings-ordering-head.md](findings-ordering-head.md) |

## Method notes worth carrying forward

- **Pin the chunking constant.** Up to +0.067 Hits@10 comes from an undeclared overlap.
- **Test the caller, not just the callee.** `FUSION_K` was fixed in `_fuse` and the config
  while the primary `query()` call site kept not passing it; a spy on `_fuse` from `query()`
  is what proves the wiring. The same miss hit the coverage-redundancy flag.
- **A replica is not the system.** `tools/beir.py` reimplements the channels and scores whole
  documents, overstating the real engine by **0.0144** on SciFact; `tools/engine_beir.py`
  runs the engine itself when a claim is about the engine.
- **Improvements do not compose additively.** The sparse channel is worth **+0.0179** (p=0.025)
  under fixed chunking but only **+0.0048** (ns) once corpus-aware chunking is the default —
  both recover the same chunked-representation loss. Report each increment against the current
  default, never as a sum.
- **Engine-level coverage is uneven, and the multi-hop half is now measured.** `tools/engine_beir.py`
  measures the engine itself on BEIR. The multi-hop rows came only from the 256-character harness
  (`tools/multihop_official.py`) until `tools/engine_multihop.py` ran the real engine under its own
  corpus-aware chunking on the harness's own 400-query seed-17 subset: the engine is **+0.0535
  MAP@10 [+0.0387, +0.0680]** above the harness's hybrid first stage and level with its
  cross-encoder row (**+0.0071, ns**), so the replica *understates* the engine on MultiHop-RAG,
  the opposite of the +0.0144 overstatement on BEIR. The full 2,255-query run is still owed; see
  [findings-engine-multihop.md](findings-engine-multihop.md).
- **Verification.** `make test` is 746 offline tests and `make test-real` is 4 end-to-end tests
  against the real checkpoint (checkpoint load and embed, Laya relevance beating cosine on a
  probe, answerability gating, full pipeline retrieving the right file). Both pass under the
  current defaults.
- **Report the pool-reorder ceiling and fact recall** beside any reranking result, so a
  selection gain is never confused with a recall gain.
- **Use grouped splits** for transfer claims; a random half reuses the same question types.
- **Label every generator-attributed number** with its model id.
- **A metric-contract bug is silent.** `official_metrics` takes a list of per-query ranked
  lists; a flat list scored everything zero including the control, which is why the metric
  path is now a single tested helper.
