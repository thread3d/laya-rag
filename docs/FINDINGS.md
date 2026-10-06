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
| Engine-level MultiHop | The 256-token harness **understates** the engine, the opposite sign to BEIR: over the **full 2,255 queries** the engine is **+0.0499 MAP@10 [+0.0432, +0.0566]** with `bge-small` fused, and the **shipped default** (Laya encoder, dense weight 0, lexical only) is better still at **0.3128**, **+0.0618 [+0.0547, +0.0688]** over the harness. The dense channel is a measured **trade**: it costs **+0.0119 MAP@10** and **+0.0187 MRR@10** to leave off, and buys nothing significant in Hits@4/@10. The first full run read 0.2216 because it reused a **partial index** missing `lexical.json`, now a fixed engine bug | [findings-engine-multihop.md](findings-engine-multihop.md) |
| Engine-level MuSiQue | **The MultiHop win does not travel, and the full set confirms it.** Over all **2,417** validation queries the engine is **−0.0091 MAP@10 [−0.0120, −0.0062]** against the harness's fused first stage and below *every* harness arm (`blend-0.75` −0.0390, `ce` −0.0271, `covquota` −0.0223); the 200-query subset read −0.0097. Pool depth is not the cause (+0.0019). Together with MultiHop — where leaving the dense channel off is worth **+0.0119** — the dense channel's contribution **reverses between corpora**, which is the argument for routing rather than one global configuration | [findings-engine-musique.md](findings-engine-musique.md) |
| Cross-encoder + coverage at the engine | The auto coverage default is **safe on BEIR** — re-verified under the corpus-aware chunking default: 0.7417 with and without the CE (p=0.99) — and **4.3× faster** (167 s against 709 s), because the 322M decision model then only gates the kept passages | [findings-engine-cross-encoder.md](findings-engine-cross-encoder.md) |
| Corpus-aware chunk size | The engine's chunking gap was the **chunked representation**: deriving the size from the corpus (p90, clamped to 2,000) gives **+0.0153 nDCG@10** (0.7264 → 0.7417, p=0.023, 40% fewer chunks) on SciFact and MRR **0.6429 → 0.7143** on the vendored docs; `CHUNK_CHARS_AUTO` now defaults on | [findings-chunk-merge.md](findings-chunk-merge.md) |
| Second-stage threshold | The engine cut a substituted cross-encoder at Laya's 0.5; `RERANK_THRESHOLD` now keeps the pool, worth **+0.0801 MAP@10** [+0.0674, +0.0936] (p=0.0002) and +0.30 Hits@10 | [findings-rerank-threshold.md](findings-rerank-threshold.md) |
| Semantic cache | Bounded LRU of query embeddings (`QUERY_CACHE_SIZE`); repeated questions skip their one dense cost, and it caches vectors, never results | [findings-lexical-backend.md](findings-lexical-backend.md) |
| Lexical backend | Eager BM25 has **identical ranking** (200/200 and 300/300 parity) at **11–13×** the query speed and **~5×** smaller index | [findings-lexical-backend.md](findings-lexical-backend.md) |
| Evaluation rigor | Paired permutation + BH-FDR (16/32 cells survive), a minimum-detectable-effect column (MAP@10 MDE ≈ 0.008–0.011 on 2,255 queries), and a selective-prediction suite: the gate's false-accept rate is 40.7% at 80% coverage vs the scalar's 51.0% | [findings-evaluation.md](findings-evaluation.md) |
| Answerability transfer | The 0.815 AUC is **within-question-type**; leave-one-type-out gives **0.7666** [0.6712, 0.8267], worst on comparison questions (0.6640) | [findings-evaluation.md](findings-evaluation.md) |
| Answerability calibration row | The first reliability diagram for the engine's answerability decision, and the case for the unclaimed axis: on MultiHop-RAG's 601 labelled queries the **shipped scalar is at chance and miscalibrated** — AUC **0.5087**, ECE **0.3085**, Brier **0.3650**, risk@80 **0.4927** — while the held-out feature gate is calibrated in-domain, ECE **0.0650** vs the scalar's 0.2714 and Brier 0.1768 vs 0.3678. The lowest probability bin is right 49.5% of the time and the highest 47.0%. The boundary is stated: no other corpus labels a negative class, so this is not yet a transfer result | [findings-calibration-row.md](findings-calibration-row.md) |
| Evidence-shape gate features | Three appended raw signals (max single-passage coverage, coverage spread, pairwise redundancy) improve the gate significantly on five of six metrics: AUC 0.8146 → **0.8290**, risk@80 0.4069 → **0.4009**; old nine-feature artefacts still load by name. They also **carry the gate across the chunking-default change**: the nine-feature gate falls to 0.7862 AUC while the twelve hold at 0.8323 | [findings-gate-shape-features.md](findings-gate-shape-features.md) |
| Trust tiers + tenant ACL | Deterministic path tiers and a per-tenant allow-list enforce isolation fail-closed; `adjudicate_claims` applies the same filter as `query` | [findings-trust.md](findings-trust.md) |
| Adversarial audit | Nine deterministic checks (poison exclusion, gold retention, negative rejection, injection flags, tenant isolation) run in `make adversarial-audit --check` | [findings-adversarial-audit.md](findings-adversarial-audit.md) |
| Adversarial noise + injection variants | The owed variants, with frozen seeds: **12 checks** now pass. Vocabulary-sharing noise leaves negative rejection at **1.0000** at every dose and gold retention at **1.0000** to 20 documents, falling to **0.6000** at 100 — a density curve, not a cliff; **seven injection payloads** covering all six markers at **recall 1.0000 with 0 false positives** on two controls, including a near-miss that must not fire | [findings-adversarial-variants.md](findings-adversarial-variants.md) |
| Structural context prefix | Indexing a breadcrumb lifts MRR **0.6429 → 0.8333** on the vendored docs, with no query worse | [findings-context-prefix.md](findings-context-prefix.md) |
| BRIGHT retrieval | Six subsets, macro mean nDCG@10: hybrid **0.1436** > BM25 0.1324 > dense 0.0991; dense alone is significantly worse than BM25 on pony and biology | [findings-bright.md](findings-bright.md) |
| Encoder distillation | The 3-model ensemble (478M) distils into one 33M `bge-small`: NFCorpus **0.3385 → 0.3505** (+0.0120 [+0.0022, +0.0216], significant), recovering ~29% of the ensemble's +0.0413 — adapted, not zero-shot | [findings-distillation.md](findings-distillation.md) |
| Trust report | One deterministic, offline artefact (`make trust-report`, `--check` gates a change): **12 checks pass** — poison leak 0.20 → **0.00** at **zero** false refusals, tenant isolation and anonymous denial hold, a fabricated number and an injection marker are flagged; and the abstention gate halves the shipped scalar's false-accept rate at 80% coverage (**0.4069** vs 0.5099, AUC 0.8146 vs 0.5199) | [findings-trust-report.md](findings-trust-report.md) |
| End-to-end pilot | The first `engine + <generator>` row: 12 answerable + 8 null records through the engine at shipped defaults with local `ornith-1.5:35b`. **EM 0.000 against gold-contained 0.667** on the same answers — prose against short entity golds — and **~36 s/query**, i.e. the full 2,255-record row is a ~22–25 h job on this host | [findings-endtoend-pilot.md](findings-endtoend-pilot.md) |
| Three-state claim adjudication | A claim about a topic the corpus never mentions was `not_measured`; with an NLI scorer configured it becomes `contradicted` or `no_coverage`. Demonstrated with `nli-deberta-v3-base` on one passage: entailment **0.994** for a supported claim, contradiction **0.99996** for a contradicted one, and ~0 on both for an unmentioned topic. Opt-in — the default path is unchanged, and a single-label head reports **no** contradiction column rather than guessing one | [findings-claim-states.md](findings-claim-states.md) |
| Cost axis | Per-query latency, throughput, peak RSS and index bytes against an existing index: the shipped default is **1.66 s mean / 2.14 s p95 / 0.60 q/s / 7.0 GiB peak RSS** and a 31 MiB index, against the hybrid's 1.88 s / 2.41 s / 0.53 q/s / 7.8 GiB. The default wins on cost **and** quality on this corpus; the ~7 GiB is Laya's 322M decision head, not the retriever | [findings-cost-axis.md](findings-cost-axis.md) |
| FRAMES retrieval | The **fourth channel reversal**, and the sharpest contrast yet: dense beats lexical on every metric (recall@1 **0.8464 vs 0.7224**, nDCG@10 **0.7505 vs 0.6586**) — the opposite of `webis-touche2020`, also long Wikipedia text but queried topically. The **three-model ensemble then beats dense** (+0.051 nDCG@10), the lever's sixth corpus, while **fusing BM25 into it loses 0.041 recall@1** — the fifth fusion failure. Resolving redirects raised gold coverage 0.8792 → 0.9977 and moved each arm 0.08–0.10 nDCG@10, more than the gap being measured | [findings-frames.md](findings-frames.md) |
| Lexical vs published BM25 | The lexical arm is the one component checkable against an independent implementation, so it is a chart (`docs/charts/lexical_vs_published.png`) rather than an assumption. On **10 measured corpora the lexical arm agrees within ±0.008 on 5 and within ±0.02 on 8**; the two beyond the loose bound are **Quora −0.020** and **HotpotQA +0.034** (a 600-query subset). An earlier reading that the gap grows with corpus size is **falsified**: `dbpedia-entity` at **4.6M documents** is only −0.008. The engine's settings are ruled out too — on all five small sets the published-settings control row is *further* from the anchor than the engine row. The two outliers are both natural-language question datasets and their cause is unidentified | [findings-beir-breadth.md](findings-beir-breadth.md) |
| BEIR breadth (10 of 18) | Membership moves a BEIR mean more than any method difference it adjudicates: adding `quora` took the fused mean **0.4515 → 0.5201 (+0.069)** and adding `webis-touche2020` took it back down to **0.4959 (−0.024)**. On touche **fusion loses 0.091 to BM25 alone** (0.3509 vs 0.4419) and dense is under half the lexical row — the **fifth** corpus where an instrument reverses | [findings-beir-breadth.md](findings-beir-breadth.md) |
| Cross-document consistency gate | `claim_check` pins prose to a *report*; it does not pin documents to *each other*, and the audit found the coverage count stated as **5 / 7 / 10 of 18**, the published `voyage-02` row truncated in one table and rounded in another (0.662/0.479 vs 0.663/0.480 against the paper's 0.6625/0.4795), the recalibration half-split count as **300** in its report and **500** in two documents, and the lexical-control count as "8 of 9 within ±0.008" in the index against the report's **5 of 10**. `make consistency-check` now computes the coverage count from `eval/beir/`, pins every registered site to one authority **rounded half-up**, and quarantines ten pre-fix BEIR reports with machine-readable markers | [findings-consistency-gate.md](findings-consistency-gate.md) |
| MTEB v2 `SearchProtocol` adapter | MTEB's "it scores a model, not an engine" objection expires in v2: `tools/mteb_search.py` implements `index`/`search` over the engine's own query path and builds its `ModelMeta` lazily, so the retrieval *system* is representable. Verified against mteb 2.22.5 (`isinstance(..., SearchProtocol)` and `ModelMeta.load_model()`), **14 tests** on the corpus-id plumbing. A full SciFact evaluation forked ~33 harness workers and stalled in this environment, so **no MTEB score is claimed** and the run/upload stays external | [findings-mteb-search-protocol.md](findings-mteb-search-protocol.md) |
| Verifier precision | The four advisory detectors on a constructed labelled set: **precision 1.0000 and recall 1.0000** each, with **0 false alarms on 200 faithful controls**, clearing the ≥0.9 bar. The honest limits are the near-misses at **0.3333 (2 of 6)** — a date in another format fires *two* spurious kinds, and rounding is inherent to literal comparison. Neither changes the `clean` verdict | [findings-verifier-precision.md](findings-verifier-precision.md) |
| Citation mechanics on composed answers | The engine's half of the end-to-end row, over the pilot's 14 answers: **no answer meets the deterministic citation bar**, dominated by **39 of 81 sentences carrying no citation** (42% on answerable, 59% on null). Quote validity is **0/2 on the null records** against **20/23 (0.870)** on the answerable ones — the verifier catching fabricated quotes exactly where the corpus had nothing to quote. The null side is 2 quotes, so it is a direction, not a rate | [findings-citation-mechanics.md](findings-citation-mechanics.md) |
| Quantized ANN at 1M | IVF-PQ (m=48) is **49.5 MB against the flat index's 1,536 MB (31×, 48 B/vector)**, and with `nprobe=32` plus **500 candidates rescored** retains **recall@10 0.998** at 0.51 ms against exhaustive 10.45 ms. **The candidate set, not the rescore, is the constraint**: at `nprobe=8`/100 candidates recall is 0.37–0.41, and fixed `nprobe` is not scale-invariant (`nlist` grows with √n) | [findings-ann-scale.md](findings-ann-scale.md) |
| Distilled encoder export | The package MTEB would load is **faithful**: it loads as `[Transformer, Pooling, Normalize]`, encodes 384-d unit vectors, and matches its source checkpoint at **max abs diff 0** over three probes. An export that loads but differs silently would corrupt every downstream number without a metric flagging it, so the check is exact and a missing package reports absent rather than passing | [findings-distillation.md](findings-distillation.md) |
| On-disk build memory | The plan's target for the 10M+ corpora was wrong twice: peak RSS is flat in vocabulary to 400k terms, and the *persistent* state is only **7.4 MB against a 943 MB peak** — the peak is transient. Assigning term ids inline cuts it **−8 to −23%** at identical index bytes with ranking parity proven. The transient peak is the **block**, not the merge — more runs means *less* memory (100k/1M/10M token caps → 340.7/641.4/911.8 MB) — but it is a tradeoff, since a lower cap multiplies run files, so the default was left alone. Note a **~15% run-to-run spread** | [findings-ondisk-memory.md](findings-ondisk-memory.md) |

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
| Relational (stance) reranker on ArguAna | Contradiction−entailment ordering of the dense pool is **far worse than the dense order itself** (nDCG@10 0.3767 vs 0.6467, **−0.2700 [−0.2921, −0.2474]**) and below BM25's ~0.417, which is the roadmap's own falsification criterion — a relational NLI score is not a better ArguAna ordering signal than the similarity it was meant to correct | [findings-relational-rerank.md](findings-relational-rerank.md) |
| Cross-encoder adapted on MuSiQue's own train split | **Worse than its zero-shot base on every metric but Hits@10**: MAP@10 **−0.0334 [−0.0395, −0.0273]**, Hits@4 −0.0236, MRR@10 −0.0635, and below the plain `fused` first stage on MAP@10 (−0.0151) and MRR (−0.0313). With 2.1 and 2.3 this closes Phase 2: none of the three ordering heads beats first-stage + coverage | [findings-adapted-scorer.md](findings-adapted-scorer.md) |

## Method notes worth carrying forward

- **A result off by an order of magnitude is a bug signal, not a finding — and here it was mine.**
  HotpotQA read nDCG@10 0.0516 against a published 0.603, and nine other corpora agreed within 0.02 on
  the same path. Rather than publish a twelve-fold gap, I diagnosed it: the gold documents ranked #1 and
  #2 in a corpus slice, so scoring was fine, and SciFact then read 0.2047 against 0.6762 under the same
  new `--limit-queries` flag. It was the dilution trap this project had already recorded twice — full
  qrels passed alongside a subset's rankings, so every unsampled query scored zero. The prediction from
  dilution (0.0516 x 7405/600 = 0.637) matched the corrected 0.6365 to three digits.
- **The dilution trap is worth auditing, not just re-checking.** An audit after the HotpotQA fix found it in **two more tools** — `ondisk.py --limit-queries` and `engine_beir.py --limit` — both passing full qrels with a capped ranking set. It has now been found **four times** here. A behavioural test in `tests/test_qrels_dilution.py` demonstrates the effect directly (a perfect 3-of-10 result scores 0.3 against the full qrels, 1.0 against a restricted one), so the reason for the guard is pinned rather than the guard's spelling. Reassuringly, **no recorded report used a limited run**, so no published number was affected — which was checked rather than assumed.
  times in this repository — `arguana_stance`, `order_eval`, and the new query-subset flag — each time
  because a partial ranking set was evaluated against a complete qrels.


- **Checkpoint the row you are actually waiting on.** The engine lexical row was checkpointed because it
  was the shipped arm; the *control* row — published settings, no stemming, a far larger vocabulary — was
  not, and on NQ it became a three-hour blind wait twice. Resumability belongs to whichever call is
  slowest, which is not always the one being reported.


- **A uniform partial result is an API *semantic*, not a data problem.** Fetching the FRAMES corpus
  stored **exactly one article per 20-title batch** — a 5% yield as regular as a metronome. It reads
  like broken title normalisation and is not: the TextExtracts extension returns **one full extract
  per request however `exlimit` is set** (three titles in, three pages back, exactly one with text), so
  a 50-title batch fetched 50 pages and stored one. Two lessons. The regular yield was the tell — a
  data problem is noisy, a contract is uniform. And the first diagnosis, HTTP 429, was **wrong**: the
  throttling was real and separate, and I attributed the systematic pattern to it because I had just
  seen a 429. `prop=revisions&rvprop=content` honours the batch and fetched all 2,482 articles.


- **Memoise the pure-Python stemmer.** `snowballstemmer` is implemented in Python and a corpus
  repeats its words heavily — 6.2M tokens in a 20,000-document sample collapsed to **74,334 distinct
  stems** — so the same token was being stemmed over and over. Caching by token took a lexical build
  from **165.1 s to 5.5 s (30×, 39k → 1,131k tokens/s)** with byte-identical tokens, which turned a
  ~54-minute corpus build into ~2 minutes. The memo is bounded by *vocabulary*, not corpus length, so
  it does not reintroduce the unbounded accumulator the build was just cleaned of.


- **Checkpoint before the marker, and verify on load.** A checkpoint written before its `.done` marker is
  untrusted if interrupted; a part *with* a marker but corrupt contents must still be re-read defensively,
  or a torn write is trusted forever. A test caught the second case in a change written to handle the first.
- **A checkpoint granularity that changes the numbers is part of the cache key.** Chunked embedding is
  float-equivalent to a single pass but not bit-identical (4.17e-07 on `bge-small`), so the chunk size is
  recorded with the cache and a mismatch re-embeds instead of silently mixing two numerics.


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
- **An index is only complete when every artifact is.** A build killed between the dense save and
  the lexical save leaves a directory with half an index, and `has_index()` accepted it — so a
  re-run reused a dense-only index and the full 2,255-query measurement read MAP@10 **0.2216**
  while the same engine on a 400-query subset of the same queries read **0.3057**. The dense
  vectors were byte-identical; only `lexical.json` was missing. `has_index()` now requires the
  lexical artifact too, and a test pins that the channel really is absent when it is.
- **A checkpoint's fingerprint does not cover the index it was measured against.** jobkit keys a
  resumed run to its *settings*, so re-running after a failed attempt reuses per-query results that
  may have been produced against a different index. When the index changes, the run needs `--fresh`.
- **A metric that averages over the qrels dilutes a subset.** `evaluate_rankings` scores every query
  in the qrels, so a `--limit 8` run against the full qrels read nDCG@10 **0.0045** — indistinguishable
  from a broken retriever. Limited runs must score against the queries they ranked.
