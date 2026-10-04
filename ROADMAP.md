# Roadmap — the six steps toward a winnable "best in the world"

Companion to [`PLAN.md`](PLAN.md) (the closed seven-gap programme), [`RESEARCH.md`](RESEARCH.md)
(the outward survey) and [`IMPROVEMENTS.md`](IMPROVEMENTS.md) (the in-repo diagnosis). This
document is the **forward** plan: what is done next, in what order, how each item is verified,
how long jobs survive interruption, and what is explicitly not attempted.

Status is updated as work lands. Nothing here is measured yet; every row is `planned` until it
has a report under `eval/`.

---

## 0. The target, stated so it can be falsified

"Best RAG system in the world" is not winnable on the axes the field currently grades:

| axis | 2026 leader | this host |
| --- | --- | --- |
| MTEB/BEIR-18 retrieval | `Gemini Embedding 2` 67.71, `NV-Embed-v2` 62.65, `Qwen3-Embedding-8B` ~62 nDCG@10 — API or multi-GPU | CPU-only, 33M–478M encoders |
| Multi-hop | iterative retrieve–read–decide (`CoRAG-8B`) | single-pass, no owned generator |
| End-to-end | answer accuracy + faithfulness + cost (TREC RAG, RAGPerf, BrowseComp-Plus, MTRAGEval) | **no end-to-end row at all** |
| Multilingual / multimodal | MMTEB, MIRACL-VISION, ViDoRe V3 | text-only, English-measured |

So the target is narrowed to one that the architecture can actually win:

> **The best fully-local, CPU-only, no-API-key RAG system — with the most honest,
> reproducible evaluation in the field — graded on retrieval, multi-hop set coverage,
> one labelled end-to-end row, and a deterministic trust report.**

Three bars, all stated before the work so a regression cannot read as a gain:

1. **Own best rows** — BEIR-5 ensemble+BM25 **0.495** (single config 0.507); MultiHop
   covquota-3-0.7 **0.3192** full / **0.3280** subset with paired CIs; `ceiling-decomp.md`
   headroom (ordering **+0.1254**, selection **+0.0634**, signal **+0.0992**).
2. **Published rows** — MultiHop `voyage-02` + reranker MAP@10 **0.480**; BEIR-4 matrix
   BGE-large-v1.5 **0.491**.
3. **Cost** — latency, peak RSS, and index size per query, reported beside every quality number
   (RAGPerf's axis), because "best per unit of compute" is the actual claim.

---

## 1. Operating rules (non-negotiable)

- **Sequential, one long job at a time.** No `&`, no process pools, no concurrent agents on the
  same corpus, index or cache. A job that must be split runs as sequential shards, never
  concurrently. `bench_index_parallel.py` and `bench_biencoder_parallel.py` measure parallelism;
  they are not an execution strategy.
- **Resumable by construction.** Every long job checkpoints per unit of work and is re-run by
  repeating the exact same command. Killing a job may lose at most the unit in flight.
- **Every code change ships with an offline test**, and new behaviour is opt-in and defaults to
  the current behaviour, so existing numbers stay comparable.
- **Differences carry a paired bootstrap over queries**; a family of cells carries permutation
  p-values with BH-FDR and an MDE. No bare point estimates.
- **Negatives are recorded**, not deleted — the do-not-retry ledger is
  [`RESEARCH.md`](RESEARCH.md) §2 plus [`docs/FINDINGS.md`](docs/FINDINGS.md).
- **A replica is not the system.** Any claim about the engine is measured by the engine
  (`tools/engine_beir.py` pattern), never only by `tools/beir.py`.
- **Generator-attributed numbers are labelled** `engine + <model>` and never presented as the
  engine's.

---

## 2. The resumability contract (new shared infrastructure)

**This is Phase 1's first item and a prerequisite for every long job in Phases 2–5.** The
current caches are all-or-nothing (`beir.cached_embed` writes one `.npy` keyed by the doc-id
list) and `tools/evalkit.write_json` is not atomic, so an interrupted run can lose hours and a
half-written report can be read as a result.

### New module: `tools/jobkit.py`

| piece | contract |
| --- | --- |
| `write_json_atomic(path, payload)` | write a sibling `*.tmp`, `flush`+`fsync`, `os.replace`; readers never see a partial file |
| `config_fingerprint(**kwargs)` | sha256 over sorted JSON of model ids, chunking, overlap, depth, split, and a `CODE_VERSION` constant; a changed fingerprint invalidates loudly instead of silently mixing |
| `ResumableRun(root, name, fingerprint, shard=(i, n))` | loads `<name>.jsonl`; exposes `completed: set[str]`, `pending(items)`, `record(key, value)`, `finish(summary)`, `manifest()` |
| work key | caller-chosen and stable (`str(query_id)` in the retrofit), never a loop index |
| `record` | append one JSON line, `flush`+`fsync`, then return — a kill loses at most one unit |
| `shard_items(items, i, n)` / `parse_shard("i/n")` | contiguous over a deterministically ordered sequence, so concatenating shards restores order |
| `merge_shards(stem, n, build=...)` | folds sequential shards, refuses mismatched fingerprints, and lets a tool rebuild its own payload shape |
| `META_KEY` | reserved row for shared, non-per-unit state, so a sharded job needs no sidecar |
| manifest | `<stem>.manifest.json`: status, fingerprint, shard, unit counts, timestamp; `iter_manifests` powers `make job-status` |

### Retrofit and Makefile

| item | target | change | status |
| --- | --- | --- | --- |
| P1.1 | `tools/jobkit.py` | the module above + `tests/test_jobkit.py` (30 tests: kill/resume equivalence, shard partition and merge, fingerprint refusal, torn-line tolerance, atomic writes) | done |
| P1.2 | `multihop_ce_cache.py`, `multihop_diagnostics.py`, `engine_beir.py`, `engine_multihop.py`, `engine_musique.py` | per-query checkpoint, `--fresh`, `--shard i/n`, `--merge N`. These are the tools whose cost is per-unit and independent. **Deferred, with reasons:** `tools/multihop_official.py` — its expensive step is the cross-encoder rerank, which `multihop_ce_cache.py` already checkpoints, so retrofitting it would duplicate the cache rather than add resumability; `tools/beir.py` — a replica, and `engine_beir.py` is the jobkit-backed harness for the claims that matter | done |
| P1.3 | `tools/finetune.py` | step checkpoints and optimiser-state resume: `--checkpoint-steps N`, `--checkpoint-dir`, `--checkpoint-keep`, `--resume-checkpoint`. This is distinct from the pre-existing `--resume`, which loads weights and restarts training; `--resume-checkpoint` restores the optimiser and scheduler too. `tools/distill_eval.py` has no training loop, so nothing to resume | done |
| P1.4 | `Makefile` | `make job-status` prints every manifest (`tools/job_status.py`) | done |
| P1.5 | `tests/test_jobkit.py` | kill-and-resume equivalence; sharded merge equals unsharded; fingerprint mismatch refuses | done |
| P1.6 | `tools/engine_multihop.py` | the engine on MultiHop-RAG under its own chunking, resumable from the first commit (`--fresh`, `--shard`, `--merge`) | built, not yet run |

### Session protocol for a long job

```sh
# one job, one log, re-run the identical command to resume
mkdir -p eval/logs
make <target> 2>&1 | tee eval/logs/<target>.log
.venv/bin/python tools/... --shard 3/10      # a bounded sequential unit
```

Within an agent session, a long step is **one** background `bash` job; the next step starts only
after it is collected. `make job-status` is the first command of any resumed session.

---

## 3. Phase 0 — repair the claim ledger (hours)

Commit `c415ece` corrected the mis-normalised reorder ceiling in the tool, the README, `RESEARCH.md`,
`PLAN.md`, `IMPROVEMENTS.md` and the research briefs — but **not in the findings index or its
source report**, and the correction it left behind is not internally consistent.

The authoritative regenerated report is `eval/benchmarks/multihop-rag/diagnostics.md`:

| depth | fact recall | corrected ceiling |
| --- | --- | --- |
| 10 | 0.4747 | **0.3970** |
| 100 | 0.8666 | **0.6088** |
| 500 | 1.0000 | **0.6592** |

The stale and inconsistent figures still in circulation:

| where | value | problem |
| --- | --- | --- |
| `docs/findings-multihop-diagnostics.md` | 0.7321 / 0.7304 / 0.6741 | pre-fix, mis-normalised; title and implications assert the refuted "selection, not recall" |
| `docs/FINDINGS.md:15` | 0.7307, "selection, not recall" | the index of every measurement contradicts its own report |
| `commit c415ece` message | "500 … already correct: 0.6741" | wrong: 0.6741 was the broken depth-500 figure, corrected to **0.6592** |
| `eval/benchmarks/multihop-rag/ceiling-decomp.md` | 0.6072 (CE top-100) | a *different pool* from diagnostics' 0.6088 (fused depth-100) — both may be right, neither is labelled |

| # | item | file | change | test | status |
| --- | --- | --- | --- | --- | --- |
| 0.1 | Correct the source report | `docs/findings-multihop-diagnostics.md` | retitled "recall **and** selection"; corrected table 0.3970/0.6088/0.6592; "widening the pool does not raise the ceiling" rewritten as **+0.2118** (10→100) and **+0.0505** (100→500), both significant; implications 1–2 rewritten; a `Correction, 2026-10-03` block records the normalisation bug | `make claim-check` | done |
| 0.2 | Correct the index | `docs/FINDINGS.md` | the Multi-hop selection row now says the gap is recall *and* selection, with the corrected ceiling and the 719/2,255 completeness figure | `make claim-check` | done |
| 0.3 | Reconcile the two corrected ceilings | `tools/multihop_ceiling_decomp.py` | the renderer now names its pool (the cross-encoder's depth-100 list) and states that diagnostics' 0.6088 is the *fused* pool, so the two are complementary and must not be added | `tests/test_ceiling_decomp.py` | done |
| 0.4 | Record the `0.6741` history | `docs/findings-multihop-diagnostics.md` | the correction block states that 0.6741 was the pre-fix depth-500 value and 0.6592 is correct | `make claim-check` | done |
| 0.5 | Claim-drift check | **new** `tools/claim_check.py`, `tests/test_claim_check.py`, `make claim-check` | an explicit registry of claim → document → report path, with derived statistics for the figures the report does not store; a table row must carry its own number, prose is matched per paragraph, known-stale values are asserted absent, and a missing report skips rather than fails | 13 new tests | done |
| 0.6 | Audit the remaining prose | `README.md`, `RESEARCH.md`, `PLAN.md`, `IMPROVEMENTS.md`, `docs/research/01,03,07,08` | every "36% of depth-100 pools" aligned to the report's derived figure | `make claim-check` | done, **then reversed — see 0.8** |
| 0.7 | Correct the README's derived cluster | `README.md` | the README carried **five** hand-typed numbers with no derivation — 811 pools, ceilings 0.451/0.695, fused 0.155/0.310, +0.086 | `make claim-check` | done, **then reversed — see 0.8** |
| 0.8 | Regenerate the report, and re-derive everything from it | `tools/multihop_diagnostics.py`, `tools/claim_check.py`, all prose | Phase 1's diagnostics retrofit made the report reproducible again, and regenerating it showed that 0.6/0.7 had aligned the prose to a **stale** `diagnostics.json` — generated before `8f7feb6` fixed the recall denominator. Re-derived: incomplete pools **799 / 35.4%** (not 719 / 31.9%), ceilings **0.4511 / 0.6952**, fused **0.1564 / 0.3102**, headroom **+0.086**, and depth-500 recall **0.9811**, not 1.000 | `make claim-check` (39 checks) | done |

**What 0.6/0.7 got wrong, and how 0.8 caught it.** The lesson is worth more than the numbers.
`8f7feb6` changed `recall_at_depth`'s denominator from "the facts the deepest pool matched" —
which made `pool_fact_recall@500` equal to **1.0000 by construction** — to the query's true gold
count. The `diagnostics.json` in `eval/` was never regenerated afterwards, so it carried the
*pre-fix* recall rows while the ceilings were current. Phase 0's prose pass read those recall
rows, derived 719 incomplete pools from them, and "corrected" the README's approximately-right
36% / 0.451 / 0.695 / +0.086 cluster to a worse 31.9% / 0.4469 / 0.6845 / +0.076. `claim_check`
then pinned the wrong values, because it read the same stale report.

It was the Phase 1 retrofit that exposed it: making `multihop_diagnostics` resumable meant
re-running it, and the fresh report's derived cluster differed from the pinned one. **The gate
caught the error, which is the point — but the gate is only as good as the report it reads.**
`make multihop-diagnostics` is now the first step of any claim audit, and `claim_check` pins the
recall rows as well as the ceilings so a stale report fails loudly.

**Measured.** `make claim-check` — 33 checks passed against `diagnostics.json`. Suite
**914 passed, 4 deselected** (901 before; 13 new claim-check tests).

**Why this is first.** The project's whole moat is that its claims are checkable. A findings
index that contradicts its own regenerated report is the single cheapest credibility loss
available, and item 0.1 is direction-changing: the refuted conclusion de-prioritised the recall
programme that `ceiling-decomp.md` now says is worth +0.0992.

---

## 4. Phase 1 — measure the engine where the claim is about the engine (days)

[`docs/FINDINGS.md`](docs/FINDINGS.md) names this itself: *"the engine's multi-hop retrieval under
its own corpus-aware chunking is **not measured**. That is the largest known gap in the evidence."*
The 256-character harness (`tools/multihop_official.py`) is a replica, and the BEIR replica was
already shown to overstate the engine by **0.0144**.

| # | item | file | change | falsify / signal | status |
| --- | --- | --- | --- | --- | --- |
| 1.1 | Resumable job infrastructure | §2 above | `tools/jobkit.py` + the CE-cache retrofit + `make job-status`; the remaining four tools are still un-retrofitted | kill/resume and shard/merge equivalence tests | done (retrofit partial) |
| 1.2 | Engine-level MultiHop-RAG | `tools/engine_multihop.py` | run `RagEngine` on MultiHop-RAG under its own corpus-aware chunking; report Hits@10/Hits@4/MAP@10/MRR@10, default path and `bge-small` hybrid, beside the harness row | if the engine beats the harness, the harness is the weak link and the headline changes | **done, full 2,255 in both configurations** — shipped default MAP@10 **0.3128** (**+0.0618 [+0.0547, +0.0688]** over the harness hybrid) and `bge-small` hybrid **0.3009** (+0.0499); the dense channel is a measured trade, costing 0.0119 MAP@10 and 0.0187 MRR to leave off. The first full run read 0.2216 against a **partial index** (fixed engine bug) |
| 1.3 | Engine-level MuSiQue | `tools/engine_musique.py` | run `RagEngine` on the MuSiQue validation split under its own chunking, full corpus, queries optionally narrowed; pair MAP@10 against the harness by qid | the per-corpus bar is stated, not averaged | **done, full 2,417** — hybrid MAP@10 **0.3761** against the harness `fused`'s 0.3852, **−0.0091 [−0.0120, −0.0062]**, and below every harness arm (`blend-0.75` −0.0390, `ce` −0.0271, `covquota` −0.0223); it replicates the 200-query −0.0097. Shipped default on 200: a further **−0.0540**, so the dense channel **reverses** between corpora |
| 1.4 | Publish the harness-vs-engine delta | `tools/engine_vs_harness.py`, `docs/findings-engine-multihop.md`, `docs/FINDINGS.md` | paired comparison that re-derives the subset and refuses a mismatch; every MultiHop row labelled by which harness produced it | `make engine-multihop-compare` | done |

**Gate:** no Phase 2 result is quoted against the engine until 1.2 has a report under `eval/`.
That gate is **satisfied on the full 2,255 queries**, in both configurations: the shipped default
(MAP@10 **0.3128**) and the `bge-small` hybrid (**0.3009**), each paired against the harness.

**The runs that produced 1.2.**

```sh
# done: the full answerable set, shipped default — the engine's best MAP@10 (0.3128)
make engine-multihop ENGINE_MULTIHOP_ARGS="--out eval/benchmarks/multihop-rag/engine-default"
# done: the full set with the dense channel on (0.3009), and the 400-query seed-17 subset (0.3057)
make engine-multihop ENGINE_MULTIHOP_ARGS="--embed-model BAAI/bge-small-en-v1.5 \
    --out eval/benchmarks/multihop-rag/engine-hybrid"
make engine-multihop-compare
```

Re-running any of those identical commands continues a killed run; `make job-status` shows how
far it got. The report is not written until the run is whole, so a `running` manifest is never a
quotable number. **An index change requires `--fresh`:** jobkit's fingerprint covers a run's
settings, not the index it was measured against.

---

## 5. Phase 2 — a trained, set-aware ordering head (weeks; the real lever)

`ceiling-decomp.md` measures the headroom and it is mostly ordering (**+0.1254**), then signal
(**+0.0992**), then selection (**+0.0634**). Every model-free optimiser over the *current*
features is already null (LTR, IDF weights, beam search, per-question-type routing), so the next
movement must come from **new information in the scorer**. This is the first phase that needs a
training run or a model download.

| # | item | needs | change | falsify | status |
| --- | --- | --- | --- | --- | --- |
| 2.1 | Late interaction done properly | `[model]` (~130 MB, MPS) | `answerai-colbert-small-v1` (33M) MaxSim over the cached pool, plus a recall-fixed arm that reorders the CE's own top-10; `tools/multihop_ordering.py` | beats covquota **0.3280** with a paired CI | **done, negative** — MaxSim is below even the unordered `fused` first stage (−0.0430 MAP@10), and reordering the CE's own top-10 still costs −0.0257, so the +0.1254 ordering headroom is not reachable by this instrument |
| 2.2 | Train on bridge conventions | `[model]` MPS | cross-encoder adapted on pairs mined from the MuSiQue **train** split — positives from gold supporting paragraphs, negatives as same-record distractors; `tools/order_train.py` + `tools/order_eval.py` | held-out MAP@10 over the adapted vs zero-shot ranker; label the result **adapted** | **done, negative** — the adapted model is **worse than its own zero-shot base** on MAP@10 (−0.0334 [−0.0395, −0.0273]), Hits@4 and MRR, and below plain `fused` on MAP@10 and MRR |
| 2.3 | Route by relation | `[model]` (~180 MB) | `cross-encoder/nli-deberta-v3-base` scored as `P(contradiction) − P(entailment)` over the dense pool; `tools/arguana_stance.py` | ArguAna recovers toward the dense 0.596 without losing BEIR-5; loses to BM25 → hypothesis wrong | **done, negative** — nDCG@10 **0.3767 vs the dense order's 0.6467** (−0.2700 [−0.2921, −0.2474]), below BM25's ~0.417, so the roadmap's own falsification criterion fires |
| 2.4 | Listwise fallback | `[dep]` | `jina-reranker-v3` is **CC BY-NC 4.0, non-commercial** | if it does not beat 2.2 on this host, drop it | **not adopted, on licence** — a non-commercial 0.6B model is not a shippable dependency for this project, and Phase 2.2's negative removes the reason to spend the reproduction budget. Recorded as a decision, not a measurement |
| 2.5 | Report ordering vs signal vs selection | `docs/findings-ordering-head.md`, `docs/findings-relational-rerank.md`, `docs/findings-adapted-scorer.md` | state each new rule's number against the decomposition, so an ordering gain is never read as a recall gain | — | done |

**Phase 2 verdict: negative, and coherently so.** All three ordering heads tested — topical late
interaction (2.1), relational NLI (2.3), and a cross-encoder adapted on the corpus's own train split
(2.2) — lose to the existing first-stage-plus-coverage rule, and two of the three lose to the
simplest baseline available. The **+0.1254** ordering headroom the decomposition measures is real,
but none of these instruments reaches it. Two of the three also repeat the project's oldest prior:
the failures are "use a stronger borrowed model or objective", the wins are "change the objective
or the routing". What is *not* falsified: cross-corpus training, hard negatives mined from the
retrieval pool, a pairwise/listwise objective, or a GPU-scale run — see the per-finding write-ups.

**Do not retry:** any pointwise MS MARCO reranker at depth 100 (loss on all five BEIR datasets;
ArguAna 0.568 → 0.250), full-split fine-tuning with naive mined negatives (below zero-shot),
BM25 PRF, another optimiser over the cached channels, **a general-purpose topical
late-interaction reranker** — `answerai-colbert-small-v1` MaxSim is below the unordered fused
first stage on MultiHop-RAG and makes the CE's own top-10 worse when it only reorders it
(Phase 2.1, `docs/findings-ordering-head.md`) — or a **relational NLI stance score as a
replacement** for similarity on ArguAna, which is worse than the dense order it reorders and
below BM25 (Phase 2.3, `docs/findings-relational-rerank.md`), or **adapting a cross-encoder on the
corpus's own train pairs**, which loses to its own zero-shot base and to the plain first stage
(Phase 2.2, `docs/findings-adapted-scorer.md`). All three point the same way: on this task the
ordering headroom is not reachable by these instruments.

---

## 6. Phase 3 — one labelled end-to-end row (weeks)

"There is no end-to-end RAG ranking, because no end-to-end benchmark has been run to completion
here." That is the largest credibility hole, and the project's own `engine + <model>` rule already
solves the attribution problem.

| # | item | change | falsify / signal | status |
| --- | --- | --- | --- | --- |
| 3.1 | FRAMES adapter | **new** `tools/frames.py`; 824 questions, ~0.5 MB, ships the Wikipedia links needed to build the corpus; report retrieval recall@k and exact-match accuracy with a **labelled** local generator | if the generator dominates the variance, report retrieval-only and say so | pending — at ~36 s/query the corpus download plus a full run is an `[llm]` job |
| 3.2 | MultiHop answer accuracy + rejection | `tools/endtoend_multihop.py` at the engine's shipped defaults with a labelled local generator | the gate's rejection rate is reported beside accuracy, with the AUC and the transfer caveat | **done as a pilot** — 12 answerable + 8 null, `ornith-1.5:35b`; EM 0.000 against **gold-contained 0.667**, abstention 25% vs rejection 37.5% (n=20, not evidence) |
| 3.3 | Citation precision/recall | `tools/citation_report.py`, `make citation-report` | `verify.py` over composed answers: does every cited span exist, and does the answer cite the evidence that supports it? | this is the engine's own number, not the generator's | **done on the pilot's 14 answers** — **no answer meets the deterministic citation bar**, dominated by **39 of 81 sentences carrying no citation**; and quote validity is **0/2 on the null records against 20/23 (0.870) on the answerable ones**, so the verifier catches fabricated quotes where the corpus had nothing to quote. The null side is 2 quotes, recorded as a direction rather than a rate |
| 3.4 | Cost axis | `tools/cost_report.py`, `make cost-report` | latency, throughput, peak RSS per query, index bytes/query, beside quality | **done** — against an existing index, the shipped default is **1.66 s mean / 2.14 s p95 / 0.60 q/s**, peak RSS **7.0 GiB**, index 31.2 MiB (7,132 B/chunk); the hybrid is 1.88 s / 2.41 s / 0.53 q/s at 7.8 GiB. The default wins on cost **and** MAP@10 here, and the ~7 GiB is Laya's decision head rather than the retriever |
| 3.5 | Publish the row | `BENCHMARKS.md`, `README.md` | replace "cannot be claimed" with the labelled row and its limits | **done for the pilot** — both claim documents now describe the pilot, its EM-versus-containment trap, and the 22–25 h full-run cost, and no longer say "no end-to-end benchmark has been run" |

**Phase 3 verdict: the harness exists, the row is a pilot, and the blocker is the generator.** The
first `engine + <model>` row is produced and labelled, and the two things it establishes are that
(a) EM against short entity golds is the wrong metric for a prose answer — 0.000 EM beside a 0.667
containment rate on the same answers — and (b) at **~36 s/query** the full 2,255-record row is a
22–25 h job on this host, so it belongs in the external list rather than the local one.

---

## 7. Phase 4 — scale to 18 datasets, and own the encoder (weeks; mixed)

| # | item | needs | change | falsify | status |
| --- | --- | --- | --- | --- | --- |
| 4.1 | Cheaper per-block build state | `src/laya_rag/ondisk.py` | assign term ids inline so a document's counts dict is transient, not one of ~10,000 alive per block | ranking parity against the in-memory index; peak RSS lower at fixed `block_tokens` | **done, partial** — peak RSS **−22.6% / −9.7% / −8.0%** at 25k / 50k / 100k documents (−25.5% under tracemalloc), at **identical index bytes and unchanged build time**, with the 15 on-disk tests green including ranking parity and block-size stability. The saving *shrinks* with corpus size, so the wall is reduced and not removed: what remains is the run/merge arrays and allocator retention, which the profile could not separate. **The lever is the block, and it is a tradeoff:** 100k / 1M / 10M token caps give **340.7 / 641.4 / 911.8 MB** at a fixed corpus, so *more runs means less memory* and the merge is cheap — an earlier claim here that the next lever was the merge was **wrong and is retracted**. The default was not changed: a lower cap multiplies runs (tens of thousands at 10M documents), which the 100k-document experiment cannot cost out |
| 4.2 | The 13 remaining BEIR datasets | `[data]` disk | build sequentially, one dataset at a time; record mirror stubs as blocked rather than skipped silently | full BEIR-18 average, stated with its dataset list | **in progress — 6 of 18.** `quora` measured (bm25-engine 0.7689, dense 0.8876, fused 0.8629; the fused mean moves **0.4515 → 0.5201**, so the membership is recorded with it). `webis-touche2020` is **downloaded and extracted** (702 MB, 382,545 docs, 49 scored queries) but **not evaluated**: its lexical pass is single-threaded and the 382k-document embedding is still ahead, and `tools/beir.py`'s `cached_embed` writes the `.npy` only at the end, so an interrupted run **restarts from zero** — which is why an hour of it was lost. **Resumable embedding is the prerequisite** (see the measurement below). `specter`/`signal1m` are 404 on this mirror; Robust04/TREC-NEWS stub; BioASQ/FEVER/NQ await 4.1 |
| 4.3 | Distilled encoder packaging | `tools/distill_verify.py`, `make distill-verify` | confirm the exported SentenceTransformer package is faithful to the checkpoint, so MTEB can load it | the export must be a copy, and a missing package must report absent rather than pass | **packaging verified; submission is external.** The package loads as `[Transformer, Pooling, Normalize]`, encodes 384-d unit vectors, and matches its source checkpoint at **max abs diff 0** over three probes — so the only remaining work is uploading/submitting, which is not offline work |
| 4.4 | MTEB submission | network, external | the leaderboard wants a **model**, and one is now packageable; submit the distilled encoder | an external row with the model card's measured numbers | external |
| 4.5 | ANN quality at scale | `tools/ann_bench.py`, `make ann-bench` | verify IVF+PQ recall against exact at 1M+ with fp32 rescore | recall retained and index reduced, measured not cited | **done at 1M** — IVF-PQ (m=48) is **49.5 MB against the flat index's 1,536 MB (31× smaller, 48 B/vector)** and, with `nprobe=32` and **500 candidates rescored**, retains **recall@10 0.998** at 0.51 ms against the exhaustive scan's 10.45 ms (1.000 at 1,000 candidates, 0.838 ms). **The candidate set, not the rescore, is the constraint**: at `nprobe=8`/100 candidates recall is only 0.37–0.41, and a fixed `nprobe` is not scale-invariant |

**Do not** chase MTEB nDCG@10 with a 33M encoder against API-scale models. 4.4 is about having a
submittable artefact and a full-dataset average, not about winning the top row.

---

## 8. Phase 5 — make trust the headline (days–weeks; the moat)

This is the axis no 8B API model leads on, and the one a CPU-only system can win outright.

| # | item | file | change | test | status |
| --- | --- | --- | --- | --- | --- |
| 5.1 | Trust report | `tools/trust_report.py`, `make trust-report` | one deterministic, offline report aggregating poison exclusion, gold retention, negative rejection, injection flags, tenant isolation, deterministic verification, and the abstention selective-prediction suite; `--check` gates a change | `make trust-report TRUST_ARGS="--check"` | done — 12/12 checks pass; poison leak 0.20 → 0.00 at zero false refusals, gate risk@80 0.4069 vs the scalar's 0.5099 |
| 5.2 | Three-state adjudication | `src/laya_rag/engine.py`, `src/laya_rag/answerability.py` | a claim the relevance gate rejected becomes `no_coverage`/`contradicted`/`supported`, not `not_measured`, via the shipped `NLIAnswerability` hook (`verdict()` returns both NLI columns) | `tests/test_adjudicate_states.py` | **done, opt-in** — with `nli-deberta-v3-base`, one passage separates a supported claim (entailment **0.994**), a contradicted one (contradiction **0.99996**) and an unmentioned topic (~0 on both); the default path is unchanged. **The hand-labelled no-coverage recall is still owed** — the demonstration is three constructed claims, not 300 labelled pairs |
| 5.3 | Numeric / negation / unit / date checks | `tools/verify_precision.py` | the four advisory detectors (`number_not_in_evidence`, `date_not_in_evidence`, `unit_mismatch`, `possible_negation_flip`) measured on a labelled set; ship only at precision ≥ ~0.9 | `make verify-precision VERIFY_ARGS="--check"` | **done on a constructed set** — all four at **precision 1.0000 / recall 1.0000**, with **0 false alarms on 200 faithful controls**; the honest limits are the hard near-misses at **0.3333 (2 of 6)**: cross-format dates (which fire *two* spurious kinds) and rounding. **Hand-labelled composed answers still owed** |
| 5.4 | Grow the adversarial suite | `tools/adversarial_audit.py` | poison / noise / negative rejection / injection with frozen seeds; poison 5 documents → ASR measured, never cited | `make adversarial-audit --check` | **done for the dose response** — `poison_dose_probe` measures the leak rate at doses 0/1/2/5 with a query-independent stuffed document: the control reads 0.0000 and the curve is **flat at 0.2000 after one document**, so the published "5 documents → 90%" is a property of another corpus and is not quoted here. Noise and injection variants with frozen seeds remain |
| 5.5 | Trust as the README claim | `README.md` | lead with the measured trust report, not with a retrieval rank the hardware cannot hold | `make claim-check` | **done** — the "where this RAG is better" bullet now carries the trust report's numbers (poison leak 0.20 → 0.00 at zero false refusals; gate risk@80 0.4069 against the scalar's 0.5099) and links the write-up |

---

## 9. Execution order and session protocol

Strictly sequential. A phase starts only when the one before it has a report under `eval/`.

1. **Phase 0** (hours) → `make claim-check` green.
2. **Phase 1** (days) → jobkit + engine-level MultiHop/MuSiQue. *Everything after this resumes
   from a manifest.*
3. **Phase 2** (weeks) → ordering head; the first `[model]` work.
4. **Phase 3** (weeks) → the end-to-end row.
5. **Phase 4** (parallel across sessions, but sequential within each) → BEIR-18, distillation,
   MTEB.
6. **Phase 5** (days) → trust report; can run whenever a slot is free, since it depends on
   nothing in 2–4.

Every phase ends with:

```sh
make test            # offline suite, must stay green
make lint
make claim-check     # the numbers quoted in prose resolve against the reports
make job-status      # nothing half-finished is left as a result
```

and a report under `eval/` plus a `docs/findings-*.md` entry added to
[`docs/FINDINGS.md`](docs/FINDINGS.md) — updated in the same commit as the code, per `PLAN.md`'s rule.

---

## 10. What is external, and stated as remaining

| item | blocked by |
| --- | --- |
| MTEB submission (4.4) | upload + submission; no code blocker left once 4.3 packages the student |
| TREC RAG, KILT | NIST corpus/gold-nugget access; an entity linker and a full submission stack |
| BioASQ / FEVER / NQ at full size | disk — 28 GB free against 14.9M / 5.4M / 2.7M documents |
| MIRACL / MMTEB | per-language Wikipedia indexes; the multilingual encoder would need measuring |
| ViDoRe / MIRACL-VISION | the engine is text-only: no OCR or vision path |
| An LLM decomposer / agentic loop | `[llm]`; every resulting number belongs to the generator, not the engine |

---

## 11. Status log

- 2026-10-03 — roadmap created. Phase 0 items 0.1–0.6 and the Phase 1 resumability contract are
  the immediate work. No item is measured yet.
- 2026-10-03 — **Phase 0 done (0.1–0.8).** The source finding, the findings index and every
  prose site carry the corrected numbers; `tools/claim_check.py` pins 39 of them against
  `diagnostics.json` and the gate is `make claim-check`. The depth-500 ceiling is 0.6592 (not
  0.6741). The 0.6/0.7 prose pass then had to be **reversed by 0.8**: it had aligned the prose to
  a `diagnostics.json` generated before `8f7feb6` fixed the recall denominator, so 719 / 31.9% /
  0.4469 / 0.6845 / 0.1498 / 0.3052 / +0.076 were all wrong and the README's original
  36% / 0.451 / 0.695 / +0.086 cluster was approximately right. Reproducible values now:
  incomplete **799 (35.4%)**, ceilings **0.4511 / 0.6952**, fused **0.1564 / 0.3102**, headroom
  **+0.086**, depth-500 recall **0.9811**. Suite **914 passed, 4 deselected**. Next: Phase 1,
  starting with the resumability contract.
- 2026-10-03 — **Phase 1 infrastructure landed.** `tools/jobkit.py` (append+fsync per unit,
  atomic finish, fingerprint refusal, contiguous shards, `META_KEY` for shared state,
  `merge_shards(build=...)`), `make job-status` over the manifests, and the first retrofit:
  `tools/multihop_ce_cache.py` now checkpoints per query and takes `--fresh`/`--shard`/`--merge`.
  `tools/engine_multihop.py` — the engine on MultiHop-RAG under its own chunking, the Phase 1
  headline — is built and tested but **not yet run**: it is the first real long job, and its
  commands are in §4. Suite **961 passed, 4 deselected** (914 before; 47 new tests across
  jobkit, the CE cache and the engine harness). Next: run `1.2`, then 1.3/1.4, then Phase 2.
- 2026-10-03 — **Phase 1 measured (1.2 on the 400-query subset, 1.4 done).** The real engine on
  MultiHop-RAG under its own corpus-aware chunking (2,000 characters derived from the corpus,
  4,592 chunks) is **+0.0535 MAP@10 [+0.0387, +0.0680]**, +0.0900 Hits@4, +0.1100 Hits@10 and
  +0.0559 MRR@10 over the 256-token harness's hybrid first stage, all significant; against the
  harness's *reranked* row it is level on MAP@10/Hits@4/MRR@10 and better on Hits@10, so the
  reranker's +0.0464 was largely recovering what the harness chunking cost. The replica
  **understates** the engine here — the opposite sign to BEIR's +0.0144 overstatement. The engine
  is still below the subset's coverage-plus-quota bar (0.3057 against 0.3280), so Phase 2's
  target survives the change. `tools/engine_vs_harness.py` makes the pairing reproducible and
  refuses a mismatched subset. Still owed: the full 2,255 and the shipped default path. Suite
  **973 passed, 4 deselected**. Next: 1.3/1.4 remainder, then Phase 2.
- 2026-10-04 — **Phase 1.2 completes on the full set, and it replicates.** The engine over all
  **2,255** MultiHop answerable queries, corpus-aware 2,000-character chunking, `bge-small` hybrid,
  no cross-encoder: MAP@10 **0.3009** against the 256-token harness's 0.2510 —
  **+0.0499 [+0.0432, +0.0566]** — with Hits@4 **+0.0772**, Hits@10 **+0.1060** and MRR@10
  **+0.0587**, every interval excluding zero. That matches the 400-query subset's +0.0535, so the
  finding is not a sampling artefact. The first full run read **0.2216** — below the harness — and
  the cause was a **partial index** missing `lexical.json`, reused because `has_index()` accepted
  the dense half; that is a fixed engine bug with six tests. Suite **1066 passed, 5 deselected**.
  Still owed: the shipped-default (Laya-encoder, dense-weight-zero) path.
- 2026-10-04 — **Phase 1 complete on the full set, in both configurations.** The **shipped
  default** — Laya's encoder, dense weight resolves to zero, lexical recall, no cross-encoder — is
  the engine's **best** MultiHop-RAG configuration: MAP@10 **0.3128** against the 256-token
  harness's 0.2510, **+0.0618 [+0.0547, +0.0688]**, with Hits@4 +0.0710, Hits@10 +0.0989 and MRR@10
  **+0.0775**. The `bge-small` hybrid scores 0.3009, and paired on the same queries the dense
  channel is a **trade, not a gain**: leaving it off is worth **+0.0119 MAP@10** [+0.0061, +0.0180]
  and **+0.0187 MRR@10** [+0.0062, +0.0316], while Hits@4 and Hits@10 move by −0.006 and −0.007
  (both ns). That is the project's "the dense channel never earned its weight" finding reproduced
  at the engine level on multi-hop. Still below the harness's coverage-plus-quota rule (0.3192), so
  Phase 2's target stands. Suite **1067 passed, 5 deselected**. Next: the full MuSiQue engine run,
  then Phase 4.
- 2026-10-04 — **The dense channel's contribution reverses between corpora.** MuSiQue's 200 queries
  with the shipped default (lexical only) score MAP@10 **0.4156** against the hybrid's **0.4695** —
  **−0.0540 [−0.0718, −0.0363]** — and −0.0500 Hits@4, −0.0550 Hits@10, −0.0730 MRR@10, all
  significant; the default is also **−0.0656** below the harness's fused stage on the same queries.
  On MultiHop-RAG the same comparison runs the other way: leaving dense off is worth **+0.0119
  MAP@10**. So the dense channel is a **per-corpus instrument**, exactly like coverage and the
  cross-encoder before it, and the engine's shipped default is right for one corpus and leaves
  ~0.054 MAP@10 on the table for the other — which is the case for routing rather than a single
  global configuration. Suite **1067 passed, 5 deselected**. Still owed: the full 2,417 MuSiQue run
  and the MuSiQue default at full size.
- 2026-10-04 — **Phase 5.2 done, opt-in.** A claim the relevance gate rejected used to be
  `not_measured`, which conflates "the corpus says nothing" with "the corpus says the opposite" —
  the limit `README.md` stated as "a fabricated claim about a topic the corpus never mentions is not
  refuted". With `LAYA_RAG_ANSWERABILITY_NLI_MODEL` configured the rejected claims are asked about
  anyway and become `supported`, `contradicted` or `no_coverage`; `NLIAnswerability.verdict()`
  returns both NLI columns, and a single-label head reports **no** contradiction column (`None`)
  rather than inventing one, which would turn silent into refutes. Demonstrated with
  `nli-deberta-v3-base` on one passage: entailment **0.994** for a supported claim, contradiction
  **0.99996** for a contradicted one, ~0 on both for an unmentioned topic. The default path is
  untouched and the existing adjudication tests pass unchanged; nine tests pin the states. **Owed:**
  the ~300 hand-labelled pairs and a no-coverage recall — the demonstration is three constructed
  claims, not a labelled evaluation. Suite **1076 passed, 5 deselected**.
- 2026-10-04 — **Phase 3.4 done: the cost axis, which is the axis this project can win.**
  `make cost-report` measures latency, throughput, peak RSS and index bytes against an **existing**
  index, so nothing is rebuilt and the system measured is the one the quality numbers describe.
  MultiHop-RAG, 4,592 chunks, 60 timed queries after warm-up: the **shipped default** is
  **1.66 s mean / 2.14 s p95 / 0.60 q/s**, peak RSS **7.0 GiB**, index **31.2 MiB** (7,132 B/chunk);
  the **hybrid** is 1.88 s / 2.41 s / 0.53 q/s at 7.8 GiB. So on this corpus the default wins on
  cost *and* on MAP@10 (0.3128 against 0.3009) — there is no trade to make — and the ~7 GiB is
  Laya's 322M decision head, not the retriever. Suite **1085 passed, 5 deselected**. Still owed:
  the full 2,417 MuSiQue runs and Phase 4.
- 2026-10-04 — **The full MuSiQue run is checkpointed mid-flight at 525/2417.** It was stopped
  deliberately rather than left running: the query rate degraded from 3.3 to **4.2 s/query** as the
  run progressed, putting the remaining 1,892 queries at roughly two more hours, and the jobkit
  checkpoint means **no work is lost**. `index/engine-musique/engine-hybrid/` holds all three
  artifacts (`index.json`, `lexical.json`, `vectors.npy`), so a re-run resumes scoring without
  re-indexing:

  ```sh
  make engine-musique ENGINE_MUSIQUE_ARGS="--embed-model BAAI/bge-small-en-v1.5 \
      --out eval/benchmarks/musique/engine-hybrid"
  ```

  `make job-status` shows `running 525/2417`; a `running` manifest is never a quotable number, and
  the report is not written until the run is whole.
- 2026-10-04 — **Phase 5.3 done on a constructed set, with its limits named.** `make
  verify-precision` measures the four advisory detectors on 400 constructed cases: each is at
  **precision 1.0000 and recall 1.0000** with **0 false alarms on the 200 faithful controls**, so
  all four clear the ≥0.9 bar on clear-cut mutations (over two unit families for the unit check).
  A perfect score on easy cases proves little, so the harness also runs six **known-hard
  near-misses** and reports them rather than hiding them: **2 of 6 false-alarm**. Cross-format dates
  are the worst — `2024-03-12` against `March 12, 2024` fires *two* spurious kinds, the date and its
  digits read as absent numbers — and rounding (`1205` vs "about 1200") is inherent to literal
  comparison. Neither changes the `clean` verdict, and a test asserts both stay known, so a fix must
  update the write-up with it. **Owed:** the ~200 hand-labelled composed answers; this set is
  constructed, and the number must not be quoted as if it were that. Suite **1092 passed,
  5 deselected**.
- 2026-10-04 — **Phase 3.3 done: the engine's half of the end-to-end row, measured.** `make
  citation-report` re-retrieves the evidence for the pilot's 14 composed answers (deterministically,
  so it is the evidence the generator saw) and runs `verify.py` over them. **No answer meets the
  deterministic citation bar**, and the dominant cause is uncited sentences rather than wrong ones:
  **39 of 81 sentences carry no citation** — 22 of 52 on answerable records, 17 of 29 on nulls. The
  sharper result is quote validity: **0 of 2 quotes on the null `null_query` records** were in the
  evidence (one fabricated, one taken from the question), against **20 of 23 (0.870)** on the
  answerable ones — the deterministic verifier catching a fabricated quotation on exactly the
  questions the corpus could not answer, with no weights and no thresholds. The null side is 2
  quotes, so that is recorded as a direction rather than a rate. Suite **1100 passed, 5 deselected**.
- 2026-10-04 — **Phase 4.5 done: quantized ANN at 1M, and the knob is not the reranker.** IVF-PQ
  (`m=48`) holds **49.5 MB against the flat index's 1,536 MB** — 31× smaller, 48 bytes per vector
  instead of 1,536 — and with `nprobe=32` and **500 candidates rescored by the fp32 vectors** it
  retains **recall@10 0.998 at 0.51 ms/query** against the exhaustive scan's 10.45 ms (1.000 at
  1,000 candidates). The first run told a worse story and is recorded because it is the mistake the
  design invites: at `nprobe=8` with 100 candidates, recall is only **0.370** — a rescore can only
  reorder what it is given, and the PQ codes had ranked the true neighbours out of the candidate
  list. So the tuning knob is the **candidate set**, and a fixed `nprobe` is not scale-invariant
  (`nlist` grows with √n, so the same setting scores 0.646 / 1.000 / 0.370 at 10k / 100k / 1M).
  Synthetic vectors, so the mechanics and the arithmetic are established rather than a corpus
  recall. Suite **1106 passed, 5 deselected**.
- 2026-10-04 — **Phase 4.1 evidence gathered, and it corrects the plan's target.** `make
  ondisk-scale` sweeps peak build RSS on two axes independently, one subprocess per point because
  `ru_maxrss` is a high-water mark. The plan says the residual 10M-document wall is *"a bounded
  external merge for the term dictionary"*; measured, the dictionary is **not** it — an eightfold
  vocabulary increase (10k → 400k terms) moves peak RSS by **0.8%** and non-monotonically, while the
  **documents** axis moves it linearly at ~**3.8 KB/doc**. The existing `block_tokens` knob already
  controls much of the peak (**399.9 / 593.5 / 808.1 MB** at 250k / 1M / 4M tokens on a fixed
  corpus, identical index). An unbounded component is real — peak still grows with corpus size at
  fixed block size, and reaches **1,031.9 MB** with one block against 491.4 MB at a quarter of the
  corpus — but **which structure dominates is not separated**, and the candidates (the accumulating
  id/length lists, the merge buffers, allocator retention) are stated as an open question rather than
  guessed at. **No index format was changed:** the bounded rewrite stays deferred until it is
  profiled, because it is parity-critical and the 20/20 and 300/300 ranking checks must stay green.
  Suite **1113 passed, 5 deselected**.
- 2026-10-04 — **Phase 4.1 profiled, and the target moved a second time.** `tools/ondisk_scale.py
  --profile` runs a build under `tracemalloc` at 100k documents: **live Python state when it finishes
  is 7.4 MB against a 943 MB peak** (inflated by tracemalloc itself). The largest live site is the
  *tokenizer's stopword set* at 2.74 MB; the term dictionary and its document frequencies are 3.3 MB
  together. So the dictionary, the chunk-id list and the length list — everything the plan named — are
  a few megabytes at 100k documents. **The peak is transient**: the block, which `block_tokens`
  already controls (399.9 → 808.1 MB across a 16× cap range), plus retention across blocks (peak rises
  312 → 418 → 596 MB with corpus size at a *fixed* cap, where the block is constant and the live state
  is 7 MB). The tractable fix is therefore to make the per-block representation cheaper — today a block
  is `list[tuple[doc_id, dict[term, tf]]]`, one Python dict per document — replaced by doc offsets plus
  parallel term/tf arrays, which writes **byte-identical runs** and is parity-testable with the
  existing 20/20 and 300/300 checks. No format change was made. Suite **1114 passed, 5 deselected**.
- 2026-10-04 — **Phase 4.1: the parity gate is verified green before the rewrite.** The per-block
  change is a core-loop edit, so the gate it must pass was confirmed first rather than discovered
  afterwards: `tests/test_ondisk_engine.py` is **15/15 green**, and three of those tests are exactly
  what the change could break — `test_ondisk_bm25_matches_the_in_memory_ranking` (ranking parity
  against the in-memory index), `test_ondisk_bm25_ranking_is_stable_across_build_blocks` (the same
  ranking under different block sizes) and `test_ondisk_lexical_channel_reproduces_in_memory_ids`
  (the chunk-id list). The plan for the rewrite: assign term ids **during** the document loop instead
  of deferring them to `flush()`, so the per-document `counts` dict becomes transient rather than
  held for every document in the block. Term ids are then assigned in the same
  first-appearance order, so the spilled runs stay byte-identical and the three tests above are the
  proof. Suite **1114 passed, 5 deselected**; no code changed this round.
- 2026-10-04 — **Phase 4.1 implemented, and it is a partial fix.** Term ids are now assigned as
  documents stream, so a document's `term -> count` dict is transient instead of one of ~10,000 alive
  at once per block. Peak build RSS falls **−22.6% / −9.7% / −8.0%** at 25k / 50k / 100k documents
  (312.4 → 241.9, 417.9 → 377.3, 596.3 → 548.9 MB) and −25.5% under `tracemalloc` (942.8 → 703.0 MB),
  at **identical index bytes and unchanged build time** (12.4/25.1/50.9 s), with the 15 on-disk tests
  green — including ranking parity against the in-memory index, block-size stability and chunk-id
  reproduction, which is what proves the runs are unchanged. **The saving shrinks with corpus size**,
  because the per-document dicts were a constant per-block cost while the rest of the peak scales with
  the corpus: the run and merge arrays and allocator retention remain, the wall is reduced rather than
  removed, and the next lever is the merge. Suite **1114 passed, 5 deselected**.
- 2026-10-04 — **Phase 1 complete: the full MuSiQue run finished and confirms the subset.** All
  2,417 validation queries, engine hybrid: MAP@10 **0.3761** against the harness `fused`'s 0.3852,
  **−0.0091 [−0.0120, −0.0062]** — the 200-query subset read −0.0097, so the finding replicates.
  Paired against the harness's other arms on the same queries it is below all of them:
  `blend-0.75` −0.0390, `ce` −0.0271, `covquota-3-0.7` −0.0223, every interval excluding zero. So
  the engine beats the harness on MultiHop-RAG (+0.0499 to +0.0618) and loses to it on MuSiQue
  (−0.0091 to −0.0390) — **the gain is corpus-specific in both directions**, which is the case for
  routing. **Phase 1 is now complete on both corpora, both configurations, at full size or the
  documented subset.** Suite **1085 passed, 5 deselected**. Next: Phase 4.
- 2026-10-03 — **Phase 1.3 done, and it falsifies the generalisation.** `tools/engine_musique.py`
  runs the engine on the full 21,098-paragraph MuSiQue validation corpus with only the queries
  narrowed (never the corpus — `tools/musique.py --limit-queries` shrinks the corpus too, which
  would make a subset incomparable). On 200 queries the engine is **−0.0097 MAP@10
  [−0.0190, −0.0007]** against the harness's fused first stage, so the MultiHop-RAG engine win
  does **not** travel. The obvious explanation — the engine's default `retrieve_k=20` against a
  depth-100 harness — is falsified: raising it to 100 moved MAP@10 by +0.0019 and left the
  deficit significant. The engine does beat the harness's `ce` and `covquota` rules on the same
  subset. Suite **985 passed, 4 deselected**. Both Phase 1 gates now have reports; the full
  2,255/2,417 runs and the shipped-default paths remain owed. Next: Phase 2.
- 2026-10-03 — **Phase 0.8: the diagnostics report was regenerated, and Phase 0's own prose pass
  was reversed.** Retrofitting `tools/multihop_diagnostics.py` onto jobkit required re-running it,
  and the fresh report disagreed with the pinned derived cluster. Cause: the `diagnostics.json` in
  `eval/` predated `8f7feb6`'s recall-denominator fix, so it carried recall rows computed with a
  denominator that makes `pool_fact_recall@500` equal to 1.0000 by construction. Phase 0 had read
  those rows and written 719 / 31.9% / 0.4469 / 0.6845 / 0.1498 / 0.3052 / +0.076 into ten
  documents. Reproducible values: **799 (35.4%)**, **0.4511 / 0.6952**, **0.1564 / 0.3102**,
  **+0.086**, and depth-500 recall **0.9811** — so "every gold fact is in the pool at depth 500"
  is false. `claim_check` now pins the recall rows too (39 checks). Suite **991 passed,
  4 deselected**. The gate caught a real error, including one of mine.
- 2026-10-03 — **Phase 1 infrastructure complete (P1.1–P1.5).** `engine_beir.py` and
  `finetune.py` joined `multihop_ce_cache.py`, `multihop_diagnostics.py` and the two engine
  harnesses on jobkit; training runs get step checkpoints and optimiser-state resume, distinct
  from the pre-existing weight-only `--resume`. `multihop_official.py` and `beir.py` are
  deliberately not retrofitted — the first's expensive step is the CE rerank the cache already
  checkpoints, and the second is a replica with a jobkit-backed engine counterpart — and that
  reasoning is recorded rather than left as an omission. Suite **1009 passed, 4 deselected**.
  Phase 1 is complete on the 400/200-query subsets; the full 2,255/2,417 runs and the
  shipped-default paths are the next long jobs, and Phase 2 is next.
- 2026-10-03 — **Phase 2.1 done, and it falsifies the hypothesis.** A ColBERT MaxSim ordering
  head (`answerai-colbert-small-v1`, 33M, built directly as `BertModel` + the checkpoint's
  `linear` projection since the pinned `sentence-transformers` 4.1.0 has no
  `MultiVectorEncoder`) scores **worse than the unordered fused first stage**: MAP@10 0.2092
  against `fused` 0.2522 (**−0.0430 [−0.0632, −0.0229]**), and −0.0894 against the CE order and
  −0.1016 against covquota. The sharpest arm holds recall fixed by reordering the cross-encoder's
  *own* top-10 (Hits@10 +0.0000 by construction) and still loses **−0.0257 [−0.0434, −0.0079]**:
  the **+0.1254** the decomposition measures as available from reordering that set is headroom a
  better *scorer* would have to reach, not one any reranker can collect. The likely reason is that
  the model is trained for topical retrieval and MultiHop's difficulty is the low-relevance second
  hop — the project's "trust a stronger pointwise model more" prior, with late interaction over a
  topical model still topical. The harness reproduces `fused` 0.2522, `ce` 0.2986 and
  `covquota-3-0.7` **0.3280** exactly, so the baselines are the project's own. Suite **1021
  passed, 5 deselected** (the model sanity check is a `real_weights` test). Next: 2.2, a
  cross-encoder trained on bridge conventions.
- 2026-10-04 — **Phase 2.3 done, also negative.** An NLI stance reranker
  (`cross-encoder/nli-deberta-v3-base`, scored `P(contradiction) − P(entailment)`) over the
  ArguAna dense pool scores nDCG@10 **0.3767** against the dense order's **0.6467** on the same
  pool — **−0.2700 [−0.2921, −0.2474]** — and below BM25's ~0.417, which is the roadmap's own
  falsification criterion. Probable causes, recorded as hypotheses: 512-token truncation of
  300–500-word arguments, sentence-level NLI as a proxy for a discourse relation, and discarding
  the topical signal instead of blending with it. Phase 2 now has **two falsified ordering heads**
  (topical late interaction, relational NLI) against one strong baseline rule, which is the
  argument for 2.2. Also found and fixed a reusable harness trap: `evaluate_rankings` averages
  over every qrels query, so a limited run must score against the subset it ranked — the `--limit 8`
  smoke read nDCG@10 0.0045 until it did. Suite **1027 passed, 5 deselected**. Next: 2.2.
- 2026-10-04 — **Phase 5.1 done.** `make trust-report` states the deterministic trust surfaces in
  one artefact and `--check` gates a change: **12/12 checks pass** — planted poison leak rate
  0.20 → **0.00** with **zero** false refusals, tenant isolation and anonymous denial hold, and a
  fabricated number plus an injection marker are flagged. The abstention half pairs the
  12-feature gate against the shipped scalar out of sample: AUC **0.8146 vs 0.5199**, false-accept
  at 80% coverage **0.4069 vs 0.5099**, ECE 0.065 vs 0.271. It aggregates existing tools and
  re-measures nothing, which the write-up says plainly. Suite **1034 passed, 5 deselected**. Next:
  2.2 (the data path exists — `tools/fetch_musique.py --split train`), then Phase 3.
- 2026-10-04 — **Phase 2.2 done, negative; Phase 2 is complete.** Adapted
  `cross-encoder/ms-marco-MiniLM-L-6-v2` on 20,000 pairs mined from 4,000 MuSiQue train queries
  (gold supporting paragraphs as positives, same-record distractors as negatives; 594 steps,
  held-out loss 0.1813) and evaluated it against its own zero-shot base over the same validation
  pool: MAP@10 **0.4035 → 0.3701** (**−0.0334 [−0.0395, −0.0273]**), Hits@4 −0.0236, MRR@10
  −0.0635, and the adapted model is below plain `fused` on MAP@10 (−0.0151) and MRR (−0.0313).
  `fused` reproduces the harness's 0.3852 exactly, so the evaluator is faithful. **All three
  Phase 2 ordering heads are now falsified** (topical late interaction, relational NLI, adapted
  cross-encoder); 2.4 is not adopted on licence (CC BY-NC). Also fixed a real trainer bug found in
  the output: the first full run silently **resumed a smoke run's checkpoint** — step checkpoints
  now carry a jobkit fingerprint and refuse a mismatch. Suite **1045 passed, 5 deselected**. Next:
  Phase 3 (an end-to-end row; Ollama is running on this host).
- 2026-10-04 — **Phase 3.2 done as a pilot, and the cost is the finding.** `tools/endtoend_multihop.py`
  runs the engine at its **shipped defaults** (gates active, so abstention is a decision) and composes
  only when it chooses to answer, reusing the Phase 1.2 index so the retrieval half is the measured
  system. 12 answerable + 8 null records with local `ornith-1.5:35b`: **EM 0.000, F1 0.026, gold
  contained 0.667** on the same nine answers — prose against short entity golds, so any end-to-end row
  must report containment or a judge and not EM alone. Abstention 25% on answerables against 37.5% on
  nulls, which on n=20 is consistent with the properly measured 30.9%/28.7% and is not evidence. The
  decisive number is **~36 s/query**, extrapolating the full 2,255-record row to **22–25 h** on this
  host, so the remainder of Phase 3 (FRAMES, citation precision, RSS/index cost) is an `[llm]` job.
  Suite **1053 passed, 5 deselected**.
- 2026-10-04 — **A full-set engine run was measured against a partial index, and that is now a
  fixed engine bug.** The first full 2,255-query MultiHop hybrid run reported MAP@10 **0.2216**,
  against **0.3057** for the same engine on a 400-query subset of the same queries. The cause was
  not sampling: the two indexes' dense vectors are **byte-identical** (max abs diff `0.0`) and
  `index/engine-hybrid/` was simply **missing `lexical.json`**, because the run that built it was
  killed mid-save and the re-run reused it. `RagEngine.has_index()` accepted the dense half as a
  complete index, so every later query degraded to dense-only recall silently — the engine already
  warned at query time, but after the numbers were wrong. `has_index()` now requires the lexical
  artifact (the eager or plain one matching the configured backend, and both directories on the
  on-disk backend) and six tests pin it. The full-set measurement is re-run against a rebuilt index
  with `--fresh`, since checkpoints from the degraded run are not reusable — jobkit's fingerprint
  covers a run's *settings*, not the index it was measured against.
