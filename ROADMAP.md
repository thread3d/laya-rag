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
| 1.2 | Engine-level MultiHop-RAG | `tools/engine_multihop.py` | run `RagEngine` on MultiHop-RAG under its own corpus-aware chunking; report Hits@10/Hits@4/MAP@10/MRR@10, default path and `bge-small` hybrid, beside the harness row | if the engine beats the harness, the harness is the weak link and the headline changes | **done on the 400-query subset** — engine **+0.0535 MAP@10 [+0.0387, +0.0680]** over the harness hybrid, level with its reranker (+0.0071, ns), still below covquota 0.3280; full 2,255 owed |
| 1.3 | Engine-level MuSiQue | `tools/engine_musique.py` | run `RagEngine` on the MuSiQue validation split under its own chunking, full corpus, queries optionally narrowed; pair MAP@10 against the harness by qid | the per-corpus bar is stated, not averaged | **done on 200 queries** — engine **−0.0097 MAP@10 [−0.0190, −0.0007]** vs the harness `fused`, and raising the pool to depth 100 moves it only +0.0019, so the MultiHop win does not travel and pool depth is not the cause |
| 1.4 | Publish the harness-vs-engine delta | `tools/engine_vs_harness.py`, `docs/findings-engine-multihop.md`, `docs/FINDINGS.md` | paired comparison that re-derives the subset and refuses a mismatch; every MultiHop row labelled by which harness produced it | `make engine-multihop-compare` | done |

**Gate:** no Phase 2 result is quoted against the engine until 1.2 has a report under `eval/`.
That gate is now satisfied for the **400-query subset**; the full-set run is still owed, so a
Phase 2 result may be quoted against the subset engine number (0.3057) and not against a
full-set one.

**The run that produced 1.2, and what is still owed.**

```sh
# done: 400 queries, seed 17 — the same subset the harness's 0.2986/0.3280 bars are quoted on
make engine-multihop ENGINE_MULTIHOP_ARGS="--limit 400 --sample-seed 17 \
    --embed-model BAAI/bge-small-en-v1.5 --out eval/benchmarks/multihop-rag/engine-hybrid-400"
make engine-multihop-compare

# owed: the full 2,255 (~2 hours at the measured 3.1 s/query), and the shipped default path
make engine-multihop ENGINE_MULTIHOP_ARGS="--out eval/benchmarks/multihop-rag/engine-default"
make engine-multihop ENGINE_MULTIHOP_ARGS="--embed-model BAAI/bge-small-en-v1.5 \
    --out eval/benchmarks/multihop-rag/engine-hybrid"
```

Re-running any of those identical commands continues a killed run; `make job-status` shows how
far it got. The report is not written until the run is whole, so a `running` manifest is never a
quotable number.

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
| 2.2 | Train on bridge conventions | `[model]` GPU-preferred | cross-encoder trained on multi-hop pairs mined from *train* splits (MuSiQue / 2Wiki / HotpotQA), positives from gold facts, hard negatives from the pool — not MS MARCO points; `tools/order_train.py` | held-out MAP@10 over the trained vs zero-shot ranker; label the result **adapted** | planned |
| 2.3 | Route by relation | `[model]` (~180 MB) | `cross-encoder/nli-deberta-v3-base` scored as `P(contradiction) − P(entailment)` over the dense pool; `tools/arguana_stance.py` | ArguAna recovers toward the dense 0.596 without losing BEIR-5; loses to BM25 → hypothesis wrong | **done, negative** — nDCG@10 **0.3767 vs the dense order's 0.6467** (−0.2700 [−0.2921, −0.2474]), below BM25's ~0.417, so the roadmap's own falsification criterion fires |
| 2.4 | Listwise fallback | `[dep]` | `jina-reranker-v3` is **CC BY-NC 4.0, non-commercial** — reproduce on the five sets + MultiHop before believing any self-reported row | if it does not beat 2.2 on this host, drop it | planned |
| 2.5 | Report ordering vs signal vs selection | `docs/findings-ordering-head.md` | state each new rule's number against the decomposition, so an ordering gain is never read as a recall gain | — | done for 2.1 |

**Do not retry:** any pointwise MS MARCO reranker at depth 100 (loss on all five BEIR datasets;
ArguAna 0.568 → 0.250), full-split fine-tuning with naive mined negatives (below zero-shot),
BM25 PRF, another optimiser over the cached channels, **a general-purpose topical
late-interaction reranker** — `answerai-colbert-small-v1` MaxSim is below the unordered fused
first stage on MultiHop-RAG and makes the CE's own top-10 worse when it only reorders it
(Phase 2.1, `docs/findings-ordering-head.md`) — or a **relational NLI stance score as a
replacement** for similarity on ArguAna, which is worse than the dense order it reorders and
below BM25 (Phase 2.3, `docs/findings-relational-rerank.md`). Both negatives point the same way:
the next ordering head must be trained on this task's own relevance conventions, not borrowed.

---

## 6. Phase 3 — one labelled end-to-end row (weeks)

"There is no end-to-end RAG ranking, because no end-to-end benchmark has been run to completion
here." That is the largest credibility hole, and the project's own `engine + <model>` rule already
solves the attribution problem.

| # | item | change | falsify / signal | status |
| --- | --- | --- | --- | --- |
| 3.1 | FRAMES adapter | **new** `tools/frames.py`; 824 questions, ~0.5 MB, ships the Wikipedia links needed to build the corpus; report retrieval recall@k and exact-match accuracy with a **labelled** local generator | if the generator dominates the variance, report retrieval-only and say so | planned |
| 3.2 | MultiHop answer accuracy + rejection | extend `tools/engine_multihop.py` under the benchmark's own answer protocol, judge labelled | the gate's rejection rate is reported beside accuracy, with the AUC and the transfer caveat | planned |
| 3.3 | Citation precision/recall | `verify.py` over composed answers: does every cited span exist, and does the answer cite the evidence that supports it? | this is the engine's own number, not the generator's; ship only with a labelled corpus of composed answers | planned |
| 3.4 | Cost axis | latency, throughput, peak RSS per query, index bytes/query, beside quality | makes "best per unit of compute" measurable rather than asserted | planned |
| 3.5 | Publish the row | `BENCHMARKS.md` | replace "cannot be claimed" with the labelled row and its limits | planned |

---

## 7. Phase 4 — scale to 18 datasets, and own the encoder (weeks; mixed)

| # | item | needs | change | falsify | status |
| --- | --- | --- | --- | --- | --- |
| 4.1 | On-disk term dictionary + chunk-id list | `[offline]` | SQLite FTS5 with `contentless_delete=1` (stdlib, no dep) or a bounded external merge; removes the residual 10M-document lexical wall | ranking parity against the in-memory index; RSS flat as corpus grows | planned |
| 4.2 | The 13 remaining BEIR datasets | `[data]` disk | build sequentially, one dataset at a time, each a resumable job; record Robust04/TREC-NEWS mirror stubs as blocked rather than skipped silently | full BEIR-18 average, stated with its dataset list | planned |
| 4.3 | Multi-corpus distillation | `[model]` | extend `tools/finetune.py` / `distill_eval.py` beyond NFCorpus (where the 33M student already recovers **~29%** of the 478M ensemble, 0.3385 → 0.3505); package in SentenceTransformers format; upload | the packaged student beats zero-shot on every training corpus, or the artefact is not shipped | planned |
| 4.4 | MTEB submission | network, external | the leaderboard wants a **model**, and one is now packageable; submit the distilled encoder | an external row with the model card's measured numbers | external |
| 4.5 | ANN quality at scale | `[offline]` | `index_store.py`/`ondisk.py`: verify IVF+PQ / RaBitQ recall against exact at 1M+ with fp32 rescore | recall retained and RSS reduced, measured not cited | planned |

**Do not** chase MTEB nDCG@10 with a 33M encoder against API-scale models. 4.4 is about having a
submittable artefact and a full-dataset average, not about winning the top row.

---

## 8. Phase 5 — make trust the headline (days–weeks; the moat)

This is the axis no 8B API model leads on, and the one a CPU-only system can win outright.

| # | item | file | change | test | status |
| --- | --- | --- | --- | --- | --- |
| 5.1 | Trust report | `tools/trust_report.py`, `make trust-report` | one deterministic, offline report aggregating poison exclusion, gold retention, negative rejection, injection flags, tenant isolation, deterministic verification, and the abstention selective-prediction suite; `--check` gates a change | `make trust-report TRUST_ARGS="--check"` | done — 12/12 checks pass; poison leak 0.20 → 0.00 at zero false refusals, gate risk@80 0.4069 vs the scalar's 0.5099 |
| 5.2 | Three-state adjudication | `src/laya_rag/verify.py` | a claim about an unmentioned topic becomes `no_coverage`/`contradicted`, not `not_measured`, via the shipped `NLIAnswerability` hook; report no-coverage recall on hand-labelled pairs | `tests/test_verify_states.py` | planned |
| 5.3 | Numeric / negation / unit / date checks | `src/laya_rag/verify.py` | deterministic span checks; ship only at precision ≥ ~0.9 on 200 labelled composed answers | `tests/test_verify_numeric.py` | planned |
| 5.4 | Grow the adversarial suite | `tools/adversarial_audit.py`, `eval/` | poison / noise / negative rejection / injection with frozen seeds; poison 5 documents → ASR measured, never cited | `make adversarial-audit --check` | planned |
| 5.5 | Trust as the README claim | `README.md`, `BENCHMARKS.md` | lead with the measured trust report, not with a retrieval rank the hardware cannot hold | `make claim-check` | planned |

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
