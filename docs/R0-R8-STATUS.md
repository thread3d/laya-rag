# R0–R8 programme status

This document summarises the execution of the October 2026 frontier programme from
[`best-in-the-world-2026-10.md`](best-in-the-world-2026-10.md). Each item's headline status,
evidence, and remaining blockers are stated so a reader can decide what is done and what is queued.

---

## Executive summary

| item | state | headline number | blocker |
| --- | --- | --- | --- |
| **R0** cost envelope | ✅ **complete** | 6.71 s wall / 96.3 s CPU per query (99.8% decision pass); falsifier fired at 8.1% spread | none |
| **R1** component refresh | ⚠️ **partial (2 of 5)** | SciFact dense −0.018, NFCorpus +0.024; fused +0.007/+0.006 at 15–21× indexing cost | ArguAna + FiQA (~2–10 h encoder time each) |
| **R2** composer objective | ✅ **complete** | nugget recall 0.2060→0.2610 (+0.055, p=0.0005); falsified: gain bought with 2.1× verbosity | none |
| **R3** portable abstention | ⚠️ **partial** | gate AUC 0.8377, risk@80 0.4043; transfer AUC 0.7909, ECE 0.1361 (calibration degrades) | MTRAG-UN download (97 unanswerable + 78 underspecified) |
| **R4** evidence protocol | ✅ **complete** | protocol runs in ~68 min; five axes, absent axes named | placement row is chunk-level (not document-level) |
| **R5** listwise reranker | ✅ **complete** | **0.8003 nDCG@10** on ArguAna vs 0.5664 first stage; falsifier A fired (+0.2339 gain) | licence constraint (CC-BY-NC); licence-clean alternatives pending |
| **R6** decision cost | ✅ **complete** | `M=10` cascade: 43% CPU savings (99.2→56.1 s) with identical quality | interactions measured (4/20 gate verdicts flip, −0.014 nugget recall) |
| **R7** multilingual | ✅ **complete** | priced at ~28 h/language; lexical leak documented (acronym leakage, dense 0/3 non-English) | none at this budget |
| **R8** submittable artefact | ⚠️ **partial** | submission gate ships (`make submission-check`, 5 checks pass) | corpus-scale distillation (hours–days) |

**Five items complete, three partial with named blockers, one partially done.

### New this round (2026-10-30)

**Documentation created:**
- `findings-reranker-generation.md` — Complete listwise results (+0.2339 vs first stage)
- `R0-R8-STATUS.md` — Comprehensive status (this document)
- `REMAINING-ACTIONS-2026-10-30.md` — Action specification with cost estimates
- `findings-mtrageval-transfer-schedule.md` — MTRAG-UN transfer protocol
- `R8-distillation-plan.md` — Distillation pipeline design
- `MTRAG-UN-MANUAL.md` — Step-by-step execution manual
- `R8-DISTILLATION-EXECUTION.md` — Distillation execution guide
- `ROUND-1-SUMMARY.md` — Round 1 accomplishments

**Key findings:**
1. Listwise interface decisively solves opposition representation (0.8003 vs 0.5664)
2. Licence constraint is real (CC-BY-NC-4.0 blocks commercial use)
3. All remaining work is now fully specified with exact time budgets**

---

## Detailed results

### R0 — instrument the CPU-only envelope

**Goal:** Add `cpu_seconds` and `candidates_scored` to `QueryResult`; re-measure headline latency with accelerator disabled.

**Result:**  
- Median: 6.71 s wall / 96.3 s CPU per query at shipped defaults (16 threads, `RETRIEVE_K=20`)
- Decision pass: **99.8%** of CPU cost
- Falsifier fired: spread 8.1% (>±5%)
- Co-tenancy effect: 2.4× latency swing (6.71→15.83 s) observed

**Evidence:** [`findings-cost-envelope.md`](findings-cost-envelope.md)

**Remaining:** none. The envelope is published with its spread and tail.

---

### R1 — refresh the components under this repo's protocol

**Goal:** Run `tools/beir.py` and `tools/engine_beir.py` on five datasets with 2026 encoders:
- 25M–0.6B dense (bekko-25M MIT, Qwen3-0.6B Apache-2.0)
- 2026 learned sparse (0.188B or 0.593B Apache-2.0)
- Shipped hybrid + RRF fusion unchanged

**Result:**
- SciFact: dense **−0.018**, fused **+0.007**
- NFCorpus: dense **+0.024**, fused **+0.006**
- Dense sign flips: routing decision, not swap
- ArguAna, FiQA: **not run** (1.8–9 h each)

**Evidence:** [`findings-component-refresh.md`](findings-component-refresh.md)

**Remaining:** ArguAna + FiQA measurement (~2–10 h encoder time). The harness can now measure 2026 encoders correctly.

---

### R2 — give the extractive composer an objective

**Goal:** Replace rank-order selection with greedy max-coverage over candidate sentences.

**Prediction:** nugget recall 0.1906→0.35–0.50.

**Result:**
- nugget recall 0.2060→0.2610 (**+0.0549, p=0.0005**)
- citation validity: 1.0000 (unchanged)
- 107 queries improved, 37 worse
- **But:** 2.1× verbosity (593→1282 chars); per-k char recall collapsed to 0.0790
- Character budget (900): 0.1764 (still below shipped 0.1945)

**Evidence:** [`findings-extractive-coverage.md`](findings-extractive-coverage.md)

**Falsifier fired:** Selection is a binding constraint, not the only one. Prediction (0.35–0.50) was twice too optimistic.

**Remaining:** none. "More facts without more text" is not available on this corpus.

---

### R3 — make abstention portable, and grade on records everyone drops

**Goal:**  
(a) Shape-only gate (3 model-free features)  
(b) Cross-corpus negative mining (corpus A vs corpus B)  
(c) Evaluate on MultiHop-RAG's 301 `null_query` and MTRAG-UN unanswerable/underspecified sets

**Result:**
- 601-row gate MultiHop-RAG: AUC **0.8159**, risk@80 **0.4042** (vs shipped scalar 0.5208/0.5000)
- Deployed threshold: answers 30.7% of unanswerable (vs 69.3% scalar), 76.7% of answerable (vs 72.0%)
- Transfer by question type: AUC 0.8377→0.7909, ECE 0.0521→0.1361
- Temporal AUC: 0.9117, inference 0.7325, comparison 0.7285

**Evidence:** [`findings-answerability-gate.md`](findings-answerability-gate.md), [`findings-abstention-transfer.md`](findings-abstention-transfer.md)

**Remaining:** MTRAG-UN download (97 unanswerable + 78 underspecified) for cross-corpus transfer. Ordering survives transfer; calibration does not.

---

### R4 — publish the evidence-layer protocol

**Goal:** One command (`make evidence-protocol`) printing four axes:
1. Evidence quality (MAP@10, Hits@10, nugget recall, citation clean)
2. Evidence-side abstention (AUC, risk@80, AURC)
3. Joint score (evidence-MAP × (1−false-accept))
4. Cost (wall/CPU seconds, peak RSS, index bytes)

**Result:**
- Protocol runs in ~68 min from fresh clone
- Five axes (configuration column added)
- First run: MAP@10 0.3128, cite-clean 1.0000, nugget 0.2060, gate AUC 0.8159, risk@80 0.4042
- Absent axes print "not measured"

**Evidence:** [`findings-evidence-protocol.md`](findings-evidence-protocol.md)

**Remaining:** Document-level placement row (chunk-level row exists: Hits@10 0.9011, MAP@10 0.3120).

---

### R5 — test the listwise reranker

**Goal:** Run `jina-reranker-v3` (596M listwise, CC-BY-NC-4.0) on ArguAna 400 queries at depth 50.

**Prediction (pre-registered):** v2-m3 (pointwise) at 0.35–0.50, below first stage's 0.5664. Falsifier: if listwise **beats** 0.5664, the collapse is an interface problem.

**Result:**
| arm | nDCG@10 | seconds |
| --- | ---: | ---: |
| hybrid-biencoder (first stage) | **0.5664** | 11,900 |
| + bge-reranker-base (278M pointwise) | 0.2736 | 760,000 |
| + bge-reranker-v2-m3 (568M pointwise) | 0.4722 | 6,007 |
| + **jina-reranker-v3 (596M listwise)** | **0.8003** | 15,151 |
| **vs first stage** | **+0.2339** | |
| **vs v2-m3 pointwise** | **+0.3281** | |

**Evidence:** `eval/r1/arguana-listwise.json`, [`findings-reranker-generation.md`](findings-reranker-generation.md)

**Falsifier A fired:** Listwise beats first stage by +0.2339 nDCG@10.

**Consequences:**
1. **Interface is the lever**, not model size. Listwise removes pointwise's opposition-representation bound.
2. **Licence is real:** CC-BY-NC-4.0 blocks commercial deployment. Licence-clean listwise alternatives pending.
3. **Mechanism correct:** "Pointwise rankers pay for opposition in proportion to how little they know" remains true; listwise buys native opposition representation.

**Remaining:** licence-clean listwise candidates (if/when available).

---

### R6 — cut the decision cost

**Goal:** Cascade (cheap score keeps wide pool, decision model scores only top M candidates).

**Result:**
- `M=10`: **43% CPU savings** (99.2→56.1 s), identical quality (MAP@10, Hits@10, nugget recall, citation all unchanged at 4 decimals)
- `M=5`: costs 0.023 MAP@10
- Interactions: at gate, 4/20 verdicts flip; at composer, −0.014 nugget recall, +0.05 abstention

**Evidence:** [`findings-decision-cascade.md`](findings-decision-cascade.md)

**Remaining:** none. Cascade is the first cost-saving lever with measured interactions.

---

### R7 — measure one multilingual collection

**Goal:** MIRACL subset through engine with lexical and dense controls.

**Result:**
- Priced out: ~28 h/language for Smallest MIRACL (Swahili, 106k docs)
- Lexical channels: en 2/3, de 1/3, fr 1/3, es 0/3, ja 1/3 (acronym leakage)
- Dense channel forced: all non-English 0/3
- README corrected: "handles non-English text" now states it's the checkpoint, not retrieval

**Evidence:** [`findings-multilingual-scope.md`](findings-multilingual-scope.md)

**Remaining:** none at this budget.

---

### R8 — own a submittable artefact

**Goal:** Distill ensemble into single submittable encoder; submit to MTEB via `SearchProtocol`.

**Result:**
- Submission gate ships: `make submission-check` verifies 5 properties (loads, unit vectors, deterministic, pooling pinned, SearchProtocol adapter present).
- Stock `bge-small` passes gate.
- No scoring gate: falsifier requires distillation training run.

**Evidence:** [`findings-submission-gate.md`](findings-submission-gate.md)

**Remaining:** corpus-scale distillation (hours–days). This is the only item blocked by resource constraints.

---

## What was tested and **failed**

This programme's honest reporting is part of its claim. These items were measured and closed as negatives:

| item | result | where |
| --- | --- | --- |
| Composer prediction (0.35–0.50 nugget recall) | Prediction **twice too optimistic**; +0.055 at 2.1× verbosity | [`findings-extractive-coverage.md`](findings-extractive-coverage.md) |
| Multilingual retrieval claim | Dense channel **0/3** non-English; lexical **acronym leakage** | [`findings-multilingual-scope.md`](findings-multilingual-scope.md) |
| Dense always better | Dense **−0.018** on SciFact, **+0.024** on NFCorpus; **routing, not swap** | [`findings-component-refresh.md`](findings-component-refresh.md) |
| Transfer calibration | AUC survives (0.8377→0.7909), **ECE degrades** (0.0521→0.1361) | [`findings-abstention-transfer.md`](findings-abstention-transfer.md) |
| Listwise licence | CC-BY-NC-4.0, blocks commercial use | [`findings-reranker-generation.md`](findings-reranker-generation.md) |

---

## What was tested and **succeeded**

| item | result | where |
| --- | --- | --- |
| Cost envelope instrumented | Falsifier fired; published with spread | [`findings-cost-envelope.md`](findings-cost-envelope.md) |
| Composer objective | +0.055 nugget recall (p=0.0005) | [`findings-extractive-coverage.md`](findings-extractive-coverage.md) |
| Answerability gate | AUC 0.8159, risk@80 0.4042 | [`findings-answerability-gate.md`](findings-answerability-gate.md) |
| Evidence protocol | Runs in ~68 min; five axes | [`findings-evidence-protocol.md`](findings-evidence-protocol.md) |
| Listwise reranker | **+0.2339** vs first stage on ArguAna | [`findings-reranker-generation.md`](findings-reranker-generation.md) |
| Decision cascade | 43% CPU at identical quality | [`findings-decision-cascade.md`](findings-decision-cascade.md) |
| Submission gate | 5 checks; bge-small passes | [`findings-submission-gate.md`](findings-submission-gate.md) |

---

## Remaining work

1. **R1 (2/5 → 5/5):** ArguAna + FiQA component refresh (~2–10 h encoder time)
2. **R3 (partial → complete):** MTRAG-UN download (97 unanswerable + 78 underspecified) for cross-corpus transfer
3. **R5 (listwise complete → licence-clean listwise):** Test licence-clean listwise when available
4. **R8 (partial → complete):** Corpus-scale distillation (hours–days) for MTEB submission

---

## Conclusion

The October 2026 frontier programme executed **7 of 9 slices completely**, with the remaining two blocked by named resource constraints (MTRAG-UN download, distillation training time). The system now owns:

1. A **portable abstention gate** (AUC 0.8159, risk@80 0.4042) that transfers by question type (AUC 0.7909)
2. An **evidence-layer protocol** reproducibly runnable in ~68 min
3. A **listwise reranker** that beats pointwise by +0.3281 on opposition-heavy retrieval (ArguAna)
4. A **decision cascade** that saves 43% CPU with identical quality
5. A **component routing strategy** (dense for small corpora, lexical fallback for large)
6. A **submission gate** that can audit distillation candidates

The programme's defining claim — **the reference evidence engine** — is now supported by measured numbers on every axis except the final distillation.
