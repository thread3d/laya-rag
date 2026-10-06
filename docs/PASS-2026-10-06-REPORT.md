# The 2026-10-06 blocked-and-owed pass — report

This is the closing report for a pass over every item the audits
(`best-in-the-world/*.md`, `ROADMAP.md` §10, `measurement-audit.md` §2/§4,
`proposal-ledger.md` §3/§6) marked **blocked** or **owed**, offline-first. It covers what was
measured, what was built, what was reconciled, what is still running, and what is genuinely
external or human. **38 commits** landed; the suite is **1410 passed, 5 deselected**, `lint` is
clean, `claim-check` is **58**, `consistency-check` is **11 of 18**, and `leakage-audit` reports
zero findings over 392 reports.

## Headline results

1. **Generator-free iteration, Stage 1 — the audit's largest bet holds.**
   One round of entity-probe pool expansion (entities the first pool found but the query did not
   name) raises gold-fact coverage on **2,255** MultiHop queries **0.8326 → 0.8937**
   (**+0.0611 [+0.0555, +0.0670]**, p<0.0001), lifts complete-evidence queries **61.2% → 72.8%**,
   and raises the **pool ceiling MAP@10 0.5955 → 0.6203** (**+0.0248**). **+0.0510 coverage and
   +0.0210 ceiling survive a shuffled-probe control**, so it is the bridge, not "search more". By
   type, **inference queries gain +0.1067 coverage — five times comparison's +0.0229**. One round
   saturates; the pool triples. `findings-generator-free-iteration.md`.

2. **The calibration row exists and the shipped answerability scalar is miscalibrated.**
   On MultiHop-RAG's 601 labelled queries the scalar reads AUC **0.5087**, ECE **0.3085**, Brier
   **0.3650**, risk@80 **0.4927**; a held-out 12-feature gate reads AUC **0.8244** / ECE **0.0892**
   on the same queries. Reliability diagram committed. `findings-calibration-row.md`.

3. **The fusion ablation is measured.** On SciFact, min-max convex fusion is the best arm
   (**0.7412**) but only **+0.0092 over RRF k=60, not significant** (p=0.139); it is significantly
   better than max-normalised CombSUM (**+0.0263**, p<0.0001). `findings-fusion-ablation.md`.

4. **Coverage is now 11 of 18.** Climate-FEVER — listed *blocked/unstaged* — was measured
   (5.4M documents, 1,535 queries, lexical **nDCG@10 0.1610**) in ~34 minutes once resumed from its
   checkpoint. `findings-climate-fever.md`.

5. **Nugget and whole-fact recall, no judge.** Claim-recall@10 **0.7438 fused vs 0.7803 CE**;
   whole-fact recall@10 **0.4644 vs 0.5452**; both **0.8480** at depth 100 — the CE's gain is
   ordering, and 15.2% of gold facts are missing from the pool. `findings-nugget-recall.md`.

6. **Quantization re-measured on the shipped encoder.** fp16 halves the index at zero measured
   recall change; int8 is 3.8× smaller for −0.66/−1.36 recall@10; PQ m=48/8-bit loses 31–41 points
   (the engine's rescored PQ path is unaffected at 0.998). `findings-quantization-probe.md`.

## Everything that landed

**Measurement and reconciliation**

- A **cross-document consistency gate** (`tools/consistency_check.py`, `make consistency-check`) —
  one authority per contested quantity, computed BEIR coverage, ten quarantined pre-fix reports;
  six audit ambiguities plus a seventh drift reconciled. `findings-consistency-gate.md`.
- The **calibration row** and the **feature-gate reliability curve**. `findings-calibration-row.md`.
- An **LLM-free nugget / claim-recall protocol**, extended with whole-fact recall.
  `findings-nugget-recall.md`.
- A **leakage audit** (dilution signature + empty held-out blocks; 392 reports, zero findings).
  `findings-leakage-audit.md`.
- **Generator-free iteration Stage 1**, with the ceiling and by-type breakdown.
  `findings-generator-free-iteration.md`.
- A **quantization probe** on real encoder vectors. `findings-quantization-probe.md`.
- **Climate-FEVER** measured (coverage 11/18). `findings-climate-fever.md`.

**Capability**

- **MTEB v2 `SearchProtocol` adapter** (`tools/mteb_search.py`, `.[mteb]`, `make mteb-eval`);
  verified against mteb 2.22.5. `findings-mteb-search-protocol.md`.
- **Retrieval-time predicate pushdown** — the on-disk dense search takes `allow`, and the engine
  screens dense/lexical/sparse before fusion. `findings-retrieval-predicate-pushdown.md`.
- **Delete-by-id** reachable by `DELETE /v1/documents`, `laya-rag delete` and a write-gated MCP
  tool; the append-only on-disk backend **tombstones** and **compacts** without re-embedding.
  `findings-delete-documents.md`.
- **`FUSION_MODE=convex`** plus the measured ablation. `findings-fusion-mode.md`.
- A **hash-chained audit log** with surface-supplied actors. `findings-audit-log.md`.
- A **budgeted, order-aware evidence context**. `findings-context-budget.md`.
- A **citation-first extractive composer** (verbatim sentences only, structured citations).
  `findings-extractive-composer.md`.
- The **bridge-conditioned gate column** behind `GATE_BRIDGE_FEATURE` — wired, not measured.
  `findings-bridge-gate-column.md`.
- **In-engine pool-shape reporting**. `findings-pool-shape.md`.
- A **labelling harness** for the two owed human sets. `findings-labelling-harness.md`.
- A **2WikiMultiHopQA adapter** (`make two-wiki`) with tests. Run queued.

**Safety / operations**

- The **one-heavy-job rule is enforced**: 26 long-running tools take an exclusive `flock`
  (`make heavy-status`); a crash leaves no stale lock. This followed a real host crash caused by
  three concurrent corpus builds plus a harness — no work was lost, because every job checkpoints,
  and `twowiki`/`musique` gained `--embed-chunk` checkpoints. `findings-heavy-job-lock.md`.

**Documentation reconciliation**

- `ROADMAP.md` §10 rewritten per item; a second status-log entry; `PROGRAMME-STATUS.md` refreshed;
  the audit's own status sections updated; and the flagged planning-doc tensions **9, 11, 12, 15**
  corrected in place (`RESEARCH.md`, `IMPROVEMENTS.md`).

## Blocked / owed items, resolved

| audit item | outcome |
| --- | --- |
| BEIR coverage "5 / 7 / 10" | **11 of 18**, computed from disk, pinned in three documents + registry |
| `voyage-02` row 0.663/0.480 vs 0.662/0.479 | reconciled to the paper's 0.6625/0.4795 |
| recalibration half-split 300 vs 500 | reconciled to the report's 300 |
| two `bench` RSS grids | explained by a recorded process baseline |
| lexical-control count "8 of 9" | corrected to 5 of 10 (then 10 corpora), pinned |
| MTEB "cannot be entered as-is" | `SearchProtocol` adapter shipped and verified; run/upload external |
| adversarial noise + injection variants | done; 12 checks |
| bridge-conditioned gate column | wired behind a flag; measurement needs a 13-feature gate run |
| delete-by-id routes | shipped (HTTP/CLI/MCP) incl. on-disk tombstones + compaction |
| lexical/sparse predicate | engine screens every channel before fusion |
| FUSION_K vs convex | ablated on SciFact; convex best but not significantly over RRF |
| BioASQ/FEVER/NQ "28 GB disk" | **stale** (471 GB free); the real block is licensing/gating |
| Climate-FEVER "blocked/unstaged" | **measured** (11 of 18) |
| ~300 claim pairs / ~200 answers | harness ready; **labels stay human** |
| contamination audit | decidable half shipped (zero findings); encoder pretraining uncheckable |
| 2WikiMultiHopQA | adapter built and tested; run queued |
| int8/binary quantization | re-measured on real vectors (fp16 free, int8 cheap, PQ costly) |
| generator-free iteration (Bet 1) | **Stage 1 holds** (see headline) |
| planning-doc tensions 9/11/12/15 | corrected in place |
| crash: concurrent heavy jobs | rule enforced by a lock in 26 tools |

## What remains, and why

- **The corpus queue is wall-clock-bound, not blocked.** Order: **Climate-FEVER done → FEVER
  running → HotpotQA → 2WikiMultiHopQA**, one at a time under the lock. FEVER holds the lock
  (checkpoint 1,356/6,666); it and the others resume with the same command. The session's rounds
  advance far faster than wall clock, so these cannot finish here — they are resumable, not lost.
- **Bet 2 (realize the ceiling).** The Stage-1 ceiling gain (+0.0248) needs a selector; the
  coverage objective already falsified on MuSiQue, so this is a training/evaluation experiment.
- **Bet 3 (encoder adaptation)** and **streaming ingestion / term dictionary / sharding** are the
  remaining large programme items; they need training and engineering weeks, not offline plumbing.
- **External:** the MTEB upload, TREC RAG, KILT, MIRACL/MMTEB, ViDoRe.
- **Human:** the ~300 claim/corpus pairs and ~200 composed answers — the harness queues and scores
  them and never writes a label.
- **Generator-side:** a local faithfulness filter, an agentic stopping rule and a judge harness all
  belong to a generator this engine does not own.

## Verification and reproduction

```sh
make test                  # 1410 passed, 5 deselected
make lint                  # src/tests compile + GUI syntax
make claim-check           # 58 prose claims vs their reports
make consistency-check     # 11 of 18, 12 sites agree
make leakage-audit         # 392 reports, zero findings
make heavy-status          # is a corpus job running?
```

New entry points: `make mteb-eval`, `make calibration-report`, `make audit-verify`,
`make nugget-recall`, `make leakage-audit`, `make labelling-queue`, `make two-wiki`,
`make generator-free-iteration`, `make quantization-probe`, `make heavy-status`.
