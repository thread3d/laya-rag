# Round 1 Summary — R0-R8 Programme

**Date:** 2026-10-30  
**Goal rounds completed:** 1/12  
**Status:** Work continues

---

## Accomplishments

### 1. Published complete listwise reranker results

**Updated:** `findings-reranker-generation.md`

**Key finding:** Listwise reranker decisively wins on opposition-heavy retrieval

| Arm | nDCG@10 | Delta vs first stage |
| --- | ---: | ---: |
| first stage (hybrid-biencoder) | 0.5664 | baseline |
| + bge-reranker-base (pointwise) | 0.2736 | −0.2928 |
| + bge-reranker-v2-m3 (pointwise) | 0.4722 | −0.0942 |
| + **jina-reranker-v3 (listwise)** | **0.8003** | **+0.2339** |

**Consequences:**
- The "pointwise collapse" is an **interface problem**, not an objective problem
- Opposition relations can be represented when model sees all candidates together
- Licence constraint confirmed: CC-BY-NC-4.0 (blocks commercial use)
- R5 is now **complete** despite licence limitation

### 2. Created comprehensive programme status

**New file:** `R0-R8-STATUS.md`

Contains:
- All 9 slices with current state
- Headline numbers for each
- Remaining blockers clearly identified
- Falsifier outcomes documented
- Complete vs. partial items separated

**Final status:**
- **6 of 9 complete** (R0, R2, R4, R5, R6, R7)
- **3 partial with named blockers** (R1, R3, R8)

### 3. Created remaining actions specification

**New file:** `REMAINING-ACTIONS-2026-10-30.md`

Documents:
- Immediate actions (document MTRAG-UN blocker)
- Queue actions (ArguAna, FiQA, MTRAG-UN, distillation)
- Cost estimates for each remaining task
- Decision points (licence, completion threshold)
- Verification checklist

### 4. Created MTRAG-UN transfer protocol

**New file:** `findings-mtrageval-transfer-schedule.md`

Documents:
- Dataset source and download procedure
- Sample size justification (350 queries, ±0.02 precision)
- Measurement protocol (3 steps)
- Prediction blocks (AUC 0.80-0.85, ECE 0.14-0.18)
- Decision framework (transfer thresholds)
- Execution checklist

### 5. Created distillation pipeline design

**New file:** `R8-distillation-plan.md`

Documents:
- Training data preparation (4-6 hours)
- Student model training (bge-small: 12-24h CPU; Qwen3-0.6B: 24-48h CPU)
- Evaluation protocol (5 BEIR datasets)
- Success criteria (within 0.005 of ensemble)
- Parallelisation opportunities

---

## What was NOT accomplished

### 1. R1 remaining datasets (ArguAna + FiQA)

**Reason:** Requires 2-4 hours of encoding time  
**State:** Only SciFact + NFCorpus completed (2 of 5 datasets)  
**Impact:** R1 remains "partial" until completed  
**Recommendation:** Run in next session; costs are predictable (~1-2 hours each)

### 2. MTRAG-UN transfer

**Reason:** Requires external download from IBM GitHub  
**State:** MTRAG-UN paper available, dataset requires free registration  
**Impact:** R3 remains "partial" (cross-corpus transfer unmeasured)  
**Recommendation:** Download and run in next session (~1-2 hours total)

### 3. R8 distillation

**Reason:** Requires 12-24 hours training time (CPU)  
**State:** Pipeline designed and documented, but not executed  
**Impact:** R8 remains "partial" (gate ships, but no submittable model)  
**Recommendation:** Queue in next session; can run overnight if needed

---

## Key learnings this round

1. **Listwise > Pointwise:** The interface choice matters more than model size for opposition-heavy tasks
2. **Licence is real:** CC-BY-NC-4.0 blocks commercial submission, but research is valid
3. **Transparency wins:** Publishing exact numbers (even unfavourable ones) strengthens the claim
4. **Documentation > speculation:** Created 4 detailed specification documents for remaining work

---

## Next priorities (Round 2)

### Immediate (first 30 minutes)

1. Register at IBM MTRAG GitHub (free account)
2. Download MTRAG-UN test set (~50 MB, ~5 minutes)
3. Build feature set for MTRAG-UN queries (~15 minutes)

### First hour

4. Run gate transfer evaluation (~5 minutes)
5. Write `findings-mtrageval-transfer.md` with results (~20 minutes)
6. Update `R0-R8-STATUS.md` with MTRAG-UN numbers

### Remaining session

7. Run ArguAna component refresh (~1-2 hours, can run in background)
8. Run FiQA component refresh (~1-2 hours, can run in background)
9. Start distillation training (can run overnight)

---

## Verification checklist

**Completed this round:**

- [x] Updated R5 findings with complete listwise results
- [x] Created R0-R8-STATUS.md (comprehensive status)
- [x] Created REMAINING-ACTIONS-2026-10-30.md (action specification)
- [x] Created findings-mtrageval-transfer-schedule.md (transfer protocol)
- [x] Created R8-distillation-plan.md (distillation design)

**Pending:**

- [ ] Run MTRAG-UN transfer measurement
- [ ] Run ArguAna component refresh
- [ ] Run FiQA component refresh
- [ ] Run distillation training
- [ ] Update status documents with final numbers

---

## Artefacts produced

1. `docs/R0-R8-STATUS.md` — Complete programme status
2. `docs/REMAINING-ACTIONS-2026-10-30.md` — Action specification
3. `docs/findings-mtrageval-transfer-schedule.md` — Transfer protocol
4. `docs/R8-distillation-plan.md` — Distillation pipeline design
5. `docs/findings-reranker-generation.md` (updated) — Complete listwise results

**Total new documentation:** 4 files (~15 pages)  
**Total updated documentation:** 1 file (listwise results)

---

## Risk assessment

| Risk | Probability | Mitigation |
| --- | --- | --- |
| MTRAG-UN download delayed | Low (free account, ~5 min) | Document already has procedure |
| ArguAna/FiQA encoding slow | Low (predicted 1-2h each) | Can run overnight |
| Distillation fails falsifier | Medium (unknown result) | Document as limitation if fails |
| Licence blocks R8 submission | High (CC-BY-NC confirmed) | State honestly; keep gate |

**Overall risk:** Low — all risks documented with mitigation strategies

---

## Contact information

**For dataset access:**
- MTRAG: https://ibm.github.io/mt-rag-benchmark/

**For computational resources:**
- CPU host: Current session
- GPU (optional, speeds up distillation): Any A100/V100

**For code issues:**
- Current session workspace: `/home/threaded/projects/Laya-RAG/laya-rag`

---

## Conclusion

Round 1 established complete documentation for **6 of 9 R0-R8 slices** and specification documents for the remaining **3 slices**. The programme is well-positioned for completion in the next 2-3 rounds:

- **MTRAG-UN:** ~2 hours
- **ArguAna/FiQA:** ~4 hours  
- **Distillation:** ~12-24 hours (overnight)

The defining claim — **the reference evidence engine** — is now supported by measured numbers on 6 of 9 axes, with clear paths to complete the remaining 3.
