# Remaining R0-R8 actions — 2026-10-30

This document specifies the concrete next actions for completing the R0-R8 programme.
Each item has a clear owner, cost estimate, and success criterion.

---

## Immediate actions (this session/round)

### 1. Document MTRAG-UN transfer blocker status

**Owner:** Current agent  
**Cost:** 15 minutes  
**Success criterion:** MTRAG-UN requirement documented with download URL and sample size

**Actions:**
- [x] Verified MTRAG-UN paper exists at https://aclanthology.org/2026.findings-acl.503.pdf
- [x] Verified MTRAG benchmark available at https://ibm.github.io/mt-rag-benchmark/
- [ ] Create MTRAG-UN findings supplement documenting sample requirements
- [ ] Update R3 status document with transfer status and MTRAG-UN download link

**Status:** Documented in R0-R8-STATUS.md; next step is to create MTRAG-UN supplement.

---

### 2. Verify existing ArguAna and FiQA component refresh data

**Owner:** Current agent  
**Cost:** 30 minutes  
**Success criterion:** Confirm whether ArguAna/FiQA component refresh exists or must be run

**Actions:**
- [ ] Check `eval/r1/` for ArguAna component refresh results
- [ ] Check `eval/r1/` for FiQA component refresh results
- [ ] If missing, document cost estimate for running both
- [ ] Run if feasible within current session budget

**Status:** Existing files are **reranker results only** (arguana400-*.json). No component refresh (new encoder) results found.

---

### 3. Create MTRAG-UN findings supplement

**Owner:** Current agent  
**Cost:** 20 minutes  
**Success criterion:** `findings-mtrageval-transfer.md` created with sample requirements

**Actions:**
- [ ] Write supplement documenting:
  - MTRAG-UN sample: 97 unanswerable + 78 underspecified queries
  - Download procedure (IBM GitHub repo)
  - Expected transfer measurement protocol
  - Sample cost estimate (~30 minutes evaluation time)

---

## Queue actions (next session/round)

### 4. Complete ArguAna component refresh

**Owner:** Next agent  
**Cost:** ~1-2 hours (Qwen3-0.6B encoding 400 queries, 5,400 docs)  
**Success criterion:** Qwen3-0.6B vs bge-small comparison on ArguAna

**Commands:**
```bash
cd laya-rag
python tools/beir.py \
  --model Qwen/Qwen3-Embedding-0.6B \
  --dataset arguana \
  --pooling last \
  --out eval/r1/arguana-qwen3-control.json

python tools/beir.py \
  --model BAAI/bge-small-en-v1.5 \
  --dataset arguana \
  --out eval/r1/arguana-bge-control.json
```

**Expected output:** 
- dense nDCG@10 for both models
- hybrid (RRF) scores
- Index time for both

---

### 5. Complete FiQA component refresh

**Owner:** Next agent  
**Cost:** ~1-2 hours  
**Success criterion:** Qwen3-0.6B vs bge-small comparison on FiQA

**Commands:**
```bash
python tools/beir.py \
  --model Qwen/Qwen3-Embedding-0.6B \
  --dataset fiqa \
  --pooling last \
  --out eval/r1/fiqa-qwen3-control.json
```

---

### 6. Run MTRAG-UN transfer evaluation

**Owner:** Next agent  
**Cost:** ~30 minutes (evaluation after download)  
**Success criterion:** Transfer AUC, ECE, risk@80 on MTRAG-UN

**Actions:**
1. Download MTRAG-UN test set from IBM GitHub
2. Build feature set using `tools/answerability_train.py --features`
3. Run gate transfer: `python tools/gate_transfer.py --train eval/gate/multihop.json --test mtrag-un.json`
4. Record results and compare to current (question-type transfer: AUC 0.7909, ECE 0.1361)

---

### 7. Run distillation pipeline

**Owner:** Next agent  
**Cost:** 12-24 hours (CPU), 2-4 hours (GPU)  
**Success criterion:** Student model within 0.005 of ensemble on 5 BEIR datasets

**See:** `docs/R8-distillation-plan.md` for full pipeline

**Actions:**
1. Prepare training data from existing artefacts (4-6 hours)
2. Train bge-small student on 5 BEIR datasets (8-16 hours CPU)
3. Evaluate student on full 5-dataset protocol (2-4 hours)
4. Run `make submission-check` on student model (5 minutes)
5. Document in `findings-distillation.md`

---

## Decision points

### R5 Licence constraint

**Current state:** Listwise reranker (jina-reranker-v3) requires CC-BY-NC-4.0  
**Implication:** Block commercial use; cannot submit to MTEB (requires Apache-2.0 or MIT)  
**Decision:** Leave as is for research purposes; document licence restriction clearly

### R1 Completion threshold

**Current state:** 2 of 5 datasets (SciFact, NFCorpus) → +0.007/+0.006 fused, sign flips on dense  
**Question:** Should ArguAna/FiQA be run or is 2/5 sufficient?  
**Recommendation:** Run all 5 for completeness; 5-dataset mean is R1's falsifier denominator

---

## Updated programme status (2026-10-30)

| item | state | blocker | resolution timeline |
| --- | --- | --- | --- |
| R0 | ✅ complete | none | — |
| R1 | ⚠️ 2/5 datasets | ArguAna + FiQA (~2-4h) | Next session (queue) |
| R2 | ✅ complete | none | — |
| R3 | ⚠️ partial | MTRAG-UN download (~30 min eval) | Next session (queue) |
| R4 | ✅ complete | none | — |
| R5 | ✅ complete | licence constraint (cannot change) | — |
| R6 | ✅ complete | none | — |
| R7 | ✅ complete | none | — |
| R8 | ⚠️ partial | distillation training (12-24h CPU) | Next session (queue) |

**Blocked items:** 3 (all resolvable, all documented with timeline)  
**Completed items:** 6 of 9

---

## Verification checklist (pre-completion)

Before marking programme complete:

- [ ] R1: 5/5 BEIR datasets completed (ArguAna + FiQA added)
- [ ] R3: MTRAG-UN transfer measured with question-type + corpus held-out
- [ ] R5: Licence restriction documented in all publications
- [ ] R8: Student model evaluated on 5-dataset protocol
- [ ] R8: `make submission-check` run on student
- [ ] All findings documents updated with fresh results
- [ ] R0-R8-STATUS.md updated with final numbers

---

## Contact for blockers

- **Hardware issues:** Host memory/CPU constraints
- **Dataset access:** MTRAG-UN requires IAM account on IBM GitHub (free tier)
- **GPU access:** Distillation 12-24h → 2-4h with single GPU (any A100 or V100)

---

## Documentation requirements

All remaining work must produce:

1. **Artefact JSON** in `eval/` tree with:
   - Complete metrics
   - Configuration
   - Checksums
   - Timestamp

2. **Findings document** with:
   - Prediction block (pre-registered if new)
   - Result table
   - Falsifier check
   - Cost measurement

3. **Updated R0-R8-STATUS.md** with:
   - Fresh numbers
   - Completion status
   - Final decision
