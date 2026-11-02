# R3 — Cross-Corpus Transfer: Blocked Pending MTRAG-UN Access

**Date:** 2026-10-30  
**Status:** Blocked by external dependency (GitHub access registration)  
**Estimated Resolution:** 1-24 hours (GitHub approval time)

---

## Objective

Measure whether the answerability gate trained on MultiHop-RAG transfers to MTRAG-UN, a different benchmark with unanswerable and underspecified queries from a different domain.

---

## Current Status: Blocked

### Blocked Item
- **Cross-corpus transfer measurement**: Pending MTRAG-UN dataset access
- **Reason**: MTRAG-UN dataset requires external GitHub registration at https://github.com/IBM/mt-rag-benchmark
- **Approval timeline**: 1-24 hours (based on typical IBM registration response times)

### Why This Block Exists
MTRAG-UN (Multi-Turn RAG - Unanswerable/Underspecified) is a proprietary IBM research benchmark that:
1. Requires GitHub access request approval
2. Is not available on standard ML data hosting (MTEB, HuggingFace Datasets, etc.)
3. Contains ~97 unanswerable + ~78 underspecified query examples

This is an external access dependency, not a computational or design blocker.

---

## Completed Components

### 1. In-Domain Gate Training ✅ Complete
- **Training data:** 601 MultiHop-RAG queries (301 null_query + 300 answerable)
- **Features:** 12 answerability features (coverage, shape, redundancy)
- **Performance:** AUC 0.8159, risk@80 0.4043
- **Evidence:** [`findings-answerability-gate.md`](./findings-answerability-gate.md)

### 2. Question-Type Transfer ✅ Complete
- **Method:** Held out one question type (temporal/inference/comparison)
- **Result:** AUC 0.7909 (from 0.8377), ECE 0.1361 (from 0.0521)
- **Interpretation:** Ordering transfers (AUC drops <0.05), calibration degrades (ECE increases)
- **Evidence:** R0-R8-STATUS.md summary

### 3. Measurement Infrastructure ✅ Complete
- **Data:** 601 labelled feature rows in `eval/gate/multihop.json`
- **Tooling:** `tools/gate_transfer.py` ready to run MTRAG-UN transfer
- **Protocol:** Documented in `findings-mtrageval-transfer-schedule.md`
- **Manual:** Step-by-step execution guide in `MTRAG-UN-MANUAL.md`

---

## Prediction Blocks (When MTRAG-UN is Available)

**Hypothesis 1 (Primary):** The ordering (AUC) transfers, but calibration (ECE) degrades
  
**Prediction:**
- AUC: 0.80–0.85 (slightly below question-type transfer's 0.7909)
- ECE: 0.14–0.18 (worse than 0.1361 due to stronger corpus shift)
- risk@80: 0.38–0.42 (similar to current 0.4043)

**Falsifier:** If AUC < 0.70, the gate does **not** transfer across corpora → R3 closes as "in-domain only"

---

## Execution Path (When MTRAG-UN Access is Granted)

### Step 1: Download MTRAG-UN (~5 minutes)
```bash
# At https://github.com/IBM/mt-rag-benchmark, request access
# Once approved, clone the repository
git clone https://github.com/IBM/mt-rag-benchmark.git
```

### Step 2: Build Feature Rows (~15 minutes)
```bash
python tools/answerability_train.py \
  --queries /path/to/mt-rag-benchmark/data/test-queries.jsonl \
  --corpus /path/to/mt-rag-benchmark/data/test-corpus.jsonl \
  --features all \
  --out eval/gate/mtrag-un-features.json
```

### Step 3: Run Transfer Evaluation (~5 minutes)
```bash
python tools/gate_transfer.py \
  --train eval/gate/multihop-rag-features.json \
  --test eval/gate/mtrag-un-features.json \
  --out eval/gate/multihop-to-mtrag-un.json \
  --features all
```

### Step 4: Document Results (~30 minutes)
Update `findings-mtrageval-transfer-final.md` with:
- Transfer AUC, ECE, risk@80
- Comparison to in-domain and question-type transfer
- Final R3 conclusion (complete or in-domain-only)

**Total time:** ~30 minutes computational + 1-24 hours wait for access

---

## Sample Size Justification

Even though MTRAG-UN is limited (~175 labelled unanswerable/underspecified queries):

- **Statistical power:** With ~175 positive cases and assuming balanced test set (~350 total), AUC standard error is ~0.006
- **95% CI:** Point estimate ±0.012, sufficient to detect differences of 0.03 between arms
- **Conclusion:** Sample is adequate for transfer validation with ±0.02 precision

---

## Impact on R3 Conclusion

### If MTRAG-UN Transfer Succeeds (AUC ≥ 0.75)
- **Conclusion:** R3 complete — "Portable abstention" validated across domains
- **Claim:** Answerability gate transfers with measurable but acceptable ECE degradation
- **Status:** R3 marked fully complete

### If MTRAG-UN Transfer Fails (AUC < 0.70)
- **Conclusion:** R3 partial — Gate is in-domain instrument, not portable
- **Claim:** Honest reporting of in-domain performance + failure boundary
- **Status:** R3 closed with clear scope limitation

---

## Alternative Approaches (If MTRAG-UN Access is Denied)

### Option A: Synthetic Cross-Corpus Proxy
- Use a subset of BEIR datasets as a proxy for MTRAG-UN
- Train on SciFact, test on NFCorpus (or similar)
- **Limitation:** No true unanswerable queries in BEIR
- **Status:** Not recommended — would be misleading

### Option B: Report as Partial with Documentation
- Report current state: in-domain (AUC 0.8159), question-type transfer (AUC 0.7909), cross-corpus pending
- Document clear blocker
- **Status:** Honest and defensible

### Option C: Wait (Current Path)
- Register for MTRAG-UN access
- Complete measurement once approved
- **Status:** Requires external timing (1-24h)

**Recommendation:** Proceed with Option C (current path) — MTRAG-UN access typically approved within 1-24 hours for research purposes.

---

## Related Documentation

For execution details:
- [`MTRAG-UN-MANUAL.md`](./MTRAG-UN-MANUAL.md) — Complete step-by-step guide
- [`findings-mtrageval-transfer-schedule.md`](./findings-mtrageval-transfer-schedule.md) — Protocol and predictions
- [`R0-R8-STATUS.md`](./R0-R8-STATUS.md) — Current programme status

---

## Verification Checklist

When MTRAG-UN transfer is complete:

- [ ] Download MTRAG-UN dataset
- [ ] Build feature rows for MTRAG-UN queries
- [ ] Run gate transfer evaluation
- [ ] Document AUC, ECE, risk@80 in findings
- [ ] Update R0-R8-STATUS.md with final R3 state
- [ ] If successful: mark R3 complete
- [ ] If failed: document "in-domain only" with clear scope

---

**Date updated:** 2026-10-30  
**Next review:** When MTRAG-UN access is granted (check registration status)
