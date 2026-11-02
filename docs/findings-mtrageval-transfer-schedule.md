# R3 supplement — MTRAG-UN transfer protocol

This document specifies the MTRAG-UN transfer measurement that completes R3's portable abstention claim.

## Background

The current abstention gate:
- **Fitted on:** MultiHop-RAG (601 rows: 301 null_query + 300 answerable)
- **In-domain AUC:** 0.8159-0.8377 (held-out splits)
- **Question-type transfer:** AUC 0.8377 → 0.7909 (ordering survives)
- **Question-type transfer:** ECE 0.0521 → 0.1361 (calibration degrades)
- **Question-type transfer:** Spread 0.18 between temporal (0.9117), inference (0.7325), comparison (0.7285)

**Missing:** Cross-corpus transfer (the original R3 promise)

## MTRAG-UN dataset

**Source:** IBM MTRAG Benchmark  
**Paper:** https://aclanthology.org/2026.findings-acl.503.pdf  
**Repository:** https://ibm.github.io/mt-rag-benchmark/  
**Download:** Requires free GitHub account registration

**Available subsets:**
- **MTRAG-UN:** 97 unanswerable queries + 78 underspecified queries
- **Total:** 175 query labels

**Format:** JSON with fields:
```json
{
  "query": "...",
  "answerability": "unanswerable|underspecified|answerable",
  "question_type": "factoid|temporal|inference|comparison|...",
  "corpus": "...",
  "gold_documents": [...]
}
```

## Measurement protocol

### Step 1: Download and prepare

```bash
# Clone and register with MTRAG
git clone https://github.com/IBM/mt-rag-benchmark.git
cd mt-rag-benchmark
# Register for test access (free on GitHub)
# Download test set after approval (~1 minute approval, ~5 minutes download)
```

### Step 2: Build feature set

Run queries through the engine to build feature rows:

```bash
python tools/answerability_train.py \
  --queries mtrag-un/test-queries.json \
  --corpus mtrag-un/test-corpus \
  --features all \
  --out eval/gate/m trag-un-features.json
```

**Expected time:** 10-15 minutes

### Step 3: Run transfer evaluation

Compare against MultiHop-RAG-fitted gate:

```bash
python tools/gate_transfer.py \
  --train eval/gate/multihop-rag-features.json \
  --test eval/gate/mtr ag-un-features.json \
  --out eval/gate/multihop-to-mtrag-un.json

python tools/gate_transfer.py \
  --train eval/gate/multihop-rag-features.json \
  --test eval/gate/mtrag-un-features.json \
  --features shape \
  --out eval/gate/multihop-to-mtrag-un-shape.json
```

**Expected time:** 2-3 minutes

### Step 4: Compute metrics

For each transfer arm:
- AUC (ROC)
- ECE (Expected Calibration Error)
- risk@80 (false accept rate at 80% coverage)
- abstention precision (P(actual unanswerable | predicted unanswerable))
- abstention recall (P(predicted unanswerable | actual unanswerable))

**Expected time:** 1 minute

## Prediction blocks

### Primary hypothesis

The **ordering** (AUC) transfers, but the **calibration** (ECE) degrades.

**Prediction:**
- AUC: 0.80-0.85 (slightly below question-type transfer's 0.7909)
- ECE: 0.14-0.18 (worse than 0.1361 due to corpus shift)
- risk@80: 0.38-0.42 (similar to current 0.4043)

**Falsifier:** If AUC < 0.70, the gate does not transfer across corpora → R3 closes as "in-domain only"

### Secondary hypothesis

The **shape features** (3 model-free) transfer better than full features (12).

**Prediction:**
- Shape AUC: 0.75-0.82 (within 0.05 of full features)
- Full AUC: 0.78-0.84
- Gap: < 0.03

**Falsifier:** If shape features underperform full by > 0.05 AUC, model-dependent features are required for transfer

## Sample size justification

**Power calculation:**

- Transfer sample: 97 unanswerable + 78 underspecified = 175 positive cases
- Assuming balanced answerable/unanswerable in test set: ~350 total queries
- AUC standard error: ~1/√(n_pos × n_neg) = 1/√(175×175) ≈ 0.006

**Confidence interval:**
- Point estimate 0.80 ± 0.012 (95% CI)
- Can detect difference of 0.03 between arms with power > 0.8

**Verdict:** Sample is sufficient for transfer measurement with ±0.02 precision

## Artefacts to produce

1. **eval/gate/mtrag-un-features.json**
   - 175+ query feature rows
   - Fields: query, label, features (dict)

2. **eval/gate/multihop-to-mtrag-un.json**
   - Transfer metrics (AUC, ECE, risk@80)
   - Per-type breakdown
   - Feature subset comparison

3. **findings-mtrageval-transfer.md**
   - Prediction block
   - Results table
   - Falsifier check
   - Comparison to MultiHop-RAG (in-domain, question-type transfer)

## Cost estimates

| Task | Time | Dependencies |
| --- | --- | --- |
| Download MTRAG-UN | 10-30 min | GitHub account (free) |
| Build features | 10-15 min | Engine installed |
| Transfer evaluation | 5 min | Feature files |
| Document | 30 min | Results |
| **Total** | **1-2 hours** | One-time download |

## Integration with current results

Current R3 status:
- In-domain: AUC 0.8159, risk@80 0.4042 ✅
- Question-type transfer: AUC 0.7909, ECE 0.1361 ✅
- Cross-corpus transfer: **Not measured** ⚠️

With MTRAG-UN:
- In-domain: AUC 0.8159, risk@80 0.4042
- Question-type: AUC 0.7909, ECE 0.1361
- Cross-corpus: AUC [PREDICTED], ECE [PREDICTED]
- **Complete R3 claim:** "Portable abstention" becomes "Portable within domain, measurable transfer"

## Decision framework

**If transfer AUC ≥ 0.75:**
- Claim: "Gate transfers across corpora with < 0.07 AUC degradation"
- Action: Complete R3, publish full numbers

**If transfer AUC < 0.75 but ≥ 0.65:**
- Claim: "Gate shows weak transfer; domain-specific fitting required"
- Action: State as limitation, publish honestly

**If transfer AUC < 0.65:**
- Claim: "Gate does not transfer; in-domain instrument only"
- Action: R3 closes as "partial" with clear scope

## Execution checklist

- [ ] Register at IBM MTRAG GitHub (free)
- [ ] Download test set (~50 MB)
- [ ] Run `tools/answerability_train.py` on MTRAG-UN
- [ ] Run `tools/gate_transfer.py` for transfer metrics
- [ ] Create `findings-mtrageval-transfer.md`
- [ ] Update `R0-R8-STATUS.md`
- [ ] Update `REMAINING-ACTIONS-2026-10-30.md`
- [ ] Update this document with actual results

---

## Related documentation

- [`findings-answerability-gate.md`](findings-answerability-gate.md) - Gate construction and in-domain performance
- [`findings-abstention-transfer.md`](findings-abstention-transfer.md) - Question-type transfer
- [`R0-R8-STATUS.md`](R0-R8-STATUS.md) - Complete programme status
- [MTRAG-UN paper](https://aclanthology.org/2026.findings-acl.503.pdf) - Dataset description
- [MTRAG GitHub](https://ibm.github.io/mt-rag-benchmark/) - Dataset download
