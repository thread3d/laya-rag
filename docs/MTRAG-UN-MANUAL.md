# MTRAG-UN Transfer — Manual for Human Execution

This document provides step-by-step instructions for downloading MTRAG-UN and running the transfer evaluation. This is a **2-hour task** that can complete R3.

---

## Step 1: Register for MTRAG-UN Access (5 minutes)

1. Go to: https://github.com/IBM/mt-rag-benchmark
2. Click "Request Access" or "Open an Issue"
3. Request access for research purposes (usually approved within 1-24 hours)
4. Clone the repository:
   ```bash
   git clone https://github.com/IBM/mt-rag-benchmark.git
   cd mt-rag-benchmark
   ```

**Expected output:** `test-queries.jsonl`, `test-corpus.jsonl`, and possibly `dev-queries.jsonl` in `data/`

---

## Step 2: Convert MTRAG-UN to Feature Rows (15 minutes)

Run this script to extract answerability features from MTRAG-UN queries:

```bash
cd /home/threaded/projects/Laya-RAG/laya-rag

python tools/answerability_train.py \
  --queries /path/to/mt-rag-benchmark/data/test-queries.jsonl \
  --corpus /path/to/mt-rag-benchmark/data/test-corpus.jsonl \
  --features all \
  --out eval/gate/mtrag-un-features.json
```

**What this does:**
- Loads each query
- Retrieves top-20 documents using the engine
- Extracts 12 answerability features (coverage, shape, redundancy)
- Outputs JSON with: `{query, label, features: {...}}`

**Expected output:** `eval/gate/mtrag-un-features.json` with ~350 rows

---

## Step 3: Run Transfer Evaluation (5 minutes)

Compare MTRAG-UN against MultiHop-RAG-fitted gate:

```bash
python tools/gate_transfer.py \
  --train eval/gate/multihop-rag-features.json \
  --test eval/gate/mtrag-un-features.json \
  --out eval/gate/multihop-to-mtrag-un.json \
  --features all

python tools/gate_transfer.py \
  --train eval/hop-rag-features.json \
  --test eval/gate/mtrag-un-features.json \
  --out eval/gate/multihop-to-mtrag-un-shape.json \
  --features shape
```

**What this does:**
- Loads the gate trained on MultiHop-RAG
- Scores it on MTRAG-UN (cross-corpus transfer)
- Computes AUC, ECE, risk@80, balanced accuracy
- Compares all-features vs shape-only features

**Expected output:** 
- `eval/gate/multihop-to-mtrag-un.json`
- `eval/gate/multihop-to-mtrag-un-shape.json`

---

## Step 4: Analyze Results (20 minutes)

Compare the results to the prediction:

### Prediction (from `findings-mtrageval-transfer-schedule.md`):

| Metric | Prediction | Decision Threshold |
|--------|------------|-------------------|
| AUC (all features) | 0.80-0.85 | ≥0.75: transfer confirmed |
| AUC (shape features) | 0.75-0.82 | ≥0.70: shape features transfer |
| ECE | 0.14-0.18 | <0.15: calibration acceptable |
| risk@80 | 0.38-0.42 | <0.45: acceptable risk |

### If results match prediction:

Update `findings-mtrageval-transfer.md`:
```markdown
# MTRAG-UN Transfer: Cross-Corpus Evaluation

The answerability gate trained on MultiHop-RAG transfers to MTRAG-UN with:

- AUC: **[ACTUAL] vs prediction of 0.80-0.85**
- ECE: **[ACTUAL] vs prediction of 0.14-0.18**
- risk@80: **[ACTUAL] vs prediction of 0.38-0.42**

Transfer is confirmed: ordering is preserved within [X] AUC, calibration degrades by [Y] ECE.
```

### If AUC < 0.70:

Update `findings-mtrageval-transfer.md`:
```markdown
The answerability gate does **not transfer** to MTRAG-UN (AUC < 0.70).
The feature distribution shift is too severe for cross-corpus deployment.
R3 closes as "in-domain only" with the MultiHop-RAG and question-type transfer
results published honestly.
```

---

## Step 5: Update R0-R8-STATUS.md

Replace the partial R3 entry with:

```markdown
| R3 | ✅ complete | Transfer AUC [X.XXX], ECE [0.XXX], risk@80 [0.XXX] |
```

or

```markdown
| R3 | ⚠️ in-domain only | MTRAG-UN AUC [X.XXX] (<0.70) |
```

---

## Step 6: Update REMAINING-ACTIONS-2026-10-30.md

Mark MTRAG-UN as complete:

```markdown
- [x] Register at IBM MTRAG GitHub
- [x] Download test set
- [x] Build MTRAG-UN features
- [x] Run transfer evaluation
- [x] Create findings-mtrageval-transfer.md
- [x] Update R0-R8-STATUS.md
```

---

## Troubleshooting

### Error: "No queries" or "Empty features"

Check that `test-queries.jsonl` is in the correct format:
```json
{"query": "...", "answerability": "unanswerable|underspecified|answerable"}
```

### Error: "Corpus not found"

Make sure `test-corpus.jsonl` has documents:
```json
{"doc_id": "doc_123", "text": "..."}
```

### Error: "Missing feature 'coverage'"

Check that the engine is working. Run a test query first:
```bash
cd laya-rag
python -c "from laya_rag.index_store import IndexStore; print(IndexStore.exists('.'))"
```

If index doesn't exist, build it first:
```bash
make index
```

### Gate file not found

Check for existing MultiHop-RAG features:
```bash
find . -name "*multihop*features*.json" | head -5
```

If not found, build from MultiHop-RAG:
```bash
python tools/answerability_train.py \
  --queries eval/benchmarks/multihop-rag/queries.jsonl \
  --corpus eval/benchmarks/multihop-rag/corpus.jsonl \
  --features all \
  --out eval/gate/multihop-rag-features.json
```

---

## Success Criteria

MTRAG-UN transfer is complete when:

- [ ] Transfer AUC measured (≥0.70 = useful transfer, <0.70 = limited)
- [ ] ECE and risk@80 recorded
- [ ] Comparison to question-type transfer documented
- [ ] `findings-mtrageval-transfer.md` created
- [ ] `R0-R8-STATUS.md` updated

---

## Time Budget

| Task | Time |
|------|------|
| GitHub registration | 5 min |
| Wait for access | 1-24 hours (do this first) |
| Feature extraction | 15 min |
| Transfer evaluation | 5 min |
| Analysis | 20 min |
| Documentation | 30 min |
| **Total active time** | **~1 hour** |
| **Elapsed time** | **1-24 hours (wait time)** |

---

## Contact / Support

If issues arise:
1. Check that all files are in expected locations
2. Verify the engine can retrieve results
3. Check that `tools/answerability_train.py` runs without errors
4. Inspect the raw data format

Review the existing gate training code in `tools/answerability_train.py` for any format mismatches.
