# R8 Distillation — Execution Plan

This document provides the exact steps to complete the R8 distillation pipeline. This is a **12-24 hour task** that can be run overnight.

---

## Objective

Train a distilled student model that:
1. Matches or comes within 0.005 of the ensemble's fused score on 5 BEIR datasets
2. Passes the submission gate (`make submission-check`)
3. Is submittable to MTEB (Apache-2.0 or MIT licensed)

---

## Prerequisites

- Python 3.10+ with virtual environment
- sentence-transformers library
- BEIR datasets installed (SciFact, NFCorpus, ArguAna, FiQA, and 1 more)
- GPU recommended (optional, speeds up training 4-8x)

---

## Step 1: Prepare Training Data (4-6 hours)

### 1.1 Download BEIR datasets

```bash
cd laya-rag

# Download 5 BEIR datasets for training
python tools/beir.py --datasets scifact,nfcorpus,arguana,fiqa,trec-covid \
  --out eval/beir/datasets
```

**Expected output:**
- `eval/beir/datasets/scifact/`
- `eval/beir/datasets/nfcorpus/` (110k train pairs)
- `eval/beir/datasets/arguana/`
- `eval/beir/datasets/fiqa/`
- `eval/beir/datasets/trec-covid/`

### 1.2 Generate teacher rankings

Run the ensemble on each dataset's train split to get soft labels:

```bash
python beir.py \
  --dataset scifact \
  --model BAAI/bge-small-en-v1.5,BAAI/bge-base-en-v1.5,BAAI/bge-large-en-v1.5 \
  --fusion hybrid \
  --train-split \
  --out eval/distill/teacher-rankings-scifact.json
```

**Expected output:** Top-K rankings for each train query, fused from 3 teachers

### 1.3 Create distillation dataset

```bash
python eval/distill_prepare.py \
  --teacher-rankings eval/distill/teacher-rankings-*.json \
  --out eval/distill/train-data.json
```

**Expected output:** `train-data.json` with query-positives labeled by teacher soft scores

---

## Step 2: Train Student Model (8-16 hours CPU, 2-4 hours GPU)

### 2.1 Run fine-tuning

```bash
python finetune.py \
  --dataset nfcorpus \
  --max-pairs 0 \
  --epochs 3 \
  --hard-negatives 1 \
  --hard-negative-pairs 40000 \
  --save-model eval/distill/student-nfcorpus \
  --batch-size 64
```

**Configuration options:**
- NFCorpus: 110k pairs, 3 epochs, ~8h CPU
- Other datasets: ~4h CPU each

**Expected output:** `eval/distill/student-nfcorpus/` with fine-tuned model

### 2.2 Evaluate student on 5 datasets

Run the evaluation script:

```bash
python distill_eval.py \
  --datasets scifact,nfcorpus,arguana,fiqa,trec-covid \
  --student BAAI/bge-small-en-v1.5 \
  --adapted eval/distill/student-nfcorpus \
  --teachers BAAI/bge-small-en-v1.5,BAAI/bge-base-en-v1.5,BAAI/bge-large-en-v1.5 \
  --out eval/distill/results.json
```

**Expected output:**
```json
{
  "scifact": {
    "zero_shot": 0.7200,
    "adapted": 0.7250,
    "teacher_ensemble": 0.7400
  },
  ...
}
```

---

## Step 3: Check Success Criteria (30 minutes)

### 3.1 Compute deltas

```python
import json

results = json.load(open("eval/distill/results.json"))

for dataset, metrics in results.items():
    delta = metrics["adapted"] - metrics["zero_shot"]
    gap = metrics["teacher_ensemble"] - metrics["adapted"]
    print(f"{dataset}: +{delta:.4f} vs zero-shot, -{gap:.4f} vs teacher")
    
# Check if all gaps <= 0.005
all_within_tolerance = all(
    metrics["adapted"] >= metrics["teacher_ensemble"] - 0.005
    for metrics in results.values()
)
```

### 3.2 Decision framework

| Outcome | Action |
|---------|--------|
| **All gaps ≤ 0.005** | Model is submittable; run `make submission-check` |
| **Some gaps > 0.005 but < 0.01** | Document honestly; submit as "near-miss" |
| **Any gap ≥ 0.01** | Iterate with more epochs or harder negatives |

---

## Step 4: Run Submission Gate (5 minutes)

```bash
make submission-check \
  model=eval/distill/student-nfcorpus \
  --out eval/distill/submission-check.json
```

**Expected output:**
```json
{
  "verdict": "SUBMITTABLE",
  "checks": {
    "loads_at_known_dimension": true,
    "outputs_unit_vectors": true,
    "deterministic": true,
    "pooling_pinned": true,
    "revision_pinned": true,
    "searchprotocol_adapter": true
  }
}
```

---

## Step 5: Document Results (30 minutes)

Create `findings-distillation.md`:

```markdown
# R8 — Distillation Results

## Training Configuration

- Student: `BAAI/bge-small-en-v1.5` (33M params)
- Teachers: `bge-small`, `bge-base`, `bge-large` (ensemble)
- Training data: NFCorpus train split (110k pairs)
- Epochs: 3
- Hard negatives: 40k triplets

## Results

| Dataset | Zero-shot | Teacher | Adapted | Gap vs Teacher |
|---------|-----------|---------|---------|----------------|
| SciFact | 0.7200 | 0.7320 | [ACTUAL] | [ACTUAL] |
| NFCorpus | 0.3391 | 0.3640 | [ACTUAL] | [ACTUAL] |
| ArguAna | [ACTUAL] | [ACTUAL] | [ACTUAL] | [ACTUAL] |
| FiQA | [ACTUAL] | [ACTUAL] | [ACTUAL] | [ACTUAL] |
| trec-covid | [ACTUAL] | [ACTUAL] | [ACTUAL] | [ACTUAL] |
| **Mean** | **[ACTUAL]** | **[ACTUAL]** | **[ACTUAL]** | **[ACTUAL]** |

## Falsifier Check

**Prediction:** Student within 0.005 of teacher ensemble on 5 datasets

**Result:** ✓ PASS / ✗ FAIL — [explanation if fail]

**Submission gate:** ✓ PASS / ✗ FAIL — [details if fail]
```

---

## Time Estimates

| Phase | CPU | GPU (A100) |
|-------|-----|------------|
| Download datasets | 1-2h | 1-2h |
| Generate teacher rankings | 4-6h | 2-3h |
| Train student (NFCorpus) | 8h | 2-3h |
| Evaluate on 5 datasets | 4-6h | 1h |
| Check criteria | 30 min | 30 min |
| Run submission gate | 5 min | 5 min |
| Document | 30 min | 30 min |
| **Total** | **18-24h** | **6-10h** |

---

## Parallelisation Opportunities

1. **Download datasets**: Run all 5 in parallel (-1-2h vs 5-10h)
2. **Teacher rankings**: Run each dataset independently
3. **Evaluation**: Score each dataset in parallel

**Optimised path:**
- Download all 5 datasets in parallel
- Generate teacher rankings while training student
- Evaluate all 5 datasets in parallel

**Best case with GPU:** **4-6 hours total**

---

## Artefacts to Produce

1. `eval/distill/teacher-rankings-*.json` (5 files)
2. `eval/distill/train-data.json` (soft labels)
3. `eval/distill/student-*/` (fine-tuned model)
4. `eval/distill/results.json` (evaluation)
5. `findings-distillation.md` (complete findings)

---

## Dependencies

- [ ] MTRAG-UN transfer completed (to unblock R3)
- [ ] ArguAna component refresh completed (to unblock R1)
- [ ] FiQA component refresh completed (to unblock R1)

---

## Success Metrics

**Quantitative:**
- Mean delta vs teacher: ≤ 0.005 (primary)
- Per-dataset max delta: ≤ 0.010 (secondary)
- Training time: < 24h CPU (efficiency)

**Qualitative:**
- Reproducible pipeline documented
- Artefacts versioned and checksummed
- Submission gate passes

---

## If Distillation Fails

### Gap > 0.01 on 1-2 datasets:
- Add more training epochs
- Increase hard-negative pool size
- Try higher learning rate or different schedule

### Gap > 0.01 on all datasets:
- Re-check teacher rankings are correctly fused
- Verify soft labels are properly computed
- Try fine-tuning on larger dataset (MS MARCO if available)

### Submission gate fails:
- Check pooling method matches family table
- Verify dimension is correct (768 for bge-small)
- Test determinism with repeated embeddings

---

## Related Documentation

- [`tools/finetune.py`](../tools/finetune.py) - Fine-tuning implementation
- [`tools/distill_eval.py`](../tools/distill_eval.py) - Evaluation script
- [`R8-distillation-plan.md`](./R8-distillation-plan.md) - Pipeline design
- [`REMAINING-ACTIONS-2026-10-30.md`](./REMAINING-ACTIONS-2026-10-30.md) - Overall status
