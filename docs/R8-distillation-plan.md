# R8 — distillation pipeline implementation plan

This document specifies the distillation pipeline that will produce the R8 submittable artefact from
the current ensemble performance.

## Goal

Distill the ensemble performance into a single submittable encoder that:
1. Matches or comes within 0.005 of the ensemble's fused score on 5 BEIR datasets
2. Passes the submission gate (`make submission-check`)
3. Can be submitted to MTEB via SearchProtocol

## Current state

**Ensemble performance (5 datasets):**
- SciFact: `hybrid-biencoder` 0.7320
- NFCorpus: `hybrid-biencoder` 0.3633
- SciFact 2026 encoder: Qwen3-0.6B hybrid 0.7389
- NFCorpus 2026 encoder: Qwen3-0.6B hybrid 0.3694

**Submission gate (complete):**
- Loads at known dimension ✓
- Outputs unit vectors ✓
- Deterministic across calls ✓
- Pooling from verified family table ✓
- Revision pinned ✓
- SearchProtocol adapter present ✓

## Pipeline design

### Phase 1: Training data preparation

```python
# tools/distillation_prepare.py

1. Generate query pairs: (query, positive_doc, negative_docs) from:
   - BEIR 5 datasets (SciFact, NFCorpus, ArguAna, FiQA, plus 2 more)
   - MultiHop-RAG (if transferring to reasoning-heavy)
   
2. Create soft labels:
   - Run full ensemble on all pairs
   - Record scores from: bm25, dense (if used), hybrid, reranker
   - Compute weighted average as soft label

3. Output format:
   {
     "query": "...",
     "positive": "doc_text",
     "negatives": [...],
     "soft_label": float,
     "dataset": "SciFact"
   }
```

**Cost estimate:** 2-4 hours for 5 datasets at ~500k queries total

### Phase 2: Student model training

```python
# tools/distillation_train.py

Options:
1. Fine-tune bge-small (33M, fastest, most memory-efficient)
2. Fine-tune Qwen3-0.6B (596M, better distillation target)

Training configuration:
- Loss: CosineSimilarityLoss with soft labels
- Batch size: 64-128 (depends on GPU, if available)
- Epochs: 3-5
- Learning rate: 2e-5 (default for BGE fine-tuning)

Output: Fine-tuned model checkpoint
```

**Cost estimate:**
- bge-small: 1-2 hours on single GPU, 8-16 hours CPU
- Qwen3-0.6B: 3-6 hours on single GPU, 24-48 hours CPU

### Phase 3: Evaluation

```python
# tools/distillation_eval.py

1. Run student on 5 BEIR datasets
2. Compare to ensemble:
   - Per-dataset nDCG@10
   - Macro average
   - Paired bootstrap interval

3. Falsifier check:
   - If any dataset > 0.005 below ensemble, NOT SUBMITTABLE
   - Document all gaps

4. Gate check:
   - Run `make submission-check`
   - Verify still passes
```

**Cost estimate:** 1-2 hours for full evaluation

## Implementation steps

### Step 1: Prepare training data (4-6 hours)

Use existing evaluation runs to generate soft labels. Query the existing artefacts:
- `eval/benchmarks/multihop-rag/*.json`
- `eval/beir/*.json` (5 datasets)
- `eval/r1/*.json` (reranker results)

Extract query-document scores from each component, compute weighted average.

### Step 2: Train student model (8-48 hours)

**Option A (bge-small):**
- Start from `BAAI/bge-small-en-v1.5`
- Fine-tune on prepared data
- Expected: ~0.01-0.02 below ensemble on most datasets

**Option B (Qwen3-0.6B):**
- Start from `Qwen/Qwen3-Embedding-0.6B`
- Fine-tune on prepared data
- Expected: ~0.005-0.01 below ensemble

**Decision:** Start with Option A (bge-small) - faster iteration, still competitive.

### Step 3: Evaluate (2-4 hours)

Run full evaluation suite:
- 5 BEIR datasets
- MTEB SearchProtocol compatibility check
- Submission gate verification

### Step 4: Document (1 hour)

Write `findings-distillation.md` with:
- Training configuration
- Per-dataset results vs ensemble
- Submission decision
- Artefact artefacts (model config, checkpoint path, hashes)

## Artefacts

Expected outputs:
1. `eval/distillation/train-data.json` - prepared training data
2. `eval/distillation/student-model/` - fine-tuned model
3. `eval/distillation/results.json` - evaluation results
4. `findings-distillation.md` - complete findings document
5. `make submission-check` → SUBMITTABLE

## Blocker resolution

**Current blocker:** Corpus-scale distillation (hours-days)

**Resolution:** This pipeline is designed to complete within 12-24 hours on a CPU host, 2-4 hours with one GPU. The bottleneck is data preparation, not training.

**Parallelisation opportunities:**
- Data preparation can run per-dataset independently
- Evaluation can run in parallel across datasets
- Only training must be sequential

## Success criteria

1. **Primary:** Student passes all 5 BEIR datasets within 0.005 of ensemble
2. **Secondary:** Submission gate passes
3. **Tertiary:** Documented and reproducible

**If success criteria not met:** Document the gap honestly, keep the gate output, and either:
- Iterate with better training (more data, different loss)
- Submit as "demonstration gate" rather than "submission-worthy model"

---

## Related findings

- `findings-distillation.md` - initial distillation on NFCorpus only
- `findings-submission-gate.md` - submission criteria
- `best-in-the-world-2026-10.md` R8 - original specification
