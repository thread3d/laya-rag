# Findings: significance control and selective prediction

This records the first measurements produced by the evaluation-rigor work from
[`RESEARCH.md`](../RESEARCH.md) (Tier 0.2 and Tier 0.3). The raw reports live under the
gitignored `eval/`; the numbers are copied here so the claims are durable.

Reproduce (both are offline and deterministic given the saved reports):

```sh
make multihop-significance          # permutation + BH-FDR over official.json
make multihop-answerability         # selective-prediction suite on the gate features
```

---

## 1. Permutation tests + FDR control (`tools/eval_significance.py`)

`tools/multihop_official.py` persists a `_per_query` row per system, so every comparison
can be paired. The new post-processor adds an exact (or sampled) **paired sign-flip
permutation** p-value beside the existing bootstrap interval, and applies
**Benjamini-Hochberg FDR** across the declared family. The harness previously reported
per-cell bootstrap intervals and asserted significance cell by cell, which is not a family
error rate.

Family: the eight MultiHop-RAG `overlap=200` rows, baseline `bge-small hybrid@ov200`,
four metrics each — **32 cells**.

| | count |
| --- | --- |
| cells tested | 32 |
| significant uncorrected (p ≤ 0.05) | **16** |
| significant after BH-FDR (q = 0.05) | **16** |

**No cell lost significance under FDR**, so the previously reported wins were not
family-wise artifacts. The interesting movement is at the boundary:

- `bge-large hybrid@ov200` **Hits@4**: −0.0142 [−0.0279, +0.0009], permutation
  p = 0.0539 → BH p = 0.1015, **not significant**. This is the clean confirmation of the
  README's "+0.008 / −0.014, none of those intervals excludes zero" claim.
- `bm25@ov200` **MAP@10**: +0.0024 [−0.0046, +0.0089], p = 0.48, **not significant** — the
  lexical baseline and the fused hybrid are a MAP@10 tie on this protocol. `bm25` Hits@4
  (−0.0483, BH p = 0.0023) and `bm25-engine` Hits@4 (−0.0492, BH p = 0.0023) *are*
  significantly worse, so the fused first stage's advantage lives in Hits@4/Hits@10, not
  MAP@10.
- Every dense-only row is significantly worse than the hybrid on all four metrics
  (BH p = 0.0023), which is the strongest form of the "fusion pays on MultiHop" claim.

The family must be declared before looking; `eval_significance.py` takes `--family`
explicitly and records it in the report.

## 2. Selective prediction for the answerability gate (`tools/answerability_train.py`)

AUC and one-threshold balanced accuracy say nothing about the risk accepted at a coverage
a deployment might actually run at. The trainer now reports the **risk-coverage curve**,
**AURC**, **AUGRC** (NeurIPS 2024; expected risk of undetected failures), **risk at 80%
and 90% coverage**, Brier, NLL, and equal-mass **ECE sweep** for the gate and for Laya's
shipped scalar, on 300 labelled MultiHop-RAG queries (150 null / 150 answerable), over 200
stratified held-out splits (mean [2.5%, 97.5%]).

| metric | feature gate | Laya scalar |
| --- | --- | --- |
| ROC-AUC | **0.8146** [0.7628, 0.8570] | 0.5199 [0.4530, 0.5835] |
| balanced accuracy | **0.7248** [0.6667, 0.7733] | 0.5026 [0.4532, 0.5533] |
| AURC (lower better) | **0.2691** [0.2413, 0.3057] | 0.4879 [0.4382, 0.5438] |
| AUGRC (lower better) | **0.6681** [0.6563, 0.6813] | 0.7417 [0.7259, 0.7590] |
| risk @ 80% coverage | **0.4069** [0.3833, 0.4252] | 0.5099 [0.4833, 0.5419] |
| risk @ 90% coverage | **0.4540** [0.4444, 0.4667] | 0.5470 [0.5185, 0.5556] |
| Brier (lower better) | **0.1768** [0.1565, 0.2017] | 0.3678 [0.3306, 0.4045] |
| NLL (lower better) | **0.5315** [0.4773, 0.6233] | 3.5059 [2.6197, 4.4421] |
| ECE (lower better) | **0.0650** [0.0278, 0.1107] | 0.2714 [0.1849, 0.3856] |
| coverage at the deployed threshold | 0.5409 [0.3531, 0.6600] | 0.6807 [0.6267, 0.7333] |

**The decision-relevant statement the project could not make before:** at 80% coverage
the feature gate's selective risk (false-accept rate among accepted queries) is **40.7%**,
against the scalar's **51.0%**; at 90% coverage the gap is 45.4% vs 54.7%. The gate is also
much better calibrated (ECE 0.065 vs 0.271) and has far lower log loss, so its probability
is usable as a threshold, not merely as a rank.

The honest reading is unchanged: this is a gate fit and scored on **one domain** and its
features come from the engine's own retrieval, so it is "recalibrate on the target domain",
not transferable abstention. What is new is that the abstention claim is now a
risk-coverage statement rather than a single AUC.

### 2b. The 0.815 is a within-question-type number

A random stratified half reuses queries of the same question type on both sides, which is the
weakest transfer test: the gate can learn that type's idiosyncrasies. Holding out an entire
answerable question type (`--group-by question_type`, leave-one-type-out with the null
queries distributed round-robin) gives a lower and more honest number:

| split | gate ROC-AUC | balanced accuracy | risk @ 80% |
| --- | --- | --- | --- |
| random stratified half | 0.8146 [0.7628, 0.8570] | 0.7248 | 0.4069 |
| **leave-one-question-type-out** | **0.7666** [0.6712, 0.8267] | 0.7044 | 0.4206 |

Per held-out type: `comparison_query` **0.6640** (n=105), `inference_query` 0.8083 (n=103),
`temporal_query` 0.8276 (n=92). The gate transfers worst to comparison questions, whose
answerable/not distinction evidently differs most from the other types.

The correction matters: **0.815 should be read as within-question-type transfer, and 0.767 as
transfer to an unseen question type.** The gate still beats the scalar decisively (0.5235 on
the same grouped folds), so the conclusion that the signal exists holds — but the headline
number is a best case, not the portable figure.

## 3. What changed in the code

| change | file |
| --- | --- |
| paired permutation test (exact below 12 queries) | `tools/evalkit.py` |
| Benjamini-Hochberg step-up FDR | `tools/evalkit.py` |
| risk-coverage, AURC/AUGRC, risk@coverage, Brier/NLL, equal-mass ECE sweep | `tools/evalkit.py` |
| selective-prediction columns + coverage at the operating point | `tools/answerability_train.py` |
| permutation + FDR CLI over a saved comparison grid | `tools/eval_significance.py` |
| offline tests (21 new) | `tests/test_evalkit.py`, `tests/test_eval_significance.py`, `tests/test_answerability_train.py` |
