# Findings: evidence-shape features improve the answerability gate

The shipped gate is a logistic model over **nine score features** — the relevance and recall
the engine produced, document spread, passage count, evidence length, and query-term
coverage. An earlier experiment added *derived* columns (margins, products, squares) and
every metric moved the wrong way. These three are different in kind: they describe the
**shape of the evidence set**, not a rescaling of the nine.

| feature | what it measures |
| --- | --- |
| `max_passage_coverage` | can any **single** passage answer the question by itself? |
| `coverage_spread` | passages a greedy set cover needs to cover the query terms — high means the answer is distributed across passages |
| `pairwise_redundancy` | mean Jaccard overlap between passages — high means the recall budget went to near-duplicates |

Reproduce (the harness run is ~12 minutes against the real Laya checkpoint):

```sh
HF_HUB_OFFLINE=1 .venv/bin/python tools/multihop_gate.py --null-limit 150 --pos-limit 150 \
    --out eval/benchmarks/multihop-rag/gate-extra.json
.venv/bin/python tools/answerability_features.py \
    --features eval/benchmarks/multihop-rag/gate-extra.json --extras --splits 200
```

## Result (300 labelled queries, 200 stratified splits, paired over splits)

| metric | raw nine | + three shape features | delta | paired 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| **auc** | 0.8146 | **0.8290** | **+0.0144** | [+0.0123, +0.0163] | <0.0001 |
| **balanced_accuracy** | 0.7248 | **0.7428** | **+0.0180** | [+0.0146, +0.0214] | <0.0001 |
| **risk@80** | 0.4069 | **0.4009** | **−0.0060** | [−0.0073, −0.0048] | <0.0001 |
| **aurc** | 0.2691 | **0.2604** | **−0.0087** | [−0.0104, −0.0069] | <0.0001 |
| **augrc** | 0.6681 | **0.6645** | **−0.0036** | [−0.0041, −0.0031] | <0.0001 |
| ece | 0.0650 | 0.0683 | +0.0033 | [−0.0004, +0.0069] | 0.085 (ns) |

Five of six improve significantly, and the one regression (calibration) is not significant.
This is the opposite of the derived-feature result, which supports the reading that the gain
is from genuinely new information rather than from giving the linear model more freedom.

## Robustness: the shape features survive a chunking change, the nine do not

Round 35 made `CHUNK_CHARS_AUTO` the engine's default, which changes passage lengths and
therefore every evidence-shape feature. Re-collecting the same 300 MultiHop queries through a
fresh corpus-aware index (`tools/multihop_gate.py --index-dir .../index-auto`) and repeating
the comparison quantifies exactly what the compatibility note predicted:

| feature set | fixed 1200/400 | corpus-aware | change |
| --- | --- | --- | --- |
| nine score features — AUC | 0.8146 | 0.7862 | **−0.0284** |
| nine score features — risk@80 | 0.4069 | 0.4295 | **+0.0226** |
| **twelve (with shape)** — AUC | 0.8290 | **0.8323** | +0.0033 |
| **twelve (with shape)** — risk@80 | 0.4009 | **0.4065** | +0.0056 |
| shape-feature delta — AUC | +0.0144 | **+0.0461** [+0.0437, +0.0486] | — |
| shape-feature delta — risk@80 | −0.0060 | **−0.0230** [−0.0246, −0.0214] | — |

Two readings, both worth having:

1. **A persisted nine-feature gate is genuinely out of distribution after the chunking
   change** — AUC falls 0.028 and selective risk rises 0.023. This is the caveat from the
   chunking finding, now measured rather than asserted.
2. **The shape features are what carries the gate across it.** Their contribution grows from
   +0.0144 to **+0.0461** AUC (and from −0.0060 to **−0.0230** risk@80) on the new index, all
   p<0.0001, and they leave the twelve-feature gate's own AUC slightly *better* than it was
   under fixed chunking. The features that describe evidence shape are exactly the ones that
   absorb a change in evidence shape.

Operationally: retraining is required either way, but a retrained gate now tolerates the
chunking default, and the measurement says which features do that work.

## Honest limits

- **The paired interval resamples *splits*, not queries.** Both arms share the same 300
  records and the same split seeds, so the test answers "is the improvement consistent across
  resamplings of this set?" — it does, decisively — but the real sample size is **300
  queries**, and the interval is narrower than that limit implies. Do not quote it as if the
  effect were measured on thousands of independent queries.
- The shape features are **lexical** (token sets, Jaccard): they cannot see paraphrase, so a
  passage that answers a question in different words scores low on coverage. A learned
  coverage feature would be a different experiment.
- One hypothesis was tested here after one earlier negative, so no multiplicity correction is
  claimed beyond that.

## Shipped

The three features are appended to `laya_rag.answerability.FEATURE_NAMES`,
`extract_features` computes them, and `LogisticGate` now **selects its columns by name**: a
persisted nine-feature artefact keeps loading (the names are a prefix of the current vector)
while a file naming a column this build does not define still fails loudly. A fresh harness
run against the real checkpoint, re-recorded with all twelve columns, reproduced the table
above **exactly** — AUC 0.8146 → 0.8290, risk@80 0.4069 → 0.4009 — so the shipped
implementation is the one that was measured, not a re-derivation.

```sh
# Record all twelve columns, then compare the nine score features with the full vector
HF_HUB_OFFLINE=1 .venv/bin/python tools/multihop_gate.py --null-limit 150 --pos-limit 150 \
    --out eval/benchmarks/multihop-rag/gate-shape.json
.venv/bin/python tools/answerability_features.py \
    --features eval/benchmarks/multihop-rag/gate-shape.json --prefix 9 --splits 200
```

Retraining a deployed gate is the operator's step (`tools/answerability_train.py`); until it
is retrained, an existing nine-feature artefact simply keeps its old behaviour.
