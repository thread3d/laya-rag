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
- **Not yet shipped.** Promoting them means appending to
  `laya_rag.answerability.FEATURE_NAMES`, computing them in `extract_features`, and keeping
  existing nine-feature gate artefacts loadable (name-subset selection) before retraining.
  Until that lands, no production path changes.
