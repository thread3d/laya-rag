# Findings: the answerability gate does not need engineered features

The shipped gate is a pure-NumPy logistic model over nine retrieval/evidence signals. A
linear model over raw columns can miss simple interactions those signals imply — a
relevance *margin*, whether recall and relevance agree, whether query coverage multiplies a
confident top score. This experiment asks whether adding those columns improves the
held-out decision, using only the recorded features (no model, no re-run).

Reproduce:

```sh
make answerability-features
```

## Method

300 labelled MultiHop-RAG queries (150 null / 150 answerable), the same 200 stratified
splits the trainer uses. `LogisticGate` is fit on the raw nine columns and on the raw nine
plus nine derived columns, and both are scored on the untouched half at the threshold
chosen on the training half.

| metric | raw nine | + derived | delta |
| --- | --- | --- | --- |
| ROC-AUC | **0.8146** [0.7628, 0.8570] | 0.8106 [0.7633, 0.8576] | −0.0040 |
| balanced accuracy | **0.7248** [0.6667, 0.7733] | 0.7242 [0.6667, 0.7733] | −0.0006 |
| AURC (lower better) | **0.2691** [0.2413, 0.3057] | 0.2706 [0.2427, 0.3040] | +0.0015 |
| AUGRC (lower better) | **0.6681** [0.6563, 0.6813] | 0.6691 [0.6564, 0.6813] | +0.0010 |
| risk @ 80% coverage | **0.4069** [0.3833, 0.4252] | 0.4097 [0.3917, 0.4333] | +0.0028 |
| ECE (lower better) | **0.0650** [0.0278, 0.1107] | 0.0695 [0.0306, 0.1131] | +0.0045 |

## Verdict

**No gain, so the nine features stay.** Every metric moves the wrong way, and the AUC and
risk intervals overlap heavily — the derived columns add variance, not signal, on 300
samples. The gate's remaining error on this domain is therefore not a
feature-*representation* problem that hand-built interactions fix. That is a useful negative:
it closes the "add more signals to the gate" idea and points the effort back at the
retrieval features themselves (which the gate already reads) rather than at their algebra.

**Later superseded in one direction.** That verdict concerns *derived* columns of the nine.
A follow-up tested three genuinely **new raw** signals — evidence-shape features (max
single-passage coverage, coverage spread, pairwise redundancy) — and they improve the gate
significantly on five of six metrics; see
[findings-gate-shape-features.md](findings-gate-shape-features.md). The distinction is the
point: more freedom for the linear model did not help, new information did.

## Honest limits

- One domain, 300 queries, nine hand-chosen interactions. A nonlinear model (gradient
  boosting) or a searched interaction set might find more, but that would add a
  scikit-learn dependency to a module deliberately built on NumPy only, and it was not
  tested.
- The derived interactions were fixed before looking at the result, so the comparison is not
  the winner of a search; it is one honest attempt, reported as such.
