# Findings: the gate's decision boundary and calibration need no change

The gate module states that because `fit` weights classes inversely to their frequency, the
intercept is weighted for a rebalanced distribution, so **threshold selection should be done
on held-out data rather than assumed to be 0.5** — while the engine hard-codes
`ANSWERABILITY_THRESHOLD=0.5` and reports the raw probability. This tests both, on the
300 labelled MultiHop queries recorded by the gate harness (twelve features, 200 stratified
splits).

Reproduce:

```sh
make gate-calibration   # tools/gate_calibration.py --splits 200
```

It fits the gate on one half of the 300 records, learns a threshold (`best_threshold`) and a
temperature (grid search minimising NLL) on that half, and scores the other half.

## 1. A learned threshold is worth +0.0004 balanced accuracy

| | threshold | balanced accuracy | coverage |
| --- | --- | --- | --- |
| fixed `0.5` | 0.5 | 0.7424 (sd 0.0274) | 0.5312 |
| learned on the training half | **0.470** (sd 0.060, range 0.333–0.600) | **0.7428** (sd 0.0287) | 0.5486 |

The learned threshold is not 0.5, and the range is wide, but the balanced-accuracy optimum is
**flat across it**: choosing the best cut per split buys +0.0004, and mainly moves the
operating point (coverage 0.531 → 0.549) rather than improving decisions. Note the learned
arm even had an advantage — its threshold was tuned on the same half the gate was fit on,
which is optimistic — and it still did not help.

## 2. Temperature scaling makes calibration worse

Fitting one temperature per split on the training half (mean fitted value **0.921**, i.e. it
mostly sharpens) and applying it to the held-out half:

| metric (lower better) | raw probability | temperature-scaled |
| --- | --- | --- |
| ECE | **0.0683** (sd 0.0226) | 0.0719 (sd 0.0239) |
| Brier | **0.1706** (sd 0.0133) | 0.1717 (sd 0.0145) |
| NLL | **0.5164** (sd 0.0422) | 0.5242 (sd 0.0516) |

All three get worse. The gate's probabilities are already calibrated well enough that a
fitted temperature only adds variance — the same overfitting pattern the derived-feature
experiment showed.

## Verdict

**Keep `0.5` and the raw probability.** Both changes would add artefact-format complexity and
a retraining step for no measured gain, so neither was implemented; the reasoning is recorded
in the `ANSWERABILITY_THRESHOLD` comment so it is not re-litigated.

## Honest limits

- 300 queries, one domain, one gate. A deployment whose gate is trained on a very different
  class balance, or whose cost asymmetry is not balanced accuracy, may need a different cut —
  which is exactly why the threshold is configurable.
- The threshold sweep used balanced accuracy as the objective. A deployment that cares about
  false accepts at a fixed coverage would choose differently; the probe reports coverage
  beside balanced accuracy so that tradeoff is visible.
- ECE depends on the binning; the sweep uses `ece_sweep`'s equal-mass bins consistently for
  both arms, so the comparison is like-for-like even if the absolute value shifts with the
  bin count.
