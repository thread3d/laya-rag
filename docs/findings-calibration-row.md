# The calibration row: reliability, ECE, Brier, NLL and risk–coverage

The audit's Bet 4 is the clearest novel claim available to this project: BrowseComp-Plus now grades
calibration error (with submissions spread over **9.11%–68.96%**), `AbstentionBench` finds abstention
unsolved and degraded by reasoning fine-tuning, and **no standard benchmark grades a retriever's
relevance or answerability calibration at all**. This is the first artifact for that row:
`tools/calibration_report.py` (`make calibration-report`) consolidates the per-query answerability
probabilities the engine already produces and renders the reliability diagram.

## The measured row

MultiHop-RAG is the one corpus that labels both classes — 301 `null_query` records (negative) and
300 answerable queries. The shipped scalar head, on all **601**:

| reading | shipped scalar head |
| --- | --- |
| ROC-AUC | **0.5087** |
| ECE (equal-width, 10) | **0.3085** |
| ECE (equal-mass, 3 bins) | **0.2979** |
| Brier | **0.3650** |
| NLL | **3.1775** |
| AURC | **0.5093** |
| AUGRC | **0.7497** |
| risk@80 | **0.4927** |
| risk@90 | **0.5397** |

The reliability table is the picture the point estimates hide: the lowest bin (predicted 0.0–0.1,
n=105) is right **49.5%** of the time and the highest bin (predicted 0.9–1.0, n=117) is right
**47.0%** of the time. The shipped scalar's probability is close to *anti*-calibrated on this corpus,
which is exactly the out-of-domain result the audit predicted, now with a curve instead of a caveat.

The feature gate, held out over 300 labelled queries (`answerability-selective.json`, 9 features):

| metric | feature gate | shipped scalar |
| --- | --- | --- |
| AUC | **0.8146** [0.7628, 0.8570] | 0.5199 [0.4530, 0.5835] |
| ECE | **0.0650** [0.0278, 0.1107] | 0.2714 [0.1849, 0.3856] |
| Brier | **0.1768** [0.1565, 0.2017] | 0.3678 [0.3306, 0.4045] |
| NLL | **0.5315** [0.4773, 0.6233] | 3.5059 [2.6197, 4.4421] |
| risk@80 | **0.4069** [0.3833, 0.4252] | 0.5099 [0.4833, 0.5419] |
| AURC | **0.2691** [0.2413, 0.3057] | 0.4879 [0.4382, 0.5438] |

The report now also trains the 12-feature gate on one stratified split of `gate-shape.json`
(seed 17, 150 test queries) and scores the scalar head on the **same** queries, so the two
reliability curves are directly comparable. Reliability diagram `docs/charts/reliability.png`
carries both:

| reading | feature gate | scalar head (same split) |
| --- | --- | --- |
| ROC-AUC | **0.8244** | 0.4679 |
| ECE (equal-width, 10) | **0.0892** | 0.3274 |
| Brier | **0.1745** | 0.3840 |
| NLL | **0.5117** | 3.9274 |
| risk@80 | **0.3917** | 0.5250 |
| AURC | **0.2691** | 0.5402 |

Two readings, and the first is the publishable one. **The gate is calibrated where the scalar is
not** — ECE 0.065 against 0.271, Brier 0.177 against 0.368, and the intervals do not overlap. The
second is the load-bearing caveat: this is an **in-domain** number, and answerability does not
transfer across corpora (AUC 0.509, half-split balanced accuracy 0.511). Publishing the curve *and*
the boundary is more valuable than either alone, because it defines a problem the field has not
measured.

## Reproduction

```sh
make calibration-report                 # eval/calibration.{json,md} + docs/charts/reliability.png
.venv/bin/python tools/calibration_report.py --chart docs/charts/reliability.png
```

The chart is `docs/charts/reliability.png`: observed frequency against predicted probability, with
the perfect-calibration diagonal, equal-width and equal-mass series, annotated with n and ECE.

## What this does not show

- **One corpus.** BEIR-10 has relevance qrels but no answerability labels; MuSiQue was run for
  retrieval only; FRAMES is answerable by construction. A cross-corpus calibration row needs
  negative labels on those corpora, which do not exist in the artifacts on disk.
- **Retrieval *relevance* calibration is not here either.** It needs per-candidate gold labels
  against the engine's per-passage `relevance`; the reports persist per-query metrics, not
  per-candidate labels. That is the next artifact, and the phantom-hit detector depends on it.
- **The aggregate gate row is the 9-feature arm** reported by
  `answerability-selective.json` (AUC 0.8146). The 12-feature arm's held-out reading (AUC 0.8290,
  risk@80 0.4009) is in [`findings-gate-shape-features.md`](findings-gate-shape-features.md); the
  reliability diagram's gate curve is a single-seed 12-feature split (AUC 0.8244), so the two are
  close but not the same estimate.
- **The scalar's NLL is large by construction** (probabilities near 0 on answerable queries), and
  NLL is dominated by those; Brier and ECE are the more stable summaries here.
