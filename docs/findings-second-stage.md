# The second stage is optional, and turning it off unblocks two measurements

The engine applies Laya's calibrated relevance decision to **every candidate** before ordering and
gating. That is the shipped behaviour, it is what every recorded number here was taken with, and it
is also the thing that makes two wanted measurements impossible: MTEB's `SearchProtocol` is supposed
to score the retrieval *system*, and §P5a's convex-fusion question is about the *fusion rule* — both
were paying a 322M-parameter forward per candidate for a decision neither asks about.

`SECOND_STAGE=0` (CLI: `tools/engine_beir.py --no-second-stage`, `tools/mteb_search.py
--first-stage-only`) returns the **first stage's own order**, with no decision model in the path.

## What it returns, and what it refuses to claim

`candidates` at that point is already the fused, deduplicated, trust- and ACL-screened list, so its
top *is* the retrieval ranking. The result reports:

- `passages` — the first-stage order, `top_k` honoured, nothing filtered;
- `answerable=True` and `answerability_measured=False`, with **`answerability` absent from the
  payload** rather than present-and-null. Nothing decided it, so a verdict would be a number no
  model produced, and a caller cannot misread an absent key as a low probability;
- `rerank_seconds=0.0` and `pool["second_stage"]=False`, which mark the configuration;
- `relevance` and `recall_score` **both carrying the first-stage score**. `relevance` is documented
  as a calibrated probability and in this mode nothing calibrated it — but it is the field a caller
  orders by, so leaving it at zero would hand back an arbitrary order. The pool marker is what says
  the field is not a probability, and no consumer should threshold it.

## Measured: the engine-level convex row §P5a was missing

SciFact, 300 queries, `bge-small`, the `scifact-base` index, depth 10, second stage off:

| arm | nDCG@10 | seconds |
| --- | ---: | ---: |
| RRF `k=5` (shipped fusion) | 0.7269 | 37 |
| convex, dense weight 1.0 | 0.7280 | 37 |
| **convex, dense weight 2.0** | **0.7331** | 37 |
| *recorded:* full engine with the second stage, RRF | 0.7264 | ~600 |

**+0.0062 for the convex rule at the tuned weight over the shipped RRF**, engine-level, on the
chunked path — where the harness's own SciFact comparison had convex *worse* (0.7362 against
0.7408) at the same weight. That sign difference is exactly why the engine-level row was worth
measuring rather than inferring: the replica is whole-document and the engine collapses its top-10
passages to distinct documents, so the two are different retrieval problems.

Two side readings, both worth stating:

- **The second stage is worth +0.0005 on this corpus** (0.7269 first-stage-only against 0.7264 with
  it). That is consistent with the earlier finding that Laya's decision is a better *filter* than a
  *ranker* — on SciFact it barely reorders — and it is not a claim that the second stage is
  worthless: it is the answerability gate and the trust filter on every other corpus.
- **The wall clock falls from ~10 minutes to 37 seconds per configuration** (the ~10 minutes is
  `findings-engine-parity.md`'s measured figure). A 16× reduction is what makes a fusion sweep at
  the engine level affordable at all, and it is the same reduction MTEB needs to fit an operator's
  patience.

## What this does not show

- **One corpus.** SciFact only. The four other full-channel members and the fusion sweep are
  unrun at the engine level.
- **No interval.** 300 queries, point estimates. The +0.0062 is inside the range this repository
  normally brackets, and nothing here is bracketed.
- **The decision model is still the shipped default.** This is an opt-in mode for measurement; no
  default changed, and the test suite pins that (`test_the_default_runs_the_second_stage`).
- **MTEB is wired but not run.** The adapter now has `--first-stage-only` and prints the
  configuration it used; a submission still needs the task to complete end to end.

## Verification

Seven tests on the mode: the default still calls the model; the mode never does
(`predict_many_calls == 0`); no verdict is claimed; both score fields carry the first-stage score;
`top_k` is honoured and nothing is filtered; the pool is still reported; and the config flag reads
`LAYA_RAG_SECOND_STAGE`. Three more on the adapter's flag and progress reporting.
`make test`: **1,642 passed, 5 deselected**.
