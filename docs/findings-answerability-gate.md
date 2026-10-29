# The fitted answerability gate, shipped — and the deployment trade it makes

`docs/best-in-the-world-2026-10.md` R3/R4. The evidence protocol's first run established that the
twelve-feature gate carries answerability signal on the benchmark's own negative class (**AUC 0.8159**)
while the shipped scalar is at chance (**0.5208**) and would answer **69.3%** of the unanswerable
queries it sees. This finding is the other half: the artefact, the command that makes it, and what
happens at the operating point a deployment actually chooses.

## The command

```sh
make gate-dataset     # labelled rows: every null_query + a seeded sample of answerable ones
make gate-train       # fit the twelve-feature gate and persist the artefact
export LAYA_RAG_ANSWERABILITY_GATE_PATH=eval/gate/answerability-gate.json.gate
```

`make gate-train` runs `tools/answerability_train.py`, which fits by gradient descent on a
class-weighted log-loss, evaluates over 40 stratified splits with bootstrap intervals, and persists a
`.gate` artefact that `RagEngine.answerability_gate` loads by path. A **damaged or stale artefact
degrades to Laya's own decision rather than breaking every query** — pinned by a test, because the
alternative is a service that stops answering because a JSON file went bad.

Two plumbing defects had to be fixed to get here, and both are the same class as the ones this
repository has already recorded:

- The trainer read only the older `null`/`positive` grouped feature file, so
  `tools/gate_dataset.py`'s flat `rows` shape was unusable by it. **One loader reads both**, like
  `extract_features` reads both passage shapes.
- The *scalar baseline* is read from a `p` column. The newer dataset records the same quantity as
  `answerability`, so the baseline was silently all zeros — reporting the shipped system at AUC
  **0.500** for the right reason by accident. It is now mapped, and the baseline below is a real
  measurement rather than a constant.

## The measurement

601 rows (all 301 `null_query` records plus 300 sampled answerable ones), 13 features, 40 stratified
splits, bootstrap intervals:

| reading | gate | shipped scalar |
| --- | ---: | ---: |
| AUC | **0.8377** [0.8073, 0.8661] | 0.5002 |
| balanced accuracy | **0.7546** [0.7132, 0.7835] | 0.4889 |
| ECE | **0.0521** [0.0289, 0.0812] | 0.2829 |
| Brier | **0.1656** | 0.3687 |
| risk@80 | **0.4043** [0.3957, 0.4169] | 0.5132 |

And at the point a deployment actually uses — the decision threshold — on the held-out half:

| scorer | unanswerable queries answered (false-accept) | answerable queries answered (true-accept) |
| --- | ---: | ---: |
| scalar at 0.5 | **0.6933** | 0.7200 |
| gate at 0.5 | **0.3067** | **0.7667** |
| gate at its fitted 0.523 | 0.3000 | 0.7467 |

**This is not a trade-off.** At the same threshold the gate more than halves the false-accept rate
(69.3% → 30.7%) *and* raises the true-accept rate (72.0% → 76.7%). Every number the repository had
before this — "abstention at the gate's rate is 28.7% on answerable MultiHop queries", "the scalar is at
chance out of domain" — described the system the gate replaces.

### The same result through the engine, not from the features

The numbers above are computed from recorded feature rows. The shipped path is different code — the
gate is consulted inside `rerank` — so it was checked end to end on the real checkpoint and the real
index, eight `null_query` records and eight answerable ones through `engine.query`:

| artefact | null_query records answered | answerable records answered |
| --- | ---: | ---: |
| none (shipped scalar) | **4 / 8** | 6 / 8 |
| `eval/gate/answerability-gate.json.gate` | **0 / 8** | **7 / 8** |

Eight of each is a demonstration, not an estimate — the estimate is the table above — but it is the
demonstration that the artefact is actually in the decision path rather than merely loadable.

## What this does and does not establish

**Established.** The feature set decides answerability on real negatives where the production scalar
does not, it is well calibrated on them (ECE 0.052), and the artefact path works end to end: train,
persist, load, decide. The AUC is the same in kind as the in-domain 0.8146 measured on the older
300-query protocol, so the features are not an artefact of that protocol.

**Not established.** Transfer *across corpora*: the gate is fitted and scored on halves of one
benchmark, so it generalises across queries. `tools/gate_transfer.py` is the instrument for the
cross-corpus question and its two-corpus pilot was undecidable at 19 rows; **MTRAG-UN's 97 unanswerable
and 78 underspecified turns** are the sample that would settle it.

**Two properties a deployment must supply.** The dataset is balanced by construction (301 negatives
against 300 positives), so AUC survives that but **risk@80 and the false-accept figures do not
describe any particular prevalence** — a corpus with 2% unanswerable questions sees a different
operating point for the same probabilities. And the gate is trained on news questions; a
documentation corpus or a code corpus is exactly the domain shift the transfer tool exists to measure.

## The product-level number: on the extractive route, the engine answers 85% of what it cannot answer

Every abstention number in this document so far has been a **gate** number — what the decision layer would
say if you asked it. The product question is different: if a user asks this engine something the corpus
cannot answer, does it quote them something?

Twenty of the benchmark's own `null_query` records, through `query_extractive` at shipped defaults:

| | abstained | answered anyway | mean sentences |
| --- | ---: | ---: | ---: |
| shipped defaults, extractive route | **15%** | **85%** | 4.25 |

**17 of 20 unanswerable questions get a quoted answer**, with 4.25 verbatim sentences and not one empty
response. That is consistent with the 78.6% the scalar gate reports at its deployed threshold, and it is the
more damning statement of the same fact because it is the one a user experiences: the citations are
faithful, the sentences are real, and the question was not answerable.

Two caveats, both real: **n=20**, and the answerability gate artefact is not configured in this run, so this
is the *shipped scalar* behaviour rather than the fitted gate's. That is the point — it is the default a user
gets.

### And the composer's objective is invisible here, which is the desired behaviour

The same twenty queries were run under both composer objectives, because R2 changed selection and nobody had
checked whether that reaches the abstention decision:

| arm | abstention | mean sentences | empty answers |
| --- | ---: | ---: | ---: |
| rank order (shipped) | 0.15 | 4.25 | 0 |
| coverage objective | 0.15 | 4.25 | 0 |

**Identical.** Abstention is decided upstream by the answerability gate, before selection runs, so a
selection objective cannot move it. That is the correct architecture, and it is now measured rather than
inferred: a lever that improves answer coverage on answerable questions without making the system more
willing to answer unanswerable ones.

## The defect that measurement found: the extractive path ignored the verdict

Configuring the fitted gate on the same twenty `null_query` records moved **mean answerability from 0.637
to 0.218** — a large, correct drop — and left the composer emitting **the same 4.25 sentences with the same
15% abstention rate**:

| arm | mean `P(answerable)` | abstention | mean sentences |
| --- | ---: | ---: | ---: |
| shipped scalar | **0.637** | 0.15 | 4.25 |
| fitted gate configured | **0.218** | 0.15 | 4.25 |

The verdict moved and the answer did not. Reading `RagEngine.query_extractive` explains it: the method calls
`self.query(...)`, which computes `answerable` through the gate, and then hands `result.passages` to
`compose_extractive`, which knows only whether a quotable span exists. **`result.answerable` was never
consulted.** The decision layer's question — *can this corpus answer the question* — and the composer's —
*is there a faithful sentence to quote* — are different, and the answer path answered the first with the
second. That is why the product-level number (85% answered) is worse than the gate's own (78.6% called
answerable): some of the 85% were questions the engine had already judged unanswerable.

**These twenty records are part of the gate's own fit data**, so this is in-sample. What it establishes is
that the lever reaches the decision layer, and that the product path did not consume it — not how good the
gate is.

### The fix, deliberately opt-in

`query_extractive(..., honour_answerability=True)` now abstains when the decision layer said the corpus
cannot answer, reporting the verdict beside the abstention instead of contradicting it. **The default is
`False`**, and that is not timidity: every recorded extractive row — including the full-set row running
while this was written — was measured with the old behaviour, and changing a default under a running row is
the provenance failure this repository catalogues five times. The next full-set row should be run with the
flag on, and the difference reported as the lever's product-level effect —
`make engine-extractive ENGINE_EXTRACTIVE_ARGS="--honour-answerability"`. The harness carries the flag in
its **resume fingerprint and both settings records**, for the reason the composer's objective does: a
configuration that changes which rows abstain must not be mixed under one fingerprint, which is the failure
that once reported two identical arms to four decimals.

## Held out by question type: the number the headline should have been

The `0.8377` above comes from `--splits` random stratified halves, and the trainer's own docstring names that
regime for what it is: *"A random stratified half reuses queries of the same question type on both sides, which is
the weakest transfer test: the gate can learn that type's idiosyncrasies. This holds out an entire answerable group
instead."* The grouped mode existed; **it had never been run on the shipped rows.**

It has now. The benchmark's four `question_type`s are held out whole — three answerable types, with the negatives
distributed round-robin so every fold keeps both classes:

| metric | random halves (the recorded headline) | **by question type** |
| --- | ---: | ---: |
| gate AUC | 0.8377 | **0.7909** |
| balanced accuracy | 0.7546 | **0.7213** |
| **ECE** | **0.0521** | **0.1361** |
| risk@80 | 0.4043 | 0.4243 |
| risk@90 | 0.4541 | 0.4679 |

**The ranking survives; the calibration does not.** AUC falls by 0.047 and stays far above the scalar's 0.5002 —
the gate carries answerability signal that is not merely a memorised question-type signature. But **ECE more than
doubles**, and calibration is precisely what the shipped gate is advertised for: the deployment argument is that
its probabilities can be thresholded, which is worth much less at 0.136 than at 0.052.

### And the folds disagree with each other

| held-out type | gate AUC | n |
| --- | ---: | ---: |
| `temporal_query` | **0.9117** | 177 |
| `inference_query` | 0.7325 | 218 |
| `comparison_query` | 0.7285 | 206 |

A 0.18 spread. The aggregate 0.7909 is an average over three quite different instruments, and a deployment whose
traffic is mostly comparison questions should read the 0.73, not the 0.79. That heterogeneity was invisible in
every number this document previously quoted.

### What this changes, and what it does not

- The gate's **ordering** claim stands, and now stands under a regime it was not tuned on.
- The gate's **calibration** claim is much weaker than published: 0.1361, not 0.0521.
- **This is transfer across question types, not across corpora.** The corpus-level question — a gate fitted on news
  answering questions about code — remains open and is what the MTRAG-UN sample would settle. Do not read this as
  the cross-corpus result.

`eval/gate/multihop-grouped.json` is the 601 rows with `question_type` joined back from the benchmark by query
text; the grouped report and its persisted `.gate` artefact sit beside it.

## Still owed

- **Default it, deliberately.** The gate is opt-in (`ANSWERABILITY_GATE_PATH`). The numbers above
  argue for a default *when a fitted artefact for the corpus is present*, which needs a
  provenance field on the artefact (trained on what, how many rows) so a mismatched gate cannot be
  silently adopted. That is the next change, not this one.
- A prevalence-weighted reading, so the false-accept column can be quoted at a stated base rate.
- The cross-corpus sample above.
