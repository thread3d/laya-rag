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

## Still owed

- **Default it, deliberately.** The gate is opt-in (`ANSWERABILITY_GATE_PATH`). The numbers above
  argue for a default *when a fitted artefact for the corpus is present*, which needs a
  provenance field on the artefact (trained on what, how many rows) so a mismatched gate cannot be
  silently adopted. That is the next change, not this one.
- A prevalence-weighted reading, so the false-accept column can be quoted at a stated base rate.
- The cross-corpus sample above.
