# R8 — the submission gate: what makes an exported encoder believable

`docs/best-in-the-world-2026-10.md` R8 asks the programme to own a **submittable artefact** rather than
only compare itself with other people's. The training run that would produce a *winning* artefact is not
something this host can do quickly — the ensemble distillation is measured on one dataset and the
student has never been fitted at scale. What stops a submission from being *credible* is cheaper and
more urgent, and this repository has already paid for it once: a cache keyed by a model's directory
*basename* silently reused a previous model's vectors and produced three benchmarks' worth of wrong rows.

So R8's first half is the gate, not the artefact.

## The command

```sh
make submission-check                        # the stock encoder
make submission-check SUBMISSION_MODEL=eval/distill-nfcorpus-model   # an exported student
```

`make submission-check SUBMISSION_ARGS=--check` exits 1 on a failure, which is what a CI step needs.

Five checks, each of which can fail, none of which scores anything:

| check | what a failure would mean |
| --- | --- |
| **loads and returns embeddings** | the package is broken, or the wrong dimension |
| **outputs are unit vectors** | a cosine index built on un-normalised vectors retrieves garbage while looking plausible |
| **deterministic across calls** | a reported score is not reproducible |
| **revision is pinned** | MTEB skips a result computed on a different dataset revision; the same discipline applies to the model, and a local export must carry a content fingerprint |
| **SearchProtocol adapter present** | the route by which the *engine*, not just an encoder, can be entered |

## The stock encoder, checked

```
verdict: SUBMITTABLE
dimension 384, pooling `cls`, revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
  loads and returns embeddings      pass
  outputs are unit vectors          pass
  deterministic across calls        pass
  revision is pinned                pass
  SearchProtocol adapter present    pass
```

That is not a claim that `bge-small` is competitive — it is not, on the 2026 board — it is a
demonstration that the gate passes an encoder that deserves to and would refuse one that does not. The
pooling comes from `laya_rag.model.family_pooling`, the table whose entries were each verified by loading
the model, rather than from a guess.

## What the gate deliberately does not do

**It does not score.** A submission gate that measured retrieval would be a second evaluation harness
with a second set of numbers to keep consistent; the numbers come from `tools/beir.py` and
`tools/engine_beir.py`, and the gate's job is to check that the artefact those numbers were measured on
is the artefact being shipped.

**It does not decide whether to submit.** R8's own falsifier is a scoring criterion — *if the distilled
student does not reproduce the ensemble's fused score within +0.005 on the five measured datasets, do not
submit* — and it stays a scoring criterion. The gate answers "believable", not "good".

## What is still owed, and it is the hard half

The artefact itself. `tools/finetune.py --teacher-models` distils a three-model ensemble into one 33M
encoder and measured **+0.0120 on NFCorpus** (about 29% of the ensemble's +0.0413);
`tools/package_model.py` exports the result, `tools/distill_verify.py` checks the export is faithful to
its source **exactly** (max |Δ| 0 over three probes), and this gate checks the shipped thing is
believable. What has never been run is the
distillation across the five measured datasets, which is the run that would decide whether an MTEB
submission is worth making. That is a training job, not a measurement job, and it is the one item in
R0–R8 this host cannot finish quickly.

## Reproducing the check

```sh
make submission-check SUBMISSION_ARGS="--check --out eval/submission-check"
```

writes `eval/submission-check.md` and `.json`. The eight offline tests drive every check with a stub
embedder — no download, no network — including the two failures that matter in practice: non-unit
vectors and non-determinism.
