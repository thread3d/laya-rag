# The bridge-conditioned gate column, measured: a negative

Horizon 1 asked for the bridge-conditioned score to be **wired into the answerability gate as an
additional column and measured**. The wiring landed earlier (`GATE_BRIDGE_FEATURE`, a 13th
`FEATURE_NAMES` entry, `extract_features(..., bridge_conditioned=...)`); the column was never
collected, so there was no number. This is the measurement, and it is negative: with the true
bridge the gate gets **slightly worse**, and the shuffled-bridge control is neutral — so the cost
is attributable to the conditioning content, not to a thirteenth column.

## Protocol

The 300-query protocol and split seeds of `findings-gate-shape-features.md`, so the number is
comparable to the shape-feature result on the same queries:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python tools/multihop_gate.py --null-limit 150 --pos-limit 150 \
    --bridge-feature --bridge-arm real     --out eval/benchmarks/multihop-rag/gate-bridge-real.json
HF_HUB_OFFLINE=1 .venv/bin/python tools/multihop_gate.py --null-limit 150 --pos-limit 150 \
    --bridge-feature --bridge-arm shuffled --out eval/benchmarks/multihop-rag/gate-bridge-shuffled.json
.venv/bin/python tools/answerability_features.py \
    --features eval/benchmarks/multihop-rag/gate-bridge-real.json --prefix 12 --splits 200
```

`--prefix 12` makes the base arm the twelve features this repository already ships (nine score +
three shape) and the augmented arm the same twelve plus `bridge_conditioned`. The bridge is the
query's **own top-ranked kept passage**; the column is the maximum conditioned decision `s(q, b, c)`
over the rest. The control rotates the bridges so every query is conditioned on *another* query's
top passage: identical instruction text, identical scored count, wrong content.

## Result (300 labelled queries, 200 stratified splits, paired over splits)

| metric | twelve features | **+ `bridge_conditioned` (real)** | delta | paired 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| **auc** | 0.8108 | 0.8085 | **−0.0023** | [−0.0032, −0.0014] | <0.0001 |
| risk@80 | 0.4073 | 0.4085 | **+0.0012** | [+0.0003, +0.0022] | 0.0096 |
| aurc | 0.2741 | 0.2757 | +0.0016 | [+0.0011, +0.0023] | <0.0001 |
| balanced_accuracy | 0.7325 | 0.7298 | −0.0027 | [−0.0051, −0.0005] | 0.0204 |
| augrc | 0.6690 | 0.6696 | +0.0006 | [+0.0004, +0.0008] | <0.0001 |
| ece | 0.0750 | 0.0751 | +0.0001 | [−0.0027, +0.0029] | 0.962 (ns) |

**The control arm**, bridges rotated to a different query:

| metric | twelve features | + `bridge_conditioned` (shuffled) | delta | paired 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| **auc** | 0.8108 | 0.8102 | **−0.0006** | [−0.0016, +0.0004] | 0.239 (ns) |
| risk@80 | 0.4073 | 0.4069 | −0.0004 | [−0.0013, +0.0005] | 0.371 (ns) |
| balanced_accuracy | 0.7325 | 0.7269 | −0.0056 | [−0.0082, −0.0030] | <0.0001 |

214 of the 300 column values differ between the arms, so the control is a real perturbation.

**Reading.** Every real-arm metric moves the wrong way, and the two that matter for a gate —
ranking (AUC) and selective risk — move significantly. The shuffled arm is flat on both, so this is
not "a thirteenth column costs a little": the true first hop makes the gate worse than an
arbitrary one. The bridge signal the *ranker* found (true bridge separates gold facts from the rest
at AUC 0.559 against a shuffled control's 0.530, +0.0293 [+0.0112, +0.0496]) does not transfer to the
gate as a single max-conditioned score; the gate's twelve retrieval/evidence-shape columns already
carry what the conditioned maximum adds, and the extra column spends the linear model's capacity on
a noisier view of the same evidence.

The shape features, measured the same way on the same protocol, gained **+0.0144** AUC. The bridge
column's −0.0023 is an order of magnitude smaller and opposite.

## A control that matches its treatment is a bug signal

The first run of both arms produced **identical** columns, to four decimals, on all 300 queries —
and the comparison printed a significant **+0.0040** AUC gain. The cause: the engine returns
passages as **dicts** on the JSON-facing path, and a dict has no attributes, so the harness's
`getattr(passage, "text", "")` read every passage as empty. Both arms conditioned on an empty
bridge and measured the wording, not the conditioning. `passage_text` (object, mapping, or a
retrieval `Hit`'s `chunk.text`) now reads both shapes, and two regression tests pin it; the runs
above are the re-collected ones.

Two lessons, both already in this project's methodology list and both repeated here: **a result off
by an order of magnitude is a bug signal**, and a control arm that agrees with its treatment to
four decimals is not a null result. The plausible-looking number went into a table before the arms
were diffed.

## What this does not show

- **300 queries, 200 splits.** The paired interval resamples splits, not queries; the real sample
  is 300. The effect is significant and consistent across splits, which is exactly what a large
  effect on a small set looks like — but it is small in absolute terms (−0.0023 AUC) and should be
  read as "does not help, slightly hurts", not as a measured magnitude.
- **One bridge choice.** "Top-ranked kept passage" is a guess; the validating experiment used the
  benchmark's own bridge. A different bridge, a different aggregation than the maximum, or a gate
  trained with the column from scratch could behave differently. What is falsified here is the
  specific owed configuration, on this protocol.
- **One corpus, in-domain.** The answerability gate's out-of-domain behaviour is already known to
  be at chance (AUC 0.509 against MultiHop-RAG); nothing here speaks to transfer.
- **The gate is unchanged in production.** `GATE_BRIDGE_FEATURE` remains off by default; this
  measurement is the reason it should stay off.

## Reproduction

```sh
make test                                   # the accessor and arm tests
HF_HUB_OFFLINE=1 .venv/bin/python tools/multihop_gate.py --null-limit 150 --pos-limit 150 \
    --bridge-feature --bridge-arm real --out eval/benchmarks/multihop-rag/gate-bridge-real.json
.venv/bin/python tools/answerability_features.py \
    --features eval/benchmarks/multihop-rag/gate-bridge-real.json --prefix 12 --splits 200
```

The harness runs two Laya forward passes per query (~18 min per arm on this host) and resumes
nothing: each arm is one pass over 300 queries, and the engine's index is reused.
