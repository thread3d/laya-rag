# The bridge-conditioned gate column

Horizon 1 names this explicitly: *"wire the bridge-conditioned score into the answerability gate as
an additional column and measure it, since that is explicitly owed."* The mechanism was already
validated — the true bridge separates gold facts from the rest at **AUC 0.559** against a
shuffled-bridge control's 0.530 (**+0.0293 [+0.0112, +0.0496]**) — but as a *ranker* the head was
the limit, and the valid place for the signal is the gate. This wires the column.

## What shipped

- `FEATURE_NAMES` gains a thirteenth entry, **`bridge_conditioned`**, after the three shape
  features. `SHAPE_FEATURE_NAMES` is now the explicit slice `FEATURE_NAMES[9:12]` rather than
  `[-3:]`, because a column now follows them.
- `extract_features(query, passages, *, bridge_conditioned=None)` appends the value, or `0.0` when
  it is not supplied — so a gate trained on the recorded 12-feature data is **unchanged** (the
  13th column is constant zero there).
- `LayaReranker.rerank(..., bridge_feature=False)`, when `GATE_BRIDGE_FEATURE=1` and a feature gate
  is configured, computes `score_relevance_conditioned(query, bridge, rest)` with the bridge being
  the **top-ranked kept passage**, takes the maximum over the rest, and feeds it to the gate. A
  model failure degrades to `0.0` rather than failing the query.
- Off by default: it costs one extra Laya forward pass per query, and no 13-feature gate has been
  trained.
- **The column is now collectable.** `tools/multihop_gate.py --bridge-feature` records the
  conditioned maximum as the 13th column beside the other twelve, and `--bridge-arm shuffled`
  records the control (another query's top passage, by rotation) in a second file with identical
  instruction text and scored count. `laya_rag.answerability.bridge_conditioned_max` and
  `rotated_bridge_texts` make both importable and unit-tested without a model, so
  `tools/answerability_features.py --prefix 12` can compare the twelve- and thirteen-feature gates
  on one file, and the real-vs-shuffled pair is the control.

## What this does not show

- **The column is wired, not measured.** No 13-feature gate was trained or evaluated, so there is
  no `GATE_BRIDGE_FEATURE=1` AUC or risk@80 here. Making the measurement runnable is what this
  change does.
- **The bridge choice is a guess.** "Top-ranked kept passage" is a plausible first-hop proxy, but
  the validating experiment used the benchmark's own bridge; a different choice is a different
  feature, and `extract_features` accepts whatever the caller computes.
- **The recorded feature files have no bridge values.** Retraining `tools/answerability_train.py`
  on `gate-shape.json` yields a gate with the column identically zero until a run computes and
  persists the conditioned scores per query.
- **It is a 13-wide vector now.** A saved 12-feature gate selects its twelve columns by name, so
  loading old artefacts is unaffected; the feature-count constant and its test were updated.

## Verification

Three new tests: the column defaults to `0.0` and carries a supplied value without changing the
other columns; with the flag on, the conditioned maximum reaches the gate and the vector is
13-wide; with it off, the gate receives `None`. The feature-name test now pins the 13-name order and
`SHAPE_FEATURE_NAMES == FEATURE_NAMES[9:12]`. `make test`: **1383 passed, 5 deselected**.
