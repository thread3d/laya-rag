# A convex-combination fusion mode

The audit's Bet 2 caution and the Horizon-1 list both ask for the same cheap control: ablate the
fusion rule before claiming a fusion gain. Specifically, "ablate `FUSION_K` ∈ {5, 10, 60} against a
**tuned convex combination**". The first half already exists — `eval/beir/rrf-{3,5,7,10,30,60,100}.json`
are the measured `FUSION_K` arms — but the engine had only RRF, so the convex arm could not be run.

## What shipped

- `convex_combination(channels, *, limit, weights)` in `src/laya_rag/lexical.py`: each channel is
  min-max normalised to `[0, 1]` independently, then the weighted scores are summed and sorted. A
  flat channel contributes `1.0` to everything it contains, so *presence* still counts; a weight of
  `0` drops a channel, as in RRF.
- `FUSION_MODE` (`EngineConfig.fusion_mode`, default `rrf`) selects the rule; `RagEngine._fuse`
  dispatches, and an unknown value raises rather than silently falling back to RRF. Both the query
  path and `adjudicate_claims` pass it.
- RRF stays the default and is unchanged.

## Why convex, and why behind a flag

RRF is rank-based, which is why it is robust across BM25's unbounded scores and cosine in
`[-1, 1]` — but it also discards *how much* better the top hit is. The external evidence says a
tuned convex combination can beat RRF in-domain and out-of-domain; the same evidence also says the
choice is not neutral and needs tuning. Shipping it behind a flag keeps the measured default while
making the comparison runnable.

## What this does not show

- **The convex arm is not yet measured here.** This change makes the ablation runnable; it does not
  produce a number, and no fusion gain is claimed. Running `tools/beir.py` (or the engine) with
  `FUSION_MODE=convex` against the recorded `rrf-*` arms is the measurement.
- **The weights are not tuned.** `dense_weight`/`lexical_weight`/`sparse_weight` are reused as the
  convex weights; a tuned convex combination is exactly the thing the ablation has to find.
- Normalisation is per query and per channel, so a channel with a single candidate contributes
  `1.0`; that is the documented choice, not a neutral one.

## Verification

Eight tests: independent per-channel normalisation; a flat channel contributes presence; a zero
weight drops a channel; mismatched weights raise; `limit` truncates best-first; the engine answers
under both `rrf` and `convex`; an unknown mode raises. `make test`: **1304 passed, 5 deselected**
plus these.

```sh
FUSION_MODE=convex .venv/bin/python -m pytest tests/test_fusion_mode.py -q
```
