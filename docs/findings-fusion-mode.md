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

- **The convex arm is now measured on six corpora, and it wins.**
  [`findings-fusion-tuning.md`](findings-fusion-tuning.md) tunes the dense weight on a seeded half
  and reads it on the other half, against the `k=5` this engine ships: significant wins on ArguAna
  (+0.0266), FiQA (+0.0296) and Quora (+0.0215, 10,000 queries), ties on SciFact and SciDocs, and a
  −0.0038 near-negative on NFCorpus. A single **global** dense weight of 2.0 beats the shipped RRF
  by **+0.0127** on the held-out halves. The earlier SciFact-only +0.0092 against `k=60`
  ([`findings-fusion-ablation.md`](findings-fusion-ablation.md)) understated it: part of that gap
  was the `k` and part the fusion rule.
- **The weights are now tuned**, on a held-out half, over a grid of dense weights
  `{0 … 3}` with the lexical weight fixed at 1.0. What that grid does not cover is a joint
  two-channel search or a `max`-normalised convex arm.
- **The engine's own path is still unmeasured.** These are the replica's whole-document rows; the
  engine chunks corpus-aware and applies Laya's decisions. Shipping convex is one flag
  (`FUSION_MODE=convex` plus a dense weight) and one `tools/engine_beir.py` run, not this table.
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
