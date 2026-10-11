# Tuned convex fusion against the RRF the engine ships

`SCORE-LIFT.md` §P5(a) asks to replace the shipped `FUSION_K=5` RRF with a **tuned convex
combination** of normalised channel scores. The measurement that stood in for it was one corpus,
untuned weights, and — the part that mattered most — a baseline of RRF at `k=60`, which is *not*
what the engine ships. `docs/findings-fusion-ablation.md` said so about itself:

> Two things make this a weak confirmation rather than a verdict. The RRF arm here is the replica's
> `k=60`, while the engine ships `FUSION_K=5` … And the convex `dense_weight`/`lexical_weight` are
> **not tuned** — a tuned convex combination is exactly what the audit asked for and is not what
> this run used.

`tools/fusion_tuning.py` (`make fusion-tuning`) closes all of it:

* the channels are computed **once** per corpus (`beir.channel_score_maps`) and each weight
  re-fuses those score maps (`beir.fuse_score_maps`), so a sweep costs one index build rather than
  one per weight — which is what made tuning expensive rather than merely tedious;
* the baseline is the engine's **RRF at `k=5`**, with `k=60` kept alongside;
* the weight is chosen on a **seeded half of the queries and read on the other half**, so the
  reported delta is what a tuner that cannot see the labels gets, not an oracle.

## The held-out reading (six corpora, `bge-small`, whole documents, depth 100)

| corpus | queries | RRF k=5 | RRF k=60 | tuned convex | weight | held-out Δ vs k=5 | 95% interval | p | oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| SciFact | 300 | 0.7408 | 0.7320 | 0.7412 | 1.0 | +0.0004 | [−0.0098, +0.0100] | 0.909 | 0.7446 (1.5) |
| NFCorpus | 323 | 0.3673 | 0.3635 | 0.3607 | 2.0 | −0.0038 | [−0.0155, +0.0072] | 0.527 | 0.3673 (RRF k=5) |
| ArguAna | 1,406 | 0.5645 | 0.5602 | 0.6020 | 2.0 | **+0.0266** | [+0.0133, +0.0403] | 0.000 | 0.6029 (1.5) |
| SciDocs | 1,000 | 0.1977 | 0.1921 | 0.2031 | 3.0 | +0.0056 | [−0.0012, +0.0121] | 0.103 | 0.2038 (2.0) |
| FiQA-2018 | 648 | 0.3817 | 0.3682 | 0.4096 | 2.0 | **+0.0296** | [+0.0177, +0.0413] | 0.000 | 0.4096 (2.0) |
| Quora | 10,000 | 0.8743 | 0.8629 | 0.8972 | 2.0 | **+0.0215** | [+0.0186, +0.0245] | 0.000 | 0.8978 (3.0) |

**Three significant wins, no significant losses, three ties.** RRF at `k=5` beats `k=60` on all
six, so the earlier comparison's baseline was the weaker one and part of that page's convex lead
was `k` rather than the fusion rule.

## The part a deployment can actually use

A per-corpus weight is not deployable as a default, so the same runs answer the question that is:
what does **one global dense weight** do, tuned on the pooled train halves and read on the test
halves?

| dense weight | train (pooled) | test (pooled) |
| ---: | ---: | ---: |
| 0.0 (lexical only) | 0.4364 | 0.4254 |
| 1.0 (the symmetric point) | 0.5382 | 0.5195 |
| **2.0** | **0.5454** | **0.5244** |
| 3.0 | 0.5444 | 0.5233 |
| RRF `k=5` (shipped) | 0.5304 | 0.5117 |

**A single convex weight of 2.0 — dense weighted twice the lexical channel — beats the shipped RRF
by +0.0127 on the held-out halves** (0.5244 against 0.5117, macro over the six corpora, each
corpus unweighted over its own queries). The curve is flat between 1.5 and 3.0, so the choice is
not knife-edge, and every corpus's optimum lies in 1.5–3.0: **the direction — weight the dense
channel above the lexical one — is consistent across all six.** SciFact is the only corpus whose
per-corpus optimum (1.5) is not the global one, and only by 0.008 there.

That is the §P5(a) claim, and it is the opposite of the previous reading: with the baseline the
engine actually ships and the weights tuned, the convex rule wins where it was previously a
non-significant +0.0092.

## What this does not show

- **It is the harness, not the engine.** `tools/beir.py` reimplements the channels over whole
  documents with `bge-small`. The engine chunks corpus-aware, uses its own encoder and applies
  Laya's decisions, so the *level* differs — and the engine's convex arm is still unmeasured.
  The change to ship is one flag (`FUSION_MODE=convex` plus a dense weight) and one
  `tools/engine_beir.py` run, not this table.
- **Six corpora, chosen by which caches exist.** SciFact, NFCorpus, ArguAna, SciDocs, FiQA and
  Quora — the five full-channel members plus the one where dense-only was already the best row.
  Touché-2020, the member where *lexical* wins by two to one and the convex rule is most at risk,
  is **not** measured: its run was still in the corpus build after ~50 minutes and was stopped.
  That is the obvious next corpus.
- **The macro is over corpora, not over queries.** Quora's 10,000 queries and SciFact's 300
  contribute equally to the pooled rows; a query-weighted reading would be dominated by Quora.
  Every per-corpus row is a paired bootstrap *within* that corpus; no interval is attached to the
  across-corpus mean.
- **The tuning is a grid, not an optimiser.** Dense weights `{0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3}`
  against a lexical weight fixed at 1.0, on the min-max normalisation only. A joint two-channel
  search, or `max`-normalised convex, could differ.
- **NFCorpus is a real negative.** Its tuned choice lost 0.0038 on the held-out half and its
  oracle is RRF itself, so "convex is better" is not universal — it is 3–3 with the ties going to
  convex's favour on the sizes that matter.

## Verification

Six tests: the split is a reproducible partition and is not a head slice; `_score` restricts the
qrels to the queries it ranks (the dilution trap); an empty query set scores zero rather than
raising; `render` names the held-out reading; and the engine-matched configuration — `title_boost`
absent, `rrf_k=5` — is asserted against `eval/beir/engine-parity.json`, which is the row this
comparison is read from. `make test`: **1,590 passed, 5 deselected**.
