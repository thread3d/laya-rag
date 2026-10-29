# R6 — the decision cascade: 43% less CPU at identical quality

`docs/best-in-the-world-2026-10.md` R6, and the answer to the cost finding in R0. The decision pass is
**99.8% of query CPU** — 5.29 s of CPU per candidate against 0.03 s for retrieval — and linear in the
number of candidates. So the one knob that moves the cost axis by a multiple rather than a percent is
**how many candidates the decision model is asked about**.

`LAYA_RAG_SECOND_STAGE_TOP_M` does exactly that: the fused candidate list is truncated to its first
`M` before the cross-encoder and the decision model see it. `0` (the default) keeps every candidate,
because the setting encodes a *quality* claim — that what the first stage ranked below `M` was not
going to be kept — and this repository has already found that such claims reverse between corpora.

## The measurement

MultiHop-RAG, the engine's own index (4,592 chunks), 150 answerable queries at seed 17, gates zeroed
so the row is comparable with the retrieval rows. Quality from `tools/engine_extractive.py`; cost from
the engine's own `total_cpu_seconds` over 6 fixed queries at each setting, 24 torch threads, CPU-only.

| `SECOND_STAGE_TOP_M` | scored per query | CPU per query | ratio | MAP@10 | Hits@10 | answer nugget recall | citation clean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` (all) | 20.0 | **99.2 s** | 1.000 | **0.3128** | 0.9067 | 0.2060 | 1.0000 |
| `10` | 10.0 | **56.1 s** | **0.566** | **0.3128** | 0.9067 | 0.2060 | 1.0000 |
| `5` | 5.0 | **33.5 s** | 0.337 | 0.2900 | 0.7867 | 0.2027 | 1.0000 |

**M = 10 is free.** Every quality metric is identical to four decimals — MAP@10 0.3128, Hits@10 0.9067,
nugget recall 0.2060, citation validity 1.0000 — for **43% less CPU**. On this corpus the decision
model's work on candidates 11–20 is entirely wasted: the first stage's ordering already puts everything
that survives the gate in the top ten.

**M = 5 costs 0.023 MAP@10** (0.3128 → 0.2900, −7.3% relative), and its Hits@10 drop is partly
structural rather than a quality signal: with five candidates, at most five passages can be kept, so a
metric that asks whether gold reached the top ten is capped by the pool. MAP@10 is the honest comparator
here and it says five is a real loss, ten is not.

## What this establishes, and what it does not

**Established.** On this corpus and this channel mix (lexical recall, no dense weight, no
cross-encoder), halving the decision depth costs nothing measurable. That is the cheapest 2× in the
programme so far, and it composes with everything else: the gate, the composer objective and the trust
filter all operate on the same truncated pool.

**Not established.** `M = 10` is a *corpus-specific* setting, for the same reason every other routing
decision in this repository is: the claim is that the first stage's ordering is good enough that
positions 11+ never survive, and that is false wherever the first stage's ordering is weak — on a corpus
where fusion is the weak link (Touché, FRAMES-shaped ensembles) the truncation would remove exactly the
candidates the decision model exists to rescue. A deployment should set it per corpus from a measured
curve, and `tools/engine_extractive.py` plus `LAYA_RAG_SECOND_STAGE_TOP_M` is the four-line recipe.

**A confound worth naming.** The baseline keeps 20 candidates and *keeps* ~5 of them, so the cascade is
partly free because the relevance gate was already discarding 15. On a corpus where the gate keeps
nearly everything, truncation would bind sooner. The curve above is therefore a statement about
MultiHop-RAG's gate behaviour as much as about the ordering, and the honest way to read it is: *on this
corpus, the gate's cut and the cascade's cut coincide.*

## The pool now says so

A row measured under a cascade has to be identifiable as one, so the pool reports both numbers:
`scored` (what the decision model saw) and `dropped_before_second_stage` (what the cascade removed),
with `candidates` the sum. A measurement that cannot say which configuration produced it is the failure
the resume fingerprint exists to prevent, in a different place.

## Still owed

- The same curve on a corpus whose first stage is weak (MuSiQue, Touché), where the prediction is that
  M = 10 costs real MAP — this is the falsifier, and it is not yet run.
- A cascade that *adapts*: score until the gate's decision is stable rather than at a fixed M. That is a
  stopping rule, not a depth, and it is the version that could travel.
- Composition with the cross-encoder path, where the second stage is the CE rather than the decision
  model; the truncation applies identically but has not been measured there.
