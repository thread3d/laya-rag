# R5 — is the reranker generation the lever? The prediction, recorded before the run

`docs/best-in-the-world-2026-10.md` R5. The 2026 controlled grid (one first stage, one harness,
`arXiv 2509.25085`) measures rerankers over the *same* fused top-100: small pointwise cross-encoders add
**+0.47 to +0.70 nDCG@10**, a 0.6B listwise model adds **+6.04**, and late interaction *loses* (−1.32).
This repository's own cross-encoder result is the small-pointwise row of that grid, reproduced exactly:
`bge-reranker-base` lifts gold-fact recall on MultiHop but lowers nDCG@10 on all five BEIR datasets.

R5's question is narrower than "should we rerank": **did the generation of reranker move?** A 568M
`BAAI/bge-reranker-v2-m3` is now licence-clean and CPU-loadable, where the 2024 choice was a 278M base
model.

## Pre-registered, before the run

Recorded here before the numbers exist, in the form `RESEARCH.md` established:

- **Prediction.** On MultiHop-RAG (150 answerable queries, seed 17, the engine's own 4,592-chunk index),
  `bge-reranker-v2-m3` clears `bge-reranker-base` by **+0.01 to +0.03 MAP@10**, and lands in the
  **0.30–0.33** band against the no-cross-encoder baseline's **0.3128**. It is not expected to change the
  BEIR-shaped verdict, because the mechanism that broke a pointwise ranker on ArguAna — an *opposition*
  relation a topical scorer cannot represent — is a property of the task, not of the model's size.
- **Falsifier.** If v2-m3 does not clear the base model by **≥0.02 MAP@10**, the generation is not the
  lever and R5 closes: the remaining headroom is in the *objective* (listwise/set-aware) or in the pool,
  not in the size of the pointwise scorer.
- **Cost, recorded in advance.** Both arms are scored on the same 150 queries and the same pool, so the
  comparison is paired; the 568M model costs roughly twice the 278M one per passage, and that cost is
  reported beside the quality rather than after it.

The three arms, in order: no cross-encoder (already measured), `bge-reranker-base`, `bge-reranker-v2-m3`.

## The result: the generation **is** the lever, by +0.031 over the 2024 model

MultiHop-RAG, 150 answerable queries at seed 17, the engine's own 4,592-chunk index, gates zeroed,
`RERANK_THRESHOLD` at 0 (the pool is kept; the ordering decides). All three arms on the same 150
queries and the same pool, so the comparison is paired:

| arm | MAP@10 | Hits@10 | answer nugget recall | citation clean | run time |
| --- | ---: | ---: | ---: | ---: | ---: |
| no cross-encoder (shipped) | 0.3128 | 0.9067 | 0.2060 | 1.0000 | 736 s |
| `bge-reranker-base` (2024, 278M) | 0.3398 | 0.9333 | 0.2281 | 1.0000 | 434 s |
| `bge-reranker-v2-m3` (2026, 568M) | **0.3710** | **0.9400** | **0.2348** | 1.0000 | 995 s |

Paired bootstrap over the 150 queries, 10,000 resamples:

| comparison | metric | delta | 95% CI | p |
| --- | --- | ---: | --- | ---: |
| no CE → base | MAP@10 | +0.0269 | [−0.0034, +0.0581] | 0.040 |
| base → v2-m3 | MAP@10 | **+0.0313** | [+0.0052, +0.0578] | **0.0089** |
| no CE → v2-m3 | MAP@10 | **+0.0582** | [+0.0324, +0.0855] | **0.0001** |
| no CE → v2-m3 | nugget recall | +0.0288 | [+0.0037, +0.0553] | 0.013 |

**The prediction was right where it mattered and wrong where it was confident.** The registered claim
was that v2-m3 clears the base model by +0.01 to +0.03 MAP@10 — the measured +0.0313 is at the top of
that band, the interval excludes zero, and the falsifier (≥0.02, else R5 closes) does not fire. The
registered *level* — "0.30–0.33" — was wrong: the arm landed at **0.3710**, above the band. A prediction
with the right sign and the wrong magnitude is worth recording as such, not rounded into a hit.

Three things follow.

**The reranker generation is a lever, and the reason is the first stage.** The 2026 grid that made large
pointwise rerankers look pointless measured them over a *strong bi-encoder* first stage (+0.47 to +0.70
nDCG@10). Here the first stage is the shipped lexical channel, which is much weaker, and the same class
of model buys **+0.058 MAP@10** over no reranker and **+0.031** over its 2024 predecessor. This is the
repository's own long-standing statement — *"a cross-encoder helps a weak first stage, not a strong
one"* — arriving with the 2026 numbers attached, and it means the two literatures are not in conflict:
they measured different first stages.

**The gain is in the ordering, not in completeness.** Hits@10 rises 0.9067 → 0.9400 and nugget recall
0.2060 → 0.2348, but the base-to-v2-m3 step improves MAP@10 significantly (p=0.0089) while its nugget
recall gain is **not** significant (+0.0067, p=0.27). A better ranker moves the right passages *up*; it
does not put more facts into five sentences. The composer's objective is the tool for that, and the two
are complementary rather than substitutes.

**The cost is real and now measurable.** 995 s against 434 s for the same 150 queries — 2.3× the
decision work for +0.031 MAP@10. On this corpus that is a good trade; on a corpus where the first stage
is already strong it is the trade the BEIR rows rejected. Between this and R6's cascade (M=10 for free,
43% less CPU) the cost axis has both a lever that spends and a lever that saves, which is the first time
this programme has had either.

## What is not established

- **One corpus.** The BEIR-sized question — whether v2-m3 rescues the five datasets that `bge-reranker-base`
  *lowered*, ArguAna especially — is not answered. The mechanism that broke a pointwise ranker there
  (opposition, not topicality) is a property of the task, and nothing in this result suggests a bigger
  pointwise model changes it. R5's listwise arm remains untested because the licence-clean listwise
  candidates need a generative interface this harness does not have.
- **One threshold.** `RERANK_THRESHOLD=0` keeps the pool, so this measures *ordering* with filtering held
  constant. The repository's measured +0.0801 from raising that threshold is a separate, additive effect
  and the two have not been run together.
- **No distillation.** The 568M model's cost is what R6's remaining stages would attack; the cascade
  already removed 43% of the decision work at equal quality, and the two have not been combined.

---

## The ArguAna question — pre-registered before the run

R5's outcome above ends with an untested claim: *"nothing in this result suggests a bigger pointwise
model changes [ArguAna's collapse]"*. That is a mechanism argument, not a measurement, and this
repository has a rule about those. The measurement is cheap now that the 568M model is on disk.

**The claim under test.** `bge-reranker-base` over the fused top-100 lowered ArguAna's nDCG@10 from
**0.568 to 0.250** (−0.318), and the diagnosis was that ArguAna's relevance is an *opposition* relation
— an argument countering the query's — which a pointwise topical scorer cannot represent. If that
diagnosis is right, capacity does not help: a 568M pointwise model should collapse too.

**Prediction.** At `--rerank-depth 50` (half the recorded depth, for compute), with `bge-small` as the
first stage: `bge-reranker-v2-m3` lands **below** the no-cross-encoder hybrid's **0.568** — my point
estimate is **0.35–0.50 nDCG@10**, i.e. a collapse of between a third and a half, better than the 2024
model's 0.250 but nowhere near the first stage.

**Falsifier.** If v2-m3 **beats** the no-CE hybrid on ArguAna by any margin, the collapse is a capacity
artefact rather than an objective one, R5's mechanism claim is wrong, and the repository's "cross-encoders
hurt on opposition" negative has to be re-stated as "cross-encoders *of that size* hurt". Either answer is
a result; the second is the more interesting one.

**Cost, recorded in advance.** Three arms on the same first stage and the same 340 ArguAna queries; the
568M arm costs about twice the 278M one per candidate, and both are reported beside the quality.
