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

### The 2024 collapse reproduces at half the recorded depth

Before the 568M arm finished, the 278M arm established that the instrument is sound. ArguAna, seeded
400-query subset (`--query-seed 17`), depth 50 — half the depth at which the −0.318 collapse was
originally recorded:

| arm | nDCG@10 |
| --- | ---: |
| `hybrid-biencoder` (first stage) | 0.5664 |
| + `bge-reranker-base` | **0.2736** |

That is **−0.293** against the recorded −0.318 at depth 100, and the first stage reproduces its own row
(0.5664 against 0.568). So the collapse is a property of the ranker and the corpus, not of the depth it
was measured at — which is the precondition for comparing a different ranker at this depth. The 568M arm
is running at the same depth and on the same subset; its number lands in the next commit, because a
prediction and its measurement should not arrive in the same edit.

---

## The ArguAna answer: prediction confirmed, magnitude two-thirds recovered

The prediction above was registered before the run. The measurement, on the same seeded 400-query
subset and the same depth as the instrument check:

| arm | nDCG@10 | nDCG@100 | MAP@100 | MRR@10 | seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hybrid-biencoder` (first stage) | **0.5664** | 0.8475 | 0.4843 | 0.3960 | 11.9 |
| + `bge-reranker-base` (2024, 278M) | 0.2736 | 0.4950 | 0.2274 | 0.2064 | 1,900 |
| + `bge-reranker-v2-m3` (2026, 568M) | **0.4722** | 0.7175 | 0.4101 | 0.3960 | 6,007 |

Paired over the 400 matched queries, bootstrapped:

| comparison | Δ nDCG@10 | 95% interval |
| --- | ---: | --- |
| first stage **−** v2-m3 | **+0.0942** | [+0.0530, +0.1335] |
| first stage − 2024 model | +0.2928 | [+0.2522, +0.3328] |
| v2-m3 **−** 2024 model | **+0.1986** | [+0.1566, +0.2407] |

**The prediction holds.** It said v2-m3 would land *below* the no-CE hybrid, point estimate 0.35–0.50;
it landed at **0.4722**, inside the band. The falsifier — v2-m3 beating the first stage by any margin —
did not fire: the 2026 model still loses, on 290 of 400 queries.

**And the magnitude moved far more than the sign did.** Capacity recovers **two thirds** of the damage
(−0.293 → −0.094) without reversing it. That makes the repository's standing negative precise rather than
wrong: it was never "a pointwise ranker cannot represent opposition", it is **"a pointwise ranker pays for
opposition in proportion to how little it knows"**, and 568M of parameters buys back 0.199 of the 0.293 —
at **3.2× the compute** and while still losing to the first stage it was given.

That is the difference between a mechanism story and a mechanism *measurement*, and it is why the
unregistered sentence at the top of this document had to be tested rather than left as reasoning. The
mechanism survived; the quantitative claim I would have made from it did not.

### What this does not license

It does not license "use a bigger reranker and the ArguAna problem goes away": the trend across two model
sizes is suggestive and two points are not a trend. It does not license a claim about *listwise* models
either — none was run (see the section above on why), and the opposition argument predicts a listwise
ranker would behave differently, which remains untested rather than assumed.

---

## The listwise blocker, resolved and bounded — and the prediction registered before the run

R5's outcome left the listwise arm untested for two stated reasons: no licence-clean candidate supported by
the harness, and most listwise models being generative-interface-only. Both were **checked** rather than
assumed, and both have moved.

**The interface moved.** `jinaai/jina-reranker-v3` ([paper](https://arxiv.org/abs/2509.25085v4),
[model card](https://huggingface.co/jinaai/jina-reranker-v3)) is a **596M listwise** reranker — "Last but not
Late Interaction for Listwise Document Reranking" — and it loads and runs here: `AutoModel` with
`trust_remote_code`, exposing `model.rerank(query=…, documents=[…])`, which scores a *whole candidate list
in one pass* rather than one (query, document) pair at a time. That is the interface the pointwise models
this document measured do not have, and it is the one ArguAna's opposition relation would need.

**The licence did not move, and that bounds the conclusion.** The model is **CC-BY-NC-4.0**: it can be
measured as a research instrument, and it cannot be shipped inside this engine's dependency path. The other
candidate, `BAAI/bge-reranker-v2-gemma`, is Apache-2.0 but **2.5B parameters with a generative interface** —
4.4× the compute of the model already measured, and a different harness path again. So the listwise arm can
now be *measured* and still not *shipped*, and any conclusion below is stated with that in view.

### The prediction, registered before the run

Same protocol as the pointwise measurements: ArguAna, the seeded 400-query subset, depth 50, the same
`bge-small` hybrid first stage, nDCG@10.

- **Prediction.** Listwise lands **between the pointwise model and the first stage**: nDCG@10 in
  **0.50–0.56**. That is +0.03 to +0.09 over `v2-m3`'s 0.4722, and still 0.01–0.07 *below* the first stage's
  0.5664. The reasoning: a listwise pass can see the pro and con arguments together, which is closer to
  what an opposition relation needs, but the corpus's relevance is a *stance* judgement that a topical
  reranker has no supervision for at any interface.
- **Falsifier A.** If listwise **beats 0.5664**, the collapse is an **interface** property rather than an
  objective one, this repository's standing negative must be re-stated to name the interface, and the whole
  "pointwise rankers cannot do opposition" claim collapses.
- **Falsifier B.** If listwise lands **below 0.4722**, the interface argument is refuted in the other
  direction: seeing the candidates together buys nothing, and the objective story stands on its own.
- **Cost.** One listwise pass per query over fifty candidates, 400 queries. Recorded when it runs; the
  pointwise arm of the same size took 6,007 s.

Either falsifier is a better outcome than the prediction, which is the point of stating it first.

### The interface was proved before the run was paid for

A five-query smoke test at depth 20 established that the adapter works end to end — pool built by the
harness's own `hybrid-biencoder` functions, one listwise call per query, the returned positions mapped back
to document ids — and it cost 120 s. That is worth doing before committing hours: the two things that can
be wrong without a metric noticing are the **pool** (a different pool answers a different question) and the
**position mapping** (a plausible ranking of the wrong documents), and both are now covered by tests.

The smoke test also caught a real defect in the adapter: `evaluate_rankings` was scoring all 1,406 ArguAna
queries while five had been ranked, so every unsampled query counted as a zero and the summary read
**nDCG@10 0.0028 over 1,406 queries**. The harness's own rule — restrict the qrels to the sampled queries —
was missing. With it, the same five queries read **0.80**. A number that looks like a catastrophic result
and a number that looks like a good one were one line apart, and neither would have been flagged by a table.

The full 400-query, depth-50 run is queued behind the heavy-job lock. Its licence is recorded beside its
number wherever it is quoted.

**Its cost, projected from measurement rather than guessed.** The re-smoke after a cache fix — two queries at
depth 10 — took **16.4 s**, which is ~8 s per query at that depth. Depth 50 is five times the candidates in
one pass, so the queue's expectation is **~40 s per query at eight threads, roughly 1.5–2 h for 400 queries
on 24**. If it comes in far outside that, the likely cause is a slow tail rather than a wrong measurement —
the same 10× spread the cost envelope's finding records on this benchmark.
