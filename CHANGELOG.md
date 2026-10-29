# Changelog

All notable changes to `laya_rag`. Versions follow [semantic versioning](https://semver.org/).

Earlier development predates this file, so the history below begins at the first tagged release;
the measurement programme's own record is `SCORE-LIFT.md` §10 and `docs/FINDINGS.md`, which are
kept per-result rather than per-release.

## 1.4.0 — 2026-10-29

The programme's own entry point brought current, and pinned so it cannot drift like the prose it
replaced.

### `docs/PROGRAMME-STATUS.md` now carries the October round

The page that a reader lands on still described the system as it stood before R0–R7. It now has that
round's table: the cost envelope (6.71 s wall / 96.3 s CPU per query, decision pass 99.8% of it, and the
falsifier that fired at 8.1%), the encoder refresh (routing, not a swap), the composer objective
(0.2060 → 0.2610, prediction twice too optimistic), the shipped gate (30.7% against 69.3% false accepts
*and* 76.7% against 72.0% true accepts), the evidence protocol (~68 min from a fresh clone), the
reranker generation (+0.0313 MAP@10), the cascade (43% less CPU for nothing), and the multilingual scope
correction. R8 is listed as not started.

### The numbers are pinned, and the pin caught itself

`tests/test_programme_status_numbers.py` reads each registered quantity out of the artefact that
produced it and asserts the page still prints it — the claim-ledger discipline applied to one page,
skipping when the gitignored reports are absent. On its first execution it caught a **real ambiguity in
the page**: the R3 row mixed two runs, quoting the trainer's 40-split AUC while the protocol artefact
held a different half-split one. The page now says which run each number is from, and the pins are split
accordingly. A ledger that cannot disagree with its own document is not a ledger.

Its own implementation also learned something small and useful: one of the report keys is literally
`scalar_false_accept_at_0.5`, so the field paths are `/`-separated rather than `.`-separated, and a pin
carries a scale for quantities the page prints as percentages.

### Still owed

- R8: a submittable artefact, which needs the ensemble distillation finished and a training run.
- R1's ArguAna and FiQA, R5's listwise arm, and the cross-corpus abstention sample (MTRAG-UN).

## 1.3.9 — 2026-10-29

A capability claim corrected, and a collection priced rather than truncated.

### "Multilingual" is a property of the checkpoint, not of the retrieval

The README said the default checkpoint "handles non-English text". Nothing had tested what that means
for retrieval, and the engine's shipped recall channel is lexical. Measured on this repository's own
index — the same three questions in five languages, targets being the documents that answer the English
versions:

| configuration | en | de | fr | es | ja |
| --- | ---: | ---: | ---: | ---: | ---: |
| shipped (lexical, `DENSE_WEIGHT=0`) | 2/3 | **1/3** | **1/3** | 0/3 | **1/3** |
| dense forced on (`DENSE_WEIGHT=1`) | 2/3 | 0/3 | 0/3 | 0/3 | 0/3 |

Every non-English match is acronym leakage — `nDCG@10`, `BEIR`, `SciFact` survive translation and BM25
matches them. **Forcing the dense channel on takes every non-English language to zero** while English is
unchanged: the multilingual encoder reads the text but is not cross-lingual for retrieval, which is one
specific fact further than its measured nDCG@10 of 0.014 on SciFact. Three questions per language is a
probe, not a measurement; the README claim was corrected rather than the probe dressed up.

### R7's collection is priced, not approximated

The smallest MIRACL collection with a published leaderboard row — Swahili — is 106,000 documents, about
**28 hours** of embedding for one language at this host's measured 7.9 chunks/s. That is a hardware limit,
and truncating it would repeat the mistake this repository quarantined once already (a 40,000-document
TREC-COVID subsample read 0.3062 against the full collection's 0.6188). The next step is cheap and named:
a retrieval-trained multilingual bi-encoder through `EMBED_MODEL`, then this same five-language probe as
the falsifier before any 28-hour run is worth its wall clock.

## 1.3.8 — 2026-10-29

R5's cheaper half answered: the pointwise reranker generation moved, by +0.031 MAP@10 over the model
this repository measured.

### `bge-reranker-v2-m3` against `bge-reranker-base`, on the same pool

MultiHop-RAG, 150 answerable queries at seed 17, the engine's own 4,592-chunk index, gates zeroed,
`RERANK_THRESHOLD` at 0 so the measurement isolates ordering:

| arm | MAP@10 | Hits@10 | answer nugget recall | citation clean | run time |
| --- | ---: | ---: | ---: | ---: | ---: |
| no cross-encoder (shipped) | 0.3128 | 0.9067 | 0.2060 | 1.0000 | 736 s |
| `bge-reranker-base` (2024, 278M) | 0.3398 | 0.9333 | 0.2281 | 1.0000 | 434 s |
| `bge-reranker-v2-m3` (2026, 568M) | **0.3710** | **0.9400** | **0.2348** | 1.0000 | 995 s |

Paired bootstrap, 10,000 resamples: base → v2-m3 **+0.0313 [+0.0052, +0.0578], p=0.0089**; no CE → v2-m3
**+0.0582 [+0.0324, +0.0855], p=0.0001**.

The prediction was recorded before the run and is reported as it fell: **right in direction and at the
top of its +0.01–0.03 band, wrong in level** (predicted 0.30–0.33, measured 0.3710).

### Why this does not contradict the 2026 grid

That grid measured large pointwise rerankers over a *strong bi-encoder* first stage, where they add
+0.47 to +0.70 nDCG@10. This engine's shipped first stage is lexical, which is much weaker, and the same
class of model buys +0.058 MAP@10. It is this repository's own finding — *a cross-encoder helps a weak
first stage, not a strong one* — with the 2026 numbers attached, and it means the two literatures
measured different first stages rather than disagreeing.

### The gain is ordering, not completeness

Base → v2-m3 improves MAP@10 significantly (p=0.0089) while its answer nugget recall gain is **not**
significant (+0.0067, p=0.27). A better ranker moves the right passages up; it does not put more facts
into five sentences. That is the composer objective's job, and the two are complementary.

### Still owed

- The listwise arm R5 actually named. The licence-clean candidates need a generative interface this
  harness does not have, and `jina-reranker-v3` is CC BY-NC.
- Whether v2-m3 rescues the five BEIR datasets that `bge-reranker-base` *lowered* — ArguAna above all,
  where the mechanism is opposition rather than topicality.
- Composition with R6's cascade (43% less decision CPU at equal quality) and with the raised
  `RERANK_THRESHOLD` (+0.0801 measured separately).

## 1.3.7 — 2026-10-29

The evidence protocol becomes reproducible elsewhere, which is the falsifier it set for itself.

### `make multihop-fetch`: the corpus arrives verified, not by hand

The four-axis report rests on two files this repository does not ship, and until now they arrived
manually — which made the report unreproducible and R4's own falsifier (*reproducible from a fresh clone
in under an hour*) unmet. `tools/multihop_fetch.py` downloads them and **asserts the dataset's shape
before any measurement is derived from it**: 2,556 records, of which exactly **301 are `null_query`**
(the negative class the abstention axis is scored on), and 609 documents. A truncated or re-versioned
download fails there rather than producing a smaller number that reads as a result — the same discipline
`tools/bioasq.py` applies to a 26 GB dump. `--check` re-verifies without the network, and a second run is
a no-op.

### The chain, with the wall clock each step actually took

| step | command | measured |
| --- | --- | --- |
| fetch and verify | `make multihop-fetch` | **12 s** |
| corpus and index | `laya-rag index eval/benchmarks/multihop-rag/engine-corpus --rebuild` | **580 s** (4,592 chunks) |
| one extractive arm | `tools/engine_extractive.py --limit 150 --sample-seed 17` | **736 s** |
| labelled gate rows | `make gate-dataset` | **~2,760 s** |
| fit the gate | `make gate-train` | **2 s** |
| the report | `make evidence-protocol` | **1 s** |
| **total** | | **~68 min** |

Ten minutes over R4's bar at full size, and under an hour at `--limit 100` for the gate rows (401 rows,
~31 min). The protocol reports its own `n`, so a reduced sample is visible in the report rather than
implied to be the full one.

### Still owed

- A second corpus for the abstention axis (MTRAG-UN), which is what turns "generalises across queries"
  into "travels" — the one thing in this section that reproducibility cannot substitute for.

## 1.3.6 — 2026-10-29

The decision cascade: half the decision work for nothing, measured on the benchmark.

### `SECOND_STAGE_TOP_M`: 43% less CPU at identical quality

R0 established that the decision pass is **99.8% of query CPU** — 5.29 s of CPU per candidate against
0.03 s for retrieval — and linear in the number of candidates. `LAYA_RAG_SECOND_STAGE_TOP_M` truncates
the fused candidate list before the cross-encoder and the decision model see it.

MultiHop-RAG, 150 answerable queries at seed 17, gates zeroed; quality from
`tools/engine_extractive.py`, cost from the engine's own `total_cpu_seconds` over 6 fixed queries at 24
threads, CPU-only:

| `SECOND_STAGE_TOP_M` | scored | CPU per query | ratio | MAP@10 | Hits@10 | nugget recall | citation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` (all) | 20.0 | **99.2 s** | 1.000 | **0.3128** | 0.9067 | 0.2060 | 1.0000 |
| `10` | 10.0 | **56.1 s** | **0.566** | **0.3128** | 0.9067 | 0.2060 | 1.0000 |
| `5` | 5.0 | **33.5 s** | 0.337 | 0.2900 | 0.7867 | 0.2027 | 1.0000 |

**M = 10 is free**: every quality metric identical to four decimals for 43% less CPU. On this corpus
the decision model's work on candidates 11–20 is entirely wasted, because the first stage's ordering
already puts everything that survives the gate in the top ten. **M = 5 costs 0.023 MAP@10**, and its
Hits@10 drop is structural — five candidates cap a metric that asks about the top ten — so MAP@10 is the
honest comparator.

It stays off by default. The setting encodes a *quality* claim — that what the first stage ranked below
M was not going to be kept — which is false exactly where the ordering is weak, and this repository has
already measured a channel whose contribution reverses between corpora. The pool now reports `scored`
and `dropped_before_second_stage`, so a row measured under a cascade can say so.

### Still owed

- The same curve on a corpus whose first stage is weak (MuSiQue, Touché) — the falsifier, not yet run.
- An adaptive cascade that scores until the gate's decision is stable, rather than a fixed depth.
- The distillation stages of R6, and composition with the cross-encoder path.

## 1.3.5 — 2026-10-29

The fitted answerability gate ships, and the deployment trade is not a trade: at the same threshold it
more than halves the false-accept rate *and* raises the true-accept rate.

### `make gate-dataset` → `make gate-train` → `LAYA_RAG_ANSWERABILITY_GATE_PATH`

`tools/answerability_train.py` was written against one labelled-feature shape (`null`/`positive`
groups) and `tools/gate_dataset.py` writes another (a flat `rows` list), so the newer dataset could not
train a gate at all. **One loader reads both.** And the trainer's *scalar baseline* is read from a `p`
column, which the newer dataset names `answerability` — so the baseline was silently all zeros,
reporting the shipped system at AUC **0.500** for the right reason by accident. It is mapped now, and
the comparison below is a measurement rather than a constant.

601 rows — every one of MultiHop-RAG's **301 `null_query`** records plus 300 sampled answerable ones —
13 features, 40 stratified splits with bootstrap intervals:

| reading | gate | shipped scalar |
| --- | ---: | ---: |
| AUC | **0.8377** [0.8073, 0.8661] | 0.5002 |
| balanced accuracy | **0.7546** | 0.4889 |
| ECE | **0.0521** | 0.2829 |
| Brier | **0.1656** | 0.3687 |
| risk@80 | **0.4043** | 0.5132 |

At the threshold a deployment actually uses, on the held-out half:

| scorer | unanswerable answered (false accept) | answerable answered (true accept) |
| --- | ---: | ---: |
| scalar at 0.5 | **0.6933** | 0.7200 |
| gate at 0.5 | **0.3067** | **0.7667** |

Lower false accepts *and* higher true accepts at the same threshold. Every abstention number this
repository published before this — "28.7% abstain on answerable queries", "the scalar is at chance out
of domain" — described the system the gate replaces.

It stays **opt-in**, and the README says why: it is fitted to a corpus, and a gate trained on news
questions is a different instrument on a code corpus. A damaged or stale artefact degrades to Laya's
own decision rather than failing queries — tested, because the alternative is a service that stops
answering because a JSON file went bad.

### Still owed

- A provenance field on the artefact (trained on what, how many rows), which is the precondition for
  ever defaulting it.
- A prevalence-weighted reading, so the false-accept column can be quoted at a stated base rate.
- The cross-corpus sample (MTRAG-UN's 97 unanswerable + 78 underspecified turns).

## 1.3.4 — 2026-10-29

The evidence-layer protocol, run. One report over four axes, and the number it produced on its first
run is the strongest thing this repository has measured about its own abstention.

### `make evidence-protocol`: four axes, one command, absence named rather than zeroed

`tools/evidence_protocol.py` reads the artefacts the programme already produces and prints evidence
quality, evidence-side abstention, the joint quality × (1 − false-accept) score, and cost — every
number naming its file and field, and every axis whose artefact is missing printed as **not measured**
with the command that would produce it. That last property is tested: a protocol gameable by omission
is worse than no protocol. No generator, judge model or API key is in any path.

It also gained `tools/gate_dataset.py` and `make gate-dataset`, which build the labelled rows —
**every** `null_query` record plus a seeded sample of answerable ones — and `make gate-transfer` for
the cross-corpus question.

### The first run: the gate carries answerability, and production does not

601 rows from MultiHop-RAG (all **301 `null_query`** records — the benchmark's own negative class,
which its retrieval metric excludes — plus 300 sampled answerable), held out half:

| scorer | n | AUC | risk@80 | AURC |
| --- | ---: | ---: | ---: | ---: |
| 12-feature gate | 300 | **0.8159** | **0.4042** | **0.2603** |
| shipped scalar | 300 | 0.5208 | 0.5000 | 0.4878 |

At its **deployed** threshold the shipped scalar would answer **69.3%** of the benchmark's
unanswerable queries on the held-out half, and **72.1%** over all 301 (mean scalar 0.627, median 0.731).
The relevance gate does part of the work — 25 of the 301 never reached the answerability decision
because the evidence set was emptied first — and of the 276 that did reach it, **78.6%** were called
answerable.

The gate's 0.8159 is essentially the in-domain 0.8146 measured on the older 300-query protocol, which
is the useful part: the features are not an artefact of that protocol. What it is *not* is a transfer
result — the gate is fitted on half of these rows and scored on the other half, so it generalises
across queries, not across corpora.

### Still owed

- A cross-corpus negative set (MTRAG-UN's 97 unanswerable + 78 underspecified turns), which is what
  turns "generalises across queries" into "travels".
- A deployment-prevalence weighting, so `risk@80` and the joint score can be read at a base rate.
- Shipping the fitted gate: these numbers are an argument for making the twelve-feature gate the
  default when an artefact is present, rather than an opt-in.

## 1.3.3 — 2026-10-29

The extractive objective measured on the benchmark, the gate dataset builder that gives R3 its
negative class, and a resume-fingerprint fix that two identical experiment arms exposed.

### The coverage objective: **0.2060 → 0.2610** answer nugget recall, p=0.0005

MultiHop-RAG's corpus built into the engine's own index (**4,592 chunks**, the figure every other row
in this repository was measured on), 150 answerable queries, gates zeroed:

| arm | answer nugget recall | sentences per answer | citation clean |
| --- | ---: | ---: | ---: |
| rank order (shipped) | **0.2060** | 5.00 | **1.0000** |
| coverage objective | **0.2610** | 5.00 | **1.0000** |
| paired delta | **+0.0549** [+0.0240, +0.0841], p=0.0005 | | |

107 queries improved, 37 worse, 6 unchanged. Both arms emitted exactly 5.00 sentences per answer, so
the gain is selection and not volume — a check that had to be made, because the claim counts differ
(653 against 727) and made the first read look like a length effect.

**The prediction was wrong by about a factor of two, and the falsifier fires.** R2 pre-registered
0.35–0.50 and a 0.10 bar; the measured rise is 0.055. So the objective is real and significant, and
selection is *a* binding constraint rather than *the* one. The remaining gap to the pool's 0.7438 is
most plausibly the five-sentence budget — the next experiment, not an assumption.

### `tools/gate_dataset.py`: R3's negative class, at a size that can decide

MultiHop-RAG ships 301 `null_query` records whose gold answer is "Insufficient information", which its
own retrieval metric excludes. This tool runs each query through the engine and writes a feature row
per query — every null kept, the answerable records sampled — so the gate's transfer question can be
asked on ~300 real negatives instead of the 19 the documentation pilot had. Resumable through
`jobkit`, one row per query, fsynced. `laya_rag.answerability.passage_view` is the typed adapter the
tools share; `extract_features` is already dual-shape, so the adapter is a boundary, not a fix, and the
test says so rather than inventing a failure it prevents.

### Two identical arms, twice, and the fingerprint that let it happen

The first attempt at the comparison set `EXTRACT_COVERAGE=1`, which does nothing: this repository's
variables carry the `LAYA_RAG_` prefix. Both arms therefore composed identically — 0.2060, 653 claims,
both — and the identical numbers were the only signal. It is the third time this session that an arm
matching its control to four decimals meant a bug rather than a null.

`tools/engine_extractive.py` now puts `LAYA_RAG_EXTRACT_COVERAGE` into its **resume fingerprint**: the
composer's objective is a configuration of the row, so a resumed run that differs only there used to
skip every completed row and report the other arm's numbers.

### Still owed

- The 400-query and full 2,255-query rows (150 is a subset; the direction and significance are
  established, the absolute level is not the published one).
- The `max_sentences` sweep, where the remaining 0.2610 → 0.7438 gap most plausibly lives.
- The R3 run itself on the 301 nulls, now that the dataset builder exists.

## 1.3.2 — 2026-10-29

The abstention-transfer protocol, and the second dataset's verdict on the 2026 encoder. One new tool,
no behaviour change.

### `tools/gate_transfer.py`: the transfer question gets an instrument

The abstention claim is the strongest thing this architecture has and the least-evidenced: the gate is
held-out AUC **0.8146** in domain and the shipped scalar is at chance out of domain (**0.5087**, risk@80
0.4927). No benchmark in the 2026 landscape grades a retrieval-only answerability predictor with a
held-out AUC, so this is instrumented here.

The tool fits the gate on one corpus's feature rows and scores another, reporting **in-domain**,
**transfer** and **shuffled-control** arms with `risk@80` beside every AUC, over either the thirteen
shipped features or the three **model-free** shape features (max passage coverage, coverage spread,
pairwise redundancy) — the latter because they need no decision-model call, so negatives can be mined at
corpus scale. Seven offline tests let a gate fail as well as succeed.

The first pilot (two documentation corpora, 19 and 17 rows) is **not decidable and says so**: transfer
AUCs of 0.514 and 0.811, a permutation null whose 95th percentile is **0.82 at 19 rows**, and a shuffled
control that matched its treatment to two decimals (0.822 against 0.811). The sample the claim needs is
MultiHop-RAG's 301 `null_query` records plus MTRAG-UN's unanswerables.

### NFCorpus reverses the encoder verdict on dense

| dataset | `bge-small` dense | Qwen3-0.6B dense | fused, `bge-small` | fused, Qwen3 |
| --- | ---: | ---: | ---: | ---: |
| SciFact | **0.7200** | 0.7012–0.7023 | 0.7320 | 0.7389 |
| NFCorpus | 0.3391 | **0.3633** | 0.3633 | 0.3694 |

The dense sign flips between corpora (−0.018, +0.024) while the fused row is positive on both
(+0.007, +0.006) at below-MDE size, for 15–21× the indexing. That is the dense channel's own
"per-corpus instrument" behaviour reproduced at the encoder level, and it makes R1 a **routing**
decision rather than a swap.

### Still owed

- The full 2,255-query extractive row (R2) and the ~300-negative gate run (R3); ArguAna and FiQA (R1).

## 1.3.1 — 2026-10-29

An objective for the extractive answer, and the second dataset's verdict on the 2026 encoder. Both
opt-in or measurement-only; retrieval behaviour is unchanged at the defaults.

### `EXTRACT_COVERAGE`: the extractive answer now has an objective

`compose_extractive` selected sentences by their **parent document's rank** and stopped at five. Its
measured row is half perfect and half costly: citation clean **1.0000** (0 of 9,806 sentences uncited)
against answer nugget recall **0.1906** where the pooled evidence carries **0.7438** — three quarters
of the facts never reach the answer.

`EXTRACT_COVERAGE=1` (or `compose_extractive(..., coverage=True)`) selects by submodular fact coverage
over the same candidates: new numbers/dates/ordinals/acronyms/proper nouns, new content at a quarter
weight, the sentence's inherited relevance at half weight, and a Jaccard ≥ 0.7 redundancy guard so
near-duplicates cannot fill the budget. Query terms are deliberately not the unit — every candidate
was retrieved for the query, so they all contain its terms. The guarantees are identical: verbatim
spans, in-range markers pointing at the document they quote, abstention when nothing qualifies.

The objective's own first version failed its own test — a near-duplicate differing by one word was
selected, and five of those carry one fact, which is precisely the failure the objective exists to
remove. The guard and the case are now a test.

Measured on the real corpus and real weights, five-sentence budget: two of three queries go from one
document and 4 and 1 fact tokens to three and two documents and **14 and 7** — 3.5× and 7×. Off by
default, so every recorded end-to-end row stays reproducible, and the published 0.1906 does not move
until the full 2,255-query run. See
[`docs/findings-extractive-coverage.md`](docs/findings-extractive-coverage.md).

### The BEIR harness can now measure a 2026 encoder, and the first two datasets say it does not pay

`tools/beir.py` pooled `cls`/`mean` only and applied one hardcoded bge instruction to every family, so
a Qwen3-class encoder would have been measured under a pooling and a prompt it was never trained for —
neither raises, both read as a negative about the model. `last` pooling and per-family query prefixes
now delegate to the engine's verified table, and `--query-instruction-text` sets an explicit line.

SciFact, 300 queries, with both controls reproducing the recorded rows to the printed precision:

| encoder and instruction | dense nDCG@10 | hybrid | index time |
| --- | ---: | ---: | ---: |
| `bge-small` (2023, 33M) | **0.7200** | 0.7320 | **183 s** |
| Qwen3-0.6B, no instruction | 0.6825 | 0.7316 | 3,810 s |
| Qwen3-0.6B, generic instruction | 0.7023 | 0.7352 | cached |
| Qwen3-0.6B, SciFact instruction | 0.7012 | **0.7389** | cached |

On dense retrieval the 2026 model loses to the 2023 one by 0.018–0.038; the best fused row is +0.007,
inside the repository's own MDE band on 300 queries, for 21× the indexing.

### Still owed

- The full 2,255-query extractive row, which is what turns R2's mechanism into a number.
- R1's third and fourth datasets (NFCorpus is running; ArguAna next) before the encoder refresh is
  either adopted or moved to the do-not-retry ledger.

## 1.3.0 — 2026-10-29

The cost axis gets an instrument, and the number it returns is not the one the README quotes. Two
additive fields on the query result and its HTTP model; no retrieval behaviour changes.

### `cpu_seconds` and `device` travel with every result

`QueryResult.timings` and the HTTP `Timings` model now carry `retrieval_cpu_seconds`,
`rerank_cpu_seconds` and `total_cpu_seconds` beside the wall clock, and `QueryResponse` carries
`device`. Wall clock says how long a caller waited; CPU time says how much work was done. On a
shared host the first is contaminated by other tenants — this repository has withdrawn three speed
claims on exactly that — so the cost axis is read from the second. `process_time()` sums every
thread the process ran, which is what torch's intra-op pool consumes; it is therefore comparable
only at a fixed thread count, which `tools/cost_report.py` now records beside it.

`device` is the device the decision model **actually** ran on, not the one that was requested:
`DEVICE=auto` is a request and the loader decides, so `LayaModel.device` reads the resolved module
back and `BiEncoderModel.device` returns what it loaded with. A result that says `cpu` can support a
CPU-only claim; a result that says `auto` cannot.

`tools/cost_report.py` reports both clocks, the thread count, the resolved device, and prints an
explicit note when a row was taken on an accelerator.

### The CPU-only envelope, measured

With the accelerator refused (`LAYA_RAG_DEVICE=cpu`), 16 torch threads, `RETRIEVE_K=20`, on the
345-chunk corpus and the multilingual checkpoint: **6.71 s median wall, 96.3 s median CPU per
query**, and **5.29 s of CPU per candidate** against 0.03 s for retrieval — the decision pass is
over 99% of the work. The README's "2–5 s" is the same engine on MPS; the CPU-only envelope is the
one the project's own claim rests on, and it is now measured rather than assumed.

The measurement's own falsifier was pre-registered (±5% across repeats at fixed depth) and it
**fired**: per-query repeat spreads ran 0.5%–8.1%, three of five above the bar. So the cost axis is
published as medians with the observed spread, not as point estimates. The same configuration also
read **15.83 s** while two unrelated processes were fetching web pages and **6.71 s** after they
stopped — a 2.4× swing from co-tenancy alone. See
[`docs/findings-cost-envelope.md`](docs/findings-cost-envelope.md).

### Still owed

- R1–R8 of [`docs/best-in-the-world-2026-10.md`](docs/best-in-the-world-2026-10.md) are unstarted;
  R0's outcome changes R6's target (the cascade has a clean 99% to attack) and leaves the rest
  unchanged.
- The figures in this repository were produced under Laya 0.3.23 and remain compatibility-checked,
  not measurement-re-verified, under 0.4.14.

## 1.2.2 — 2026-10-29

Correctness and gate repair. No retrieval behaviour changes: every number in this repository stands.
What changed is the machinery around them — a crash on the branch that reports good news, a checker
that computed its arithmetic from the wrong directory, a tie-break that depended on the NumPy build,
and the CI that would have caught all three.

### The consistency gate crashed on the case where nothing is wrong

`make consistency-check` computes the §1.2 deficit from `eval/beir/*.json`: for each BEIR member, the
gap between this engine's best measured row and the best published model column, summed.

```python
total = sum(Decimal(str(-c["gap"])) for c in cells if c["gap"] < 0)
macro = total / DENOMINATOR
```

A `sum()` over an **empty** sequence returns the *integer* `0`, so `macro` was a `float`, and
`macro * Decimal("0.9")` raises `TypeError: unsupported operand type(s) for *: 'float' and
'decimal.Decimal'`. The empty case is not an edge: it is the branch reached when **no member sits
below its anchor** — the best outcome the document can report, and exactly what a run over a subset
of well-measured cells produces. Four tests were red on it.

Fixed with an explicit `Decimal(0)` start. The tests were red for the right reason and are now the
regression test.

### The deficit arithmetic was computed from a directory nobody asked about

`check_channel_coverage(root)` read the §1.1 channel map from `root/eval/beir` and then called
`module.deficit()` **with no argument**, which reads the repository's own `eval/beir`. So a checker
handed a root carrying its own reports compared them against numbers from somewhere else — and on a
checkout with no `eval/beir` at all the cell loop was empty, the check inspected nothing, and it
passed. The tests that pinned it only passed because the author's tree had the gitignored reports.

The directory is now threaded through, and the fixtures carry a numeric `nDCG@10` row rather than an
empty retriever map, so §1.2 has a cell to check without reading anything the repository does not
ship. A fixture that wants the deficit checked has to state the number, which is the property the
test is about.

### A tie is now settled by the index, not by the sort implementation

`np.argpartition` selects the k largest — but *which* member of a block of equal scores it keeps is
decided by introselect, and that differs between NumPy builds and between inputs that take a
different pivot. This engine ties routinely: duplicate and near-duplicate documents, repeated
boilerplate, and the deliberately uniform test corpora. So the candidate set was a function of the
host rather than of the corpus, and — because the coverage and diversity passes reorder whatever
they are handed — so was the answer.

`index_store.top_k_indices` now returns one canonical ranking: partition once for the k-th largest
score, keep everything strictly above it in index order, fill from the tie block in index order,
then stable-sort by score. Same complexity, host-independent result, used by both the in-memory and
the on-disk scan.

One test was red on this and it is worth stating what that test found. `RERANK_MIN_DOCS=2` promises
the first two kept passages come from distinct documents; the fixture built three files of identical
text with 400-character chunks, so one file produced **43** chunks against a `retrieve_k` of 30, the
pool held a single document, and there was nothing to spread across. The fixture now fits inside the
pool and the contrast the test asserts is real (without the quota all four kept passages are the long
file; with it, the head is the long file and a second document). The guarantee is **over the recall
pool**, and the README says so now.

### There is CI

`.github/workflows/ci.yml` runs `make lint`, `make test`, `make claim-check` and
`make consistency-check` on Python 3.10 and 3.12, plus an entry-point smoke (`import laya_rag`,
`cli --help`, `doctor` expected to exit 3 with no checkpoint). It installs the package with
`--no-deps` and states the runtime dependencies explicitly, because `pyproject.toml` declares `laya`
— which drags in torch, ~800 MB this job never exercises, since the suite runs against `StubModel`
and `model.py` imports `laya` lazily.

It deliberately does **not** run the measurement harnesses, download the 2.3 GB of checkpoints, or
call a generator; `make test-real`, `make eval` and `make bench` stay manual. Both documentation
gates skip cleanly when their gitignored artefacts are absent, so on a fresh checkout they pass by
having nothing to pin — they are in CI to keep the invocation working.

The matrix stops at 3.12 because `numpy<2` publishes no cp313 wheel and the source build fails; that
was measured, not assumed (a 3.13 venv cannot install the declared pin). Python 3.13 users are not
newly broken by this — the package ran on it before and runs on it after — but CI cannot pin the
declared version there.

### The lint gate is real, and it found real defects

`make lint` was `compileall` plus `node --check`. It now runs `ruff` with the **error class only**
(`E9`, `F`): undefined names, unused imports and variables, shadowed redefinitions, broken f-strings.
The house style is deliberately not linted — this tree is 32k lines of measurement tools and a full
default rule set would report taste across all of it.

100 findings; the defects worth naming:

- `tools/frames_corpus.py` used `Mapping` in a signature without importing it (`F821`);
- `tools/bench_lexical.py` shadowed the module-level `json` with a local import;
- `tests/test_chunking.py` annotated with a `"Document"` that only resolved because a local import
  happened to sit in the body (`F821`);
- eleven variables were assigned and never used — among them a `Path(args.out)` binding left behind
  by the fix that deliberately stopped anchoring documents to `--out`, and an `index_dir_for(args)`
  binding that could disagree with the `LAYA_RAG_INDEX_DIR` the engine actually reads;
- thirty-nine unused imports and seven f-strings without placeholders.

One suppression, with its reason in the config: `tools/ann_bench.py` deletes its faiss indices at the
end of each size iteration to keep peak RSS flat across a 10k → 1M sweep, and `F821` reads the next
iteration's lambda captures as uses after that delete. The lambdas are evaluated eagerly, before the
`del`; the alternative was holding five multi-gigabyte indices resident.

### A dependency floor that the code actually needs

`api.py` passes `exclude_content_types` to `GZipMiddleware` and imports
`DEFAULT_EXCLUDED_CONTENT_TYPES`; both arrived in **Starlette 0.48**, while `fastapi>=0.110.0` alone
resolves to 0.47 on an older index, where the service raises `TypeError: unexpected keyword argument
'exclude_content_types'` **at construction**. Measured on 0.47.2 (104 tests red) and on 1.10.0 (all
green). `starlette>=0.48` is now declared in `pyproject.toml`, `make setup` and CI.

### Tests no longer depend on artefacts a fresh clone does not have

Six tests read `eval/` reports that are gitignored, one needed `faiss`, one needed the `vendor/laya`
submodule, and the docs-link checker failed on `SCORE-LIFT.md`'s link to `../best-in-the-world/` — the
audit set in a sibling checkout. All now skip or fixture their way to hermetic:

- the reversal-chart tests skip **on the missing file**, so they still assert when the data is there
  and still fail on a missing key inside a present report;
- `test_check_mode_needs_no_output_and_reports_coverage` builds a two-query BEIR-shaped dataset, so
  `--check` needs nothing from `eval/`;
- the ANN engine test skips on `faiss` (`pytest.importorskip`), matching the `[eval]` extra;
- the `vendor/laya` docs test skips on the absent submodule;
- the link checker now decides its exemption **per target** — "is this path inside `eval/` or the audit
  tree?" — rather than from whether a directory exists. The old form was order-dependent: a test
  elsewhere creates `eval/` for matplotlib's cache, and from then on every `eval/` link in the docs was
  checked against a tree holding none of the reports.

Two smaller corrections in the same family: `tools/engine_vs_harness.py` now validates the
`--engine` report's recorded seed **before** reading the `--harness` report it defaults to (it used to
report `FileNotFoundError` on an eval path instead of the argument error it was written to raise), and
`tools/dotbin_to_safetensors.py` refuses a file that is not a ZIP archive before importing torch, so
the cheapest verdict no longer needs a two-gigabyte wheel.

### Verified

- `make lint`: green (`ruff` error class, `compileall` over `src/ tests/ tools/`, `node --check`).
- `make test` on this host: **1721 passed, 41 skipped, 5 deselected** — 14 failures before this pass,
  none after. The suite collects **1766** tests (README's live count said 1764).
- The same suite in a **torch-free venv installed as CI installs it** (`pip install -e . --no-deps`
  plus the explicit runtime floors): **1685 passed, 57 skipped, 0 failed**.
- `make claim-check` and `make consistency-check`: exit 0 (both skip with no `eval/` artefacts).
- A real-weights smoke, unchanged from the 1.2.1 procedure: `doctor` loads the multilingual checkpoint
  (embedding dim 768), 78 documents index to 345 chunks in 68 s, and `/v1/query` answers.

### Still owed

- The figures in this repository were produced under Laya **0.3.23** and remain compatibility-checked
  only under 0.4.14; this pass changes no retrieval behaviour but is not a re-measurement either.
- Starlette 1.x warns that `starlette.testclient` with `httpx` is deprecated in favour of `httpx2`.
  The suite is green on 1.10.0 today; when that deprecation lands, the `dev` extra needs the new
  client, and CI will be the thing that says so.
- `RERANK_MIN_DOCS` still cannot spread a pool that holds fewer documents than the quota asks for;
  that is now documented rather than fixed, because deepening the fusion limit to satisfy it is a
  measured change and not a repair.

## 1.2.1 — 2026-10-24

Dependency and prose only: no code in this repository changed. The vendored engine moved a minor series
forward, and two numbers that had drifted out of date were corrected. Behaviour on the path this project
exercises is unchanged, and the one behavioural edge case that did move is inherited from upstream.

### The vendored Laya pin moves to v0.4.14

`vendor/laya` moves from `0362118` (v0.4.0-34) to `f903b49a` (v0.4.14) — 897 commits and fourteen tagged
releases (0.4.1 through 0.4.14). The fork's `main` had already merged upstream's 0.4.14 release commit
(`0a05a450`), so the move was a plain fast-forward over the 40 commits the fork carries of its own: the
torch-2.2 / macOS-Intel portability patches, the ONNX opset spelling, the `numpy<2` cap in the ONNX extra,
thread-serialised tokenizer encoding, docs/Pages housekeeping, and the fine-tuning additions.

The surface this project exercises is unchanged. `laya.load` keeps its parameters, `Agent.predict` and
`predict_batch` keep their signatures, `embed_fn_from_agent` and `cached_embed_fn` still take the same
arguments, and `src/laya_rag/model.py` remains the only module that touches Laya. What moved on that path is
performance and robustness: `collate_items` builds its arrays in NumPy rather than torch, the decoding loop
normalises probabilities in place, the sequence stats report their own option count instead of re-rendering
the options, routed batches memoise question-schema serialisation, and `cached_embed_fn` now maps non-finite
embedding values to `0.0` and caches the repaired row rather than refusing the call.

The update was verified rather than assumed. A controlled before/after on one 341-chunk index — the same two
questions at both pins — returns identical answerability (`0.8747` and `0.5928`), the same passages in the
same order, and identical relevance per passage, the largest absolute difference being `0.0`. The offline
suite reports 1764 passed and 5 deselected at both pins, and `make lint`, `make claim-check` (65 checks),
`make consistency-check` (47 sites at 18 of 18 coverage), `make doctor` (embedding dim 768) and the
real-weights suite (4 passed) are green.

### Two numbers that had drifted

The live suite count in `README.md` said 1229 tests; it is 1764. `docs/PROGRAMME-STATUS.md` said ~1,700; it
now says ~1,760. The vendored-submodule section was rewritten for v0.4.14 instead of left describing 0.4.0,
including the fork's relationship to upstream and what did and did not change.

### Still owed

Compatibility-verified is not measurement-re-verified. Figures in this changelog, the README and
`docs/findings-*.md` were produced under Laya **0.3.23**; the submodule has since passed through 0.3.28,
0.4.0 and 0.4.14 with compatibility checks only, so a benchmark should be re-run before any recorded number
is quoted as reproducible under 0.4.14. Two known limits are unchanged rather than fixed here: MTEB still
does not complete on this host, and the default index is the 312 MB MTEB corpus, which makes `/v1/stats`
slow enough that two `make test-gui` checks time out — under the previous pin as well, so that is not a
regression from this bump.

## 1.2.0 — 2026-10-24

Measurement-led again: the headline deficit is now known to be mostly configuration, two research bets are closed with
negatives, and two capabilities landed that were blocked on the security guard rather than on technique.

### The deficit was an upper bound, and now it has a number

A settings-matched per-field BM25F control was run against each cell's registered anchor, seven cells in all:

| cell | flat gap | per-field gap |
| --- | ---: | ---: |
| FEVER | −0.1207 | −0.0242 |
| TREC-COVID | −0.0372 | −0.0134 |
| DBPedia-Entity | −0.0077 | **+0.0144** |
| NQ | −0.0172 | **+0.0055** |
| Climate-FEVER | −0.0520 | −0.0351 |
| HotpotQA | +0.0322 | **+0.0623** |
| TREC-NEWS | −0.0629 | **−0.0653** |

**Mean −0.0379 → −0.0080**: about four fifths of the average deviation was configuration, three cells end at or above their
published anchor, and **TREC-NEWS is the counter-example** — no weighting recovers it. **Robust04 cannot be tested**: its corpus
carries no titles, so there is no second field to weight.

### The two residual cells fail for opposite reasons

- **TREC-NEWS is recall-bound with a favourable ranking** — 53 gold documents per query, only 45% found in the top 100, yet its
  first gold is usually rank 1–2 (`MRR@10` 0.7373).
- **Climate-FEVER is retrieval-bound on a sparse target** — three golds per query, 44% recalled, `MRR@10` 0.2362 (≈ rank 4).

Also recorded: `recall@100` is capped at `100/gold`, so TREC-COVID's 0.1212 is 60% of its achievable 0.203 while NQ's 0.79 is
79% of 1.0. Cross-cell recall comparisons need the ceiling.

### Closed with negatives

- **The pool bet (P11) does not survive training.** Coverage-quota converts on MultiHop-RAG (+0.0266 over `ce`, significant) but
  reverses off-corpus (MuSiQue held-out 0.3984 vs 0.4032), while a plain `blend-0.75` transfers (+0.0120 [+0.0088, +0.0150]).
  All three hard-negative variants fail: restriction collapses the ranker (0.2939 → 0.0257), weighting degrades it monotonically
  (0.2939 → 0.2780), and a **listwise** softmax is **+0.0016 [−0.0017, +0.0052]** — inside the noise — and **−0.0191
  [−0.0259, −0.0126] against the hand rule**.
- **Passage-level retrieval is disproven on genuinely long documents**: Robust04 0.4077 → 0.3522, TREC-NEWS 0.3351 → 0.2456 at a
  600-character budget.
- **The cross-encoder reports once called un-re-takeable do re-take**: their writer is `multihop_rerank_experiment.py` and all
  **18 rows of `rerank-experiment-full` reproduce to four decimals**.

### Two capabilities that were blocked on the security guard

- **`tools/dotbin_to_safetensors.py`** converts legacy `.bin` checkpoints **without `torch.load`** — a restricted unpickler whose
  `find_class` admits only tensor-rebuild callables, so a pickle naming `os.system` is refused rather than executed. Validated by
  behaviour rather than shape: the doc2query model generates *"who was convicted of fraud and conspiracy"* and
  `splade-v3-distilbert` expands *"arrested"* for a passage that never says it.
- **What it unlocked is only partly worth having.** The sparse channel pays **+0.0338** on scifact — but through the encoder
  already in the tree; the converted SPLADE is +0.0004. doc2query pays **+0.0095** (0.6762 → 0.6857) at **3.0 docs/s**, which is
  ~21 days for a 5.4M-document corpus. `naver/splade-v3` proper is **gated**, which is an access grant rather than a format
  problem.

### Harness and engineering

- **`--jobs N` pools every scoring phase** (`mp.get_context("fork")`, since macOS defaults to `spawn` and re-opens the index per
  worker); equality-gated on scifact, 3.25× on the large builds, 13 workers observed on a 5.4M-document corpus.
- **Sharded runs are safe**: `--dump-rankings` writes the per-query rankings, `--shard I/N` ranks a slice, and
  `tools/beir_merge_rankings.py` unions the sidecars and scores the union **once** against the full qrels. Both refusals are
  tested — `--shard` without a sidecar exits 2, and merging an incomplete union exits 2 — because a shard's report must never be
  mistaken for a dataset score. Verified end to end: two 150-query scifact shards merge to the same **0.6762** a single run reads.
- **Reports record their producer**: `settings.producer` and `settings.argv` on every BEIR and engine report, guarded by tests.
- **Progress is printed from the function that actually runs**; a phase counter in the wrong (in-memory) path had produced seven
  hours of silence that looked like a stall.

### Known limits

- **SPLADE-v3 proper and doc2query remain unpromoted**: v3 is gated, and doc2query's gain does not justify its cost at corpus
  scale on this host. Both would need MPS expansion or a smaller student model.
- **MTEB does not complete here**; no MTEB score is claimed.
- **Six cross-encoder reports** were re-taken and reproduce, but their writer's flags beyond the cache are not recorded, so only
  the cache-plus-defaults path is reproducible.

## 1.1.1 — 2026-10-21

The first tagged release. Its substance is measurement, not new features: the BEIR-18 coverage
claim closed, two benchmark deviations explained or refuted, and the parallel scoring path that
made the largest of those runs affordable.

### Coverage

- **BEIR-18 is complete: 18 of 18 datasets carry a measured row.** `bioasq` was the last
  (14,914,604 documents, 500 queries, `bm25-engine` nDCG@10 **0.5193** against a published 0.465),
  so the macro is now computed over the full 18 rather than a subset.
- **MS MARCO remains an extra, not coverage** — it is the collection the BEIR encoders were trained
  on, and the gate counts over an explicit membership set.

### Two deviations explained, one of them against the engine's favour

- **FEVER's −0.121 was a settings artefact.** A per-field BM25F control on the same 6,666 queries and
  5,416,568 documents reads **0.7288**, against the flat control's 0.6323 and a published 0.753: the
  gap narrows to **−0.0242**, so ~80% of it was a flat-versus-multi-field difference rather than a
  retrieval deficit. This also means the headline summed deficit (**0.6154**) is an **upper bound**
  on the engineering gap, bounded in size by the measured title-weight sweep.
- **P11, the pool bet, converts but does not transfer.** A coverage-quota blend reaches **0.3147
  MAP@10**, significantly above `ce` (**+0.0266 [+0.0230, +0.0303]**) and the learned `ltr`
  (**+0.0208 [+0.0130, +0.0287]**); off-corpus it falls **below `ce`** (MuSiQue held-out 0.3984 vs
  0.4032), while a plain **`blend-0.75` transfers** (**+0.0120 [+0.0088, +0.0150]**). Hard-negative
  restriction collapses the ranker (**0.2939 → 0.0257**) through a training/test distribution shift.
- **Passage-level retrieval is disproven on genuinely long documents**, not merely untested:
  Robust04 **0.4077 → 0.3522**, TREC-NEWS **0.3351 → 0.2456** at a 600-character budget.

### Performance

- **`--jobs N` for the on-disk scoring passes**, pooled over queries with
  `mp.get_context("fork")` — macOS defaults to `spawn`, which re-imported and re-opened the index per
  worker and could show zero pool workers as if hung. Verified equality-gated on `scifact`
  (`--jobs 1` vs `4`: identical metric dicts, only `seconds` differs; **3.25×**).
- **Progress is now printed during scoring**, in the function that actually runs. A phase counter in
  the wrong (in-memory) function had produced seven hours of silence that looked like a stall.
- The on-disk BM25F `build()` accepts an **injected vocabulary**, and run files carry the pid — the
  two preconditions for generating index runs in parallel.
- The query deepening loop is **bounded** (8 widenings), so a filter that rejects most candidates can
  no longer re-rank exponentially without terminating.

### Provenance

- **Every BEIR and engine report now records the tool that wrote it and the arguments it was given**
  (`settings.producer`, `settings.argv`), with source-level guards in the test suite. Four static
  screens had failed to identify a report's producer from its settings alone.

### Known limits

- **SPLADE-v3 and doc2query cannot run on this host.** Both ship `pytorch_model.bin`, and torch 2.2.2
  — the newest macOS x86_64 wheel — is refused by `transformers` 4.57 under CVE-2025-32434. This is a
  platform limit, not a tuning choice; bypassing the guard was declined deliberately.
- **Six cross-encoder reports cannot be re-taken**: their writer is not in the tree, and an identical
  settings block proved not to be evidence of authorship. Their numbers stand with that caveat.
- **MTEB evaluation does not complete here** (`mteb.evaluate` forks ~33 workers and stalls); no MTEB
  score is claimed.
