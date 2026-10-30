# Changelog

All notable changes to `laya_rag`. Versions follow [semantic versioning](https://semver.org/).

Earlier development predates this file, so the history below begins at the first tagged release;
the measurement programme's own record is `SCORE-LIFT.md` §10 and `docs/FINDINGS.md`, which are
kept per-result rather than per-release.

## 2.7.2 — 2026-10-30

The placement path now re-derives a report's summary from its own rows — and the full-set report passed.

### The check

`make multihop-row` reads a report's `summary`, and the summary is what gets quoted. A summary and its per-query
shard can disagree — a resumed run writing them at different times, or a summary computed over a different query set
— which is **the subset-as-member failure wearing a different hat**. The tool now re-reads the report's own `.jsonl`
and refuses the row if the query count or the headline metrics disagree.

The full-set report **passes**: every figure recomputes exactly from its 2,255 rows, to four decimals, and it records
the same `index_identity` (`947d10ba…`) as the 150-query subset — so the two are on the same index and the
subset-versus-full comparison is like-for-like.

| key | recorded | recomputed |
| --- | ---: | ---: |
| answer nugget recall | 0.1835 | 0.1835 |
| Hits@10 | 0.9011 | 0.9011 |
| Hits@4 | 0.7561 | 0.7561 |
| MAP@10 | 0.3120 | 0.3120 |
| MRR@10 | 0.6166 | 0.6166 |
| citation-clean, all | 1.0000 | 1.0000 |

Two tests drive the failure path: an inflated summary is refused with the key and both values named, and a summary
counting 2,255 queries against a one-row shard is refused with the count. The check costs one pass over a file that
is already on disk, which is the cheapest integrity gate the programme has.

## 2.7.1 — 2026-10-30

The night's results reach the R-table and the outcome table — and R2's row now records **three** falsifications.

### The status row

`PROGRAMME-STATUS.md`'s R2 entry said "done, prediction falsified" and quoted the +0.0549. It now carries what came
after: the gain is **bought with length** (2.1× the text, worse per thousand characters), the **ratio remedy
collapsed** to 0.0790, and the **character-budget remedy** read 0.1764 — *less than the shipped arm at 20% more text*.

The queue table's full-set row moves from *running* to *finished*: Hits@10 **0.9011**, MAP@10 0.3120, cite-clean
1.0000 over all 2,255 queries — and, recorded beside it, that the 150-query subset was **0.0225 optimistic** on the
metric R2's delta depends on.

### The outcome table

The assessment's §8 row for R2 now reads **"falsified three times over"**: the magnitude (0.0489 of 0.5404), the
cost (length), and both remedies. That is a different kind of entry from the original — a prediction that was wrong,
followed by two attempts to rescue it, both measured, both failed, and a conclusion that is stronger than either
number: **"more facts without more text" is not available on this corpus.**

A row that records a refutation and then shrugs is a footnote. A row that records the refutation, the two rescue
attempts and what they jointly establish is a finding.

## 2.7.0 — 2026-10-30

Two results land together: the **full-set placement row** finished, and prediction (d) is **falsified** — which
establishes that the coverage/length frontier is flat.

### The full 2,255-query row

| system | Hits@10 | Hits@4 | MAP@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| voyage-02 + rerank (published) | 0.747 | 0.663 | 0.480 | 0.586 |
| bge-large + rerank (published) | 0.718 | 0.636 | 0.476 | 0.563 |
| bge-large, no rerank (published) | 0.672 | 0.522 | 0.342 | 0.430 |
| **the engine, shipped defaults** | **0.9011** | **0.7561** | 0.3120 | **0.6166** |

Citation validity **1.0000** across all 2,255 answers, nugget recall 0.1835, abstention 0.

**The full set is not the subset**: the same arm on 150 queries read nugget recall **0.2060** — **0.0225 optimistic**,
the largest divergence and the one R2's delta depends on. The subset could not have been quoted as the member, and
this is the measurement of how much it would have mattered. The row remains **chunk-level and indicative** because
it started before the document-level fields existed; `make multihop-row` says so instead of sorting it beside the
anchors.

### Prediction (d), falsified — and it says something stronger

| arm | nugget recall | vs shipped | mean chars | vs shipped |
| --- | ---: | ---: | ---: | ---: |
| rank order (shipped) | 0.1945 | — | 568.2 | 1.00× |
| coverage, unbounded | **0.2251** | +0.0306 | 989.6 | 1.74× |
| coverage @ 900 chars | 0.1764 | **−0.0181** | 682.8 | 1.20× |
| coverage @ 600 chars | 0.1402 | **−0.0543** | 440.8 | 0.78× |

**Both bounded arms fall below the shipped arm.** At 900 characters the objective still emits 20% more text than the
shipped order and finds *fewer* facts. So:

> **"More facts without more text" is not available on this corpus.** The gain exists only at 1.7× the length, and
> every budget that brings it near the shipped length loses more coverage than the extra text buys. The shipped
> ordering is **on the efficient frontier**; the objective sits off it.

Two registered remedies — a ratio score and a character budget — and neither escaped the trade. That is the honest
completion of R2: the lever is real, it is a pure length-for-coverage exchange with no free lunch, and the shipped
default is the *efficient* choice rather than merely the incumbent one.

## 2.6.0 — 2026-10-30

Prediction (c) registered two rounds ago, measured now: **falsified, and decisively** — with the mechanism
measured separately.

### The result

| arm | nugget recall | mean chars | length vs shipped |
| --- | ---: | ---: | ---: |
| rank order (shipped) | 0.1945 | 568.2 | 1.00× |
| coverage (absolute) | **0.2251** | 989.6 | 1.74× |
| **coverage per character** | **0.0790** | 212.7 | 0.37× |

Not "most of the coverage, most of the brevity" but **less than half the shipped arm's recall** in a third of the
length. The prediction was wrong in **sign**, not magnitude — falsifier A fired far past its threshold.

### The mechanism, measured separately

Scoring `new facts / length` under a budget of **five sentences** is a ratio objective facing a count constraint,
and those do not compose. Over 43,463 corpus sentences:

| | value |
| --- | ---: |
| correlation(length, fact count) | **0.466** |
| facts per 100 chars, sentences under 100 chars | **3.27** |
| facts per 100 chars, sentences 250+ chars | **2.40** |

Short sentences are *more* fact-dense per character — so a per-character objective dutifully picks the five densest
short sentences, which carry about one fact each: ~15 facts where the absolute objective's five long sentences carry
~40. **The ratio objective optimises the wrong quantity under a constraint expressed in the other one.**

### What it leaves

The length is not a fixable artefact of bad scoring; it is what absolute fact coverage *costs* when the budget is
counted in sentences. The remedy that remains untested is falsifier B's branch: **keep the absolute objective and
constrain length directly** — a character budget. That is a different experiment, not a tweak, and it is not
registered yet.

The rejected mode stays **opt-in and non-default**, so no recorded row changes; a test pins that it changes the
selection and that the shipped path is untouched.

## 2.5.4 — 2026-10-30

The status page's queue brought current, and — more usefully — what the queue has *already settled*.

### The queue row that mattered most

| measurement | state |
| --- | --- |
| full-set rank row | **running**, 1,550 of 2,255 at 00:45 |
| listwise arm | **queued** on the lock |
| full-set coverage arm | **queued** behind it |
| `--honour-answerability` arm | not queued |
| sentence-budget sweep | not queued |
| **coverage-per-character** selection | not queued, and **must not jump the queue** — it changes the algorithm prediction (a) is measuring |

That last row is the one that reads like a warning rather than a plan, and it is: a fix registered six rounds after
a measurement started can silently turn the measurement into a test of something else.

### And a section for the items that left the queue by being *measured*

| question | answer |
| --- | --- |
| does the gate survive **transfer**, or only random splits | **ordering yes, calibration no** — type hold-out: AUC **0.8377 → 0.7909**, **ECE 0.0521 → 0.1361**, 0.18 spread between types |
| is the composer's +0.055 bought with **verbosity** | **yes** — 2.1× the text, and per thousand characters the shipped arm is the more efficient (15.07 against 8.84) |
| what does `citation_clean` certify | **structure, not truth** — pinned by a table-driven test so the two layers cannot be quietly redefined |

A queue document that only lists what is pending ages into a wish list. Listing what it has **settled** makes it a
record of decisions instead.

## 2.5.3 — 2026-10-30

The length confound's **remedy** is registered before its run — a fix, not just a caveat.

### (c) Coverage per character

The objective maximises facts covered with no cost for length, which is how it emitted 2.1× the text. The obvious fix
is to make it efficiency-aware: greedily pick the sentence with the best **new facts per character** rather than the
most new facts, at the same five-sentence budget.

- **Prediction.** At `max_sentences=5`, coverage-per-character keeps **≥80% of the gain** (≥ +0.044 of +0.0549) while
  bringing mean length to **within 20% of the shipped arm's** (~710 characters against 593.6). A *pareto* move: most
  of the coverage, most of the brevity.
- **Falsifier A.** Under half the gain (+0.027) means fact-dense sentences are the long ones for a structural reason —
  facts live in long sentences — and "coverage bought with length" is a property of the corpus, not a fixable
  artefact.
- **Falsifier B.** Keeping the gain *without* shortening the answers means a five-sentence budget does not control
  length at all, and the arm needs a character budget instead.

**Not run now, deliberately**: the queued coverage arm is measuring the *shipped* objective, and changing the
selection key before it runs would turn prediction (a) into a measurement of a different algorithm than the
150-query delta. The arm goes first, then this.

A finding that ends in a caveat has done half its job; the other half is a registered experiment that could remove
the caveat, or establish that it cannot be removed.

## 2.5.2 — 2026-10-30

The failures ledger gains its eighteenth class — **a gain bought with output volume** — and it is the one that
generalises furthest.

### The class

A comparison can be controlled on corpus, metric, budget and seed, report a paired interval excluding zero, and
still be measuring **volume**: the composer's `+0.055` nugget recall came with **2.1× the text** (593.6 → 1282.6
characters, longer on 149 of 150 queries) while the finding offered *"both arms emitted 5.00 sentences"* as its
control.

The tell: **the metric's denominator is the output, not the input.** Retrieval metrics are bounded by the corpus;
extraction and generation metrics are bounded by what the system chose to emit — and that choice is precisely what
the comparison is supposed to hold fixed.

The rule, in a form a reviewer can apply:

> When comparing two systems on a metric computed over their **output**, report output size beside it and compute the
> metric **per unit of output** as well. A per-unit view that agrees with the absolute view is evidence; one that
> disagrees is the finding.

### And the ledger now separates its two halves

Fifteen of the eighteen classes are numbers that were **wrong**. Three — 16, 17 and 18 — are numbers that were
**right** and told a story their denominator did not support. That distinction matters because the second kind
cannot be caught by re-deriving the number; it has to be caught by asking *what the number is a ratio of*. The
ledger says so, which is the most useful thing eighteen entries can do at the end of a night.

## 2.5.1 — 2026-10-30

The length lens applied across every measured arm: it **strengthens** R5's claim and localises the confound to
exactly one lever.

### One table, five arms

| arm | mean chars | nugget recall | **nuggets per 1k chars** |
| --- | ---: | ---: | ---: |
| rank order (no cross-encoder) | 593.6 | 0.2060 | 0.3471 |
| + `bge-reranker-base` (2024) | **579.8** | 0.2281 | 0.3935 |
| + `bge-reranker-v2-m3` (2026) | **559.9** | **0.2348** | **0.4195** |
| coverage objective | **1282.6** | 0.2610 | 0.2034 |
| cascade `M=10` | 593.6 | 0.2060 | 0.3471 |

**The cross-encoder arms are shorter and better.** 560 and 580 characters against the baseline's 594, while
containing *more* of the gold facts — so *"the gain is ordering, not completeness"* survives the check and is
**strengthened** by it: +21% nuggets per thousand characters (0.4195 against 0.3471) while emitting 6% less text.

### Two things the table shows for free

The **cascade arm is byte-identical** to the baseline on all three columns — 593.6 characters, 0.2060 recall,
0.3471 per 1k — an independent confirmation that the truncated run produced *the same answers*, not merely the same
metrics.

And **exactly one of five arms carries the length confound**, which turns it from *"a doubt about the measurement
setup"* into *"a property of the coverage objective"*. A lens that condemned everything would have been a lens worth
doubting; this one indicts one lever and clears four.

## 2.5.0 — 2026-10-30

R2's headline delta has a confound nobody had checked: **the two arms do not emit the same amount of text.**

### Five sentences is not a length control

Both arms emit five sentences — and the finding said so, as its control. The stored answers say the control was
insufficient:

| arm | mean chars | mean sentences | nugget recall | **nuggets per 1k chars** |
| --- | ---: | ---: | ---: | ---: |
| rank order (shipped) | 593.6 | 4.65 | 0.2060 | **15.07** |
| coverage objective | **1282.6** | 4.73 | **0.2610** | **8.84** |

149 of 150 coverage answers are longer, by **+689 characters on average**. The objective selects fact-dense
sentences, and fact-dense sentences are long ones — which is what it is for, and also exactly what a recall metric
is most sensitive to: more text contains more nuggets by construction.

### The honest form of +0.055

Not *"the composer finds more facts"* but **"the composer quotes much more text and finds somewhat more facts in
it"**. Per thousand characters the coverage arm is *less* efficient — 8.84 against 15.07 — so the objective trades
verbosity for absolute coverage. Whether that is a good trade depends on how long an acceptable answer is, and the
recall delta does not settle it.

The paired interval **+0.0549 [+0.0240, +0.0841]** is unchanged: the coverage arm really does contain more of the
gold facts. What was missing was the cost. **A length-matched comparison has not been run** — same-sentence-count is
not it; per-character normalisation or a fixed character budget would be.

`EXTRACT_COVERAGE` stays off by default, which is now better justified rather than less: the shipped behaviour emits
answers half the length for two-thirds of the absolute coverage. All three claim-bearing pages say so.

## 2.4.3 — 2026-10-30

The content layer measured on real output: it **cannot fail here**, and that is worth knowing before the field is
quoted.

### Eight real answers, both layers

| layer | result |
| --- | --- |
| structural (`clean`) | **7 of 8** |
| advisory content (`content.clean`) | **8 of 8**, zero issues |

**The first result is structural rather than empirical**: every emitted sentence is a substring of a cited passage,
so its numbers are in the evidence *by construction*. A verbatim span cannot contain a number its own source lacks.
What the layer still catches is **attribution** — a span quoted from passage 1 but cited to passage 2 has numbers
absent from passage 2 — so its value here is *"the citation points at the right passage"*, not *"the claim is
honest"*.

`content_clean_all` on extractive rows will therefore read 1.0000 **for a structural reason rather than an
evidential one**. That is stated in the finding before the number is quoted, because a 1.0000 that cannot be
anything else is not evidence of anything. Its real audience is the `engine + <model>` row, where text is generated
and can say anything.

### And the one structural failure is also structural

The seventh answer was an **abstention**, whose fixed sentence counts as one uncited claim — a documented
interaction the harness reports rather than smooths. It is also why `citation_clean_all` is 1.0000 over an
answerable-only set: abstentions are near zero there. My probe happened to include a query the engine abstained on,
which is how a documented interaction surfaced in eight samples.

## 2.4.2 — 2026-10-30

The honest transfer regime becomes what `make gate-train` produces, not something you must remember to ask for.

### Three changes, one purpose

**`make gate-train` runs both regimes.** Random halves first (the published figure), then the grouped run — and if
`GATE_GROUP_BY` is empty it prints *"no GATE_GROUP_BY set, so only the random-half regime was measured"* rather
than quietly reporting the weaker one. A capability that exists but is not wired is capability nobody has, which is
the same lesson as the fifteen invisible `make` targets.

**`gate_dataset.py` now records `question_type` on every row.** It is the grouping key, so a dataset that omits it
can *only* be scored in the weak regime — the field is not metadata, it is what makes the stronger measurement
possible. Threaded through `select()` as a `(query, label, question_type)` triple.

**And the artefact name is no longer `answerability-gate.json-grouped`** — `$(basename …)` puts it beside the
original, where `make gate-train`'s own output convention says things live.

Rows measured before today do not carry the key; the grouped report in `eval/gate/` was produced by joining it back
from the benchmark by query text, which is exactly the work the dataset tool now does for free.

## 2.4.1 — 2026-10-30

The grouped transfer figures reach all three pages that quote the gate, and a test keeps them there.

### The drift the propagation closes

Three claim-bearing pages quoted the gate — the status table, the claim boundary, and the README's known limits —
and all three said **0.8377** alone. That figure is the *random-half* regime, which the trainer's own docstring
calls the weakest transfer test. A page quoting only it presents the weaker regime as though it were the only one.

All three now carry the pair:

| page | now says |
| --- | --- |
| `PROGRAMME-STATUS.md` R3 | AUC **0.8377 → 0.7909**, accuracy 0.7546 → 0.7213, **ECE 0.0521 → 0.1361**, 0.18 spread between types |
| `BENCHMARKS.md` claim table | ordering survives under type hold-out; calibration does not |
| `README.md` Known limits | the ordering survives, **the calibration does not**, and the corpus question is still open |

A test now reads the grouped report and requires both the AUC and the ECE on all three pages — so the next page
edited to quote the gate has to quote the honest pair.

### What stays open, stated in each of them

**Cross-corpus transfer** — a gate fitted on news answering questions about code — is untouched by this. Holding
out a question type is the strongest transfer test *available in this dataset*, not the strongest imaginable, and
the three pages say so rather than letting the improvement in rigour imply more than it did.

## 2.4.0 — 2026-10-30

The gate's **transfer** number, measured: the ordering survives, the **calibration does not**.

### The regime that had never been run

`0.8377` comes from random stratified halves, and the trainer's own docstring names that for what it is —
*"the weakest transfer test: the gate can learn that type's idiosyncrasies"* — while `--group-by` exists to hold
out a whole answerable group instead. The grouped mode was written, documented, and **never run on the shipped
rows**.

Run now, joining `question_type` back onto the 601 rows by query text:

| metric | random halves (published) | **by question type** |
| --- | ---: | ---: |
| gate AUC | 0.8377 | **0.7909** |
| balanced accuracy | 0.7546 | **0.7213** |
| **ECE** | **0.0521** | **0.1361** |
| risk@80 | 0.4043 | 0.4243 |

**The ranking claim stands under a regime it was not tuned on. The calibration claim does not**: ECE more than
doubles, and calibration is exactly what the shipped gate is advertised for — the deployment argument is that its
probabilities can be thresholded, which is worth much less at 0.136.

### And the folds disagree

| held-out type | gate AUC |
| --- | ---: |
| `temporal_query` | **0.9117** |
| `inference_query` | 0.7325 |
| `comparison_query` | 0.7285 |

A 0.18 spread. The aggregate is an average over three different instruments, and a deployment heavy in comparison
questions should read 0.73.

### Stated as what it is

**Transfer across question types, not across corpora.** The corpus-level question — a gate fitted on news answering
questions about code — remains open and is what the MTRAG-UN sample would settle. This is the strongest transfer
test *available in this dataset*, which is not the same as the strongest one imaginable.

## 2.3.1 — 2026-10-30

The citation audit becomes a test, because every published row depends on what `clean` means.

### `tests/test_citation_semantics.py`

`test_verify.py` tests the two layers granularly. What was **not** tested is the semantics as a whole — which is
what a reader of this repository's tables needs and what a future change is most likely to break. The eight-case
table from the audit is now a table-driven test, with expectations for **both** layers.

Two failure modes it catches, and both would silently invalidate recorded numbers:

| change | consequence |
| --- | --- |
| make `clean` **stricter** (fold the content layer in) | every published `citation_clean_all 1.0000` becomes a claim about a different property, and the rows are not re-measurable on demand |
| make it **looser** (miss a quote, accept an out-of-range citation) | the engine's central guarantee weakens without a single number moving |

Two further assertions pin the parts that could be quietly dropped: that the advisory layer **actually fires** (a
check that never fires is not a check, however advisory), and that a **polarity flip is outside what these checks
can see** — stated as a limit in a test rather than left to be discovered, so nobody reads a passing verdict as a
semantic guarantee.

The table is the audit recorded in the finding, so document and code cannot drift apart.

## 2.3.0 — 2026-10-30

The most-quoted number in the programme audited: **"cite-clean 1.0000" certifies structure, not truth.**

### Eight adversarial cases

| case | `clean` | expected |
| --- | --- | --- |
| verbatim, cited correctly | True | True |
| **one number changed** (0.4799 → 0.4979) | **True** | False |
| **wrong citation index** | **True** | False |
| **polarity flipped** ("above" → "below") | **True** | False |
| **two sentences, one fabricated** | **True** | False |
| quote not in the evidence | False | False |
| sentence with no citation | False | False |

`clean` is **structural**: every citation in range, every quote found in the evidence, nothing uncited. That is a
real and load-bearing property for a quoting engine — a span that does not exist cannot be emitted — but **it says
nothing about the numbers or the sense of a claim.**

### The checker was right; the reporting was not

`verify.py` documents this deliberately: the content layer *"is reported in a separate `content` block and never
change[s] the `clean` verdict. A noisy check is one a reader learns to ignore."* It works — it flagged the changed
number and the fabricated figure — and **no finding in this programme ever quoted it**. So `citation_clean_all
1.0000` sat alone in every table as though it meant the answers were faithful. A reader could be forgiven for that;
the checker checked the numbers and nothing published the result.

The extractive row now records `content_clean_all`, `content_issue_total` and `negation_flags` beside the
structural figure. **Rows measured before today do not carry them**, and their 1.0000 is the structural one alone.

### And one limit stated as a limit

`_NEGATION` matches explicit negators (not, no, never, cannot, without, none, nor) and **not antonym pairs** —
above/below, higher/lower, more/fewer all pass silently. A polarity inversion is therefore outside what these
checks can detect; the defence against it is structural (the engine emits verbatim spans, so it cannot produce an
inverted claim) rather than semantic. A generator would change that, which is why the `engine + <model>` row is
labelled separately.

## 2.2.1 — 2026-10-30

The reproduce page catches up with three rounds of new capability — and with a heading that said *four* above
five items.

### Rows for the arms that now exist

`docs/REPRODUCING.md` is the page whose only purpose is that a reader can re-derive any number. Three things had
appeared since it was written:

| added | what it lets someone reproduce |
| --- | --- |
| **R2b** | the answerability lever on the *answer* path — `--honour-answerability`, with its cost and where it records itself |
| **R5b** | the listwise arm (`make listwise-rerank`, `jina-reranker-v3`, CC-BY-NC — measurable, not shippable) and its projected runtime |
| a fourth comparability rule | **check the unit before the metric** — a row records both chunk-level and `doc_` figures because the published tables score documents |

### The heading that said four

Adding that rule made the list five and the heading still read *"The four things that make a re-run comparable"*.
Fixed, and the drift is now **tested**: any `##` heading in that page containing a number word must match the
count of numbered items beneath it. A page about preventing prose drift should not be the page that drifts.

## 2.2.0 — 2026-10-30

The failures ledger gains three classes, and the three of them arrived in one night **without changing a single
metric**.

### Classes 15–17

**15. A metric computed over a different unit than the comparison.** The MultiHop rows score top-ten *chunks*
against published rows scoring top-ten *documents*, because the corpus chunks 609 documents into 4,592. Nothing in
the metric complains — it matches gold sentences by substring against whatever it is handed. The tell is a number
nobody computed: **five chunks cover 2.38 documents**, so the row asks a *harder* question wearing the same name.

**16. An argument of the right type and the wrong meaning.** `ranking` (document ids, URLs) handed to a metric
that wanted document *text*. Both `list[str]`, no exception, and every query scores **0.0** — two columns of zeros
that read like a catastrophic retrieval result rather than a broken call. It reached a shipped harness. Caught
because an analysis script made the same mistake minutes earlier and the two disagreed: **0.875 against 0.000**.

**17. A number kept in a comment.** A default in `config.py` justified by a full-protocol measurement with an
interval and a p-value — whose reports are not on disk. Correct, precise, and unreproducible, which is the one
thing this repository does not accept anywhere else.

### What the three have in common

Nothing was mis-measured in the ordinary sense. The evaluator was the benchmark's, the qrels were the benchmark's,
the runs completed. What was wrong was **what the numbers were about** — and no test that checks types or totals
can see that.

All three were caught by the same trick in different clothes: **compute the number a second way and require
agreement** — against a different unit, against a second implementation, against a report that still exists. The
ledger now says so, and notes that early failures are mechanical while late ones are semantic.

## 2.1.7 — 2026-10-30

A quota and a wider pool are not interchangeable, and the finding now says which arm can reach which.

### The distinction, from reading the call site

`engine.py` applies `rerank_coverage_min_docs` inside `if cross_scores is not None` — so:

- **without a cross-encoder** (the arm running now) the quota is **unreachable**, and the only lever is a wider
  pool: the probe measured five chunks → **2.38** documents, ten → **3.50**, twenty → **5.00**;
- **with a cross-encoder** the quota applies — and a second thing happens automatically, which is worth stating
  beside every CE row this repository has published: `resolved_rerank_coverage_lambda` returns **0.7** whenever
  a reranker is configured, so those rows are **coverage-reordered as well as cross-encoded**. *"A cross-encoder
  rerank"* is not what they are.

A finding that said "a quota or a wider pool" without that distinction would send a reader to a setting their arm
cannot use, which is the same failure as a documented flag that does nothing. A test now pins both halves — the
`if cross_scores is not None` guard in the code and the corresponding sentences in the finding.

## 2.1.6 — 2026-10-30

A caution written **before** the number lands, and an artefact-loss case found while writing it.

### What the document-level row will and will not be comparable to

The queued coverage arm runs at the shipped `top_k` — five chunks, about **2.4 distinct documents** — so its
`doc_*` fields will compare "gold fact among the top 2.4 documents" against published rows scored over **ten**.
That is *more* comparable than the chunk row and **still not the fair comparison**, and the finding now says so
in advance, because it is exactly what a reader would otherwise assume.

Making it fair needs a document quota. The engine has one — `RERANK_COVERAGE_MIN_DOCS` — and it is off by
default for a measured reason.

### The artefact-loss case that surfaced

The reason is recorded in `config.py` as a **full-protocol (2,255-query)** measurement: MAP@10 0.3165 → 0.3192
(+0.0027, p=0.037) against Hits@4 0.7694 → 0.7588 (−0.0106). Correct, interval included, honest about the trade
— and **its reports are not on disk**. A number from a complete run of the full set now lives only as a comment
in a configuration file, and cannot be re-derived.

That is the standard every other number here is held to, so it is named rather than quietly relied on. The
finding also notes that a quota or wider-pool row is *a measurement to run deliberately*, not a footnote to
attach to this one.

## 2.1.5 — 2026-10-30

Two probes disagreed, and the disagreement was a **silent zero in shipped code**.

### The bug

The direction probe used the engine's scoring helper. A follow-up probe — written to ask how pool *width*
affects the document-level row — inlined the metric call and passed `ranking`, a list of **document ids**, to
`official_metrics`, which matches gold facts by substring. A URL never contains a gold sentence:

| top_k | chunk Hits@10 | doc Hits@10 |
| ---: | ---: | ---: |
| 5 | 0.750 | **0.000** |
| 10 | 0.875 | **0.000** |
| 20 | 0.875 | **0.000** |

Two probes, same engine, same queries: `doc_Hits@10` of **0.875** and **0.000**. That is the whole value of
writing both. The second probe was wrong — and **the mistake it made was in `tools/engine_multihop.py` too**:
the document-level line added an hour earlier passed `ranking` rather than the documents' text, so every future
run would have recorded zeros in both document columns and **nothing would have failed**. A column of zeros that
reads like a catastrophic retrieval result.

Fixed in one line — build the text map the corpus writer already builds — and pinned by a test.

### What the probe established despite its bug

The chunk-level columns come from text and are unaffected. Widening the pool moves them, and moves the document
count:

| top_k | chunk Hits@10 | mean distinct documents |
| ---: | ---: | ---: |
| 5 | 0.750 | 2.38 |
| 10 | **0.875** | 3.50 |
| 20 | 0.875 | **5.00** |

So five returned chunks really do cover about two and a half documents, and a **document quota** is the obvious
lever if the document-level row is to be compared at ten. Recorded as a measurement to make, not a change to
make on a probe.

## 2.1.4 — 2026-10-30

The chunk-versus-document direction is measured, and it runs in the **cautious** direction rather than the
flattering one.

### The probe

An hour ago the finding said the bias direction "cannot be signed in advance". Honest, and now out of date. An
eight-query probe on the engine's own output signed it:

| level | Hits@10 | MAP@10 |
| --- | ---: | ---: |
| chunk text (the rows the README quotes) | 0.7500 | **0.2816** |
| distinct documents | **0.8750** | **0.4896** |

One query improved at document level, **none worsened**, and the mean number of distinct documents behind five
returned chunks was **2.38**.

That last number is the explanation, and it is the interesting part: the engine's top five chunks represent
about two and a half documents, so a *chunk-level* Hits@10 effectively asks "is the gold fact among the top two
or three documents" while every published row gets **ten**. The chunk-level metric is a structurally harder
question wearing the same name.

### Two consequences that point opposite ways

1. **The published comparison is not inflated by this defect — it is understated.** The caveat was written on
   the suspicion that the error might be embarrassing; the probe says it is conservative.
2. **That is still an error, and it still cannot be published as a result.** *"We may be better than we said"*
   is exactly as unmeasured as *"we may be worse than we said"* until the full-set document-level numbers exist.

The README's caveat and the finding both now say this, with the probe labelled a probe: it signs the direction
and does not estimate the magnitude. For that, the queued coverage arm's `doc_*` fields over 2,255 queries.

## 2.1.3 — 2026-10-29

The placement tool places the **document-level** row when one exists, and says *indicative* when it does not.

### `make multihop-row` prefers `doc_*`

Now that `engine_extractive.py` records both levels, the tool places the comparable one:

- with `doc_*` present → the rendered row says **scored over documents**, "like-for-like apart from chunking";
- without it → the row says **scored over chunks**, that the published rows are scored over documents, and
  that **this report carries no document-level counterpart, so the row is indicative rather than placeable**.

That distinction is the whole point of the last two rounds. Printing a chunk-level number in the same four
columns as document-level anchors and letting a reader assume the comparison holds is how the README's table
became a claim it could not defend.

### Why the coverage arm can supply it, and that is not a compromise

The rank arm running now started before the field existed, so its document-level counterpart cannot be derived
after the fact — the row stores answers, not document ids. The **queued coverage arm** runs the current code and
will record it.

Using the coverage arm's retrieval numbers for the placed row is legitimate because **retrieval is deterministic
and identical between the two composer arms**: the objective changes which sentences are *selected*, not which
passages are *retrieved*. The document-level metrics are the rank arm's retrieval, measured once.

## 2.1.2 — 2026-10-29

A test that cannot fail reads as coverage — so the suite is now audited for vacuous tests, and the audit's
first find was itself.

### `tests/test_no_vacuous_tests.py`

`findings-measurement-failures.md` records this class at the level of *gates* ("a gate that is green because
it had nothing to do"). It exists one level down and is easier to miss: a test function that calls a helper and
asserts nothing passes forever, shows green in a summary, and counts in a total.

The audit walks all **1,868 test functions** for a recognisable check — `assert`, `pytest.raises`/`fail`/`skip`,
`numpy.testing.assert_*`, `self.assert*`. It found **two**, and both were instructive:

- `test_blend_order_rejects_an_out_of_range_weight` used a hand-rolled `try/except ValueError: pass`, which
  **passes when anything raises**, not when the right thing does. Rewritten with
  `pytest.raises(ValueError, match="blend")`.
- The other was **the audit itself**: its helper was named `test_functions`, so pytest collected it as a test,
  and the audit flagged its own helper as vacuous. A rename fixed both — the file's first find was the file.

A third category is named rather than pattern-matched: tests whose assertion is that **nothing raises**
(`test_allow_concurrent_bypasses_the_lock`). The file states its blind spot too — semantic vacuity
(`assert result == result`) and checks hidden inside helpers are invisible to it — because a check that hides
its blind spot is the thing this file is about.

## 2.1.1 — 2026-10-29

The same fairness problem reaches the README's headline MultiHop table, and the page now says so.

### `engine_multihop.py` had the defect too, and its rows are the ones quoted

The chunk-versus-document problem fixed in the extractive row an hour ago is in `tools/engine_multihop.py` as
well — the same `official_metrics([texts], ...)` call over chunk text — and **those are the rows the README
quotes**: `0.896 Hits@10` against voyage-02's published `0.747`, in a table whose every anchor is
document-level.

The harness already computed `ranking` (document ids) per query, so the comparable metric was one call away.
Every run now records `doc_metrics` per query and a **`doc_summary`** in the report.

### And the README says the comparison is indicative until measured

**The historical reports this table came from are not on disk**, so their document-level counterparts cannot
be derived — only re-run. The README therefore now states, directly above the table, that these rows are a
**system-level measurement rather than a like-for-like placement**, that the bias direction cannot be signed
in advance, and that the document-level counterparts have **not yet been measured**.

That is a headline claim voluntarily weakened on the strength of a code reading, which is the correct response
when the alternative is a comparison nobody can defend. The rows may well survive the correction — chunks make
each slot smaller, which is the harsher direction — but *"may well"* is precisely what the programme refuses to
publish as a result.

## 2.1.0 — 2026-10-29

A fairness problem found **before** the placement was made: the row scores chunks, the published table scores
documents.

### The problem

The benchmark's table scores **ten documents** per query. The engine returns **chunks**, and this corpus
chunks **609 documents into 4,592** — 7.5 chunks per document. A chunk-level row is therefore answering a
different question in the same metric's clothes: ten slots hold a fraction of one document's text where the
published rows hold ten whole ones.

The metric does not care — `official_metrics` matches gold facts by normalised substring against whatever
strings it is handed — **which is precisely why the problem is invisible in the number.** And the direction of
the bias could not be signed: chunks make each slot *smaller* (harsher), while a chunk containing the fact can
outrank the document containing it because the document's other chunks dilute it (kinder). **A comparison
whose bias direction cannot be signed is not a comparison.**

### The fix, recorded rather than resolved

The row now carries both, named:

| field | question |
| --- | --- |
| `Hits@10`, `MAP@10`, … | the system's own: does a gold fact appear in the top ten **chunks** |
| `doc_Hits@10`, `doc_MAP@10`, … | the published table's: does it appear in the top ten **documents** |
| `doc_passages` | how many distinct documents those ten chunks covered |

The collapse reuses the harness's own `ranking_from_passages` — path → corpus id, de-duplicated in
first-appearance order — so the ranking is preserved rather than re-sorted. **The row that gets placed is the
`doc_` one.** The chunk row stays because it is the system's own measurement, and quoting a number without
its document-level counterpart is the habit the placement tool exists to prevent.

Chunking still differs between the two groups, and the finding says so rather than glossing it.

**This landed mid-run and does not disturb the row in flight**: the running arm writes the old schema, and the
queued coverage arm will write the new one — which is why the finding names both fields rather than replacing
one with the other.

## 2.0.1 — 2026-10-29

The objective's own rule is now a test: a prediction without a falsifier is caught.

### `tests/test_predictions_have_falsifiers.py`

The programme's charter says *every prediction is recorded with the measurement that would falsify it*. That
is a claim about a **documentation practice**, and every other claim in this repository is checked rather than
asserted — so it is now checked. A finding that registers a prediction without a falsifier is a prediction
that cannot lose.

The audit came back clean: **9 prediction blocks across the findings documents and 4 in the assessment, all
with a falsifier**. The test enforces it going forward, and is deliberately loose about layout (paragraph,
table cell or bolded run-in all count) and strict about presence.

**Perturbed before trusted**: appending a bare `**Prediction:**` paragraph to the composer finding fails it
with the file and line number; restored.

Its own threshold was corrected once — the first version expected ten predictions and found nine, and the
fix was to set the guard to the real count rather than to write a tenth prediction to satisfy a test. **A
threshold that gets met by adding content is not a threshold.**

## 2.0.0 — 2026-10-29

The status page carries the round's **queue** — the durable state a fresh agent needs at round 55 of a
long programme.

### What the page said, and what it left out

`docs/PROGRAMME-STATUS.md` has a section called *"What is left, and what it costs"*, which is a long
**historical** backlog — the BEIR-18 coverage story, the research bets, the blocked `.bin` checkpoints. It
ended with *"the queue is empty"*, which was true of that backlog and false of this round: five measurements
were running, queued, or owed, and a fresh agent reading the page would have started something heavy beside
one of them.

It now has a subsection listing them, with state, cost and what each closes:

| measurement | state |
| --- | --- |
| full-set rank row, 2,255 queries | **running** |
| listwise arm (`jina-reranker-v3`) | **queued** on the lock |
| full-set coverage arm | **queued**, behind the listwise arm |
| `--honour-answerability` full-set arm | not queued; waits for the three above |
| sentence-budget sweep | not queued |
| cross-corpus abstention sample (MTRAG-UN) | **blocked**: needs the dataset |
| R8's training run | **blocked**: a corpus-scale distillation |

With it, the reason the queue is safe to leave unattended: each job waits on `eval/.heavy-job.lock` itself
rather than on a process-name pattern, because a script that greps for a job's name matches its own command
line and waits forever — `make heavy-wait` is the supported queueing form.

And the reason two of the three queued arms exist at all is stated there too: they settle **registered**
predictions, and each runs with the configuration its prediction was registered against. The coverage arm
deliberately does **not** use `--honour-answerability`, because changing two things at once would make its
delta unattributable.

## 1.9.9 — 2026-10-29

The listwise path re-verified after the cache fix, and its cost projected from that measurement before the
real run starts.

### Re-smoke, because the fix touched the path the run depends on

The cache change was the only thing standing between a queued 400-query run and a corpus embedding nobody
could reuse, and the function it touches — `first_stage()`, which builds the pool with the harness's own
functions — has no unit test (its pure parts do; the `beir.py` call chain does not). So it was re-smoked:
**two queries at depth 10, 16.4 s, nDCG@10 1.0**, and the cache landed where it was told to —

```
eval/r1/listwise-dense-cache.meta.json  {'chunk': 0, 'count': 8674, 'model': 'BAAI/bge-small-en-v1.5'}
```

8,674 vectors, the whole ArguAna corpus, recorded with the model that produced them.

### The projection, written before the run

At ~8 s per query on depth 10, depth 50 is five times the candidates in one pass, so the queued run's
expectation is **~40 s per query at eight threads and 1.5–2 h for 400 on 24**. If it lands far outside that,
the finding says in advance where to look: the 10× latency tail the cost envelope records on this benchmark,
not a wrong measurement.

A cost estimate recorded before the run is a prediction; recorded after, it is an excuse.

## 1.9.8 — 2026-10-29

The listwise tool cached its corpus embedding in `/tmp` — found by reading it, not by running it.

### A defect in code written three rounds ago

`tools/listwise_rerank.py` embedded ArguAna's corpus to
`/tmp/listwise-dense-cache` with `use_cache=False`. Three things wrong with that, in order of severity:

1. **`/tmp` is not durable on this host**, so a cache written there is a cache nobody finds — and the queued
   run would have embedded 8,674 documents for nothing.
2. **`use_cache=False`** made the cache a write nobody reads: a re-run to re-derive a number would pay the
   embedding again. A corpus embedding is *the* long part of a BEIR evaluation, which is why the harness has
   a chunked, resumable cache for it.
3. It wrote outside the workspace, which the sandbox permits for `/tmp` and which is therefore easy to do
   without noticing.

It now caches to `eval/r1/listwise-dense-cache` with `use_cache=True`. **The queued run picks the fix up**: the
process starts only when the lock frees, so the file is read after the edit.

The test that pins this looks at **string literals** through the AST rather than the source text — its first
version failed on the explanatory comment that describes the bug it prevents, which is a small lesson in
writing a check that tests the behaviour instead of the prose about it.

## 1.9.7 — 2026-10-29

The dated market digest records what happened when its own listwise row was acted on.

### An addendum, not a rewrite

`docs/research/09-retrieval-market-2026-10.md` reads *"a 0.6B listwise model matches a 4B cross-encoder
(61.85 vs 61.16) — parameter efficiency of listwise ranking is real"*. Acting on that row turned out to be a
different question from reading it, so the digest now says so, dated:

> **Addendum, 2026-10-29.** `jinaai/jina-reranker-v3` loads and runs here — 596M, `AutoModel` with
> `trust_remote_code`, `rerank(query=…, documents=[…])` scoring a whole list in one pass. Its licence is
> **CC-BY-NC-4.0**, so it can be measured as a research instrument and cannot be shipped inside the engine's
> dependency path; the Apache-2.0 alternative is 2.5B with a generative interface. **The market fact and the
> deployable fact are therefore different.**

That distinction is exactly what the digest exists to keep — it already carries a *shippable?* column on its
embedding table for the same reason — and it is the difference between a quality-per-parameter result and a
procurement recommendation. The measurement the addendum enabled is registered in the findings document, so
the note points forward rather than closing the question.

Dated digests get append-only notes; rewriting one to look prescient would defeat the reason they are dated.

## 1.9.6 — 2026-10-29

The second half of registered prediction (a) is queued, not merely owed.

### Three measurements, one lock, an explicit order

The programme now has a queue rather than a to-do list:

| | state |
| --- | --- |
| full-set **rank** arm (R4b's placement row) | running, 725 of 2,255 |
| **listwise** arm (R5's named measurement) | waiting on the lock, queued first |
| full-set **coverage** arm (prediction (a)'s other half) | waiting on the lock, queued second |

All three use the same flock-based wait, so the order is enforced by the lock rather than by whoever
remembers. Each checkpoints, so an interruption resumes.

**One thing the queue is explicit about**: the coverage arm runs with the **shipped** abstention behaviour,
not with `--honour-answerability`. Prediction (a) was registered against the behaviour that produced the
150-query measurement, and changing two things at once would make the delta unattributable — the same reason
the harness carries both settings in its fingerprint. The answerability lever gets its own arm afterwards.

## 1.9.5 — 2026-10-29

The declared Python floor becomes a local check, because CI is the wrong place to find out.

### `tests/test_python_floor.py`

`pyproject.toml` says `requires-python = ">=3.10"` and CI runs 3.10 and 3.12. **The development host runs
3.13**, so a 3.11+ syntax slip — a `type` statement, PEP 695 generics, `except*` — compiles here and fails on
the first CI run, which is a push nobody can undo. That is the same class of gap as round 31's two tests
that needed torch while the CI job installed none: *green on the machine where everything is installed says
nothing about the machine that follows `pyproject.toml`.*

The test parses every shipped module — `src`, `tools` and `tests`, **300+ files** — with
`ast.parse(..., feature_version=(3, 10))`. It also asserts the **CI matrix actually runs the declared floor**,
because a floor no run exercises is a floor nobody has tested.

**Perturbed before trusted**: appending `type _FloorProbe = int` to `schemas.py` fails the test with
`Type statement is only supported in Python 3.12 and greater`; restored.

It cannot catch a *runtime* API added in a later version — the CI matrix is what covers that. It catches the
syntax, which fails loudly and cheaply.

### And the CI-shaped environment re-verified

The same round re-ran the suite in a venv installed exactly as CI installs it, after several rounds of
changes to `engine.py` and the tools: **1,822 passed, 58 skipped**. The configuration that caught round 31's
regression still passes.

## 1.9.4 — 2026-10-29

The answerability lever reaches the harness that measures it — flag, fingerprint and both settings
records.

### `--honour-answerability` on `tools/engine_extractive.py`

Round 47's defect fix added the engine-side option, deliberately default-off because every recorded
extractive row predates it. That left the lever **unmeasurable through the harness**: the row that would
show its product-level effect had no way to turn it on, and nothing would record that it had been turned on.

Now `make engine-extractive ENGINE_EXTRACTIVE_ARGS="--honour-answerability"` runs the row with the flag, and
the flag travels with it into three places:

- the **resume fingerprint**, for the reason the composer's objective does: a configuration that changes
  which rows abstain must not be mixed under one fingerprint. This repository has already reported two arms
  identical to four decimals from exactly that omission.
- **both settings records** — the META record and the final report — because a fingerprint is a hash and
  the rule is that *a configuration variable that changes what a number means has to appear in the artefact*.

Three assertions in one test cover the three places, so a future refactor cannot quietly drop one of them.

## 1.9.3 — 2026-10-29

A run that looked stalled was a slow query — and the tail it exposed belongs in the cost envelope.

### The check

The full-set row's manifest sat at 625 for 45 s while the process burned twenty cores, which looks exactly
like a hang. Reproducing the one query instead of watching the counter settled it in 30 s: `answerable[751]`
takes **29.2 s** at four threads where a typical query takes ~3 s, and the run completed 25 rows in the two
minutes afterwards. **A run that appears stuck is usually a slow query, and the way to know is to reproduce
that query** — the counter can only tell you that nothing finished, which is what a slow query looks like.

### What the tail means

The decision pass reads the whole evidence block, so a query whose top passage is long costs more, and
nothing in the engine bounds that ratio. On this corpus the tail is **~10× the median**. That is exactly
what a median hides: a capacity plan built on 6.71 s would be wrong on p95 and worse beyond it, which is why
`tools/cost_report.py` reports `p95` beside the median and why the R0 finding quotes a spread rather than a
point.

Recorded with the limit that matters: **one slow query is not a distribution.** The finding says the tail
exists, names its cause, and says where a per-corpus p95 would come from — a `--limit`-free row — rather than
presenting a single observation as a shape.

## 1.9.2 — 2026-10-29

A product defect found by measuring the gate instead of the decision layer: **the extractive path never
consulted the answerability verdict.**

### The verdict moved and the answer did not

Configuring the fitted gate on the twenty `null_query` records moved **mean answerability from 0.637 to
0.218** — a large, correct drop — and left the composer emitting **the same 4.25 sentences at the same 15%
abstention rate**:

| arm | mean `P(answerable)` | abstention | mean sentences |
| --- | ---: | ---: | ---: |
| shipped scalar | **0.637** | 0.15 | 4.25 |
| fitted gate configured | **0.218** | 0.15 | 4.25 |

`RagEngine.query_extractive` calls `self.query(...)`, which computes `answerable` through the gate, and then
hands the passages to `compose_extractive`, which knows only whether a quotable span exists. **`answerable`
was never consulted.** The decision layer's question — *can this corpus answer it* — and the composer's —
*is there a faithful sentence to quote* — are different, and the answer path answered the first with the
second. That is why the product-level number (85% answered) is worse than the gate's own (78.6% called
answerable): some of the 85% were questions the engine had already judged unanswerable.

### The fix, deliberately opt-in

`query_extractive(..., honour_answerability=True)` abstains when the decision layer said the corpus cannot
answer, reporting the verdict beside the abstention instead of contradicting it. **The default is `False`**,
and that is not timidity: every recorded extractive row — including the full-set row running while this was
written — was measured with the old behaviour, and changing a default under a running row is the provenance
failure this repository catalogues five times. The next full-set row runs with the flag on.

A test covers both branches, and the finding states that these twenty records are part of the gate's own
fit data, so the number is in-sample and establishes where the lever reaches rather than how good it is.

## 1.9.1 — 2026-10-29

The abstention number a user actually experiences: **85% of questions the corpus cannot answer get a
quoted answer** — and the composer's objective is invisible there, which is the architecture working.

### Product-level, not gate-level

Every abstention figure in this programme so far has been a **gate** number: what the decision layer says
if asked. The product question is different, and nobody had measured it. Twenty of the benchmark's own
`null_query` records through `query_extractive` at shipped defaults:

| | abstained | answered anyway | mean sentences |
| --- | ---: | ---: | ---: |
| shipped defaults, extractive route | **15%** | **85%** | 4.25 |

**17 of 20 unanswerable questions get a quoted answer**, 4.25 verbatim sentences each, not one empty
response. Consistent with the 78.6% the scalar gate reports at its deployed threshold — and the more damning
form of the same fact, because it is the one a user meets: the citations are faithful, the sentences are
real, and the question had no answer in the corpus.

Caveats stated in the finding: **n=20**, and the fitted gate artefact is not configured here, so this is the
*shipped scalar* default rather than the gate's behaviour — which is the point.

### The objective does not leak into abstention, which is the architecture working

The same twenty queries ran under both composer objectives, because R2 changed selection and nothing had
checked whether that reaches the abstention decision:

| arm | abstention | mean sentences |
| --- | ---: | ---: |
| rank order (shipped) | 0.15 | 4.25 |
| coverage objective | 0.15 | 4.25 |

Identical. Abstention is decided upstream of selection, so a selection objective cannot move it. A lever
that raises answer coverage on answerable questions **without** making the system more willing to answer
unanswerable ones is exactly the property R2 needed and had not verified.

## 1.9.0 — 2026-10-29

R5's listwise arm: the blocker was checked rather than assumed, and it has moved. The arm is built,
predicted and queued.

### The blocker, resolved at the interface and bounded by the licence

R5 was blocked on two things. Both were checked:

- **The interface moved.** [`jinaai/jina-reranker-v3`](https://huggingface.co/jinaai/jina-reranker-v3) —
  the model behind *Last but not Late Interaction for Listwise Document Reranking*
  ([arXiv 2509.25085](https://arxiv.org/abs/2509.25085v4)) — is a **596M listwise** reranker, the same size
  class as the pointwise model already measured, and it runs here: `AutoModel` with `trust_remote_code`,
  exposing `model.rerank(query=…, documents=[…])`, which scores a whole candidate list in one pass.
- **The licence did not move, and that bounds the conclusion.** It is **CC-BY-NC-4.0**: a research
  instrument, not something this engine could ship. The Apache-2.0 alternative, `bge-reranker-v2-gemma`, is
  **2.5B parameters** with a generative interface — 4.4× the compute and another harness path. So the arm
  can be *measured* and still not *shipped*, and every number is quoted with that beside it.

### The prediction and both falsifiers, registered before the run

nDCG@10 in **0.50–0.56** — above `v2-m3`'s 0.4722, still below the first stage's 0.5664. *Falsifier A:* if
it beats 0.5664, the collapse is an **interface** property and the standing negative must name the
interface. *Falsifier B:* if it lands below 0.4722, seeing candidates together buys nothing and the
objective story stands alone.

### The interface was proved before the run was paid for

A five-query smoke test at depth 20 — 120 s — established the whole path: the pool built by the harness's
own `hybrid-biencoder` functions, one listwise call per query, positions mapped back to document ids. It
also caught a real defect: `evaluate_rankings` was scoring all **1,406** ArguAna queries while five had been
ranked, so unsampled queries counted as zeros and the summary read **nDCG@10 0.0028 over 1,406 queries**.
The harness's rule — restrict the qrels to the sample — was missing. With it, the same five read **0.80**.
*A catastrophic-looking number and a good-looking number were one line apart, and no table would have
flagged either.*

Four tests cover the two things that can be wrong without a metric noticing: the pool the model sees, and
the position-to-id mapping. `make listwise-rerank` runs the full 400-query arm, queued behind the lock.

## 1.8.4 — 2026-10-29

The claim ledger reaches the index page — the surface that actually gets quoted.

### `docs/FINDINGS.md` is pinned, not just `PROGRAMME-STATUS.md`

The status page has had its numbers pinned to their reports since the October round. The **findings
index** had no such pin, though it is the larger surface — the page a reader cites when they want to know
what this programme measured — and its rows quote the same four levers.

The same registry now covers it: gate held-out AUC **0.8377**, composer's shipped arm **0.2060** and its
objective arm **0.2610**, reranker v2-m3 MAP@10 **0.3710**, and the cascade's M=10 CPU seconds when a cost
report exists (it does not here, so that pin skips — the same gap axis 4 reports).

**Perturbed before trusted**: changing the objective arm's number in the index to 0.2699 fails the test and
names the file and field that disagree; restored.

That leaves the index and the status page on one ledger, applied to the two pages a reader is most likely
to quote from.

## 1.8.3 — 2026-10-29

The two-hour full-set row is checked at 6% rather than trusted only at the end.

### A mid-flight check, and what it is not

The full-set retrieval row writes a resumable shard, so its first **134** rows can be summarised while it
runs. A measurement that turned out to have used the wrong index — or to be emitting empty answers —
should be caught at 6% rather than at 100%, which is the whole reason to look.

| metric | 150-query subset (recorded) | first 134 of the full set |
| --- | ---: | ---: |
| answer nugget recall | 0.2060 | **0.1987** |
| citation-clean, all queries | 1.0000 | **1.0000** |
| abstention rate | 0.0000 | **0.0000** |
| Hits@10 | 0.9067 | 0.9328 |
| Hits@4 | 0.7400 | 0.7985 |
| MAP@10 | 0.3128 | 0.3265 |
| MRR@10 | 0.6152 | 0.6410 |

Healthy on every property that would betray a misconfiguration — no empty answers, every answer
citation-clean at 5.00 sentences, no abstentions on an answerable-only set — and tracking the subset it was
previewed on, a little better on every retrieval column.

**The finding says plainly that this is not a result.** It is 6% of the set, read before the run finished,
on the arm whose number was never in question. The registered prediction concerns the **delta between the
two composer objectives**, and one arm is running.

## 1.8.2 — 2026-10-29

The evidence protocol's cost axis wires itself to the report the pipeline writes.

### `--cost` stops being a flag anyone has to remember

Axis 4 — seconds, CPU seconds, device — was populated only when a caller passed `--cost`. That makes the
last axis of a five-axis report depend on a flag that `make evidence-pipeline` did not pass, so the axis
that answers "what does this cost" was the one most likely to read "not measured" for reasons having
nothing to do with the system.

It now defaults to `eval/cost-report.json`, the path `make cost-report` writes. An **absent** file still
leaves the axis reported as *not measured* rather than failing the run, which is the behaviour the
protocol exists to have: a named gap rather than a number from somewhere else.

That is its state on this repository today, and the finding now says so explicitly — the cost report for
the MultiHop index has not been run beside the other two axes, so axis 4 reads "not measured" because it is.

## 1.8.1 — 2026-10-29

`docs/REPRODUCING.md`: every number this project quotes, with the command that produces it.

### The page that makes "reproducible" a property rather than a habit

The programme's whole claim on a reader's trust is that its figures can be re-derived. Nothing said how.
The new page is a table of the eight measurements — what each answers, the commands, the artefacts it
writes, its **recorded** runtime on this host, and the findings document that interprets it — from
`make cost-report` (~10 min) through `make evidence-pipeline` (~68 min from a fresh clone) to the full
retrieval row (~2 h for 2,255 queries).

It also states the four things that make a re-run comparable: the **index identity** hash, the **query
sample seed**, the **configuration block** each report prints, and the two gates. And it states what
reproduction does *not* give you: every measurement here is single-host and single-run, the intervals are
bootstrapped over queries rather than machines, and a number reproduced on a different corpus is a
**different measurement**, not a confirmation — which is the answerability gate's own first limit.

Two structural points are spelled out on the page because they are what makes the model work rather than
merely describe it: artefacts live under the gitignored `eval/`, so prose can always be traced to a file
that is regenerated instead of trusted; and the heavy-job lock plus per-run checkpointing mean an
interrupted measurement resumes rather than restarts.

A test requires every `make` target the page names to exist, since a renamed target turns "re-derive it
yourself" into "hunt for the new name".

## 1.8.0 — 2026-10-29

The assessment's closing section said none of its predictions had been measured. Eight rounds later, most
have — and the outcome table replaces the stale sentence.

### A statement about the document's own status, left standing

`docs/best-in-the-world-2026-10.md` ends with a section headed *what this document does not claim*, whose
last bullet read:

> Every prediction in §5 is a hypothesis with a stated falsifier … **None has been measured.**

That was true when written and false by the time the round finished. Left stale, it **under**-claims in a
way that misrepresents the work — it says nothing was measured when six items have been — and it sits in
the one section a skeptical reader consults first.

It is now an outcome table, item by item:

| item | outcome |
| --- | --- |
| **R0** decision share | **falsified** — 99.8% of cost is the decision pass; the falsifier fired at an 8.1% spread |
| **R1** encoder refresh | partly — routing, not a swap |
| **R2** composer objective | **falsified in magnitude** — 0.0489 of 0.5404 recovered, not 0.25 |
| **R3** portable abstention | confirmed — AUC 0.8377, 30.7% vs 69.3% false accepts *and* 76.7% vs 72.0% true accepts |
| **R4** evidence protocol | shipped, now five axes; the placement row refuses a subset |
| **R5** reranker generation | confirmed; the ArguAna prediction **held inside its band** |
| **R6** decision cost | confirmed at the small end — 43% free, with interaction costs |
| **R7** multilingual | **the claim was withdrawn** — 28 h per collection, README corrected |
| **R8** submittable artefact | half — the gate ships, the training run is the gap |

**Two falsified, one withdrawn, four confirmed, one half done.** The text ends by saying what is worth
re-reading: the falsifications, because both came from asking a question the mechanism story had made look
already answered.

A test now forbids the stale sentence from returning and requires the table to keep recording all four
outcome kinds.

## 1.7.2 — 2026-10-29

The two remaining composer measurements are registered before they run, while the answer is unknown.

### Written now, not after

The heavy-job lock is held by the full-set retrieval row, so the composer's two open questions cannot be
measured this hour. They are therefore **registered**, which is the only thing that makes them
predictions rather than descriptions:

**(a) The full-set composer row.** The `0.2060 → 0.2610` delta rests on a 150-query subset. Predicted on
the full 2,255: **+0.03 to +0.08**, i.e. within ±0.025 of the subset estimate, cite-clean 1.0000 in both
arms. *Falsifier:* below +0.02 makes the 150-query table a subset result that has to be re-stated with its
own interval; above +0.09 does the same in the other direction.

**(b) The sentence budget.** Five sentences is the obvious remaining constraint once selection is known to
be only one of several. Predicted: 5 → 8 → 10 raises nugget recall monotonically with sharply diminishing
returns — **8 sentences buys under +0.04**, and by 10 the marginal gain is under +0.01 per sentence, with
cite-clean at 1.0000 throughout. *Falsifier:* if 8 sentences buys **more than +0.08**, the budget is the
dominant constraint and R2's conclusion is wrong; if cite-clean drops below 1.0000 at any budget, the
guarantee that makes this engine quotable would be budget-dependent, which would be the more serious of the
two results.

Both carry their cost estimate and both are indexed from `docs/FINDINGS.md`, so neither can be quietly
dropped.

## 1.7.1 — 2026-10-29

The ArguAna verdict reaches the three places that carry R5, and the cascade's interactions reach the
status page.

### R5's verdict, in the pages a reader uses

`docs/PROGRAMME-STATUS.md`, `docs/best-in-the-world-2026-10.md` and `docs/FINDINGS.md` all carried R5 as
"pointwise answered; listwise untested" — a state that stopped being current the moment the ArguAna arm
landed. Each now states the verdict: the prediction held inside its registered band (0.4722 against a
0.35–0.50 estimate), the falsifier did not fire, and capacity recovers two thirds of the opposition damage
without reversing the sign.

**The "listwise arm is still untested" sentence stays in all three**, deliberately. A confirmed prediction
is not a licence to widen it, and the falsifier named the listwise question as the one the test *could not*
reach.

### R6's row gains its interactions

The cascade's row said "identical quality" — true of the retrieval metrics it measured, and now qualified
by the two interaction probes: **4 of 20 gate verdicts flip** between M=20 and M=10 while the aggregate is
nearly indifferent, and the composer loses **0.0043 of nugget recall** at M=10 with abstention rising.
"Free for the ranker, slightly expensive for the answer" is the accurate form.

Three tests pin the ArguAna numbers to the two reports that produced them and keep the untested-arm
sentence in place.

## 1.7.0 — 2026-10-29

The ArguAna prediction held — and the magnitude, not the sign, is what moved. Plus the cascade's second
interaction: it costs the composer a little of everything.

### ArguAna: the registered prediction, confirmed inside its band

Pre-registered three commits before the run: *v2-m3 lands **below** the no-CE hybrid's 0.568, point
estimate **0.35–0.50**; falsifier — if it beats the first stage by any margin, the collapse is a capacity
artefact and R5's mechanism claim is wrong.* Measured at depth 50 on the seeded 400-query subset:

| arm | nDCG@10 | seconds |
| --- | ---: | ---: |
| `hybrid-biencoder` (first stage) | **0.5664** | 11.9 |
| + `bge-reranker-base` (2024, 278M) | 0.2736 | 1,900 |
| + `bge-reranker-v2-m3` (2026, 568M) | **0.4722** | 6,007 |

Paired over the 400 matched queries: first stage **−** v2-m3 **+0.0942 [+0.0530, +0.1335]**; first stage −
2024 model +0.2928 [+0.2522, +0.3328]; **v2-m3 − 2024 model +0.1986 [+0.1566, +0.2407]**. The 2026 model
still loses, on 290 of 400 queries.

**The prediction landed at 0.4722, inside the 0.35–0.50 band, and the falsifier did not fire.** Capacity
recovers **two thirds** of the opposition damage (−0.293 → −0.094) without reversing the sign, at 3.2× the
compute. That makes the repository's standing negative precise rather than wrong: it was never "a
pointwise ranker cannot represent opposition" but **"a pointwise ranker pays for opposition in proportion
to how little it knows"**.

### The cascade at the composer: free for the ranker, not for the answer

Same twenty queries, three depths, both composer objectives, every answer through the verifier:

| `M` | arm | abstention | nugget recall | citation-clean |
| --- | --- | ---: | ---: | ---: |
| 20 | rank / coverage | 0.20 / 0.20 | 0.2005 / **0.2232** | 0.80 |
| 10 | rank / coverage | 0.25 / 0.25 | 0.1862 / **0.2097** | 0.75 |
| 5 | rank / coverage | 0.30 / 0.30 | 0.1834 / **0.2013** | 0.70 |

The composer's advantage **survives the cascade** at every depth (+0.023, +0.024, +0.018), so R2's lever is
not an artefact of having twenty candidates. But the cascade is not free here: nugget recall falls, and
abstention rises because an abstention emits the fixed no-answer sentence that the checker counts as an
uncited claim. **R6's "identical quality" is a claim about retrieval metrics**; one stage later it costs a
little of everything.

## 1.6.3 — 2026-10-29

The cascade's interaction with the gate, measured: aggregate-stable, per-query unstable.

### Two levers measured alone, then paired

R6 measured `SECOND_STAGE_TOP_M` on retrieval quality; R3 measured the answerability gate at
`retrieve_k=20`. Nobody had asked what the gate concludes when it only sees ten candidates — and the
gate's features include pool-shape signals that a truncated pool necessarily changes. Twenty answerable
MultiHop-RAG queries, both depths, paired:

| `SECOND_STAGE_TOP_M` | answerable rate | mean `P(answerable)` | mean kept |
| --- | ---: | ---: | ---: |
| 20 (all) | 0.55 | 0.5099 | 3.85 |
| 10 | 0.55 | 0.5009 | 3.30 |
| 5 | 0.65 | 0.5000 | 2.50 |

In aggregate the gate is nearly indifferent — the mean probability shifts by **−0.009** and the
answerable rate is unchanged at ten. **Per query it is not**: the verdict flips for **4 of 20** queries,
in both directions, which is exactly why the aggregate looks unmoved.

For a system whose product is *honest abstention*, "the same number of questions refused" is not the same
claim as "the same questions refused", and only the first survives a change of `M`. Twenty queries is a
probe rather than an effect size, and the finding says so; the M=5 row's *rise* to 0.65 is recorded as
something a tuner should not be surprised by rather than as a result.

## 1.6.2 — 2026-10-29

`BENCHMARKS.md` — the page that says what may and may not be claimed — now carries the October levers.

### Four additions, each with its limit, and one removal

`BENCHMARKS.md` opens its claim section with what *can* be claimed and follows it with what cannot. The
four levers the October round measured belong in exactly that structure, so each now appears with the
boundary that travels with it:

| added | allows | does not |
| --- | --- | --- |
| the 12-feature answerability gate, shipped | refusing what the corpus cannot answer, provably: held-out **AUC 0.8377**, false accepts more than halved while true accepts *rise* | transfer across corpora — halves of one benchmark, balanced by construction |
| the composer objective | **0.2060 → 0.2610** nugget recall at unchanged citation validity, significant | the gap to the pool's 0.7438; selection is *a* constraint, not *the* one |
| the reranker generation | **+0.031 MAP@10** over the 2024 model, +0.058 over none | anything on BEIR, where a pointwise ranker loses; ArguAna is being measured |
| the decision cascade | **43% less decision CPU at identical quality** here | generality — free because the first stage is strong here |

The round also **removed** a claim, and that is recorded on the same page: `docs/index.md` said
`make claim-check` "resolves every quoted figure", and it resolves 54 against 8,370 decimal figures in the
tracked prose. A page about what may be claimed is the right place to note a claim that was withdrawn.

## 1.6.1 — 2026-10-29

The CI workflow's `make` invocations become a checked contract.

### `tests/test_ci_workflow_targets.py`

`test_docs_make_targets_exist.py` holds the *documentation* to "documented means runnable". The workflow
is the other place a `make` invocation lives, and a renamed target there fails only on the next push —
after the change that renamed it has been committed, and after it has been trusted. A CI step that cannot
run is the same defect as a documented command that cannot run, one step further from the reader.

The check reads the workflow text (the commands live inside `run: |` block scalars, which a parser would
hand back unchanged) and asserts every `make <target>` names one the Makefile defines. It matches inside
comments too, deliberately: a workflow comment naming a command nothing defines misleads by the same
mechanism, and there are few enough of them that the noise a stricter scan avoids does not arise.

It found nothing — the workflow's nine invocations (`lint`, `test`, `test-mcp`, `claim-check`,
`consistency-check`, and the commands its comments name) are all real — which is worth recording as a
clean sweep beside the eight contracts that were not.

## 1.6.0 — 2026-10-29

CI installs the MCP extra and runs its suite — and doing so found two of this round's own tests that
would have broken CI on its first run.

### The MCP server was 1,000 lines with no coverage in CI

`mcp_server.py` is an optional surface behind the `[mcp]` extra, and the CI job installed the engine's
dependencies *without* it. So its **21 tests skipped on every run**, and `make test-mcp` — the target that
exists for them — was invoked by nothing but the author. The job now installs `mcp>=2.3` and runs
`make test-mcp` as a named step, so a failure points at the server rather than at the suite. Verified in a
venv installed exactly as CI installs it: **21 passed**.

### Two of my own tests would have failed there

Running the suite in that configuration — rather than in the development environment, which has torch —
found two tests added five rounds ago that need torch:

- the last-token pooling arithmetic test builds `torch.tensor`s directly;
- the pooling-argument test constructs `beir.BiEncoder`, whose constructor imports torch.

Both now `importorskip("torch")` with the reason, so they run where torch exists and skip where it does
not — the same guard the `.bin`-converter test already had. **The lesson is the one CI exists for**: a
suite that passes on the machine where everything is installed says nothing about the machine that
installs only what `pyproject.toml` declares.

## 1.5.9 — 2026-10-29

The README's Known limits carries the October round's limits, and the long-jobs section documents
`heavy-wait`.

### Four levers, four limits, in the section a careful reader reads

`README.md`'s **Known limits** said "these are measured, by `make eval` and `make bench`" — a set written
before any of October's results existed. Each of the four levers now appears there with the limit that
travels with it:

- the gate **transfers across queries, not across corpora** — fitted on halves of one benchmark, classes
  balanced by construction, news questions;
- the composer's objective is **+0.055, not the +0.25 the gap suggested** — significant, half the
  pre-registered bar, so selection is *a* constraint rather than *the* one;
- the cascade is **free on this corpus because the first stage is strong there**, and per-corpus by
  construction;
- the reranker generation is **a lever only against a weak first stage**, with its ArguAna behaviour
  being measured rather than assumed.

Each points at the findings document that supports it, which is the same structure the pre-existing
entries use.

### And the long-jobs section gains the one command that queues

`make heavy-wait` is documented where a reader looks for it — beside `heavy-status` — with the reason it
exists: a shell that waits on `pgrep -f "tools/beir.py"` matches its own command line and waits forever.

## 1.5.8 — 2026-10-29

`heavylock.py --wait`: queue behind a long job by asking the lock, not by grepping for its name.

### The trap this closes

The first version of the queued full-set launcher waited with `while pgrep -f "beir[.]py"; do sleep 30;
done`. A shell whose own command line contains that pattern matches it, so the loop waits forever even
after the job exits — the same self-match that had already killed two `pkill` calls in this session. It
is not a mistake a careful reading catches reliably; it is a mistake that *asking a better question*
removes.

`tools/heavylock.py --wait` asks the lock itself — the same question a heavy job asks when it starts:

```sh
make heavy-wait HEAVY_POLL=5 && make <long-job>
```

`wait_for_free()` acquires, releases and returns, so **nothing is held on return** and the caller's own
`heavy_job_guard` is the one that acquires — which keeps the holder's name and pid honest.
`--timeout` gives up with exit 1; `--poll` sets the interval.

Four tests: free-lock returns immediately, a held lock times out, nothing is left held on return, and
the CLI exits non-zero on timeout. The `sleep` is injectable, so the suite runs in 0.4 s rather than
sleeping for real — an earlier version took 30 s because the CLI used the default poll interval, which
was its own small lesson.

## 1.5.7 — 2026-10-29

The configuration axis checked on a non-default arm — and the full-set leaderboard row queued behind the
running measurement.

### The axis does the thing it was added for

The same protocol run over the **cross-encoder** row now shows `rerank_model=BAAI/bge-reranker-v2-m3` in
the answer row's settings, so the CE arm and the shipped arm are distinguishable at a glance rather than
by filename. `index_identity` is identical across both, which is correct: the identity hashes the corpus
and the ingest settings, not the second stage. A configuration axis that only ever printed the default
would have been untested in the case it exists for.

### R4's last measurement is queued

The full 2,255-query answerable row — the one `make multihop-row` refuses to place a subset in place of —
is queued behind the running cross-encoder arm and will start when the heavy-job lock frees. Measured
cost: ~3 hours at 736 s per 150 queries. When it lands, `make multihop-row` will place it without the
refusal, and R4's placement half will be a published-style row rather than a marked subset.

## 1.5.6 — 2026-10-29

The programme's entry point carries the placement half and the audit layer, with its numbers pinned.

### `docs/PROGRAMME-STATUS.md`, brought current again

R4's row now reports both halves — the five-axis protocol reproducible in ~68 min, and `make
multihop-row`, which refuses to place a subset while showing what the subset reads (Hits@10 0.940,
Hits@4 0.847, MRR@10 0.688 against the best published row's 0.747 / 0.663 / 0.586, MAP@10 0.371 against
0.480). R8's row moves from *not started* to *gate shipped, artefact owed*, which is where it actually
stands.

A new subsection records **the audit layer**, which turned out to be the round's second product: eight
contracts, six of which found something, and every failure in a surface that *describes* the work rather
than in the work — a settings block, a provenance ledger, a configuration table, an index, a help screen,
a version string, a tool path, and the front page's claim about its own checking. The rule they state is
in the table.

Two tests now pin the placement numbers to the cross-encoder report that produced them, and one asserts
the page keeps the word *subset* beside them — a number that is not placeable must not be printed as
though it were.

## 1.5.5 — 2026-10-29

R4's placement half: a command that formats the engine's row for MultiHop-RAG's retrieval table — and
refuses to place a subset.

### `make multihop-row`

`tools/engine_extractive.py` already reports exactly the four columns the benchmark's primary table
carries (`Hits@10`, `Hits@4`, `MAP@10`, `MRR@10`), and the table is retrieval-only, which is why this
engine can be placed in it at all. `tools/multihop_row.py` puts its row beside the published anchors and
prints the three caveats that travel with it — the chunking differs, the paper's table carries no lexical
baseline, and the engine contributes two columns it owns (citation validity, answer nugget recall).

**The refusal came first.** A report over 150 of the 2,255 answerable queries looks exactly like one over
all of them, so the tool checks the count and prints:

```
**not placeable** — 150 queries, not the 2255 answerable records the table scores — this is a subset,
and a subset presented as the member is the failure class 11 records
```

That is the eleventh class in the measurement-failures ledger turned into a gate rather than a footnote;
`--allow-subset` prints the numbers anyway, clearly marked, for internal reading.

### What the subset shows, marked as a subset

| system | Hits@10 | Hits@4 | MAP@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| voyage-02 + rerank (published) | 0.747 | 0.663 | 0.480 | 0.586 |
| **ours: engine + `bge-reranker-v2-m3`, 150-query subset** | **0.940** | **0.847** | 0.371 | **0.688** |

Three of four columns exceed the best published row and `MAP@10` trails by 0.11 — this repository's
long-standing shape (*the gold is found; the order is where the score goes*), with the 2026 reranker
having closed most of the `MRR@10` gap. None of it is placeable until the full set runs, which is ~3
hours and the one remaining measurement R4 needs.

Six tests pin the tool, including the refusal and the caveats.

## 1.5.4 — 2026-10-29

The composer objective checked against real weights, through the shipped verifier.

### A different selection, and the guarantees intact

The API test for the per-request `coverage` runs against the stub model: it proves the plumbing and says
nothing about whether the objective changes what a real model emits. On the documentation index with the
multilingual checkpoint, both arms per query, every answer passed through `verify_citations`:

| query | sentences | selection differs | rank arm clean | coverage arm clean |
| --- | ---: | --- | --- | --- |
| SciFact nDCG@10 | 5 / 5 | **yes** | yes | yes |
| how the answerability gate abstains | 5 / 5 | **yes** | yes | yes |
| which chunk sizes are chosen | 5 / 5 | **yes** | yes | yes |

Three of three selections differ, and both arms are citation-clean on every query under the same verifier
the end-to-end row uses. That is the property that had to hold — a different selection is only acceptable
if the guarantees survive it — and it holds under the real checkpoint rather than only under the stub.

## 1.5.3 — 2026-10-29

The provenance failure class gains its fifth instance and the rule the five of them state.

### Class 14, completed

The ledger entry had four instances. The fifth is this round's own: the composer's objective was put into
the resume **fingerprint** — so a resumed run could not mix arms — and into nothing a human reads, which
left the two rows behind `0.2060 → 0.2610` distinguishable only by remembering which filename they were
written to. Fixed two commits ago; recorded here because a class is worth more than its instances.

The five, stated together:

> When a configuration variable changes what a number means, it has to appear in the artefact — or the
> artefact is a filename.

That is the most transferable sentence the October round produced. It generalises past this repository:
any harness that names its outputs by parameter (which is most of them) is one omission away from an
unattributable result, and the tell — *two arms that should differ came out identical* — is cheap enough
to check on every run.

`docs/findings-extractive-coverage.md` now says how its own two arms are identified, since the rows
predate the fix.

## 1.5.2 — 2026-10-29

A row now records the composer objective it ran under — the fifth instance of the same class, and again
the round's own.

### `extract_coverage` in the settings block, not only in the fingerprint

The composer's objective went into `engine_extractive.py`'s **resume fingerprint** first, so a resumed
run could not silently mix arms. That fixed the identity problem and left the *readability* one: a
fingerprint is a hash, and `947d10ba…` does not tell a reader whether they are looking at the shipped
order or the measured one. The rows produced by the 0.2060 → 0.2610 comparison could only be told apart
by remembering which filename they were written to.

Both records now carry `extract_coverage`, and the evidence protocol's configuration axis surfaces it
alongside `second_stage_top_m`. A test asserts the field is written in **both** places — the META record
and the final report — because either alone leaves a row unidentifiable.

**Artefacts produced before this commit keep their old settings blocks**, and the protocol prints what is
recorded rather than what is true: the `rank150` row still shows a stale `index_dir` and no
`extract_coverage`. That is visible in the report (and in the fact that the directory it names does not
exist), which is the behaviour the axis exists to have.

### The class, counted

Five instances in one programme: a cache keyed by a basename, a baseline read from the wrong column, a
resume fingerprint missing the variable under test, a settings block naming a derived path, and now a
settings block **omitting** the variable that the row's whole point depends on. The pattern is stable
enough to state as a rule: **when a configuration variable changes what a number means, it has to appear
in the artefact or the artefact is a filename.**

## 1.5.1 — 2026-10-29

The composer's objective becomes a per-request choice on `POST /v1/extract`, not a service-wide setting.

### `coverage: true | false` on the extract route

R2's composer objective measured **answer nugget recall 0.2060 → 0.2610** at unchanged citation
validity, and it shipped behind `EXTRACT_COVERAGE` — a variable read once at construction. A caller
therefore could not ask for the measured arm without restarting the service, which makes the result
reachable only by whoever deploys it.

`POST /v1/extract` now accepts `coverage`, and `RagEngine.query_extractive(..., coverage=None)` takes the
same override (`None` means the service default, so nothing changes for a caller that omits it). The
guarantees are identical either way — every emitted sentence is a verbatim span of a cited passage and
the markers stay in range — which is what makes this a per-request choice rather than a deployment
decision. A test exercises all three states (omitted, `true`, `false`) and re-checks the citation contract
on each.

### A correction worth noting

The first attempt at this edit asserted on a call-site indent that was four spaces deeper than assumed
and aborted before writing anything; the second matched the real text. The stray parameter introduced
while fixing it (`include_matching`, never requested) was removed before the tests ran. Both are the
ordinary cost of scripted edits, and both were caught by the assertion rather than by a reader.

## 1.5.0 — 2026-10-29

The front page claimed its gates covered every quoted figure. They cover 54 of several thousand.

### A process claim corrected, and its count now checked

`docs/index.md` said, under the heading "Two rules that make the pages readable":

> `make claim-check` resolves **every quoted figure** against the report that produced it.

`claim_check` registers **54** claims. The tracked prose contains **8,370** decimal figures. The sentence
was false by two orders of magnitude — and it is the failure this repository catalogues as class 13, *a
correct number, over-generalised*, applied to a statement about its own machinery, printed on the page a
reader lands on.

It now states the registered count, says plainly that the great majority of figures are covered by
neither gate, and points at the measurement-failures ledger as the record of what care alone has cost
here. `tests/test_docs_claim_coverage.py` keeps the stated count equal to the number of checks that exist
(so the sentence cannot drift back into flattery) and forbids the phrase "every quoted figure" from
returning. It deliberately does *not* bound the total number of figures, which moves whenever a document
is edited.

### What this says about the audit layer

Eight contracts were added over the last eight rounds and six found something. Every failure was in a
surface that *describes* the work — a settings block, a provenance ledger, a configuration table, an
index, a help screen, a version string, a tool path — and the one that was clean was clean because 91
rows had been maintained by hand.

This one is the sharpest, because the overstated claim was about the checking itself. A project's
credibility rests on the accuracy of what it says about its own method, and that assertion had never been
counted against the machinery it described.

## 1.4.9 — 2026-10-29

`make help` now lists every target that exists — fifteen were invisible.

### The one command a reader runs first

Of the Makefile's **98 targets**, 83 appeared in `make help`. The missing fifteen included
`mteb-eval`, `calibration-report`, `nugget-recall`, `leakage-audit`, `reranker-fairness`, both
`labelling-*` queues, `two-wiki`, the two `order-eval-*` arms and `heavy-status`. Every one is documented
somewhere in the prose — which is why nobody noticed: the capability was discoverable *if you already
knew to look for it*.

A **capability a user cannot discover is one the project does not really have**, and `make help` is the
only place all of them are meant to be listed together. The fifteen now have one-line descriptions taken
from the targets' own comments, and `tests/test_make_help_lists_every_target.py` keeps it that way: every
`.PHONY` target except `help` itself must be listed, `make help` must not name a target that does not
exist, and both extractions must still see more than sixty entries so a broken regex cannot let either
pass vacuously.

## 1.4.8 — 2026-10-29

Tool paths written in prose are now checked — and the check found two of this round's own mistakes.

### `tests/test_docs_tool_paths_exist.py`

`test_docs_links_exist.py` checks markdown *links*; it does not see a backticked `tools/….py` in a
sentence, and those are how every findings document names the tool that produced its number. Of the
**96 distinct tool paths** the documentation names, **nine pointed at nothing**:

- **two introduced by this round, in the document written three rounds ago** — `distil_verify.py` for
  `distill_verify.py` (a dropped letter) and `distill.py` for the tool that is actually
  `finetune.py --teacher-models`. Both corrected.
- **seven declared planned** in the documents that name them: `generation-placement.md` heads its sketch
  "planned adapters only" and "planned, does NOT exist yet", and `SCORE-LIFT.md` names `tools/judge.py`
  under "**Specification.**". They are listed by name rather than skipped by a heuristic, which is the
  convention `test_docs_make_targets_exist.py` already sets for `PROPOSED_REFERENCES` — a checker that
  infers intent from prose is a checker nobody can trust.

A perturbation test drives both directions: a fabricated path is caught, and a declared-planned path is
exempt. A third test asserts the extraction still finds more than thirty paths, so a broken regex cannot
let the check pass vacuously.

**The lesson is the one this repository keeps paying for**: a path in prose is a claim, and nothing was
checking it. The two that were wrong were written by the tool that is supposed to be careful about
exactly this — which is why the fix is a test rather than another careful reading.

## 1.4.7 — 2026-10-29

The version becomes one value, checked in all five places it is written.

### `tests/test_version_is_one_value.py`

`docs/PROGRAMME-STATUS.md` says it plainly: *"The version is what `laya_rag.__version__` reports and what
the HTTP API publishes in its OpenAPI schema, so a client can tell which engine it is talking to."* That
holds only while five copies agree — `pyproject.toml`, `__init__.py`, `README.md`,
`docs/PROGRAMME-STATUS.md` and the newest `CHANGELOG.md` heading — and this round bumped all five by hand
fourteen times.

The test reads all five, requires one distinct value, requires the changelog's headings to be in
descending version order (so "newest heading" really is the current release), and checks the
client-facing consequence directly: `create_app().openapi()["info"]["version"] == laya_rag.__version__`.
A divergence is not cosmetic — a service answering `/openapi.json` with one version while its changelog
claims another cannot be matched to the release notes describing its behaviour.

Perturbed before trusted: setting the README to 1.4.7 alone fails the test and names the file that
disagrees; the file was restored.

## 1.4.6 — 2026-10-29

`/v1/stats` says how the engine is configured, not just what is loaded.

### A `behaviour` block on the stats payload

The October round added levers that change *behaviour* rather than performance — the second-stage
cascade, the composer's objective, the fusion rule, the recall-side selectors — and the service reported
none of them. A number quoted from a running service therefore could not name the configuration that
produced it, which is the provenance failure this repository has now recorded four instances of (a cache
keyed by a basename, a baseline read from the wrong column, a resume fingerprint missing the variable
under test, a settings block naming a derived path).

`GET /v1/stats` now carries:

```
second_stage, second_stage_top_m, extract_coverage, fusion_mode, fusion_k, dense_weight,
rerank_order, rerank_coverage_lambda, recall_dedup_threshold, recall_coverage_lambda,
answerability_gate_configured
```

The values are the *resolved* ones (`dense_weight` is what `auto` came to, not the word "auto"), for the
same reason `QueryResult.device` is the resolved device. One test pins the block's presence and the
shipped defaults, so a lever cannot be added without appearing here.

## 1.4.5 — 2026-10-29

Every findings document is now reachable from the index that claims to list them all — and one was not.

### `tests/test_docs_findings_indexed.py`

`docs/FINDINGS.md` opens with "every measurement the improvement programme produced, with the headline
result and the document that carries the detail", and `docs/index.md` sends a reader there to "see every
result". A findings document that neither the index nor any indexed document links to is therefore a
result nobody can find — the same failure as a number with no report behind it, one step further along.

The check is **one hop or fewer**, because that is how the programme writes: the index points at the
measurement, and the measurement points at the implementation record that produced it. It found one
orphan, `findings-bridge-gate-column.md` — the record of the wiring, reachable only from a pass report
that the index does not link — and the measurement document now points at it, which is where a reader
looking for "what shipped" would go anyway.

Two tests drive the checker itself: an orphan is detected on a synthetic tree, and a one-hop chain is
accepted, so neither failure nor leniency is accidental.

## 1.4.4 — 2026-10-29

The configuration table becomes a checked contract, and it holds: every one of the 91 variables the
package can read is in it.

### `tests/test_docs_env_vars.py`

`test_docs_tool_flags_exist.py` has held the CLI flags to "documented means parsed" for a while. The
environment is the larger surface — **91 variables** — and had no such check, so a new knob could ship
in `config.py` and be invisible to every reader of the configuration table, which is the only place they
are listed.

The new test extracts every variable the package can read (`_env("X")` and its typed siblings, plus any
explicit `LAYA_RAG_X`), every bare name the README's tables put in backticks, and asserts the first set
is a subset of the second. **It holds with nothing missing** — a positive result worth recording, since
91 hand-maintained rows is exactly where drift lives.

Its gate was perturbed before being trusted: a planted `_env_int("UNDECLARED_KNOB_NOT_IN_README")` fails
the test by name, and the file was restored. A third test asserts the extraction still finds more than
sixty variables, so a broken regex cannot make the other two pass vacuously. The reverse direction is
bounded rather than asserted — two variables (`AUDIT_LOG`, `EMBED_PASSAGE_PREFIX`) are read through paths
the extractor models imperfectly, and failing the suite for that would be a false alarm.

## 1.4.3 — 2026-10-29

The measurement-failures ledger gains its most transferable class, and the README's live test count is
brought current.

### Class 14: provenance that names something other than what ran

The repository's catalogue of its own measurement mistakes had thirteen classes. The October round added
a fourteenth, with four instances that arrived from four *different* mechanisms: a cache keyed by a
directory basename (three benchmarks' worth of wrong rows), a scalar baseline read from the wrong column
(AUC 0.500 for the right reason by accident), a resume fingerprint that ignored the variable under test
(two arms identical to four decimals), and a settings block naming a `--out`-derived path the run never
used.

The class, stated in the ledger: **every field that is filled in automatically is a claim about what ran,
and none of them is checked by default.** Three of the four were caught because two things that should
have differed came out identical — the cheapest bug signal this programme has — and the fourth by making
provenance visible on the report's face rather than reviewable in a diff.

### The live test count

`README.md` said 1,766 collected; it is **1,838**. That number is a live claim about the documented
command, unlike the historical counts in the changelog, which stay as written.

## 1.4.2 — 2026-10-29

A fifth axis on the evidence protocol, and the provenance defect it caught on its first run.

### "Which configuration produced these"

The report's own caveat — *quality and abstention come from different samples* — was prose a reader had
to notice. It is now **computed**: the protocol reads the settings block from each input and reports
`same_query_sample`, with a warning when it is false. Its first run on this repository's artefacts printed

```
answer_row: limit=150, sample_seed=17, rerank_model=None, index_dir=eval/…/engine-extractive-rank150
gate_rows:  limit=300, sample_seed=17, …                index_dir=eval/…/engine-default
the axes were measured on different samples; the joint score is a joint report, not a product
```

and in doing so caught a **provenance defect**: the answer row recorded the `index_dir` that `--out`
would have *derived*, while the run actually used the index named by `--index-dir` /
`LAYA_RAG_INDEX_DIR`. A row measured against an existing index could carry a settings block naming a
directory that never existed on disk. `tools/engine_extractive.py` now records the effective path.

That is the fourth defect of this class the programme has found — after a cache keyed by a model
basename, a scalar baseline read from the wrong column, and a resume fingerprint that ignored the
composer's objective — and it was found the same way: by making a number's provenance *visible* rather
than reviewable. Four tests pin the axis, including the case where the gate dataset carries its settings
at the top level rather than under `_settings`.

## 1.4.1 — 2026-10-29

R8's first half: a submission gate that decides whether an exported encoder is *believable*, and the
stock one passes it.

### `make submission-check`: five checks, a verdict, and no score

```sh
make submission-check                                              # the stock encoder
make submission-check SUBMISSION_MODEL=eval/distill-nfcorpus-model  # an exported student
make submission-check SUBMISSION_ARGS=--check                      # exit 1 on failure
```

| check | what a failure would mean |
| --- | --- |
| loads and returns embeddings | the package is broken, or the wrong dimension |
| outputs are unit vectors | a cosine index on un-normalised vectors retrieves garbage while looking plausible |
| deterministic across calls | a reported score is not reproducible |
| revision is pinned | MTEB skips results computed on a different revision; a local export must carry a content fingerprint |
| SearchProtocol adapter present | the route by which the *engine*, not just an encoder, can be entered |

`BAAI/bge-small-en-v1.5` reads **SUBMITTABLE** at 384 dimensions with pooling `cls` — taken from the
engine's verified family table, not guessed — and revision `5c38ec7c…`. That is a demonstration that the
gate passes what deserves to pass, not a claim that the encoder is competitive.

The gate deliberately **does not score**: R8's falsifier (*reproduce the ensemble's fused score within
+0.005 on the five measured datasets*) stays a scoring criterion, and the numbers keep coming from the
harnesses. Eight offline tests drive every check with a stub embedder — no download, no network —
including the two failures that matter in practice: non-unit vectors and non-determinism.

### Still owed

The artefact itself. Distillation is measured on one dataset (+0.0120 NFCorpus, ~29% of the ensemble's
gain), the export is verified faithful to its source exactly (max |Δ| 0), and the shipped thing is now
verified believable — but the five-dataset distillation is a training run this host cannot finish quickly.

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
