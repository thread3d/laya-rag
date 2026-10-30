# R4's placement half — the MultiHop-RAG retrieval row, and a refusal that came first

`docs/best-in-the-world-2026-10.md` R4 asked for two things: the evidence-layer protocol (landed, five
axes, reproducible) and **generator-free leaderboard placements**. This is the second half, as a
command — plus the number it will show once the run that fills it exists.

## The command

```sh
python tools/engine_extractive.py --zero-thresholds \
  --index-dir eval/benchmarks/multihop-rag/index/engine-default \
  --out eval/benchmarks/multihop-rag/engine-extractive-full
make multihop-row MULTIHOP_ROW_REPORT=eval/benchmarks/multihop-rag/engine-extractive-full.json
```

`tools/engine_extractive.py` already reports exactly the four columns the benchmark's table carries
(`Hits@10`, `Hits@4`, `MAP@10`, `MRR@10`). `tools/multihop_row.py` places them beside the published
anchors, and refuses to place a subset.

## The refusal is the point

A report over 150 of the benchmark's 2,255 answerable queries looks exactly like a report over all of
them. The tool checks the query count and, unless `--allow-subset` is given for internal reading, prints:

```
**not placeable** — 150 queries, not the 2255 answerable records the table scores — this is a subset,
and a subset presented as the member is the failure class 11 records
```

with the command that would make it placeable. That is the eleventh class in
[`findings-measurement-failures.md`](findings-measurement-failures.md) — *a subset read as the member* —
turned into a gate rather than a footnote.

## What the subset shows anyway, marked as a subset

Because the numbers are worth seeing while the full run is pending:

| system | Hits@10 | Hits@4 | MAP@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| voyage-02 + rerank (published) | 0.747 | 0.663 | 0.480 | 0.586 |
| bge-large + rerank (published) | 0.718 | 0.636 | 0.476 | 0.563 |
| bge-large, no rerank (published) | 0.672 | 0.522 | 0.342 | 0.430 |
| **ours: engine + `bge-reranker-v2-m3`, 150-query subset** | **0.940** | **0.847** | 0.371 | **0.688** |

On three of four columns the engine exceeds the best published row and on `MAP@10` it trails by 0.11.
That is this repository's long-standing shape — **the gold is found and the order is where the score
goes** — now with the 2026 reranker having closed most of the `MRR@10` gap (0.688 against 0.586) while
`MAP@10` stays behind.

**None of it is placeable**, and the tool says so before printing it. The full-set run is ~3 hours at the
measured 736 s per 150 queries; it is the one remaining measurement R4 needs, and it is a scheduling
problem rather than a design one.

## A mid-flight check, because a two-hour run should not be trusted only at the end

The full-set row writes a resumable shard, so its first **135** rows can be summarised while it
runs. That is worth doing: a measurement that turns out to have used the wrong index, or to be emitting
empty answers, should be caught at 6% rather than at 100%.

| metric | 150-query subset (recorded) | first 135 of the full set |
| --- | ---: | ---: |
| answer nugget recall | 0.2060 | **0.1981** |
| citation-clean, all queries | 1.0000 | **1.0000** |
| abstention rate | 0.0000 | **0.0000** |
| Hits@10 | 0.9067 | 0.9333 |
| Hits@4 | 0.7400 | 0.8000 |
| MAP@10 | 0.3128 | 0.3278 |
| MRR@10 | 0.6152 | 0.6436 |

The run is healthy on the properties that would betray a misconfiguration — no empty answers, every
answer citation-clean at 5.00 sentences, no abstentions on an answerable-only set — and its quality tracks
the subset it was previewed on, a little better on every retrieval column.

**This is not a result.** It is 6% of the set, read before the run finished, on the arm whose number was
never in question. The registered prediction in
[`findings-extractive-coverage.md`](findings-extractive-coverage.md) is about the **delta between the two
composer objectives** on the full set, and this check says nothing about it: only one arm is running.

## The full-set row, measured

The 2,255-query row finished. Over the **full answerable set**, scored over chunks, at the shipped defaults:

| system | Hits@10 | Hits@4 | MAP@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| voyage-02 + rerank (published) | 0.747 | 0.663 | 0.480 | 0.586 |
| bge-large + rerank (published) | 0.718 | 0.636 | 0.476 | 0.563 |
| bge-large, no rerank (published) | 0.672 | 0.522 | 0.342 | 0.430 |
| **the engine, shipped defaults** | **0.9011** | **0.7561** | 0.3120 | **0.6166** |

Citation validity **1.0000** over all 2,255 answers, answer nugget recall **0.1835**, abstention **0**.

**The full set is not the subset.** The same arm on 150 queries read Hits@10 0.9067, Hits@4 0.7400, MAP@10 0.3128,
MRR@10 0.6152 and nugget recall **0.2060** — so the subset was within 0.007 on the retrieval columns but **0.0225
optimistic on nugget recall**, which is the largest single divergence and the one that matters for R2's delta. A
subset read as the member was already refused by the tool; this is the measurement of how much it would have
mattered.

**The row is still chunk-level and therefore still indicative**, for the reason in the next section: it started
before the document-level fields existed. `make multihop-row` refuses to present it as like-for-like, and prints
the published anchors beside it with that caveat rather than sorting the two groups into one column.

### And the report was checked against its own rows before it was quoted

The full-set report's summary recomputes **exactly** from its 2,255 per-query rows — every headline figure, to four
decimals — and records the same `index_identity` (`947d10ba…`) as the 150-query subset, so the two are on the same
index and the comparison between them is like-for-like.

That check is now part of the placement path rather than a thing done once by hand: `make multihop-row` re-reads the
report's own `.jsonl` shard and refuses the row if the summary's query count or headline metrics disagree with the
rows they claim to summarise. A rolled-up summary and a per-query shard can diverge — a resumed run writing them at
different times, or a summary computed over a different query set — and that is **the subset-as-member failure
wearing a different hat**: the summary is what gets quoted, so the question is whether it describes *these* rows.

## A fairness problem found before the placement was made

The published table this row is meant to sit in scores **ten documents** per query. The engine returns
**chunks**, and this corpus chunks **609 documents into 4,592** — 7.5 chunks per document. So a chunk-level
row is answering a different question in the same metric's clothes: ten slots hold a fraction of one
document's text where the published rows hold ten whole ones.

The metric itself does not care — `official_metrics` matches gold facts by normalised substring against
whatever strings it is handed — which is exactly why the problem is invisible in the number. The direction of
the bias was not obvious either: chunks make each slot *smaller* (harsher for "does a gold fact appear in the
top ten"), while a chunk that contains the fact can outrank the document that contains it, because the
document's other chunks dilute it (kinder). A comparison whose bias direction cannot be signed is not a
comparison.

So the row now records **both**, and the pair is named in the artefact:

| field | question it answers |
| --- | --- |
| `Hits@10`, `MAP@10`, … | the system's own question: does a gold fact appear in the top ten **chunks** |
| `doc_Hits@10`, `doc_MAP@10`, … | the published table's question: does it appear in the top ten **documents** |
| `doc_passages` | how many distinct documents those ten chunks actually covered |

The collapse reuses the harness's own `ranking_from_passages`, which maps a returned path back to a corpus id
and de-duplicates in first-appearance order, so the ranking is preserved rather than re-sorted. **The row that
gets placed is the `doc_` one;** the chunk row stays because it is the system's own measurement and because
quoting a number without its document-level counterpart is what got this repository into the situation the
placement tool was written to prevent.

### The direction, measured: the chunk-level row is *harsher*

An hour ago this document said the direction of the chunk-versus-document bias **could not be signed in
advance**. That was honest and it is now out of date: an eight-query probe on the engine's own output signed it.

| level | Hits@10 | MAP@10 |
| --- | ---: | ---: |
| chunk text (the rows the README quotes) | 0.7500 | **0.2816** |
| distinct documents | **0.8750** | **0.4896** |

One query improved at document level, **none worsened**, and the mean number of distinct documents behind five
returned chunks was **2.38**. That last number is the explanation and it is worth sitting with: the engine's top
five chunks represent about two and a half documents, so a *chunk-level* Hits@10 is effectively asking "does the
gold fact appear among the top two or three documents", while every published row gets **ten**. The chunk-level
metric is a structurally harder question wearing the same name.

Two consequences, and they point in opposite directions for how the project should behave:

1. **The published comparison is not inflated by this defect — it is understated.** The README's caveat was
   written on the assumption that the error might be embarrassing; the probe says the error is conservative.
2. **That is still an error and it still cannot be published as a result.** "We may be better than we said" is
   as unmeasured as "we may be worse than we said" until the full-set document-level numbers exist.

**This is a probe, not a result**: n=8, single run, read against gold facts the same way the full row is. It
signs the direction and it does not estimate the magnitude — for that, the queued coverage arm's `doc_*` fields
over 2,255 queries.

### A probe that contradicted itself found a silent zero in shipped code

The direction probe above used the engine's own scoring helper. A follow-up probe — written to test how the
pool *width* affects the document-level row — inlined the metric call instead, and passed `ranking`, which is a
list of **document ids**, to `official_metrics`, which matches gold facts by normalised substring. A URL never
contains a gold sentence, so it read:

| top_k | chunk Hits@10 | doc Hits@10 | distinct documents |
| ---: | ---: | ---: | ---: |
| 5 | 0.750 | **0.000** | 2.38 |
| 10 | 0.875 | **0.000** | 3.50 |
| 20 | 0.875 | **0.000** | 5.00 |

Two probes, same engine, same queries, `doc_Hits@10` of 0.875 and 0.000 — which is the whole value of writing
both. The second probe was wrong, and **the mistake it made was in `tools/engine_multihop.py` too**: the
document-level line I added an hour earlier passed `ranking` rather than the documents' text, so every future
run would have recorded `doc_Hits@10 = 0.0`, `doc_MAP@10 = 0.0` and nothing would have failed. A column of
zeros that reads like a catastrophic retrieval result.

It is fixed and pinned by a test, and the fix is one line: build the text map the corpus writer already builds,
and hand the metric `doc_text.get(doc_id, "")`.

**What the probe did establish, despite its bug.** The chunk-level columns are computed from text and are
unaffected: widening the pool from five to ten chunks moves chunk Hits@10 **0.750 → 0.875** and raises the mean
distinct documents from **2.38 to 3.50**; at twenty it reaches **5.00**. So the engine's five-chunk answer
really does cover about two and a half documents, and a document-quota setting is the obvious lever if the
document-level row is to be compared at ten. That is a measurement to make, not a change to make on a probe.

### What the document-level row will and will not be comparable to

A caution to write **before** the number lands, because it is the kind of thing a reader will otherwise assume.

The queued coverage arm runs at the shipped `top_k`, which is **five chunks** — about **2.4 distinct
documents**. Its `doc_*` fields will therefore compare "gold fact among the top 2.4 documents" against published
rows scored over **ten**. That is *more* comparable than the chunk row and it is still not the fair comparison.

Making it fair needs either a wider pool or a document quota, and **which one is available depends on the arm**,
which is worth being precise about because the two are not interchangeable:

- **Without a cross-encoder** — the arm running now — the quota is unreachable: `engine.py` applies
  `rerank_coverage_min_docs` inside `if cross_scores is not None`, so the only lever is a wider pool. The probe
  above measured what that buys: five chunks → 2.38 documents, ten → 3.50, twenty → 5.00.
- **With a cross-encoder** the quota applies, and a second thing happens automatically:
  `resolved_rerank_coverage_lambda` returns **0.7** whenever a reranker is configured, so every CE row this
  repository has measured is *coverage-reordered as well as cross-encoded*. That is worth stating beside those
  rows, because "a cross-encoder rerank" is not what they are.

The quota is **off by default** for a measured reason recorded in `config.py`: on a full-protocol (2,255-query) run it moved MAP@10 **0.3165 → 0.3192** (+0.0027, p=0.037)
while dropping Hits@4 **0.7694 → 0.7588** (−0.0106). A real trade, which is why it is not a default.

**That full-protocol measurement is also an artefact-loss case worth naming**: its reports are not on disk, so a
number from a complete 2,255-query run now lives only as a comment in a configuration file. It is not wrong — the
code's own comment records the interval and the trade — but it cannot be re-derived, which is the standard every
other number in this repository is held to.

So the placed row will be **document-level at ~2.4 documents**, and the finding will say so rather than calling it
ten. A quota or wider-pool row is a measurement to run deliberately, not a footnote to attach.

### The document-level row comes from the coverage arm, and that is valid

The rank arm running now writes the old schema — it started before the field existed — so its document-level
counterpart cannot be derived after the fact (the row stores answers, not document ids). **The queued coverage
arm runs the current code and will record it.**

Using the coverage arm's retrieval numbers for the placed row is legitimate rather than a compromise, and the
reason is worth stating: **retrieval is deterministic and identical between the two composer arms**. The
objective changes which sentences are *selected*, not which passages are *retrieved* — `multihop_official`'s
own docstring says re-running reproduces the same ranking exactly. So the coverage arm's `doc_*` metrics are
the rank arm's document-level retrieval, measured once.

`make multihop-row` now prefers `doc_*` when present and, when it is absent, prints that the row is **indicative
rather than placeable** instead of putting a chunk-level number in the same four columns.

**What this does not fix.** Chunking still differs — the published rows re-chunk at a 200-character overlap
and the engine derives chunk size from the corpus — so `doc_Hits@10` compares *documents retrieved by
different chunkers*, which is the closest this data allows and is stated rather than glossed.

## Why the anchors are quoted rather than merged into one sorted column

Three caveats travel with the row, and the tool prints all three:

- **Chunking differs.** The published anchors re-chunk with the official harness at a 200-character
  overlap; this engine derives chunk size from the corpus (2,000 characters here). Sorting the two groups
  into one column would imply the protocols match.
- **The paper's table carries no lexical baseline.** Every published row is a dense model, so a `Hits@10`
  advantage over `bge-large` is partly lexical retrieval the table never measured.
- **The engine's row adds two columns it owns** — citation validity and answer nugget recall — which is
  the entire argument for a retrieval table that admits a non-generative system at all.

## What is owed

- The full-set run. ~3 h, one heavy job, and the numbers above are its expectation at 15× the queries
  (they will move — a subset is not the member, which is why the tool refuses to quote one).
- A `--check` in CI once the full report exists, so the row cannot be edited into a claim.
