# The MTEB v2 `SearchProtocol` adapter

MTEB's standing objection to this project was that "MTEB scores a model, not an engine, so this
system cannot be entered as-is" (`ROADMAP.md` §10, `BENCHMARKS.md`). MTEB v2 defines a
`SearchProtocol` alongside the encoder and cross-encoder protocols, so that objection expires: an
object that exposes `index` and `search` is first-class representable, and what gets scored is the
retrieval *system* — chunking, fusion, the second stage and the gate — rather than one embedding
model. `tools/mteb_search.py` (extra `.[mteb]`, `make mteb-eval`) is that adapter.

## What it does

- `index(corpus, ...)` writes the corpus once, by position, to a temporary directory (the same
  convention `tools/engine_beir.py` uses) and builds the engine's index over it.
- `search(queries, ..., top_k=100)` runs the engine's own query path and returns
  `{query_id: {doc_id: score}}`, mapping returned passage paths back to corpus ids through the
  write order. It passes `relevance_threshold=0.0` so the benchmark scores the *ranking* rather
  than the gate's kept set, and widens `retrieve_k` to at least `top_k`.
- `build_model_meta()` constructs the `ModelMeta` lazily — 322M parameters, Apache-2.0, an
  8192-token window, `model_type=["hybrid"]`, `languages=["eng-Latn"]` (honest: the lexical
  instrumentation is English-only) — so the adapter and its tests import nothing from MTEB.
- `load_laya_rag_search` is the loader; it ignores kwargs the adapter does not understand, so a
  typo in a submission argument cannot fail the run.

## Verification

Against **mteb 2.22.5**:

- `isinstance(LayaRagSearch(...), mteb.models.models_protocols.SearchProtocol)` is **True**;
- `ModelMeta.load_model()` returns a `LayaRagSearch` with the metadata assigned;
- **14 tests** pin the corpus plumbing (positional files, path→id mapping, dedup to the best score,
  `top_ranked` filtering, unknown-path rejection, temp cleanup) and the conformance itself.

The plumbing is the part that matters: an off-by-one between write order and passage path would
corrupt every MTEB number without a metric flagging it, so it is tested directly.

## The evaluation: it was not hung, it was slow, and the silence hid that

The first reading of this was wrong and is corrected here. A full `mteb.evaluate(model, task)` on
SciFact was launched three ways — default, `num_proc=1`, and `num_proc=1` with
`OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` — and each was recorded as MTEB forking **~33 worker
processes** that then stalled: "no result, no error, no progress output, for 30+ minutes", read as a
failure in the benchmark harness's process handling.

Re-run with `num_proc=1` and `faulthandler.dump_traceback_later(90, repeat=True)`, the process
**is not deadlocked**: 20 consecutive dumps over 30 minutes, every one of them inside the engine's
own work — `mteb/_evaluators/retrieval_evaluator.py` → `tools/mteb_search.py:search` →
`laya_rag/engine.py:query` → `laya_rag/rerank.py:rerank` → `score_relevance` →
`model.predict_many` → Laya's `ModernBERT` forward. The stack is a single Python thread computing;
`_create_dataloaders` passes `num_workers=0` whenever `num_proc <= 1`, so no pool is involved at
all.

**Measured wall clock: 90+ minutes and no task.** The instrumented run was left going and stopped
after **60 stack dumps over ~90 minutes without reaching a result** — every dump in the same frame
chain above. That is the number the "accept the wall clock" option is weighed against, and it is
also the strongest evidence for the observability point: an operator watching that run for an hour
and a half had **no way to tell it apart from a hang**, because the adapter printed nothing.

**The cost is structural, not incidental.** MTEB's `SearchProtocol` asks for a `top_k=100` ranking,
and the adapter widens `retrieve_k` to at least `top_k`, so every query makes the engine run its
**322M-parameter decision model on 100 candidates** before returning the order. SciFact has 300
queries: at the measured rate that is a **multi-hour** task, and `findings-engine-parity.md` already
records the same shape on BEIR ("the engine run costs ~10 minutes per configuration because Laya's
decisions run on every candidate"). Nothing was broken; a working run simply looked identical to a
hung one.

### What was fixed

`tools/mteb_search.py` now prints a progress line every 25 queries with the rate and a remaining-time
estimate, plus an explicit `--limit-queries` **smoke** cap that says in the log that a capped pass is
a rate measurement and not a score. The 33-worker reading is not reproducible under `num_proc=1` and
the earlier note's "failure is in the harness's process handling" is withdrawn.

**Two more instruments, because the same defect existed one stage earlier:**

- **The index phase now reports progress.** `index_paths` takes a progress callback (the API's
  streaming route uses it); the adapter passed nothing, so embedding 5,183 documents was as silent
  as the search phase had been. It prints stage, documents, chunks and embedded counts, throttled to
  once a second, and a `done in Ns` line.
- **`--stack-dump-every N`** installs `faulthandler.dump_traceback_later`, the instrument that
  produced this page's corrected diagnosis. "Slow" and "hung" no longer have to look identical for
  want of a process inspector.

### Where the first-stage-only run stands: the phase is now named

`--first-stage-only` removes the decision model from the **search** phase, and the engine-level
measurement of that (37 s for 300 queries with `bge-small`, against ~10 minutes) is in
[`findings-second-stage.md`](findings-second-stage.md). An end-to-end `mteb.evaluate` on SciFact was
then run through the CLI **with the index-progress instrument**, and the phase is no longer a guess:

| phase | measured |
| --- | --- |
| index (read + chunk + embed) | **5,981 chunks in 178 s** |
| search | **fewer than 25 of ~300 queries scored in 13+ minutes** — ≥31 s/query, so the task is a **multi-hour** run |

So indexing is not the wall clock and the search phase is, with the stack dumps in model inference
(`torch/nn/modules/linear.py forward`, repeatedly).

### The silence had a cause, and it was mine

The five-query stack probe answered it in one run, and the answer is a defect in this adapter's
`--first-stage-only` flag rather than anything in MTEB or the engine. `main` printed the banner and
**set nothing on the adapter**: `load_model()` builds the adapter with no config, `_engine_for`
applies `first_stage_only` when it constructs the engine, and the CLI never set the attribute — so
every "first-stage-only" MTEB run still ran the full second stage. The dumps say so plainly, at
30-second intervals through the search phase:

```
engine.py:1385 in query → rerank.py:373 in rerank → rerank.py:213 in score_relevance
  → model.py:199 in predict_many
```

That is the 322M decision on every candidate, which is what this page's *original* reading said
("~10 minutes per configuration because Laya's decisions run on every candidate") before two rounds
of hypotheses about the query embedding and the driver. It also explains the other loose end: the
**282,984-token** warning is the rerank prompt — a query concatenated with its candidate passages —
which is why it appeared with `--first-stage-only` "on", why it is absent on a three-document corpus
where few candidates survive, and why it is not a document at all.

`main` now sets `model.first_stage_only = True` (with a regression test naming the bug), and the
engine-level measurement that motivated the mode was never affected — `engine_beir.py` sets
`LAYA_RAG_SECOND_STAGE=0` through the environment, which works.

**Confirmed, and the corrected cost model.** With the flag applied the same five-query probe reads:

| phase | before the fix | after |
| --- | --- | --- |
| index (5,981 chunks) | 178–184 s | **183 s** |
| search | ≥31 s/query (the second stage on 100 candidates) | **1.05 s/query** |
| `rerank.py` frames in the stack dumps | every dump | **none** |
| 282,984-token warnings | one per run | **none** |

So an MTEB `SciFact` task is **~8.5 minutes**, not the multi-hour figure this page carried for four
rounds: a 183-second index plus ~5.3 minutes for 300 queries at the measured rate (plus MTEB's own
scoring, which is arithmetic over 5,183 × 300). The 1.05 s/query agrees with the engine-level
measurement (1.3 s/query with the same encoder), which is the cross-check that says the two
configurations are now the same configuration. **Nothing about the harness was ever the problem**:
it forks nothing under `num_proc=1`, it does nothing between `index` and `search`, and it was
faithfully driving an adapter whose flag did not work.

**The two refuted hypotheses, kept because they were part of the cost.**

*Attempt one — the query embedding.* With `--first-stage-only` the only per-query model work left is
the query embedding, so that was the hypothesis. The engine level refutes it: **20 queries in 26 s
with the same 322M adapted encoder**, 1.3 s/query, first stage only.

*Attempt two — MTEB drives `search` once per query.* `RetrievalEvaluator` carries a **`qid`**, which
made this look likely, and it would explain everything: a one-query call never reaches the 25th and
prints nothing. **Reading the driver refutes it.** `mteb/abstasks/retrieval.py:_evaluate_subset`
builds **one** `RetrievalEvaluator` per subset, passing the whole `queries` collection, and calls it
**once** — so `search` is called once with every query, `qid` is not used to split the call, and the
progress line at 25 *should* have fired within a minute of indexing finishing. The adapter still
gained the short-call report, because the instrument should not depend on that reading being right —
but it is not the explanation.

**What that leaves.** `search` is called once over ~300 queries, fewer than 25 of which were scored in
13+ minutes, and the engine answers the same corpus's queries at 1.3 s each. The remaining
differences between the two runs are small and enumerable: the adapter asks for `retrieve_k = 100`
(MTEB's `top_k` is 100, and it widens the pool to match) where the engine-level timing run used the
default, and the adapter's own loop does the path→id mapping per passage. Neither looks like a 20×
factor, which is the honest state of this: **the two measurements disagree and nothing yet explains
by how much.** The next run is cheap and does not need a guess — `--limit-queries 5
--stack-dump-every 30`, against the 178-second index, dumps the stack of every thread at 30-second
intervals while five queries are scored, which is enough to name the frame.

**The obvious explanation is measured false.** With `--first-stage-only` the only model work left per
query is the query embedding, so that was the hypothesis — and the engine level refutes it: **20
queries in 26 s with the same 322M adapted encoder** (1.3 s/query), first stage only. So ~31 s/query
is not what a query costs here, and the MTEB search phase's silence is **not** explained by the query
embedding. The next step is what `mteb.evaluate` does *between* the index call and the first `search`
call — the dumps exist to answer exactly that, and the earlier reading of them (`search` on the
stack) was taken from a run whose phase was never actually established.

**No MTEB score exists and nothing is uploaded.** The run was stopped rather than left to occupy a
shared host for hours.

### An unresolved observation, narrowed

The run logs, once, at the start of indexing:

```
Token indices sequence length is longer than the specified maximum sequence length for this model
(282984 > 8192). Running this sequence through the model will result in indexing errors
```

What has been **ruled out**: it is not a long document (MTEB's SciFact corpus has 5,183 rows whose
longest `title + text` is **10,126 characters** and whose longest query is 204), and it is not the
reranker — it appears in `--first-stage-only` mode, where no rerank runs. It is absent on a
three-document corpus with the same checkpoint, so it scales with the corpus. With the index
instrument it also has a **position**: line 194 of the log, immediately after the `index: done in
178s` line and before any `search:` progress line, i.e. on the indexing→search boundary on the first
query. Given that the search phase is ≥31 s/query with only the query embedding left in the path,
the two observations are most likely **the same fact**, and that is the hypothesis to test next. The most likely reading
is that Laya's tokenizer is handed a **batch** and reports the aggregate length, which would make it
noise rather than mis-indexing — but "most likely" is not measured, and the wording ("indexing
errors") is alarming enough that it should not be left as an assumption. Recorded here with the
evidence so the next attempt starts from it.

### What the score actually needs, and why it is not done here

The engine has **no first-stage-only mode**: `RerankResult()` is the empty case, and `order="recall"`
still *computes* the relevance probabilities before keeping the retrieval order (it changes the
ordering, not the work). So MTEB cannot score the retrieval system without the decision model in the
path. Three ways forward, in order of principle:

1. **A config flag that returns the first-stage order with no second stage** (`RERANK=off` or
   equivalent), and the adapter documenting that a submission used it. This is what "MTEB scores
   the retrieval system rather than one embedding model" should mean, and it is a change to
   `EngineConfig`/`RagEngine.query`, not to the adapter.
2. **Accept the wall clock** (~2–3 h per task at the measured rate) and run it as a long job.
3. **Lower `retrieve_k`**, which changes what the benchmark measures — MTEB asks for a 100-deep
   ranking, and a system that ranks 100 must score 100 — so this is the option to take last.

No MTEB score exists, and there is still no upload. What exists is the adapter, its conformance
tests, the diagnosis above, and a run that can be watched while it works.

## The first score under an external protocol

With the flag fixed, a full `mteb.evaluate(model, task)` on SciFact completed locally. **MTEB's own
`TaskResult` reports `main_score=0.37`**, and its cached result file carries the breakdown:

| metric | value |
| --- | ---: |
| **nDCG@10** | **0.36987** |
| nDCG@1 / @100 / @1000 | 0.2 / 0.39378 / 0.42096 |
| MAP@10 / @100 | 0.31667 / 0.32255 |
| evaluation time (MTEB's own figure) | 193.6 s |

**This is the first number in this repository that nobody in this project computed**, and it is low.
Read against the corpora this repository normally quotes, SciFact's published BM25 is 0.665 and a
`bge-small` bi-encoder reads 0.72 — so **0.37 is not a placement, it is a starting point**, and the
reasons are stated rather than explained away:

- **`--first-stage-only`.** This is the retrieval system's fusion order with the decision model
  removed. It is the right configuration for a *retrieval* benchmark and it is not the engine as
  deployed.
- **The encoder is `laya-rag/engine-multilingual`**, the decision model's multilingual encoder,
  which has never been retrieval-trained. It is scored here because that is the model the adapter
  registers — not because it is the best encoder this project has (`models/laya-retrieval` is, and
  it is a different submission).
- **MTEB's SciFact corpus is whole documents with titles**, under the engine's own chunking, so this
  is not comparable to the repository's own SciFact rows (`bm25-engine` 0.6762, `hybrid-biencoder`
  0.7413) taken over BEIR's document convention.

**The next step was one flag, and it changes the number by 0.32.** The same task, same
configuration, with `--embed-model models/laya-retrieval` and filed under its own name
(`laya-rag/engine-retrieval`, which `--embed-model` now *requires* so a run cannot be filed under a
description that is false):

| encoder | MTEB `main_score` | nDCG@10 | nDCG@100 | MAP@10 | MRR@10 | recall@10 | search rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `engine-multilingual` (decision checkpoint) | 0.37 | 0.36987 | 0.39378 | 0.31667 | — | — | 1.05 s/query |
| **`engine-retrieval` (retrieval-trained)** | **0.69** | **0.69482** | 0.72163 | 0.64817 | 0.65887 | 0.82211 | **0.21 s/query** |

**0.69482 nDCG@10 on MTEB's SciFact** — against the published BM25 the repository's own rows are
anchored to (**0.665**) and a `bge-small` bi-encoder on the same convention (**0.72**). So the engine,
scored as a *system* under an external protocol with its own retrieval-trained encoder and no
decision model in the path, sits **above published BM25 and just below `bge-small`** — which is a
much more interesting starting point than 0.37, and it is the same engine the repository's BEIR rows
describe. The rate is also 5× better (0.21 s/query), because that encoder is much smaller than the
322M decision checkpoint.

**Nothing is uploaded.** These are local MTEB runs; a submission is a separate act, and it should name
the encoder explicitly, which the CLI now enforces.

## What this does not show

- **No MTEB score exists** for the engine yet. The adapter is representable and verified and the
  evaluation is reachable; a number is not claimed, and a `--limit-queries` pass is a subset that
  must never be quoted as one.
- **`top_ranked` filtering is best-effort.** For reranking tasks the adapter runs the query and
  filters to the supplied candidate ids; a candidate outside the engine's own pool is omitted
  rather than pair-scored, because the engine has no pair-scoring path. Retrieval tasks, which pass
  `top_ranked=None`, are unaffected.
- **Each evaluation re-indexes.** The corpus is written to a temporary directory per run, so a
  large task pays the index build every time; a persistent backend for MTEB would be the next
  optimization if submissions become routine.
