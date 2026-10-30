# Reproducing a result

Every number this project quotes came from a command in this file's table, and every one of them can be
re-derived from a fresh clone. This page exists because "our numbers are reproducible" is otherwise a
claim about the author's habits rather than about the repository.

The rule that makes it work: **artefacts live under `eval/`, which is gitignored**. Nothing is committed
except the reports' *interpretation*, so a number in prose can always be traced to a file that is
regenerated rather than trusted — and the gates that check the prose skip when the artefacts are absent,
as they are on a fresh clone.

Two properties carry almost all of the weight, and both are enforced rather than assumed:

- **Every long job checkpoints.** Re-issue the same command after an interruption and it resumes; the
  checkpoint is keyed to the run's settings *and to the index it was measured against*.
- **One heavy job at a time.** Twenty-six long-running tools take an exclusive `flock` on
  `eval/.heavy-job.lock`. A second one exits `2` with the holder's name rather than piling on — and
  `make heavy-wait HEAVY_POLL=5` queues behind whatever is running, by asking the lock rather than by
  grepping for a process name.

## Before anything: what a fresh clone has, and what it must fetch

```sh
make setup                       # a venv and the engine's dependencies
make doctor                      # does the checkpoint load? (needs LAYA_RAG_MODELS)
make multihop-fetch              # 12 MB, verified: 2,556 records, 301 null_query, 609 documents
```

`make multihop-fetch` asserts the dataset's shape before any number is derived from it. That check exists
because the benchmark's two files previously arrived by hand, which made the whole evidence layer
unreproducible — R4's own falsifier.

## The eight measurements

Runtimes are on this host (24 threads, CPU only) and are **recorded rather than estimated**, because a
cost claim that is not measured is the first thing a reader should distrust.

| # | answers | commands | writes | time | findings |
| --- | --- | --- | --- | --- | --- |
| **R0** | where the CPU goes | `make index` then `make cost-report` | `eval/cost-report.{json,md}` | ~10 min | [`findings-cost-envelope.md`](findings-cost-envelope.md) |
| **R1** | does a 2026 encoder move retrieval | `make beir DATASETS=scifact,nfcorpus` with and without `--biencoder-model` | `eval/beir/` | ~1 h per pair | [`findings-component-refresh.md`](findings-component-refresh.md) |
| **R2** | does the composer's objective pay | `make multihop-fetch`, `make index MULTIHOP=1`, then `make engine-extractive ENGINE_EXTRACTIVE_ARGS="--limit 150 --zero-thresholds"` twice, once with `LAYA_RAG_EXTRACT_COVERAGE=1` | `engine-extractive-*.json` | ~12 min per arm | [`findings-extractive-coverage.md`](findings-extractive-coverage.md) |
| **R2b** | does the answerability verdict reach the *answer* | the same row with `ENGINE_EXTRACTIVE_ARGS="... --honour-answerability"` | a second report; the flag is in the fingerprint and both settings blocks | ~12 min per arm, plus the shipped arm for comparison | [`findings-answerability-gate.md`](findings-answerability-gate.md) |
| **R3** | is abstention portable | `make gate-dataset`, `make gate-train`, `make gate-transfer` | `eval/gate/` | ~46 min for the rows, 2 s to fit | [`findings-answerability-gate.md`](findings-answerability-gate.md) |
| **R4** | the evidence layer, end to end | `make evidence-pipeline` | `eval/evidence-protocol.{json,md}` | **~68 min** from a fresh clone | [`findings-evidence-protocol.md`](findings-evidence-protocol.md) |
| **R4b** | a row for the benchmark's table | the full extractive row, then `make multihop-row` | `eval/benchmarks/multihop-rag/engine-extractive-full.*` | ~2 h for the full 2,255 queries | [`findings-multihop-placement.md`](findings-multihop-placement.md) |
| **R5** | does the reranker generation move | the extractive row with `--rerank-model BAAI/bge-reranker-v2-m3`, then the ArguAna arms | `engine-extractive-ce-*.json`, `eval/r1/arguana*` | 434 s → 995 s per 150 queries; 100 min for the ArguAna pair | [`findings-reranker-generation.md`](findings-reranker-generation.md) |
| **R5b** | does a *listwise* ranker change the ArguAna collapse | `make listwise-rerank` — `jina-reranker-v3`, **CC-BY-NC-4.0**, so measurable and not shippable | `eval/r1/arguana-listwise.json` | ~41 s per query projected from a measured 8 s at depth 10; **~4.5 h for 400** — the first estimate said 1.5–2 h and assumed thread scaling that did not happen | [`findings-reranker-generation.md`](findings-reranker-generation.md) |
| **R6** | what the decision costs | `make cost-report` with `LAYA_RAG_SECOND_STAGE_TOP_M` at 20 / 10 / 5 | `eval/cost-report.*` | ~4 min per arm | [`findings-decision-cascade.md`](findings-decision-cascade.md) |
| **R7** | one multilingual collection | see the finding — it is **priced out**, not run | — | 28 h projected per collection | [`findings-multilingual-scope.md`](findings-multilingual-scope.md) |
| **R8** | is an export submittable | `make submission-check SUBMISSION_ARGS=--check` | `eval/submission-check.{json,md}` | seconds | [`findings-submission-gate.md`](findings-submission-gate.md) |

## The five things that make a re-run comparable to the recorded number

1. **Pin the index.** Rows record `index_identity`, a hash over the corpus, the chunker and the ingest
   settings. If it differs, the number is not comparable — and it changes when *any* ingest setting does,
   which is why it is a hash rather than a directory name.
2. **Pin the query sample.** Subset runs take `--sample-seed`; the row records it. A row without a seed is
   not a sample, it is an accident.
3. **Check the unit before the metric.** A row records both the chunk-level and the `doc_` document-level
   figures, because the published tables score *documents* and this engine returns *chunks* — and one corpus's
   five chunks cover about 2.4 documents. `make multihop-row` prefers the document-level row and says
   *indicative* when the report has none.
4. **Read the configuration block.** `make multihop-row` and `make evidence-protocol` both print the
   settings that produced each axis. The rule they enforce: *when a configuration variable changes what a
   number means, it has to appear in the artefact — or the artefact is a filename.* Five defects of that
   class are catalogued in [`findings-measurement-failures.md`](findings-measurement-failures.md).
5. **Run the gates.** `make claim-check` resolves the registered figures against the reports that produced
   them; `make consistency-check` catches two documents disagreeing about the same quantity. Both skip
   cleanly when the gitignored artefacts are absent — a skip is not a pass, and the CI job says so.

## What reproducing does *not* give you

The same numbers. Every measurement here is single-host, single-run; the intervals quoted are bootstrapped
over queries, not over machines. A different CPU, a different thread count, or a different `torch` build
moves the cost figures first and the quality figures hardly at all — which is why the cost rows report CPU
seconds, not wall clock, and why the quality rows report paired deltas with intervals rather than levels.

And a number reproduced on a different corpus is a **different measurement**, not a confirmation. That is
the limit the answerability gate's own finding states first: it transfers across queries, not across
corpora, and `make gate-transfer` exists to measure rather than assume the difference.
