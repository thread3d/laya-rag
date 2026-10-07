# MS MARCO at 8.8M documents: the corpus was gated, not the benchmark

The audit carried MS MARCO as *"gated"* alongside BioASQ and the four licensing-blocked corpora.
It is not gated: the UKP mirror serves `msmarco.zip` (1.08 GB) and `BeIR/msmarco` answers **200
anonymously**, exactly like the datasets already measured. The claim was wrong, and this measures
the corpus so the statement is retired with a number rather than an argument.

It also shows why the coverage count needs a stated membership. **MS MARCO is not one of BEIR's
18 datasets.** It is the training collection the BEIR models are fine-tuned on; the benchmark's
members are the list `Resources for Brewing BEIR` (arXiv:2306.07471, Table 1) tabulates — which
includes CQADupStack and excludes MS MARCO and SPECTER. `make consistency-check` now counts over an
explicit `BEIR_18` set, so this run is reported as an **extra**, not as coverage.

## The measurement

```sh
make beir BEIR_ARGS="--datasets msmarco --retrievers bm25,bm25-engine \
  --out eval/beir/msmarco --lexical-checkpoint"
```

| dataset | documents | queries | retriever | nDCG@10 | Recall@10 | nDCG@100 | Recall@100 | MRR@10 | MAP@100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| msmarco | 8,841,823 | 43 | `bm25` (published settings control) | **0.4162** | 0.1386 | 0.4626 | 0.4403 | 0.8282 | 0.2912 |
| msmarco | 8,841,823 | 43 | `bm25-engine` (shipped lexical path) | **0.4321** | 0.1284 | 0.4898 | 0.4670 | 0.8004 | 0.3104 |

The scoring pass is 868 s for the control row over 43 judged queries — 20 s/query, the cost of a
lexical scan whose postings are spread over 8.8M documents. No published BEIR anchor is quoted:
MS MARCO is not a BEIR test dataset, so there is no like-for-like row to compare against.

## What this does (and does not) change

- **It retires a false block.** Only BioASQ (14.9M, gated) and Robust04 / TREC-NEWS / `specter` /
  `signal1m` (mirror 404 with 401 HuggingFace copies) remain access-blocked.
- **It exposes the membership error the count hid.** `BENCHMARKS.md` and the audit listed the
  uncovered datasets as Robust04 / TREC-NEWS / `specter` / `signal1m` / BioASQ / MS MARCO, i.e.
  nineteen candidates for an eighteen-dataset denominator, one of which (`CQADupStack`, 457,199
  documents, public on the mirror) was never named at all. The gate now reports
  `missing: bioasq, cqadupstack, robust04, signal1m, trec-news` beside the measured set, so the
  denominator cannot drift again without a failure.
- **Coverage is still 13 of 18.** MS MARCO does not move it; TREC-COVID's full collection did.
- **`CQADupStack` is the one canonical member that is both public and unrun.** It is not a
  drop-in `--datasets cqadupstack` job: the BEIR release ships it as twelve per-forum
  sub-collections with one qrels each, so it needs an adapter that aggregates them — engineering,
  not access.

## What this does not show

- **No dense row.** The 8.8M-document corpus was run lexical-only; embedding it is a separate
  multi-hour pass.
- **No significance test.** 43 judged queries is a small set; the two lexical rows are reported
  as point estimates, and the report's own bootstrap machinery is over so few queries that an
  interval would be a statement about the query sample, not the system.
- **Nothing about MS MARCO as a task.** The 43 queries are the BEIR release's judged subset, not
  the 6,980-query dev set, so this is not an MS MARCO leaderboard row.
