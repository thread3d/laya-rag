# Climate-FEVER: the eleventh BEIR dataset, and the first of the "blocked" large corpora

Climate-FEVER was listed in the audit as **blocked/unstaged** — "5.4M documents; downloaded and
extracted, not run" — and in the roadmap as compute-bound at ~6 h per lexical row. It is now
measured. The run resumed from a 941-query checkpoint and finished the remaining ~594 queries in
about **34 minutes**, BM25 build included, because the stemmer is memoised and scoring at ~1.4 q/s
touches only a query's postings.

## The row

| dataset | documents | queries | retriever | nDCG@10 | Recall@10 | MRR@10 | MAP@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| climate-fever | 5,416,593 | 1,535 | bm25-engine | **0.1610** | 0.2093 | 0.2139 | 0.1198 |

This is the **third 5.4M-document corpus** and the **eleventh** BEIR dataset with a report, so
`make consistency-check` now computes **11 of 18** and the three documents that state the count were
updated together (the gate failed first and named them, which is the designed behaviour).

## What this does not show

- **Lexical-only.** The dense arm would need its own multi-million-document embedding pass; the row
  is `bm25-engine` under the engine's settings (stemming on, query stopwords dropped).
- **No published anchor is configured** for Climate-FEVER, so this row records our number and not a
  gap. The repository's per-corpus anchor table covers SciFact, NFCorpus, ArguAna, SciDocs, FiQA,
  Quora, Touché, NQ, DBPedia and HotpotQA; adding Climate-FEVER's would need a published BM25 figure
  from a named source, and inventing one is exactly what this repository does not do.
- **One retriever, not the fusion comparison.** The channel-reversal and lexical-control analyses
  need the dense arm; this row is a coverage addition, not a new instrument comparison.
- **The ~6 h estimate was pessimistic.** The measured time was ~34 minutes after the checkpoint
  resume, so the "compute-bound" framing for the remaining corpus rows is now known to be
  pessimistic for a lexically-memoised run.

## Reproduction

```sh
.venv/bin/python tools/beir.py --datasets climate-fever --retrievers bm25-engine \
    --lexical-checkpoint --out eval/beir/climate-fever
```

Resumable per query through `eval/beir/climate-fever.lexical.jsonl`; the heavy-job lock is taken for
the duration, so no second corpus run may start beside it.
