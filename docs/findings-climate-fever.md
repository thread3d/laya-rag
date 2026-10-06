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
- **The anchor exists, and the gap is the largest measured.** `tools/compare_more.py` already carried
  a published BM25 anchor of **0.213** for Climate-FEVER, so the row is **0.1610 against 0.213 — a
  −0.052 gap**, worse than Quora (−0.020) and NQ (−0.017). It joins the lexical control as its
  eleventh corpus and its worst outlier, which moved that sentence from "8 of 10" to "8 of 11" and its
  heading to *eleven corpora including three multi-million-document ones*. An earlier draft of this
  finding said no anchor was configured; regenerating `tools/lexical_control.py` after the run showed
  otherwise — the anchor was present, only the derived report was stale, and `make claim-check` now
  pins the corrected count.
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
