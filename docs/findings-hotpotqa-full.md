# HotpotQA: the full query set, and the seeded estimate retired

HotpotQA's 600-query subset estimate was the row the dilution bug was found and fixed on; the full
7,405-query run was then left owed because the rate had degraded to ~0.04 q/s under contention. It
finished here, running alone, resuming from a 3,061-query checkpoint.

## The row

| dataset | documents | queries | retriever | nDCG@10 | Recall@10 | nDCG@100 | Recall@100 | MRR@10 | MAP@100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hotpotqa | 5,233,329 | **7,405** | bm25-engine | **0.6352** | 0.6602 | 0.6712 | 0.7995 | **0.8065** | 0.5517 |

Against the published BM25 anchor of **0.603** that is **+0.032** — one of only two corpora where the
lexical arm beats its anchor (the other is ArguAna, +0.020). The **MRR@10 of 0.8065** is the highest
in the table and is a property of the corpus: HotpotQA questions have a single gold paragraph, so
finding it first is most of the score.

## The subset estimate is superseded

The earlier 600-query sample read **0.6365 (+0.034)**; the full set reads **0.6352 (+0.032)**. The
estimate was good to 0.0013, which is what a correctly restricted 600-query sample should be — but the
machinery did not retire it. `tools/compare_more.py` now treats `SUBSET_ESTIMATES` as a **fallback**:
when a full report exists (`eval/beir/hotpotqa.json`), the corpus is pulled in with the other
full-scale rows and the seeded estimate is skipped, so the chart and the control can never show a
sample in place of a full run again.

**Coverage does not change.** HotpotQA was already counted among the 12 of 18 — the subset report
established it — so this is a row *upgraded from a sample to the full set*, not a thirteenth dataset.
`make consistency-check` still reads 12 of 18, and the lexical-control counts are unchanged (12
corpora, 8 within ±0.02, 5 within ±0.008).

## What this does not show

- **Lexical-only.** No dense arm and no fusion; the dense arm would need its own
  multi-million-document embedding pass.
- **The anchor is a published number, not re-run here** — the same like-for-like caveats as
  `docs/findings-beir-breadth.md`.
- **Nothing about the other corpora.** HotpotQA's subset was the one case where a full run existed to
  compare against; whether the other large corpora would move as little is untested, because none of
  them has both a subset and a full run.

## Reproduction

```sh
.venv/bin/python tools/beir.py --datasets hotpotqa --retrievers bm25-engine \
    --lexical-checkpoint --out eval/beir/hotpotqa
```

Resumable per query through `eval/beir/hotpotqa.lexical.jsonl`.
