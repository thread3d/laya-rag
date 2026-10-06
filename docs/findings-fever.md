# FEVER: the twelfth BEIR dataset, and the largest lexical gap measured

FEVER was the second of the audit's "blocked/unstaged" 5.4M-document corpora. It ran lexical-only
after Climate-FEVER, resuming from its 1,356-query checkpoint, and finished all 6,666 queries.

## The row

| dataset | documents | queries | retriever | nDCG@10 | Recall@10 | nDCG@100 | Recall@100 | MRR@10 | MAP@100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fever | 5,416,568 | 6,666 | bm25-engine | **0.6323** | 0.7846 | 0.6599 | 0.9070 | 0.6022 | 0.5796 |

This is the **twelfth** dataset with a report, so `make consistency-check` now computes **12 of 18**
(the gate failed first and named the three documents + registry, which were updated together).

## The gap is the largest in the control

The lexical control carries a published BM25 anchor of **0.753** for FEVER, so this row is
**0.6323 against 0.753 — a −0.121 gap**, more than twice Climate-FEVER's −0.052 and six times Quora's
−0.020:

| corpus | ours | published | gap |
| --- | --- | --- | --- |
| **FEVER** | 0.6323 | 0.753 | **−0.121** |
| Climate-FEVER | 0.1610 | 0.213 | −0.052 |
| Quora | 0.7689 | 0.789 | −0.020 |
| NQ | 0.3118 | 0.329 | −0.017 |
| SciDocs | 0.1497 | 0.158 | −0.008 |

**It is not corpus size.** That hypothesis was already falsified (`dbpedia-entity`, 4.6M documents,
sits at −0.008), and FEVER does not rescue it: it is 5.4M documents and loses 0.121, while dbpedia is
almost as large and loses 0.008. What the four negative outliers have in common is the *query kind* —
FEVER's queries are natural-language claims, Quora's and NQ's are natural-language questions, and the
two corpora that track their anchors (SciFact, NFCorpus) or beat them (ArguAna) are short technical
queries. That is a pattern, not a cause: **the cause remains unidentified**, and the experiment that
would separate "our tokenizer/settings" from "these queries" is a published-settings control row,
which on this corpus is the same diagnostic that was abandoned on NQ on measured cost.

## What this does not show

- **Lexical-only.** No dense arm, no fusion; the dense arm would need its own multi-million-document
  embedding pass.
- **The anchor is a published number, not re-run here.** `tools/compare_more.py` carries the 0.753
  figure (BEIR's Anserini BM25); the like-for-like caveats in `docs/findings-beir-breadth.md` apply —
  this is a comparison against a published configuration, not a controlled reimplementation.
- **The large gap is unexplained**, and it is the strongest single argument that the lexical
  shortfall is about query distribution rather than scale.

## Reproduction

```sh
.venv/bin/python tools/beir.py --datasets fever --retrievers bm25-engine \
    --lexical-checkpoint --out eval/beir/fever
```

Resumable per query through `eval/beir/fever.lexical.jsonl`. The run holds the heavy-job lock, so it
cannot overlap another corpus job.
