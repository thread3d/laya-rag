# 2WikiMultiHopQA: the gold is found, the order is the bottleneck

2WikiMultiHopQA was built into the programme as a third external multi-hop set, to test whether the
coverage objective's failure on MuSiQue (−0.0109 MAP@10) was idiosyncratic or general. The adapter
runs here for the first time, on the full dev split: **12,576 questions** over the **56,651-paragraph
distractor pool** (the union of the context paragraphs the records carry, deduplicated by
`(title, text)`).

The harness is deliberately retrieval-only — BM25, a bi-encoder and their RRF fusion, **no
cross-encoder and no selection rules** — so this measures the *channel* question on a third corpus.

## The rows

| arm | Hits@10 | Hits@4 | MAP@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| bm25-engine | 0.9927 | 0.9848 | 0.5037 | 0.9293 |
| biencoder (`bge-small`) | 0.9986 | 0.9959 | 0.5223 | 0.9505 |
| **hybrid-biencoder** (RRF, k=5) | **0.9997** | **0.9986** | **0.5337** | **0.9564** |

Every paired comparison is significant over 12,576 queries (10,000 bootstrap samples, p<0.0001):

| comparison | delta | 95% CI |
| --- | --- | --- |
| biencoder − bm25, MAP@10 | **+0.0186** | [+0.0163, +0.0208] |
| hybrid − bm25, MAP@10 | **+0.0300** | [+0.0285, +0.0315] |
| biencoder − bm25, MRR@10 | +0.0212 | [+0.0174, +0.0249] |
| hybrid − bm25, MRR@10 | +0.0271 | [+0.0245, +0.0297] |

## What it says

1. **Ordering is the bottleneck, not recall.** **Hits@10 is 0.9927 while MAP@10 is 0.5037** — the pool
   almost always contains the supporting paragraphs, and the score is lost in the order they come
   back in. That is the same recall/selection split the MultiHop-RAG diagnostics found, on a third
   corpus and a different question distribution.
2. **A retrieval-trained channel wins on multi-hop questions.** The bi-encoder beats the lexical arm
   on every metric (**+0.0186 MAP@10**), and RRF fusion of the two adds again (**+0.0300 MAP@10**
   over BM25 alone). On BEIR this system's *shipped* dense channel (Laya's mean-pooled encoder) loses
   badly to BM25; this run shows the gap is the encoder's training, not dense retrieval as such — the
   `bge-small` row replaces that channel and wins.
3. **The coverage objective is still untested here.** This is the corpus the third replication was
   *for*, and it needs the selection rules added to the harness; the rows above answer only the
   channel question. Recording that explicitly is the point.

## What this does not show

- **Distractor setting, not open corpus.** The corpus is the union of the dev records' context
  paragraphs, so retrieval is measured inside the provided pool — the standard distractor evaluation
  and not a Wikipedia-scale number.
- **No generation.** There is no EM or answer-accuracy figure here; scoring answers needs a generator
  and a judge, and any such number would be a statement about them.
- **Not Laya's own dense channel.** The bi-encoder row is `bge-small`; the shipped Laya encoder is not
  in this comparison, and its BEIR behaviour is the reason it was replaced.
- **The 2Wiki coverage/selection replication is owed**, and now has a harness, a corpus and a
  checkpoint to run it on.

## Reproduction

```sh
.venv/bin/python tools/twowiki.py --split dev --out eval/benchmarks/2wiki/retrieval
```

The corpus embedding pass took **389 s** for 56,651 paragraphs on this host and is checkpointed every
20,000 documents; the whole run holds the heavy-job lock.
