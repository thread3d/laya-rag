# Findings: FRAMES — retrieval, and the fourth reversal

ROADMAP 3.1. FRAMES is 824 questions whose evidence is English Wikipedia articles, so it is a
**document-level retrieval** benchmark: each question's `wiki_links` name the articles that contain
what is needed. The plan's rule applies — if the generator dominates the variance, report retrieval
only — and retrieval is what can be measured offline and deterministically.

Reproduce:

```sh
curl -sL -o eval/frames/test.tsv \
    https://huggingface.co/datasets/google/frames-benchmark/resolve/main/test.tsv
.venv/bin/python tools/frames_corpus.py      # 2,482 articles, 51 API requests
make frames-eval                            # both arms
```

## Result — 814 scored questions over 2,482 articles

| arm | recall@1 | recall@10 | recall@20 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| lexical (BM25, engine settings) | 0.6425 | 0.8759 | 0.9079 | 0.7296 | 0.5794 |
| **dense (`bge-small`)** | **0.7432** | **0.9275** | **0.9521** | **0.8113** | **0.6594** |

**Dense beats lexical on every metric**, by 0.10 recall@1 and 0.08 nDCG@10.

## The fourth reversal, and what it says

This is the fourth corpus in this programme where a channel reverses, and it does so in the opposite
direction from the most recent one:

| corpus | who wins | why, as far as the evidence goes |
| --- | --- | --- |
| MultiHop-RAG | lexical (shipped default) | short news chunks, entity names |
| MuSiQue | dense (+0.054) | multi-hop paraphrase |
| `webis-touche2020` | **lexical, by 2:1** | long noisy web pages |
| **FRAMES** | **dense (+0.08 nDCG@10)** | natural-language multi-constraint questions |

Touche and FRAMES are both Wikipedia-derived and both "long", and they disagree — which is the useful
part. The variable is not length but **what the query looks like against the document**: touche asks a
short topical query against a page of running prose, where exact term matching is strong; FRAMES asks a
sentence of constraints ("the 15th first lady's mother's first name…") against a page that names its
subject once, where the vocabulary overlap is thin and a semantic encoder earns its keep.

That is the fifth independent instance of the programme's recurring conclusion — **route per corpus,
do not configure globally** — and this time the two sides of the contrast sit on the same source
dataset family, so it cannot be explained by domain.

## `gold_coverage` is 0.8792, and recall has to be read against it

The dataset's links are URLs while the corpus is keyed by the API's canonical titles, and redirects
(`USA` → `United States`) are **not** resolved, so 12.1% of gold titles do not match a corpus article.
Recall@20 of 0.9521 is therefore close to a ceiling set by coverage rather than by the retriever —
dense finds a linked article for essentially every question whose articles are present. Any future
comparison on this corpus should fix the redirect gap first, or it will be measuring coverage.

## What this does not show

- **Not end-to-end accuracy.** No generator ran: the roadmap's exact-match-accuracy row is not
  measured, and this says nothing about whether a model can *use* the retrieved articles.
- **Not a labelled span set.** Gold is document-level, so a retrieval can be "correct" and still not
  contain the sentence a question needs — which is what Phase 5.3's verifier would then have to catch.
- **Not redirect-aware.** As above; the coverage figure is the honest bound on every recall number here.
- **One retriever per channel.** BM25 at engine settings and one 33M encoder; the three-model ensemble
  and the cross-encoder were not run.

## Method notes

- **A document-level gold set needs its coverage reported.** Recall against a gold set that is 12%
  unresolvable is not comparable to one that is complete, and the number alone hides that.
- **TextExtracts returns one full extract per request however `exlimit` is set.** Three titles in,
  three pages back, exactly one with text — so a 50-title batch stored one article. `prop=revisions`
  honours the batch (50/request) and fetched the whole corpus in 51 requests. The tell was that the
  yield was *uniform*: a data problem is noisy, a contract is regular.
