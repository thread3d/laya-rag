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
| lexical (BM25, engine settings) | 0.7224 | 0.9349 | 0.9619 | 0.8028 | 0.6586 |
| **dense (`bge-small`)** | **0.8464** | **0.9767** | **0.9939** | **0.8931** | **0.7505** |

**Dense beats lexical on every metric**, by 0.12 recall@1 and 0.09 nDCG@10.

### The ensemble is the best arm, and BM25 fusion loses at the top

The three-model dense ensemble (`bge-small` + `bge-base` + `bge-large`, 478M parameters) against the
single encoder, and against fusing BM25 back in by RRF:

| arm | recall@1 | recall@10 | recall@20 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| dense (`bge-small`) | 0.8464 | 0.9767 | 0.9939 | 0.8931 | 0.7505 |
| **ensemble** | **0.8722** | 0.9939 | 0.9975 | **0.9193** | **0.8013** |
| ensemble + BM25 (RRF) | 0.8317 | 0.9939 | **0.9988** | 0.8939 | 0.7978 |

The ensemble beats the single encoder on every metric — **+0.051 nDCG@10, +0.026 recall@1, +0.026
MRR@10** — which is the sixth corpus on which this programme's one positive lever has held.

**Fusing BM25 into it loses 0.041 recall@1** and 0.0035 nDCG@10, buying only the last 0.0013 of
recall@20 (0.9975 → 0.9988). So on FRAMES the lexical channel is not merely redundant beside a strong
dense arm, it is harmful at the top of the ranking — the fifth corpus where fusion fails to be a free
win, and the clearest case yet, since here it is fusing into an arm that is already better than the
channel being added.

### Resolving the redirects was worth more than the comparison it enabled

The first run scored the dataset's URL spellings directly and reported `gold_coverage` 0.8792 — 12% of
the gold articles matched nothing. Resolving every link through the API's `normalized` and `redirects`
hops (`tools/frames_corpus.py --resolve-map`) took coverage to **0.9977** and moved both arms by far
more than the arms differ from each other:

| arm | metric | unresolved | **resolved** | change |
| --- | --- | --- | --- | --- |
| lexical | recall@1 | 0.6425 | **0.7224** | +0.080 |
| lexical | nDCG@10 | 0.5794 | **0.6586** | +0.079 |
| dense | recall@1 | 0.7432 | **0.8464** | +0.103 |
| dense | nDCG@10 | 0.6594 | **0.7505** | +0.091 |

So the unresolved run understated *both* retrievers by roughly 0.08–0.10 nDCG@10 — more than the
dense-lexical gap it was being used to measure — and the gap itself widened slightly (0.0800 → 0.0919).
A coverage gap does not just scale the numbers down; it changes how much they can be trusted to rank
the things being compared.

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
- **Redirects are now resolved** (`--resolve-map`, coverage 0.9977), so these numbers are the
  comparable ones; the unresolved run is kept above only to show what the gap cost.
- **One retriever per channel.** BM25 at engine settings and one 33M encoder; the three-model ensemble
  and the cross-encoder were not run.

## Method notes

- **A document-level gold set needs its coverage reported.** Recall against a gold set that is 12%
  unresolvable is not comparable to one that is complete, and the number alone hides that.
- **TextExtracts returns one full extract per request however `exlimit` is set.** Three titles in,
  three pages back, exactly one with text — so a 50-title batch stored one article. `prop=revisions`
  honours the batch (50/request) and fetched the whole corpus in 51 requests. The tell was that the
  yield was *uniform*: a data problem is noisy, a contract is regular.
