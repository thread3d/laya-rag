# Findings: BEIR breadth — quora, and how much the mean depends on its members

ROADMAP Phase 4.2. The headline BEIR claim covers **five of eighteen** datasets
(scifact, nfcorpus, arguana, scidocs, fiqa), so it is not comparable to the field's own
18-dataset averages. This adds a sixth, under exactly the retrievers the headline table uses.

Reproduce:

```sh
.venv/bin/python tools/beir.py --datasets quora \
    --retrievers bm25-engine,biencoder,hybrid-biencoder --out eval/beir/quora
```

## Result — nDCG@10, shipped configuration

| dataset | `bm25-engine` | `bge-small` dense | fused (hybrid) |
| --- | --- | --- | --- |
| scifact | 0.6762 | 0.7200 | 0.7413 |
| nfcorpus | 0.3244 | 0.3385 | 0.3673 |
| arguana | 0.4169 | 0.5956 | 0.5679 |
| scidocs | 0.1497 | 0.1973 | 0.1992 |
| fiqa | 0.2386 | 0.3848 | 0.3817 |
| **quora** | **0.7689** | **0.8876** | **0.8629** |
| **5-set mean** | 0.3612 | 0.4472 | **0.4515** |
| **6-set mean (with quora)** | **0.4291** | **0.5206** | **0.5201** |

## Two readings

**1. The mean is composition-sensitive, and by more than the thing being compared.** Adding one
easy dataset moves the fused mean from **0.4515 to 0.5201 — +0.069**. For scale, the project's
whole per-parameter claim over published BGE-large is a 0.04 margin, and the gap between the
published 4-set matrix (0.491) and its own 5-set row is comparable. So "the BEIR average" is only
meaningful with its membership named, and the README's existing caveat that the 0.495 ensemble row
is *"a single configuration rather than a per-dataset choice"* extends to the mean itself: it is a
statement about a set, and a different set is a different claim.

**2. Fusion is not a free win, again.** On quora the **fused** row (0.8629) is **0.0247 below the
dense row alone** (0.8876). That is the fourth corpus where a "helpful" instrument reverses
(MultiHop-RAG's dense channel, MuSiQue's dense channel, the cross-encoder on five BEIR sets, and
now RRF fusion on quora), and it is the same conclusion each time: pick per corpus, do not
configure globally.

## The seventh dataset: touche, and selection cutting the other way

`webis-touche2020` (382,545 documents, 49 scored queries) is now measured under the same retrievers:

| dataset | `bm25-engine` | `bge-small` dense | fused (hybrid) |
| --- | --- | --- | --- |
| **webis-touche2020** | **0.4419** | **0.2202** | **0.3509** |
| **5-set mean** | 0.3612 | 0.4472 | 0.4515 |
| **6-set mean (+quora)** | 0.4291 | 0.5206 | 0.5201 |
| **7-set mean (+touche)** | **0.4309** | **0.4777** | **0.4959** |

**Adding touche moves the fused mean the other way** — 0.5201 down to **0.4959**, −0.024 — where adding
quora moved it up by +0.069. Two datasets, opposite directions, is the composition point demonstrated
rather than asserted: a BEIR average is a statement about a set, and swapping one member can move it
by more than any method difference it is used to adjudicate.

**Fusion reverses again, and harder than anywhere else.** On touche the fused row is **0.091 below
BM25 alone** (0.3509 against 0.4419), and the dense row is *less than half* the lexical one (0.2202).
That is the fifth corpus where an instrument that helps elsewhere loses — after MultiHop-RAG's dense
channel, MuSiQue's dense channel, the cross-encoder across five BEIR sets, and RRF fusion on quora.
Touche's documents are long, noisy web pages, which is where a 33M-parameter dense encoder has least
to offer relative to exact lexical matching.

## The lexical control, and a hypothesis that did not survive

The lexical arm is the one component checkable against an independent implementation, so it is worth
drawing rather than assuming. Twelve corpora now have both our number and a published BM25 anchor:

| corpus | documents | ours | published | gap |
| --- | --- | --- | --- | --- |
| SciFact | 5,183 | 0.6762 | 0.679 | −0.003 |
| NFCorpus | 3,633 | 0.3244 | 0.325 | −0.001 |
| ArguAna | 8,674 | 0.4169 | 0.397 | **+0.020** |
| SciDocs | 25,657 | 0.1497 | 0.158 | −0.008 |
| FiQA | 57,638 | 0.2386 | 0.236 | +0.003 |
| Touché-2020 | 382,545 | 0.4419 | 0.442 | −0.000 |
| **Quora** | 522,931 | 0.7689 | 0.789 | **−0.020** |
| **NQ** | 2,681,468 | 0.3118 | 0.329 | **−0.017** |
| **DBPedia-entity** | 4,635,922 | 0.3053 | 0.313 | **−0.008** |
| HotpotQA | 5,233,329 | 0.6352 | 0.603 | **+0.032** |
| **Climate-FEVER** | 5,416,593 | 0.1610 | 0.213 | **−0.052** |
| **FEVER** | 5,416,568 | 0.6323 | 0.753 | **−0.121** |

**8 of the 12 corpora agree within ±0.02** (5 within ±0.008). ArguAna and HotpotQA sit *above* their
anchors — HotpotQA by +0.032 on the **full 7,405 queries** (its earlier +0.034 was a 600-query subset,
now superseded). **FEVER (−0.121, the largest gap measured by far)**, Climate-FEVER (−0.052), Quora
(−0.020) and NQ (−0.017) sit below. An earlier version of this line claimed "eight of nine agree
within ±0.008", which was simply a miscount; the table is the record.

**The tempting explanation — that the gap grows with corpus size — is wrong**, and DBPedia is what
kills it: at 4.6M documents, the largest corpus measured, the gap is −0.008, smaller than SciDocs at
25,657 documents. Three points had suggested a trend (Quora and NQ were the two largest *and* the two
worst); a fourth, chosen to test it, did not fit.

**The second candidate is ruled out by data already on disk.** Our `bm25-engine` row differs from a
published BM25 in two ways — it drops stopwords and it stems — so the obvious mechanism is that those
choices cost accuracy. The `bm25` **control** row runs at published settings without either, and on all
five small datasets it sits *further* from the anchor than the engine row does (SciFact −0.025 against
−0.003, NFCorpus −0.021 against −0.001). So the engine's settings are not the cause; if anything they
help.

What is left is that both outliers are **natural-language question** datasets — NQ and Quora — while
every keyword- or entity-style set agrees closely. That is an observation with two members, not an
explanation, and the cause is **unidentified**. A control row on NQ itself (published settings, another
~1 h at 2.68M documents) is the experiment that would separate "our settings" from "these queries".

## What this does not show

- **Not the 18-dataset average.** Six of eighteen. `webis-touche2020` is **downloaded and extracted**
  (702 MB, 382,545 documents, 49 scored queries) but not evaluated: its lexical pass is
  single-threaded over long documents and the 382k-document embedding is still ahead, so it was
  stopped rather than left running — a re-run skips the fetch. `specter` and `signal1m` are **404 on
  this mirror**, Robust04 and TREC-NEWS return stubs, and the largest (BioASQ 14.9M, FEVER 5.4M,
  NQ 2.7M) are past the ingestion work in Phase 4.1.
- **Not an ensemble row.** The three-model dense ensemble is not extended to quora; only the
  shipped single-encoder configuration is.
- **No BM25 control check.** The harness verifies its `bm25` control against a pinned published
  baseline and reports any drift, but no published anchor is configured for quora, so that check
  did not run for it — ours is 0.7689 and is not validated against an external number here.
- **Not a claim about quora as a task.** Its documents are short questions, which is why the
  embedding pass is cheap and the dense arm is strong; it is an easy dataset, and that is exactly
  why it moves the mean so much.

## Method notes

- **Record the membership with the mean.** The tool writes `datasets` into the report, and the
  table above names every member, because a mean without its set is not a claim anyone can check.
- **A new dataset costs one embedding pass, and nothing else.** Five of six datasets were already
  cached; adding quora was a 15 MB download, 799 s of embedding for 522,931 documents and 199 s of
  scoring, with no harness change.
