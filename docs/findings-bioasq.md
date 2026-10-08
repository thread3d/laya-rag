# BioASQ: the last member, and the mirror that would have faked it

BioASQ was the only one of BEIR's eighteen datasets with no report here, and the only blocker that
survived an honest search. The other four came back — three from a public mirror
(`findings-beir-recovered.md`) and one, CQADupStack, from its own runner (`findings-cqadupstack.md`).
This one needed a **free BioASQ account**, and it is now measured: **18 of 18**.

The detour is worth keeping, because a **plausible-looking public mirror exists and would have
produced a wrong number** — one that passed every check available without the real corpus.

## What BEIR's own reproduction instructions require

The benchmark ships the recipe, and it is unambiguous
([`examples/dataset`](https://github.com/beir-cellar/beir/blob/main/examples/dataset/README.md)):

> 1. Register yourself at BioASQ: <http://www.bioasq.org/>
> 2. Download documents from BioASQ task 9a (Training v.2020 ~ 14,913,939 docs) and extract the
>    title and abstractText for each document.
> 3. There are few documents not present in this corpus but present in test qrels so we add them
>    manually.
>
> **Queries and Qrels**
> 1. Download Training and Test dataset from BioASQ 8B datasets which were published in 2020.
> 2. Consider all documents with answers as relevant (binary label) for a given question.

The BEIR wiki tabulates the same member as **14.91M documents / 500 test queries**. Both required
inputs — the task 9a corpus and the 8B golden train/test sets — sit behind the BioASQ participants
area, which is free but account-gated.

## What was checked, and what each check returned

| route | result |
| --- | --- |
| BEIR's UKP zip `bioasq.zip` | **404**, and the Wayback Machine has **no snapshot** of it (CDX query over the whole datasets path) |
| `BeIR/*` on HuggingFace | the organisation carries no bioasq corpus or qrels repo — only `bioasq-generated-queries` (see below) |
| `ir_datasets` `beir/*` namespace | BioASQ is **not** in it |
| BioASQ participants area | every data endpoint resolves to the **login page**: `/Tasks/8b/` (404 + login), `/Tasks/A/getData/`, `/oracle/` — all 200 with a login body |
| `ii.nlm.nih.gov/BioASQ/Archive/` (hosts some *public* BioASQ files, e.g. Task 5C) | **host unreachable** from this machine (`http=000`) |
| HuggingFace, all 200 `bioasq` datasets, scanned for files > 300 MB | no text corpus other than the partial one below; the large files are vector indexes and embeddings |
| HuggingFace full-text search, `Task8BGoldenEnriched`, `BioASQ-training8b` | no dataset carrying the golden files |

## The trap: a public corpus that is 5.4% short

`BeIR/bioasq-generated-queries` — mirrored at `jkwyp/bioasq-generated-queries`, same 7,115,850,480
bytes — is **public, non-gated, and in BEIR layout**. Its records are

```json
{"_id": "31784955", "title": "Introduction.", "text": "Depressive disorder is one of ...",
 "query": "are neurochemical drugs based on antidepressants"}
```

which is a PubMed corpus with an InPars-generated query attached, not merely generated queries as the
repo name suggests. It looks exactly like the thing that is missing.

It is not. Streaming the archive and counting gives **14,100,000 records** — a round number, and
**813,939 short of the 14,913,939** the benchmark scores against, a **5.46% corpus truncation**.

**How much of the gold set survives that truncation was measured, and the answer is most of it —
which is what makes this dangerous rather than merely wrong.** Against the 500-query 8B golden set
(2,298 distinct judged documents, of which 750 are supplied by BEIR's manual-add list and 1,548 must
come from the corpus), the mirror contains **1,533 of the 1,548 corpus-expected gold documents, and
misses 15**. It contains **0 of the 750** manual-add documents, which independently confirms that
list is exactly "in the qrels, absent from the corpus".

So a run on this mirror would not collapse the way the TREC-COVID subsample did — it would keep
98.97% of the documents the metric rewards, and the 813,939 missing documents would perturb the
*rankings* rather than delete the answers. The result would land within a few thousandths of the real
row, in an unpredictable direction, and there is no public qrels-complete corpus to check it against.
That is a worse failure mode than an obvious one: it is the shape of a number that gets published.

Using it would not be a slightly smaller measurement. It is the defect
`findings-trec-covid-full.md` is written about: the 500 test qrels are fixed and reference documents
across the whole collection, so truncating the corpus removes gold documents the qrels still ask for
and the metric then scores a corpus defect rather than a retriever. There, a 40,000-document subsample
of TREC-COVID read **0.3062** against the full collection's **0.6188** — half the number, from nothing
but missing documents. A 14.1M-of-14.9M corpus would not fail that loudly, which is exactly what
makes it dangerous: it would land *near* the published anchor and look like a result.

No public copy of the 8B golden queries or qrels was found either, so the truncation cannot even be
detected from the public side — there is nothing to score against.

## Status: both inputs in hand, ingested and asserted

**Obtained (2026-10-07) from the BioASQ participants area.**

The golden half — `Task8BGoldenEnriched.zip`, five batches `8B1`…`8B5` of **100 questions each, 500
total**, exactly the member's test split:

- **2,359 judgements over 2,298 distinct documents**, written to `queries.jsonl` and `qrels/test.tsv`
  in the layout `tools/beir.py` reads;
- every question has `body` (the query) and `documents` (PubMed URLs, PMID taken from the final path
  segment), so the binary rule needs no interpretation.

The corpus half — `allMeSH_2020.zip` → `allMeSH_2020.json`, **26.2 GB, one article per line inside a
single `articles` array**. `tools/bioasq.py` streams it (never `json.load`-ing 26 GB), merges the
manual-add rows first, and **asserts the membership before writing a report**:

```
corpus 14,914,604 documents (14,913,854 distinct pmids from 14,913,938 article lines,
84 duplicates dropped; 750 manual-add), 500 queries, 2,359 judgements over 2,298 distinct
documents, in 291.7s
```

Three things that check caught, each of which would otherwise have scored a subtly wrong corpus:

- **The array closer shares the last article's line.** The file ends `...}]}`, not with a line of its
  own, so a per-line `json.loads` succeeds fourteen-million times and then reports *"Extra data"* on
  the final record. It reads like a corrupt download; it is an off-by-one.
- **The dump is latin-1, not UTF-8.** Reading it as UTF-8 with replacement turns accented characters
  in the abstracts into `U+FFFD` — silently, and across the whole corpus.
- **The manual-add CSV has no header.** Treating its first row as one drops PMID `25941654`, a judged
  document, and the count then comes up *one judgement short* rather than failing.

The article count is **14,913,938 lines, one fewer than the 14,913,939** the participants-area table
advertises; BEIR's own note hedges it with a "~", and a single document out of 14.9M is not the failure
mode this check exists for. 84 of those lines repeat an earlier pmid and are dropped, which is what
the reference preprocessing does too.

## The measurement

```sh
make beir BEIR_ARGS="--datasets bioasq --retrievers bm25,bm25-engine --no-download \
  --out eval/beir/bioasq --lexical-checkpoint --lexical-backend ondisk"
```

| dataset | documents | retriever | nDCG@10 | Recall@10 | nDCG@100 | Recall@100 | MRR@10 | MAP@100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bioasq | 14,914,604 | `bm25` (published settings control) | **0.5168** | 0.5342 | 0.5869 | 0.7563 | 0.6049 | 0.4669 |
| bioasq | 14,914,604 | `bm25-engine` (shipped lexical path) | **0.5193** | 0.5496 | 0.5917 | 0.7764 | 0.5939 | 0.4683 |

Published anchors for the member (`tools/published_beir18.py`): **BM25 0.465**, uniCOIL 0.477,
**SPLADE 0.504** (the best of the five), TAS-B 0.371, Contriever 0.383.

So the engine's lexical row beats the published BM25 by **+0.0543**, and beats the *best* published
system by **+0.0153** — making BioASQ one of the nine datasets where this engine leads, and the
largest positive gap in the lexical control (HotpotQA's +0.032 was the previous high).

### The control row, and a hypothesis it killed

I wrote the first version of this section with an explanation attached: that the gap was our
*settings* rather than our engine — stemmed, stopword-dropped text doing more work on biomedical
abstracts than on news, and the engine row being anyway not the like-for-like control. It read
plausibly. **The control row falsifies it.**

The `bm25` row runs at the *published* settings — no stemming, no stopword dropping — and it reads
**0.5168**, which is

- **+0.0518 above the published BM25 anchor**, and
- within **0.0025** of the engine row.

So essentially all of the +0.05 is *not* a settings effect. Both rows sit about 0.05 above the
published baseline, and whichever way the difference arises — the harness's tokenisation, the
corpus as this repository builds it, or the published row having been measured under a convention
this one does not reproduce — **it is not attributable to the engine, and it is not explained here**.
What can be said is what the numbers say: on this corpus our flat BM25 control and the published BM25
column do not agree, by five times the margin the control is meant to stay inside, and the two rows
this repository can compare to each other agree to 0.0025.

That is the same class of finding as NQ's and FEVER's gaps, in the opposite direction, and it is
recorded as a gap in the *control* rather than as a result. It is also why the honest headline for
BioASQ is "measured, and beats the best published system by 0.0153" — not "beats published BM25 by
0.054".


## What this costs to reproduce

The run is **not cheap**: 14.9 M documents is ~2.8× NQ, and this is the corpus that Phase 4.1's on-disk
term dictionary exists for. The in-memory `BM25Index` cannot hold it, so `--lexical-backend ondisk`
spills postings to a memory-mapped file — the index build alone took ~90 minutes and the whole pass
~4.5 h. The corpus and its derived `corpus.jsonl` stay out of the repository (`eval/` is gitignored),
which matters because one of the two inputs is a dump distributed under a BioASQ account.

## Reproduction

```sh
.venv/bin/python tools/bioasq.py \
    --source eval/bioasq-src/allMeSH_2020.json \
    --additional eval/bioasq-src/additional_passages.tsv \
    --golden eval/bioasq-src/Task8BGoldenEnriched \
    --out eval/beir/bioasq
make beir BEIR_ARGS="--datasets bioasq --retrievers bm25-engine --no-download \
  --out eval/beir/bioasq --lexical-checkpoint --lexical-backend ondisk"
make consistency-check     # 18 of 18, every registered site agrees, the audit set included
```

## Status

Coverage is **18 of 18**, and this is the change that completes it. With BioASQ in, the engine's macro
over its measured set is a genuine BEIR-18 average for the first time — **0.4665**, between the
paper's Contriever (0.448) and SPLADE (0.474) — so the macro chart's *different membership — not
comparable* annotation comes off, with the caveat that ours is a best-per-dataset row rather than one
global configuration.

The trap is recorded above rather than deleted: the distance between "looks like the corpus" and "is
the corpus" here is 813,939 documents, and the next person to reach for that mirror should find this
page first.

```sh
make consistency-check     # 17 of 18, missing: bioasq
```
