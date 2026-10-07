# BioASQ: why the last member is still not measured

BioASQ is the only one of BEIR's eighteen datasets with no report here, and the only one whose
blocker survives an honest search. The other four came back — three from a public mirror
(`findings-beir-recovered.md`) and one, CQADupStack, from its own runner (`findings-cqadupstack.md`).
This one does not, and the reason is worth writing down because a **plausible-looking public mirror
exists and would produce a wrong number**.

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

Using it would not be a slightly smaller measurement. It is the defect
`findings-trec-covid-full.md` is written about: the 500 test qrels are fixed and reference documents
across the whole collection, so truncating the corpus removes gold documents the qrels still ask for
and the metric then scores a corpus defect rather than a retriever. There, a 40,000-document subsample
of TREC-COVID read **0.3062** against the full collection's **0.6188** — half the number, from nothing
but missing documents. A 14.1M-of-14.9M corpus would not fail that loudly, which is exactly what
makes it dangerous: it would land *near* the published anchor and look like a result.

No public copy of the 8B golden queries or qrels was found either, so the truncation cannot even be
detected from the public side — there is nothing to score against.

## What would finish it

Two files, both from a **free** BioASQ account (registration is open to anyone; the account is the
gate, not a licence fee):

1. **Task 9a training v2020** (~14,913,939 documents) — or `allMeSH_2020.json`, the BioASQ 2020
   snapshot the corpus is derived from. Extract `pmid`, `title`, `abstractText`, plus the handful of
   manually-added documents BEIR step 3 names.
2. **The 8B golden train and test sets** — question `id`, `body`, and `documents`; every document in
   `documents` becomes a relevant judgement, which is BEIR's binary rule.

Together those are enough to write `corpus.jsonl`, `queries.jsonl` and `qrels/test.tsv` in the layout
`tools/beir.py` reads, and the existing harness scores it unchanged.

Two things to know before starting it. The run is **not cheap**: 14.9M documents is ~2.8× NQ, and
this repository's own history records the 5.4M-document corpora as "compute-bound at ~6 h per corpus"
— so a BioASQ lexical pass is a **~15–24 h** job, and it is why Phase 4.1's on-disk term dictionary
exists at all. And because the corpus is 14.9M and one of the two inputs is a licensed-ish dump, the
files should stay out of the repository, which is already the case (`eval/` is gitignored).

## Status

Coverage is **17 of 18** and this does not change it. Recording the negative result is the point:
the corpus mirror is a trap that a later round would otherwise step into, and the distance between
"looks like the corpus" and "is the corpus" here is 813,939 documents.

```sh
make consistency-check     # 17 of 18, missing: bioasq
```
