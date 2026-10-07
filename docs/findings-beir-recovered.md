# The three 404'd BEIR corpora, recovered and measured

Three of BEIR's eighteen datasets had been carried as **blocked on licensing** since the UKP mirror
deleted their zips: `robust04`, `trec-news` and `signal1m`. Coverage stood at 13 of 18 for that
reason. This round went looking for them elsewhere, found all three in BEIR layout on a public,
non-gated mirror, and measured them as lexical rows — taking coverage to **16 of 18**.

The recovery is a *measurement* result, not a licensing one, and the distinction is load-bearing:
the corpora are third-party redistributions of TREC-agreement collections. See "What this does not
show" before quoting the rows.

## Where they were, and where they were not

| source | `robust04` / `trec-news` / `signal1m` |
| --- | --- |
| UKP mirror (`public.ukp.../thakur/BEIR/datasets/`) | **404** — the directory listing today serves 23 zips and none of these three |
| Wayback Machine | **no snapshot at all.** A CDX query over the whole datasets path returns captures for the corpora that are *still live* and none for these; they were never crawled |
| `BeIR/*` on HuggingFace | **absent.** The organisation carries 50 repos and none of these three corpora. The **401** its API returns is not evidence of gating: it returns 401 for a name that does not exist too, so the earlier "gated HF copies" reading was unsupported |
| `ir_datasets` `beir/*` namespace | **absent.** It carries 13 BEIR datasets — and not `scidocs` either — so it is not a route to any of them |
| `Tevatron/beir-corpus` | **absent** — the same 15 corpora that are already runnable, in `.jsonl.gz`, no `robust04`/`trec-news`/`signal1m` |
| **`liuqi6777/{robust04,trec-news,signal1m}`** | **present**, public and non-gated: `corpus.jsonl`, `queries.jsonl`, `qrels.beir-v1.0.0-<name>.test.txt`, plus a `qrels/test.jsonl` |

The mirror is a third-party re-upload in exactly the layout `tools/beir.py` already reads, which is
why the adapter below is one `awk` and not a format port.

## The adapter: one column, one separator

`tools/beir.py:load_qrels` reads `qrels/*.tsv` as three tab-separated columns
(`query`, `doc`, `score`). The mirror's files are **four** columns — the TREC `qid iter docid rel`
form — and **space**-separated, not tab-separated. Handed to the loader unchanged they are silently
dropped: for `robust04` the third field is a doc id (`FBIS3-10082`), which fails the `float()` parse
and the row is skipped; for `signal1m`, whose doc ids *are* numeric, the parse succeeds and every
judgement collapses onto the doc id `"0"`. The whole adapter is:

```sh
awk '{print $1"\t"$3"\t"$4}' qrels.beir-v1.0.0-<name>.test.txt > qrels/test.tsv
```

## The corpora, verified against the expected membership

| dataset | documents | queries | judgements | dangling qrels doc ids |
| --- | --- | --- | --- | --- |
| `robust04` | 528,155 | 249 | 311,410 | **0** |
| `trec-news` | 594,977 | 57 | 15,655 | 57 (3 judged relevant; every query keeps a relevant doc) |
| `signal1m` | 2,866,316 | 97 | 1,899 | **0** |

`robust04`'s three counts match `ir_datasets`' own metadata for `trec-robust04` exactly — 528,155
documents and 311,410 qrels — which is the check that the re-upload is the full collection and not a
sample of it. `trec-news`'s 57 dangling ids are 0.4% of its judgements and touch no query's ability
to be scored.

## The measurements

```sh
make beir BEIR_ARGS="--datasets robust04 --retrievers bm25,bm25-engine --out eval/beir/robust04"
make beir BEIR_ARGS="--datasets trec-news --retrievers bm25,bm25-engine --out eval/beir/trec-news"
make beir BEIR_ARGS="--datasets signal1m --retrievers bm25,bm25-engine --out eval/beir/signal1m"
```

| dataset | documents | retriever | nDCG@10 | Recall@10 | nDCG@100 | Recall@100 | MRR@10 | MAP@100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| robust04 | 528,155 | `bm25` (published settings control) | 0.3711 | 0.1266 | 0.3432 | 0.3441 | 0.6385 | 0.1728 |
| robust04 | 528,155 | `bm25-engine` (shipped lexical path) | **0.4077** | 0.1421 | 0.3794 | 0.3821 | 0.6886 | 0.1986 |
| trec-news | 594,977 | `bm25` (published settings control) | 0.3151 | 0.1208 | 0.3585 | 0.4069 | 0.6697 | 0.1971 |
| trec-news | 594,977 | `bm25-engine` (shipped lexical path) | **0.3351** | 0.1358 | 0.3891 | 0.4461 | 0.7599 | 0.2214 |
| signal1m | 2,866,316 | `bm25` (published settings control) | **0.3640** | 0.2002 | 0.3659 | 0.3909 | 0.6517 | 0.2075 |
| signal1m | 2,866,316 | `bm25-engine` (shipped lexical path) | 0.3131 | 0.1609 | 0.3261 | 0.3573 | 0.5639 | 0.1729 |

Published BM25 anchors, from the same column as the rest of this repository's large-corpus anchors
(`tools/published_beir18.py`): **robust04 0.407**, **trec-news 0.398**, **signal1m 0.330**.

- **Robust04 is the strongest lexical evidence in the repository.** `bm25-engine` reads **0.4077**
  against a published **0.407** — the control reproduces an external anchor to one thousandth, on a
  528,155-document corpus that had never been run here. The flat control, at 0.3711, is the one
  0.036 *below* the anchor, which is the familiar convention gap rather than an engine result.
- **TREC-NEWS is a new negative**: −0.063 for the engine row, making it the second-largest gap in the
  lexical control after FEVER.
- **Signal-1M is led by the control, not the engine.** The flat `bm25` row (0.3640) beats the published
  0.330 by +0.034, while the shipped `bm25-engine` path (0.3131) is 0.017 *behind* it. The corpus
  therefore enters the BEIR-18 matrix as a win — the matrix draws the best measured row — but the win
  is a statement about BM25 convention, not about this engine, and it is annotated as such in
  `docs/COMPARISONS.md`.

## What moved as a result

- **Coverage: 13 → 16 of 18.** The two remaining members at this step were **BioASQ** and
  **CQADupStack**; CQADupStack followed in the next round (`findings-cqadupstack.md`), leaving BioASQ
  as the only one.
- **The engine's macro over its measured set: 0.4979 over 13 → 0.4737 over 16.** All three new
  corpora sit below their published best (Contriever 0.473 and 0.428, BM25 0.330), so the wider
  membership *lowers* the mean — the same lesson the breadth finding drew, from the other direction.
- **Lead/loss against the best of the five published systems: 7 of 13 → 8 of 16**, the eighth being
  Signal-1M's control row.
- **The lexical control: 12 → 15 corpora**, now **10 within ±0.02 and 6 within ±0.008** (it was 8 and
  5). `tools/compare_more.py` needed the three anchors and `tools/compare.py` the three report
  routes; the counts are generated, so the prose they feed is pinned by `make claim-check`.

## What this does not show

- **The licensing question is untouched.** Robust04 is TREC disks 4–5 and TREC-NEWS is the
  Washington Post collection; both are distributed by NIST under signed agreements, and `ir_datasets`
  is explicit that access requires them. The mirror is a public redistribution, not a licensed one.
  "Blocked (licensing)" remains the *policy* position even though the technical obstacle is gone, and
  Signal-1M is Twitter content with the same problem in a less formal register. Nothing here has been
  checked with a lawyer; a reader who needs the licence should get it from NIST.
- **No dense or hybrid row.** All six rows are lexical, consistent with the other large-corpus rows
  (NQ, DBPedia, FEVER, Climate-FEVER).
- **No significance test.** One run per corpus; the gaps are quoted as differences, not intervals.
- **No claim that the mirrors are complete.** `robust04` matches `ir_datasets` exactly and
  `trec-news` is 0.4% short on qrels, but three corpora on one person's HF account is not a
  preservation guarantee — the checks above are the reason to trust *these* files, not the mirror.
- **BioASQ was not solved.** No public drop-in copy was found. Its corpus is the BioASQ 2020 PubMed
  snapshot (`allMeSH_2020.json`) plus `additional_passages.tsv`, with queries from the Task 8B golden
  set — all behind a **free** BioASQ participants-area registration. A later 23M-document snapshot
  (`jmhb/pubmed_bioasq_2022`) exists under a CC0 packaging, but it would not reproduce BEIR's 14.9M
  collection, so it is a substitute and not the dataset.

## Reproduction

```sh
make consistency-check      # 17 of 18 today (16 of 18 when this finding was written)
.venv/bin/python tools/lexical_control.py     # 15 corpora: 6 within ±0.008, 10 within ±0.02
.venv/bin/python -m pytest tests/test_charts_compare.py tests/test_lexical_control.py -q
```

The three corpora are fetched from the mirror by hand into `eval/beir/<name>/` (that tree is
gitignored); the eval itself is `--no-download` from there.
