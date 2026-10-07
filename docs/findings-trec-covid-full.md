# TREC-COVID on the full collection: the subsample was wrong, not just small

TREC-COVID was the last BEIR dataset the audit carried as measured-but-not-counted: *"a
40,000-document subsample that is deliberately not counted"*. The full collection is public on the
same UKP mirror as every other dataset (`trec-covid.zip`, 73.9 MB), needs no licence, and the
engine's on-disk postings path had already built it once at full size for the scale work. The
subsample was a `--limit-docs 40000` run over a corpus that was otherwise complete. Running the
full collection turns a deliberately-uncounted bounded row into a counted one — **12 → 13 of 18**.

## The measurement

```sh
make beir BEIR_ARGS="--datasets trec-covid --retrievers bm25,bm25-engine \
  --out eval/beir/trec-covid --lexical-checkpoint"
```

| dataset | documents | retriever | nDCG@10 | Recall@10 | nDCG@100 | Recall@100 | MRR@10 | MAP@100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trec-covid | 171,332 | `bm25` (published settings control) | **0.5573** | 0.0159 | 0.3819 | 0.0951 | 0.7573 | 0.0607 |
| trec-covid | 171,332 | `bm25-engine` (shipped lexical path) | **0.6188** | 0.0175 | 0.4594 | 0.1219 | 0.8767 | 0.0828 |

Published anchor: **BM25 flat 0.595**, Pyserini 2CR for `beir-v1.0.0-trec-covid`
([castorini.github.io/pyserini/2cr/beir.html](https://castorini.github.io/pyserini/2cr/beir.html)),
the same source the other per-corpus lexical anchors in this repository use. The flat control sits
at **−0.038**; the engine's own settings (stemming on, query stopwords dropped) sit at **+0.024**.

## The finding: a corpus truncation is not a query subset

The 40,000-document run read `bm25-engine` **0.3062** — half the full-collection number. That is not
a sampling difference. The 50 TREC-COVID qrels are fixed and reference documents across the whole
171,332-document collection, so truncating the corpus removes gold documents the qrels still ask
for. The metric then measures a corpus defect, not a retriever.

This is why the registry carried the row as *bounded* rather than counting it, and why the correct
resolution is not to average the subsample into a BEIR figure but to replace it. The run is recorded
in `SUPERSEDED` (`trec-covid-bounded` → `trec-covid`) with a quarantine marker, so the gate refuses
to read the 0.3062 as a current result, and `BOUNDED` is now empty — a bounded run remains a
legitimate measurement the registry can hold, but no dataset currently needs one.

## What also moved

The same pass corrected a factual error the previous audit introduced: **MS MARCO is not gated.**
The UKP mirror serves `msmarco.zip` (1.08 GB) and `BeIR/msmarco` answers 200 anonymously. It is
unrun because 8.8M documents is a heavy build, not because access is denied. Only BioASQ (14.9M,
gated) and the four licensing-gated corpora — Robust04, TREC-NEWS, `specter`, `signal1m`, each 404
on the mirror with 401 HF copies even with the local token — remain access-blocked.
**Superseded in part (later round):** three of those four were subsequently recovered from a public
non-gated mirror and measured, and the `BeIR/*` "gated copies" reading turned out to be wrong — that
organisation does not carry the corpora at all, and its 401 is returned for nonexistent names too.
See `findings-beir-recovered.md`; BioASQ remains the only member without a public drop-in copy.

## What this does not show

- **No dense or hybrid row.** Both rows are lexical, consistent with the other large-corpus rows
  (NQ, DBPedia, FEVER, Climate-FEVER) but not a full channel comparison.
- **No significance test.** One run, 50 queries; the published gap is quoted as a difference, not
  as a paired interval.
- **No claim about long documents.** TREC-COVID is abstracts and titles; the earlier rejection of
  600-character passage indexing (0.306 → 0.245) was run on the subsample and is not re-measured
  here.
- **The published anchor is a reproduction, not the leaderboard.** Pyserini's 2CR BM25 flat row is
  the same convention the flat control uses (flat index, `--remove-query`), so the comparison is
  like-for-like; it is not the BEIR paper's multifield number.

## Reproduction

```sh
make beir BEIR_ARGS="--datasets trec-covid --retrievers bm25,bm25-engine --out eval/beir/trec-covid --lexical-checkpoint"
make consistency-check     # 17 of 18 today (13 of 18 when this finding was written)
```

The run resumes from `<out>.lexical.jsonl` per query; the corpus is fetched and extracted by
`tools/beir.py` on first use, so a fresh checkout pays the 74 MB download once.
