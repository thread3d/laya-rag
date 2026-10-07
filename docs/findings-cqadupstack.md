# CQADupStack: twelve sub-forums behind one name

CQADupStack was the last canonical BEIR member with no report, and the only one described as needing
an *adapter* rather than a licence. Both halves of that description turned out to be true, and
neither was the obstacle the phrase suggested: the release was never 404, the qrels were already in
the format this repository reads, and the whole member is 457,199 documents — smaller than Robust04,
which had been measured for weeks. What it needed was a **shape** the one-corpus harness does not
have.

It is now measured, and the member's flat control reproduces its published BM25 anchor to
**−0.0002**. Coverage is **17 of 18**; BioASQ is the only member left.

## What the release actually is

A range request for the ZIP64 central directory — 6,209 bytes at the end of a 5.34 GB archive, no
full download needed to look — shows **61 entries**: twelve forums (`android`, `english`, `gaming`,
`gis`, `mathematica`, `physics`, `programmers`, `stats`, `tex`, `unix`, `webmasters`, `wordpress`),
each with its own

```
cqadupstack/<forum>/corpus.jsonl
cqadupstack/<forum>/queries.jsonl
cqadupstack/<forum>/qrels/test.tsv
```

Three consequences follow, and each rules out a different explanation:

- **It is not gated.** `cqadupstack.zip` is public on the same UKP mirror that 404s the other four,
  and `BeIR/cqadupstack` and `ir_datasets beir/cqadupstack/*` carry it too.
- **It is not a format problem.** Unlike the corpora `findings-beir-recovered.md` recovers, these
  qrels are already **three-column tab-separated with the header row** `load_qrels` skips. There is
  no converter in this round.
- **It is not a scale problem.** 457,199 documents across the twelve; the largest forum, `tex`, is
  68,184.

What it is: `tools/beir.py` loads **one corpus per dataset**, and `ensure_dataset("cqadupstack")`
extracts all twelve and then fails its own dataset-level `corpus.jsonl` check. Even with that fixed,
the harness could not score it, because the member's published number is the **macro-average of the
twelve sub-forum scores** — which is why MTEB carries a single aggregated `CQADupstackRetrieval` task
rather than twelve leaderboard rows. Pooling the twelve would not be a shortcut but a different
measurement: a `tex` query could then retrieve `physics` documents, which the per-forum qrels never
judge.

## The measurement

`tools/cqadupstack.py` runs each sub-forum independently through the same ranking and evaluation code
`beir.py` uses, then averages. `make beir --datasets cqadupstack` now fails with a pointer to it rather
than extracting 5.3 GB and then failing.

```sh
make cqadupstack CQADUPSTACK_ARGS="--retrievers bm25,bm25-engine --out eval/beir/cqadupstack"
```

| sub-forum | documents | scored queries | `bm25` | `bm25-engine` |
| --- | --- | --- | --- | --- |
| android | 22,998 | 699 | 0.3752 | 0.3681 |
| english | 40,221 | 1,570 | 0.3459 | 0.3400 |
| gaming | 45,301 | 1,595 | 0.4706 | **0.4743** |
| gis | 37,637 | 885 | 0.2838 | 0.2814 |
| mathematica | 16,705 | 804 | 0.2002 | **0.2048** |
| physics | 38,316 | 1,039 | 0.3063 | **0.3203** |
| programmers | 32,176 | 876 | 0.2805 | 0.2721 |
| stats | 42,269 | 652 | 0.2733 | 0.2677 |
| tex | 68,184 | 2,906 | **0.2335** | 0.2195 |
| unix | 47,382 | 1,072 | 0.2642 | **0.2651** |
| webmasters | 17,405 | 506 | 0.3023 | 0.2971 |
| wordpress | 48,605 | 541 | 0.2493 | 0.2209 |
| **MACRO (12)** | **457,199** | **13,145** | **0.2988** | **0.2943** |

Published BM25 nDCG@10 for the member: **0.299** (`tools/published_beir18.py`). The flat control's
**−0.0002** is the tightest anchor agreement in this repository after Robust04's +0.0007, and unlike
Robust04 it is over twelve independently scored collections rather than one.

**The control significantly beats the engine row here.** Paired bootstrap over the 13,145 pooled
queries: **+0.0046 [+0.0015, +0.0078]**, p=0.0052. The engine's settings — query stopwords dropped,
English stemming on — help on the corpora where they were chosen and *hurt* on this one, which is the
same per-corpus lesson every other instrument in this programme has taught: it is not a global
setting, it is a choice. Five of the twelve forums prefer the engine configuration and seven prefer
the published one, and the largest forum (`tex`, 2,906 queries) is the second-largest engine loss.

## What moved as a result

- **Coverage: 16 → 17 of 18.** **BioASQ** is the only member left, and it is the only one that is
  genuinely gated.
- **The engine's macro over its measured set: 0.4737 over 16 → 0.4634 over 17.** CQADupStack sits
  0.046 below the best published system, so it *lowers* the mean again — the third consecutive round
  in which widening the membership moved a BEIR average more than most method differences do.
- **Lead/loss against the best of the five published systems: 8 of 16 → 8 of 17.** The ninth loss.
- **The lexical control: 15 → 16 corpora**, now **11 within ±0.02 and 7 within ±0.008** (it was 10 and
  6). CQADupStack lands inside the tight bound, so the count rises without adding an outlier.

## A number that was already right

An earlier pass over this round claimed `BENCHMARKS.md`'s "457,199 documents" was wrong and should be
470,344, on the evidence of the twelve corpus row counts the HuggingFace datasets server reports.
That was this author's error, not the document's. The server's per-config `num_rows` is **corpus plus
queries summed** — for `english`, 40,221 + 1,570 = 41,791 — and 470,344 is exactly 457,199 + 13,145.
The zip's own `corpus.jsonl` line counts, now on disk, total **457,199** documents and **13,145**
queries, matching the document exactly. The figure stands; the correction was retracted.

## What this does not show

- **No dense, fused or reranked row.** All rows are lexical, consistent with the other large-corpus
  rows. A dense arm would need an embedding pass over all twelve sub-forums.
- **No per-forum significance test.** The bootstrap pools the twelve; the per-forum deltas in the
  table are reported as differences, not as intervals. With 506–2,906 queries per forum, several of
  the five "engine wins" would not survive a per-forum interval.
- **No claim that the twelve are equally weighted.** The macro treats `webmasters` (506 queries) and
  `tex` (2,906) alike, which is the benchmark's convention and not a statement about either.
- **No new licence question.** Unlike the recovered corpora, this collection has no TREC agreement
  behind it — it is public Stack Exchange content, the same material `BeIR/cqadupstack` redistributes.

## Reproduction

```sh
make cqadupstack CQADUPSTACK_ARGS="--retrievers bm25,bm25-engine --out eval/beir/cqadupstack"
make consistency-check      # 17 of 18, every registered site agrees, the audit set included
.venv/bin/python -m pytest tests/test_cqadupstack.py -q
```

The run took ~19 minutes for both lexical rows over the twelve sub-forums; the archive is fetched and
extracted once by `tools/cqadupstack.py` on first use, and a re-run is `--no-download`.
