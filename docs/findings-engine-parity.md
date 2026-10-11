# Findings: the engine on BEIR, not the replica

Every BEIR row in this project comes from `tools/beir.py`, which *reimplements* the engine's
channels instead of calling the engine. That is exactly how the engine came to fuse at `k=60`
while the harness used `k=5` for several rounds. `tools/engine_beir.py` closes that gap: it
indexes a BEIR dataset with the real `RagEngine`, runs its test queries with the relevance and
answerability thresholds set to **0** so nothing is filtered and the returned order *is* the
retrieval order, and scores nDCG@10 at the document level.

```sh
make engine-beir ENGINE_BEIR_ARGS="--dataset scifact --tag base"
make engine-beir ENGINE_BEIR_ARGS="--dataset scifact --tag sparse --sparse"
```

## Results (SciFact, 300 test queries, `bge-small`, top-10)

| configuration | nDCG@10 | vs base | report |
| --- | --- | --- | --- |
| **engine**, dense + lexical | 0.7264 | — | `eval/engine-beir/scifact-base.json` |
| **engine + learned sparse** | **0.7443** | **+0.0179 [+0.0028, +0.0337]**, p=0.025 | `eval/engine-beir/scifact-sparse.json` |
| `tools/beir.py` replica, 2-channel, **engine-matched** | 0.7408 | — | `eval/beir/engine-parity.json` |
| `tools/beir.py` replica, 3-channel, **engine-matched** | 0.7427 | +0.0019 | `eval/beir/engine-parity.json` |
| `tools/beir.py` replica, 2-channel, **title-weighted** | 0.7413 | +0.0005 | `eval/beir/five-small-fixed.json` |

**Which rows these are, stated exactly, because §4.3 of `SCORE-LIFT.md` requires it.** The engine
row is the real `RagEngine` at `BAAI/bge-small-en-v1.5`, thresholds at 0, `top_k=10`. The replica's
2-channel row is `hybrid-biencoder` — the same model, `rrf_k=5`, whole documents at depth 100 — and
the engine's lexical channel runs the engine's own settings (stemming on, query stopwords dropped),
which the harness reproduces as `bm25-engine`, **not** as its `bm25` control: the control is the
published Anserini configuration, `k1=0.9, b=0.4`, a different system from the shipped `1.2/0.75`.

The two replica rows are the **same retriever at two configurations**, and the difference is worth
naming because it is the one that is easy to miss:

- `eval/beir/engine-parity.json` is produced with **no `--title-boost`**, which is what the engine
  does: it indexes each document's title inline, once. That is the engine-matched comparator, and
  it is the row the overstatement is computed from.
- `eval/beir/five-small-fixed.json` and the `rrf-*` fusion sweep were run with **`--title-boost 2`**,
  repeating each title twice for the lexical channel — an experiment about the lexical channel, not
  a reproduction of the engine. Re-measured here at both settings:
  `title_boost=1, rrf_k=5` → **0.7408**; `title_boost=2, rrf_k=5` → **0.7413**;
  `title_boost=1, rrf_k=60` → **0.7320**; `title_boost=2, rrf_k=60` → **0.7310**. Every recorded
  value reproduces to four decimals.

Two things to take from this:

1. **The sparse channel's gain transfers, and is larger through the engine** (+0.0179
   significant on 300 queries, against **+0.0019** for the engine-matched replica — +0.0035 in the
   title-weighted one). The channel is doing real work on the path that actually serves requests,
   not only in the harness.
2. **The engine-matched replica overstates the engine by 0.0144** (0.7408 against 0.7264). The
   replica scores whole documents; the engine chunks at 1200/400 and its top-10 *passages* collapse
   to fewer distinct documents, because several chunks of the same file occupy the list. The
   project's BEIR rows are therefore an upper bound on what the engine returns, and should be read
   that way.

**A correction to a correction, recorded because the first one was mine.** Wiring this pair into
`make consistency-check` initially produced the claim that 0.7408 "reproduces from no report on
disk" and that the overstatement is +0.0149 — inferred from two reports reading 0.7413, without
noticing that they carry `--title-boost 2` and the parity run did not. Both numbers were right all
along; only the *configuration* differed. The lesson is the section's own subject: a BEIR row is
not a retriever, it is a retriever **plus its settings**, and comparing across an unstated knob is
exactly the harness/replica confusion §4.3 exists to prevent. The pin therefore reads
`eval/beir/engine-parity.json` — a report produced at the engine-matched configuration, so the
quoted row is one a reader can regenerate with the command in that report — and the delta is
computed from the two reports rather than typed.

**Two files both claim to be "the RRF-`k` sweep", and they are different configurations.**
`docs/findings-fusion-k.md` tabulates SciFact at **0.7408 (k=5) / 0.7320 (k=60)** — the un-boosted
pair, consistent with this page — and `eval/beir/rrf-{3,5,7,10,30,60,100}.json` hold **0.7413
(k=5) / 0.7310 (k=60)**, the `--title-boost 2` sweep. Both are right; neither report says which
title boost it used in its *prose*, though both carry it in `settings`. A reader comparing the two
without that column would see a 0.0005 disagreement that is not one. `docs/findings-fusion-k.md`
is not re-derived here — its deltas are internally consistent within its own configuration.

## What the BM25 divergence means for every other row

`tools/beir.py` runs `k1=0.9, b=0.4` for its `bm25` control (Anserini/Pyserini, the published
configuration); the shipped engine runs `k1=1.2, b=0.75` (Lucene's defaults). A `bm25-engine` row
is the engine's configuration reproduced in the harness, and a `bm25` row is the published
baseline reproduced in the harness — **neither is the engine itself**, which is the whole reason
this page exists. At the engine level the two settings are indistinguishable on SciFact
(0.7264 against 0.7238, p=0.45), so `BM25_K1`/`BM25_B` exist to reproduce a baseline exactly or to
tune per corpus, not because one is better.

## Honest limits

- **One dataset** (SciFact) and 300 queries; the engine run costs ~10 minutes per
  configuration because Laya's decisions run on every candidate, so the sweep is not free.
- Thresholds are set to 0 to isolate retrieval. That is deliberate — with the gate active,
  unanswerable queries would return nothing and confound the score — but it means this is not
  a measure of the full decision pipeline, only of what the engine retrieves before deciding.
- Document-level scoring collapses repeated chunks of one file. That is the honest unit for a
  BEIR comparison, but it is also why the engine's number is lower: the engine counts
  *passages*, not documents, against `top_k`.
- The gap is not attributed to a single cause here. Chunking, passage-level `top_k` and the
  engine's own encoder settings all differ from the replica; separating them is the next
  measurement, and `RERANK_COVERAGE_MIN_DOCS` / a document-aware `top_k` is the natural
  candidate to test with this harness.
