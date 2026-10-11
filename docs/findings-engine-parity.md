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

| configuration | nDCG@10 | vs base |
| --- | --- | --- |
| **engine**, dense + lexical | 0.7264 | — |
| **engine + learned sparse** | **0.7443** | **+0.0179 [+0.0028, +0.0337]**, p=0.025 |
| `tools/beir.py` replica, 2-channel | 0.7413 | — |
| `tools/beir.py` replica, 3-channel | 0.7448 | +0.0035 |

**Which rows these are, stated exactly, because §4.3 of `SCORE-LIFT.md` requires it.** The engine
row is `eval/engine-beir/scifact-base.json` — the real `RagEngine`, `BAAI/bge-small-en-v1.5`,
thresholds at 0, `top_k=10`. The two replica rows are `hybrid-biencoder` and `hybrid-splade` in
`eval/beir/five-small-fixed.json` and `eval/beir/splade-small-fixed.json`: the same model, the same
`rrf_k=5`, scored over whole documents at depth 100. The engine's lexical channel runs the engine's
own settings (stemming on, query stopwords dropped), which the harness reproduces as
`bm25-engine`, **not** as its `bm25` control — the control is the published Anserini configuration,
`k1=0.9, b=0.4`, and it is a different system from the shipped `1.2/0.75`; quoting a fused row
against the control is not a like-for-like comparison.

Two things to take from this:

1. **The sparse channel's gain transfers, and is larger through the engine** (+0.0179
   significant on 300 queries, against +0.0035 for the replica). The channel is doing real
   work on the path that actually serves requests, not only in the harness.
2. **The replica overstates the engine by 0.0149** (0.7413 against 0.7264). The replica scores
   whole documents; the engine chunks at 1200/400 and its top-10 *passages* collapse to fewer
   distinct documents, because several chunks of the same file occupy the list. The project's
   BEIR rows are therefore an upper bound on what the engine returns, and should be read that
   way.

**This pair is now a pinned quantity, and the pinning found a drift.** An earlier draft of this
page reported the replica at **0.7408** and the overstatement at **+0.0144**. Neither reproduces:
two independent reports, `five-small-fixed.json` and `rrf-5.json`, both read **0.7413** for the
same retriever, model and `rrf_k`, so the delta is **+0.0149**. The figure had been hand-typed into
this page, `docs/FINDINGS.md`, `ROADMAP.md` and `SCORE-LIFT.md`, and it is load-bearing — §7.2
argues from it that an external harness is the only control on this one. `make consistency-check`
now pins all three numbers to the two reports, so a re-run of either moves the documents or fails
the gate rather than passing silently.

*Not swept here:* the same stale **0.7408** baseline appears in two ablation tables —
`docs/findings-fusion-k.md` §Results and `docs/findings-coverage-scope.md` — where it is the
reference arm of an RRF-`k` and a coverage-λ comparison. Those tables are not re-derived above
because their *other* column does not come from the same report, so re-reading the baseline alone
would change a delta without a matching treatment row. It is recorded here so the next pass at
either document knows the baseline is stale.

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
