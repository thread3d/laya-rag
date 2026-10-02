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
| `tools/beir.py` replica, 2-channel | 0.7408 | — |
| `tools/beir.py` replica, 3-channel | 0.7446 | +0.0038 |

Two things to take from this:

1. **The sparse channel's gain transfers, and is larger through the engine** (+0.0179
   significant on 300 queries, against +0.0038 for the replica). The channel is doing real
   work on the path that actually serves requests, not only in the harness.
2. **The replica overstates the engine by 0.0144** (0.7408 against 0.7264). The replica scores
   whole documents; the engine chunks at 1200/400 and its top-10 *passages* collapse to fewer
   distinct documents, because several chunks of the same file occupy the list. The project's
   BEIR rows are therefore an upper bound on what the engine returns, and should be read that
   way.

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
