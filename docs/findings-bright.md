# Findings: BRIGHT — reasoning-intensive retrieval, measured retrieval-only

BRIGHT is the benchmark that asks whether a retriever can find documents that require
*reasoning* to connect to the query, where the authors report that BM25 is a strong baseline
and the best published nDCG@10 is only about 21 ([BRIGHT dataset
documentation](https://github.com/xlang-ai/BRIGHT/blob/main/Dataset_documentation.md),
[paper](https://arxiv.org/abs/2407.12883)). Its official score is retrieval-only — official
gold ids, no generator — so it is one of the benchmarks the engine can legitimately enter.

Reproduce (downloads on first run, then offline):

```sh
make bright BRIGHT_ARGS="--subsets pony,theoremqa_theorems"
```

## Measured

Six CPU-cheap subsets, `bge-small` dense, no reranker, no generator:

| subset | docs | queries | bm25 | dense | hybrid |
| --- | --- | --- | --- | --- | --- |
| pony | 7,894 | 112 | 0.1221 | 0.0398 | **0.1281** |
| theoremqa_theorems | 23,839 | 76 | 0.0296 | **0.0579** | 0.0446 |
| biology | 57,359 | 103 | **0.2089** | 0.1076 | 0.1789 |
| economics | 50,220 | 103 | 0.1519 | 0.1321 | **0.1783** |
| psychology | 52,835 | 101 | 0.1282 | 0.1487 | **0.1786** |
| sustainable_living | 60,792 | 108 | 0.1535 | 0.1082 | 0.1529 |
| **macro mean** | — | 603 | 0.1324 | 0.0991 | **0.1436** |

Paired over queries (bootstrap 95% CI, nDCG@10):

| comparison | delta | 95% CI | verdict |
| --- | --- | --- | --- |
| pony dense vs bm25 | **−0.0823** | [−0.1119, −0.0555] | significant |
| biology dense vs bm25 | **−0.1013** | [−0.1553, −0.0516] | significant |
| psychology hybrid vs bm25 | **+0.0504** | [+0.0148, +0.0862] | significant |
| economics hybrid vs bm25 | +0.0265 | [−0.0035, +0.0561] | tie |
| biology hybrid vs bm25 | −0.0300 | [−0.0733, +0.0093] | tie |
| sustainable_living hybrid vs bm25 | −0.0006 | [−0.0371, +0.0351] | tie |
| theoremqa hybrid vs bm25 | +0.0151 | [−0.0063, +0.0385] | tie |

## Reading

- **The benchmark's premise reproduces.** Dense-only retrieval is *significantly worse* than
  BM25 on pony (−0.0823) and biology (−0.1013) — the queries ask for programs or infer
  relationships, and the relevant documents share few surface terms with them. Dense also
  has the lowest macro mean (0.0991 against BM25's 0.1324).
- **The fused hybrid has the best macro mean (0.1436)**, above BM25 by +0.0112 and dense by
  +0.0445, and it is the best channel on three of six subsets. Per dataset it is only
  *significantly* better than BM25 on psychology (+0.0504); elsewhere the intervals cross
  zero, which is the honest reading on 76–112 queries per subset.
- **Biology is the one place BM25 leads the hybrid** (0.2089 vs 0.1789), the same pattern as
  pony: exact terminology dominates, and fusion dilutes it slightly.
- **This is the same "fusion is robust, dense alone is not" pattern** BEIR and MultiHop
  showed, now on the reasoning-intensive benchmark, and it is the strongest argument for the
  engine shipping fusion rather than a single channel.
- The absolute numbers are far below the paper's ~21 ceiling because this is a 33M
  `bge-small` with no reranker; they are honest retrieval rows, not leaderboard positions.

## Honest limits

- **Six of twelve subsets.** `leetcode` (414k docs), `aops` and `theoremqa_questions` (188k
  each) are excluded as too large for this host; `stackoverflow` (107k) and the remaining
  three StackExchange subsets were not run.
- The published top score uses much larger encoders and reranking; these rows are not
  comparable to it, only to themselves.
- BRIGHT queries are instruction-formatted; they are used raw. The paper's own guidance is
  that an instruction-tuned query encoder helps, which `bge-small` is not.
- Per-subset query counts (76–112) make all but the largest deltas ties; the macro mean is
  reported for breadth, not as a significance claim.
