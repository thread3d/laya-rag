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

Two CPU-cheap subsets, `bge-small` dense, no reranker, no generator:

| subset | docs | queries | retriever | nDCG@10 | nDCG@100 |
| --- | --- | --- | --- | --- | --- |
| pony | 7,894 | 112 | bm25 | 0.1221 | 0.2350 |
| pony | 7,894 | 112 | dense | 0.0398 | 0.1015 |
| pony | 7,894 | 112 | **hybrid** | **0.1281** | 0.2143 |
| theoremqa_theorems | 23,839 | 76 | bm25 | 0.0296 | 0.0437 |
| theoremqa_theorems | 23,839 | 76 | **dense** | **0.0579** | 0.0941 |
| theoremqa_theorems | 23,839 | 76 | hybrid | 0.0446 | 0.0889 |

Paired over queries (bootstrap 95% CI, nDCG@10):

| comparison | delta | 95% CI | verdict |
| --- | --- | --- | --- |
| pony dense vs bm25 | **−0.0823** | [−0.1119, −0.0555] | significant |
| pony hybrid vs bm25 | +0.0060 | [−0.0231, +0.0348] | tie |
| theoremqa hybrid vs bm25 | +0.0151 | [−0.0063, +0.0385] | tie |
| theoremqa dense vs bm25 | +0.0284 | [−0.0070, +0.0662] | tie (n=76) |

## Reading

- **Pony reproduces BRIGHT's premise exactly.** Dense-only retrieval is *significantly worse*
  than BM25 (−0.0823) on programming-language questions — the query asks for a program to be
  written, and the relevant document is a language reference that shares few surface terms
  with it. BM25's exact-term strength wins, and the fused hybrid matches it.
- **TheoremQA-Theorems reverses the order**: dense leads BM25 by +0.0284, but with 76 scored
  queries the interval crosses zero, so it is a tie, not a win.
- **The engine's hybrid is a safe default on both**: it is the best channel on pony and
  within noise of the best on theoremqa. That is the same "fusion is robust" pattern BEIR and
  MultiHop showed, now on a reasoning-intensive benchmark.
- The absolute numbers are far below the paper's ceiling because this is a 33M `bge-small`
  with no reranker on 2 of 12 subsets; they are honest retrieval rows, not leaderboard
  positions.

## Honest limits

- **Two of twelve subsets.** `leetcode` (414k docs), `aops` and `theoremqa_questions` (188k
  each) are excluded by default as too large for this host; the remaining StackExchange
  subsets (~50–120k each) were not run.
- The published top score uses much larger encoders and reranking; these rows are not
  comparable to it, only to themselves.
- BRIGHT queries are instruction-formatted; they are used raw. The paper's own guidance is
  that an instruction-tuned query encoder helps, which `bge-small` is not.
