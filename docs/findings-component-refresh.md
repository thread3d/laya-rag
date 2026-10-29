# R1 — refreshing the retrieval components: the harness first, then the encoder

`docs/best-in-the-world-2026-10.md` R1 asks whether a 2026 encoder changes the repository's flagship
finding — that the dense channel never earned its weight — when the measurement runs under this
repository's own protocol. This is the first half: what had to be true before such a row could be
believed, and the control that shows it now is.

## 1. The control reproduces the published rows to four decimals

Before any new encoder is scored, the harness has to reproduce a row that is already recorded, on this
host, with a different ML stack — otherwise a "gain" could be the stack. SciFact, all four retrievers,
`eval/beir/r1-scifact-control.json`:

| retriever | this run | `RANKINGS.md` recorded | Δ |
| --- | ---: | ---: | ---: |
| `bm25` (published settings) | **0.6539** | 0.654 | 0.000 |
| `bm25-engine` (stemmed) | **0.6762** | 0.676 | 0.000 |
| `biencoder` (`bge-small`) | **0.7200** | 0.720 | 0.000 |
| `hybrid-biencoder` | **0.7320** | 0.732 | 0.000 |

All four match at the precision the project prints. The stack differs from the one that produced the
originals (torch 2.15 CPU, `sentence-transformers` 4.1, numpy 2.5 against a pinned 2.2.2 / numpy 1.x
environment), the corpus was downloaded fresh, and the numbers did not move — which is the property
that makes the next row comparable rather than merely higher. It took 183 s to embed 5,183 documents
with the 33M encoder, i.e. **28 documents/s on CPU**; that rate is the baseline every 2026 candidate is
priced against.

## 2. Two silent mismatches had to be fixed first

The harness could not have measured a 2026 decoder-class encoder *correctly*, and neither defect
raises: both produce a number that reads as a fact about the model.

**Pooling.** `tools/beir.py` pooled `cls` or `mean` only. Qwen3-Embedding (and the gte and e5-mistral
family) are trained on the **last token**, so a Qwen3 run would have been embedded under a pooling it
was never trained for. The engine had already learned this — `src/laya_rag/model.py` carries a family
table whose entries were each verified by loading the model — but the harness carried its own "CLS if
the name contains `bge`, else mean" heuristic, and the two disagreed. The harness now has a `last`
mode (which takes the last token *with attention 1*, not position `-1`, because batches pad on the
right) and delegates the name heuristic to `laya_rag.model.family_pooling`:

```
BAAI/bge-small-en-v1.5   -> cls
Qwen/Qwen3-Embedding-0.6B -> last
intfloat/e5-base-v2       -> mean
unknown/model             -> mean      (the historical default, not a guess)
```

**Query prefix.** `--query-instruction` applied one hardcoded string — bge's *"Represent this sentence
for searching relevant passages: "* — to every family. Qwen3 wants an `Instruct: … Query: ` pair and e5
a bare `query: ` token; applying one family's line to another model is the same class of silent
mismatch as the wrong pooling, and it is the third on this project's list after the pooling and the
passage prefix. The harness now asks `laya_rag.model.family_query_prefix`, and **bge falls back to its
historical string**, so no previously recorded row moves — which is the property the test pins.

Both are covered by `tests/test_beir_passages.py` against the real functions (`pool_hidden`, the
`BiEncoder` constructor) rather than a re-implementation of them.

## 3. A hazard worth recording: a fresh report breaks the consistency gate

The first control run was written to `eval/beir/`, the canonical tree. `make consistency-check` then
failed on every "of 18" claim in the repository — correctly, because its authority is the reports on
disk and one report is one dataset. The gate was doing its job; the run was in the wrong place.

Ad-hoc measurement therefore writes to its own root (`eval/r1/…`) and leaves `eval/beir/` to the
canonical programme. The general rule this establishes: **anything that adds a report to the canonical
tree has to regenerate the documents that quote it in the same commit.** That is exactly what the gate
exists to enforce, and it is worth knowing before the first 2026 row lands.

## 4. What is running now, and what would settle it

The second half is the row itself: Qwen3-Embedding-0.6B (Apache-2.0, `last` pooling, its own query
instruction) over the same SciFact protocol, scored by `tools/beir.py` and then — because a replica is
not the system — by `tools/engine_beir.py`. It embeds at roughly a twentieth of `bge-small`'s rate, so
5,183 documents is about an hour of CPU per run, and both halves are single heavy jobs by this
repository's rule that only one runs at a time.

The prediction and its falsifier are in the assessment (R1): fused nDCG@10 **≥ +0.03** over the 0.452
single-config mean, and the dense channel re-earning a non-zero weight on at least three of five
datasets; falsified if the fused row does not beat lexical-only by ≥0.02 on ≥3 of 5 at matched
chunking. The arithmetic it rests on: Qwen3-Embedding-0.6B reads **0.5552** on MTEB's BEIR-15 board
against `bge-large`'s 0.5434 and BM25's 0.3984, and discounting this repository's best-per-dataset
inflation leaves +0.03 to +0.05 from the encoder alone.

## What this does not claim

- It does not claim a 2026 encoder wins here. The control shows the harness is now a valid instrument
  and that a decoder-class model can be measured under its own protocol; it says nothing about the
  score.
- It does not claim the pooling and prefix fixes *raise* any recorded number: bge is outside both
  family tables, so its rows are unchanged by construction, and the control above is the evidence.
- The 28 documents/s figure is one host, one model, one corpus, warm cache, `OMP_NUM_THREADS=32` — the
  same caveat that applies to every timing in this repository.
