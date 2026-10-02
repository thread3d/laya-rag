# Findings: the BM25 parameters were not configurable, and do not matter here

The engine's lexical channel used Lucene's `k1=1.2, b=0.75` (the defaults in
`laya_rag.lexical`) while every BEIR row in this project ran Anserini's `k1=0.9, b=0.4`
through `tools/beir.py` — and the engine had **no way to set them**, so it could not
reproduce the harness rows it was compared against. That is the same class of
engine/harness mismatch as the fusion constant, so it was worth measuring rather than
asserting.

## The measurement

`BM25_K1` / `BM25_B` are now exposed, wired into all three lexical backends (in-memory,
eager, on-disk) and into `tools/engine_beir.py --k1/--b`. Engine-level SciFact, 300 queries,
`bge-small`, top-10:

| BM25 configuration | nDCG@10 |
| --- | --- |
| engine default `k1=1.2, b=0.75` | 0.7264 |
| Anserini/BEIR `k1=0.9, b=0.40` | 0.7238 |

Paired: **−0.0026 [−0.0093, +0.0039]**, p=0.45 — **indistinguishable**.

## Verdict

**No default change.** The mismatch is real but immaterial for the fused ranking on this
dataset, so the engine keeps its Lucene defaults and the new configuration exists to let
anyone reproduce the harness baseline exactly or tune per corpus. The audit is closed with
evidence instead of an assumption.

The rest of the lexical constants do match their harness counterparts: the engine stems by
default (`LEXICAL_STEM=true`, the harness's `stem=True` for the hybrid rows) and drops query
stopwords on both sides.

## Honest limits

- One dataset, and the comparison is of **fused** retrieval, where the lexical channel is one
  contributor among two or three — a small lexical difference is diluted. A lexical-only
  comparison could show a larger gap, which is why the parameters are now configurable rather
  than dismissed.
- SciFact queries are short and its documents short, so length normalisation (`b`) has little
  to work on; a corpus with long, variable documents is where `b` would matter most.
- These are engine-level numbers against the document-de-duplicated metric, the same protocol
  as `findings-engine-parity.md`, so they are not comparable to the harness table directly.
