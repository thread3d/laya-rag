# Findings: the engine beats the 256-token harness on MultiHop-RAG

This closes the gap [`FINDINGS.md`](FINDINGS.md) called the largest in the evidence: every
MultiHop-RAG number in the project came from `tools/multihop_official.py`, a harness that
reimplements the channels at the benchmark's own 256-token chunking. `tools/engine_multihop.py`
runs the real `RagEngine` instead, with the decision thresholds at zero so nothing is filtered
and the returned order is the retrieval order, and scores the benchmark's metrics on the chunks
the engine actually returns.

Reproduce:

```sh
make engine-multihop ENGINE_MULTIHOP_ARGS="--limit 400 --sample-seed 17 \
    --embed-model BAAI/bge-small-en-v1.5 --out eval/benchmarks/multihop-rag/engine-hybrid-400"
.venv/bin/python tools/engine_vs_harness.py \
    --engine eval/benchmarks/multihop-rag/engine-hybrid-400.json \
    --arm "bge-small hybrid@ov200"
```

The subset is not a convenience: `--limit 400 --sample-seed 17` reproduces
`tools/multihop_official.py`'s sample exactly (asserted by `tools/engine_vs_harness.py`), so the
engine lands on the same 400 questions as the harness's bar.

## Result

Engine: shipped `RagEngine`, corpus-aware chunk size (p90 → **2,000 characters**, 4,592 chunks
from 609 articles), `bge-small` dense + stemmed lexical fused, **no cross-encoder**, thresholds
0. Paired over the 400 queries, harness as base:

| metric | harness `hybrid` | harness `+rerank` | **engine** | engine − hybrid | engine − rerank |
| --- | --- | --- | --- | --- | --- |
| MAP@10 | 0.2522 | 0.2986 | **0.3057** | **+0.0535 [+0.0387, +0.0680]** | +0.0071 [−0.0112, +0.0251] (ns) |
| Hits@4 | 0.6625 | 0.7225 | **0.7525** | **+0.0900 [+0.0499, +0.1325]** | +0.0300 [−0.0150, +0.0750] (ns) |
| Hits@10 | 0.7975 | 0.8600 | **0.9075** | **+0.1100 [+0.0775, +0.1425]** | **+0.0475 [+0.0125, +0.0800]** |
| MRR@10 | 0.5342 | 0.6010 | **0.5901** | **+0.0559 [+0.0260, +0.0847]** | −0.0109 [−0.0500, +0.0268] (ns) |

Three readings:

1. **The replica understates the engine, significantly, on every metric.** +0.0535 MAP@10 over
   the harness's own first stage, all four intervals excluding zero. This is the *opposite* sign
   to the BEIR replica, which overstated the engine by 0.0144 — so "a replica is not the system"
   cuts both ways, and only engine-level measurement settles which way.
2. **The engine's own chunking captures the whole cross-encoder gain on this subset.** Against
   the harness's reranked row it is statistically indistinguishable on MAP@10, Hits@4 and
   MRR@10, and significantly better on Hits@10. The reranker was the project's largest single
   measured lever (+0.0464 over the harness first stage); that lever is now explained as
   recovering part of what the harness's 256-token chunking cost, which is consistent with the
   corpus-aware chunking finding on BEIR (**+0.0153** nDCG@10, 40% fewer chunks).
3. **The selection programme still has work.** The engine's 0.3057 sits below the subset's
   coverage-plus-quota bar of **0.3280**, so the multi-hop MAP@10 gap the roadmap's Phase 2
   targets is real at the engine level too — it is not an artefact of the harness.

## What this does not show

- **Not the full set.** 400 of 2,255 queries (seed 17). The full engine run is ~2 hours
  (3.1 s/query measured) and is the next long job; `make job-status` tracks it. The subset is the
  same one the project's own bar is quoted on, so the comparison is like-for-like, but the
  headline number is still a 400-query number.
- **Not that the engine's model is better.** The delta bundles the chunking change with the
  engine's channel wiring; the engine did not use a cross-encoder, coverage or a document quota
  here, so this is a statement about the *system under its own defaults* versus a replica of it.
- **Not a recall claim.** MAP@10 is set-coverage sensitive; the pool-reorder ceiling and fact
  recall diagnostics in [findings-multihop-diagnostics.md](findings-multihop-diagnostics.md) still
  describe where the remaining headroom is.
- **One chunking, one encoder.** `bge-small` only, corpus-aware sizing only. The shipped default
  (Laya's encoder, dense weight 0) and fixed-size ablations are separate fingerprints and separate
  runs.

## Method notes

- **Pair, do not compare summaries.** The two reports are only comparable query by query;
  `tools/engine_vs_harness.py` re-derives the subset from `MultiHopRAG.json` and the seed
  recorded in the engine report's `_settings`, and refuses to compare if the keys do not match.
- **Record the effective chunk size.** The engine derives 2,000 characters from this corpus
  (p90, clamped); the harness uses 256 *tokens*. Two thirds of the delta is a representation
  difference, and the report writes `chunk_chars_effective` beside every number.
- **The engine's per-query rows are resumable.** 3.1 s/query makes the full run a jobkit
  checkpoint, so an interrupted run resumes rather than restarting — see `ROADMAP.md` §2.
