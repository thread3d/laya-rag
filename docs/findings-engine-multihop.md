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
   to the BEIR replica, which overstated the engine by 0.0149 — so "a replica is not the system"
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

## Full set — 2,255 queries, and a defect it caught

The subset above is now backed by the whole answerable set, run against a complete index:

| metric | harness `hybrid` (2,255) | **engine (2,255)** | delta |
| --- | --- | --- | --- |
| MAP@10 | 0.2510 | **0.3009** | **+0.0499 [+0.0432, +0.0566]** |
| Hits@4 | 0.6860 | **0.7632** | **+0.0772 [+0.0590, +0.0949]** |
| Hits@10 | 0.8022 | **0.9082** | **+0.1060 [+0.0918, +0.1202]** |
| MRR@10 | 0.5396 | **0.5983** | **+0.0587 [+0.0451, +0.0724]** |

All four intervals exclude zero, and the full-set gain (+0.0499) matches the 400-query subset's
(+0.0535) — **the result replicates**. The engine's MAP@10 is now 0.3009 on the full protocol
without a cross-encoder, against the harness's 0.2510 and the coverage-plus-quota rule's 0.3192.

**It did not replicate the first time.** The first full run reported MAP@10 **0.2216**, *below* the
harness, which would have overturned this finding. The cause was not sampling: the two indexes'
dense vectors are **byte-identical** (max abs diff `0.0`) and `index/engine-hybrid/` was missing
`lexical.json`, because the run that built it was killed mid-save and the re-run reused it.
`RagEngine.has_index()` accepted the dense half as complete, so every query silently degraded to
dense-only recall. That is now a fixed engine bug with a test
(`tests/test_index_completeness.py`), and it is why the full-set number had to be re-run with
`--fresh`: jobkit's fingerprint covers a run's *settings*, not the index it was measured against.

## Shipped default — the engine's best configuration

Full 2,255 queries with no `--embed-model`: Laya's encoder resolves to a dense weight of zero, so
recall is lexical only and there is no cross-encoder. This is what the engine actually ships.

| metric | harness `hybrid` | harness `bm25-engine` | engine hybrid | **engine default** |
| --- | --- | --- | --- | --- |
| MAP@10 | 0.2510 | 0.2520 | 0.3009 | **0.3128** |
| Hits@4 | 0.6860 | 0.6368 | **0.7632** | 0.7570 |
| Hits@10 | 0.8022 | 0.7947 | **0.9082** | 0.9011 |
| MRR@10 | 0.5396 | 0.5507 | 0.5983 | **0.6170** |

Paired: against the harness hybrid, MAP@10 **+0.0618 [+0.0547, +0.0688]**, Hits@4 +0.0710, Hits@10
+0.0989, MRR@10 +0.0775; against the harness's stemmed lexical row, MAP@10 **+0.0608**, Hits@4
+0.1202, Hits@10 +0.1064, MRR@10 +0.0664 — every interval excluding zero.

Paired against the hybrid on the same queries, the dense channel is a **trade, not a gain**:

| metric | hybrid → default | delta | 95% interval | p |
| --- | --- | --- | --- | --- |
| MAP@10 | 0.3009 → 0.3128 | **+0.0119** | [+0.0061, +0.0180] | 0.0002 |
| MRR@10 | 0.5983 → 0.6170 | **+0.0187** | [+0.0062, +0.0316] | 0.0032 |
| Hits@4 | 0.7632 → 0.7570 | −0.0062 | [−0.0213, +0.0093] | 0.44 (ns) |
| Hits@10 | 0.9082 → 0.9011 | −0.0071 | [−0.0182, +0.0040] | 0.22 (ns) |

So turning the dense channel on buys a little breadth at the top of the list and costs set
coverage and ranking: **the shipped default is the best MAP@10 configuration the engine has on
this corpus (0.3128)**, and it is the one that ships. It remains below the harness's
coverage-plus-quota rule (0.3192) — the selection programme's target survives, unchanged.

## What this does not show

- **Not that the engine's model is better.** The delta bundles the chunking change with the
  engine's channel wiring; the engine did not use a cross-encoder, coverage or a document quota
  here, so this is a statement about the *system under its own defaults* versus a replica of it.
- **Not a recall claim.** MAP@10 is set-coverage sensitive; the pool-reorder ceiling and fact
  recall diagnostics in [findings-multihop-diagnostics.md](findings-multihop-diagnostics.md) still
  describe where the remaining headroom is.
- **One chunking, corpus-aware sizing only.** The hybrid rows use `bge-small`; the shipped-default
  row uses Laya's encoder with a dense weight of zero. Fixed-size chunking ablations are separate
  fingerprints and separate runs.

## Method notes

- **Pair, do not compare summaries.** The two reports are only comparable query by query;
  `tools/engine_vs_harness.py` re-derives the subset from `MultiHopRAG.json` and the seed
  recorded in the engine report's `_settings`, and refuses to compare if the keys do not match.
- **Record the effective chunk size.** The engine derives 2,000 characters from this corpus
  (p90, clamped); the harness uses 256 *tokens*. Two thirds of the delta is a representation
  difference, and the report writes `chunk_chars_effective` beside every number.
- **The engine's per-query rows are resumable.** 3.1 s/query makes the full run a jobkit
  checkpoint, so an interrupted run resumes rather than restarting — see `ROADMAP.md` §2.
