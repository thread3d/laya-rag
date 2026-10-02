# Findings: the chunking loss is the chunked representation, not its selection

Round 29's engine-level run put the engine 0.0144 below the whole-document replica. SciFact's
median document is **1,426 characters** against a 1,200-character chunk size, so most
abstracts were split into two heavily overlapping chunks — a plausible cause. This tests four
responses to it at the engine level, same 300 queries, `bge-small`, top-10. **None of the
three targeted fixes recovers the whole-document score**; the conclusion is that the loss is
in the chunked representation itself.

| configuration | chunks | nDCG@10 | vs base |
| --- | --- | --- | --- |
| default chunking, 1200/400 | 9,986 | 0.7264 | — |
| `CHUNK_MERGE_SHORT` — documents ≤ 1600 chars kept whole | 8,470 | 0.7236 | −0.0028 [−0.0130, +0.0070], p=0.59 |
| `RETRIEVE_K=40` — double the candidate pool | 9,986 | 0.7273 | +0.0009 [−0.0031, +0.0048], p=0.66 |
| `CONTEXT_PREFIX` — breadcrumb (path + title line) on every chunk | 9,986 | 0.7243 | −0.0021 [−0.0178, +0.0128], p=0.79 |
| whole-document — one chunk per abstract | 5,183 | **0.7396** | **+0.0132 [−0.0002, +0.0272]**, p=0.058 |

Reproduce:

```sh
make engine-beir ENGINE_BEIR_ARGS="--dataset scifact --tag merge --chunk-merge-short"
make engine-beir ENGINE_BEIR_ARGS="--dataset scifact --tag rk40 --retrieve-k 40"
make engine-beir ENGINE_BEIR_ARGS="--dataset scifact --tag ctxpre --context-prefix"
make engine-beir ENGINE_BEIR_ARGS="--dataset scifact --tag whole --chunk-chars 100000 --chunk-overlap 0"
```

## Three plausible fixes, none of which works

Whole-document beat the merge rule **significantly**: **+0.0160 [+0.0025, +0.0304]**, p=0.016.
Each remaining explanation was then tested directly, and all three were rejected:

| hypothesis | test | result |
| --- | --- | --- |
| merging short documents back recovers it | `CHUNK_MERGE_SHORT` (62% of the corpus) | −0.0028, p=0.59 — no |
| the pool covers too few **documents** under chunking | `RETRIEVE_K=40` (20 → 40 candidates) | +0.0009, p=0.66 — no |
| later chunks lose the document's **title/context** | `CONTEXT_PREFIX` on every chunk | −0.0021, p=0.79 — no |

So the loss is not pool depth, not the short-document split, and not missing context. It is the
**chunked representation itself**: for SciFact's abstracts a single embedding over the whole
document matches the claim better than the embedding of any of its chunks, and the
document-level ranking takes a document's best chunk as its score, so deeper pools cannot
repair it. That points at chunk *size* rather than chunk *selection* — and at a corpus-shape
decision the engine cannot make for an unknown corpus.

Note the bisection that remains: `merge_short` covered 1,200–1,600 characters and changed
nothing, so the gain must sit in the documents **longer than 1,600**. `--chunk-chars 2000`
would locate the band; it is left as the next measurement rather than run to exhaustion here.

## What shipped

`chunk_text` / `chunk_documents` / `RagEngine` gained a `merge_short` knob
(`CHUNK_MERGE_SHORT`, default **false**), and `engine_beir` gained `--chunk-merge-short`,
`--retrieve-k` and `--context-prefix` so each hypothesis above is one command. The knob is
documented as a **negative**; no default changed.

## The fix: derive the chunk size from the corpus

If the loss is the chunked representation, the response is not another constant — it is to
choose the size from the corpus. `derive_chunk_chars` takes the **90th percentile** of the
document lengths, clamped into `[CHUNK_CHARS, CHUNK_CHARS_AUTO_MAX=2000]`, so it can only ever
chunk *less* aggressively than before and never exceeds what a 512-token encoder can see.
SciFact's p90 is above the clamp, so it derives 2,000 — one chunk for 86% of abstracts.

| configuration | chunks | nDCG@10 | vs base |
| --- | --- | --- | --- |
| fixed 1200/400 | 9,986 | 0.7264 | — |
| **corpus-aware (`CHUNK_CHARS_AUTO`)** | 5,976 | **0.7417** | **+0.0153 [+0.0025, +0.0293]**, p=0.023 |
| whole-document (upper bound) | 5,183 | 0.7396 | +0.0132, p=0.058 |

The derived size is **significantly better** than the fixed default and statistically
indistinguishable from whole-document chunking (+0.0021, p=0.54) — it captures the whole
gain the fixed constant was leaving behind, with 40% fewer chunks.

The project's own corpus agrees, measured on the vendored Laya docs with the same lexical
harness (`tools/context_prefix_eval.py`):

| chunk size | chunks | MRR (no prefix) | MRR (context prefix) |
| --- | --- | --- | --- |
| fixed 1200 | 459 | 0.6429 | 0.8333 |
| derived 2000 | 220 | **0.7143** | **0.8929** |

Recall stays 1.000 and the worst rank stays 6 in both, so the gain is in where the right
document lands, not in whether it is found.

**Shipped: `CHUNK_CHARS_AUTO` now defaults to `true`.** Two corpora, one short-document and
one long-document, both improve, and the derived value can never be smaller than the old
default. `CHUNK_CHARS` remains the floor and setting `CHUNK_CHARS_AUTO=false` restores the
fixed behaviour exactly.

### Compatibility: retrain a persisted answerability gate

The gate reads evidence *shape* — `passage_count`, `log_evidence_chars`,
`max_passage_coverage`, `coverage_spread`, `pairwise_redundancy` — and all five move when
chunking does: larger chunks mean fewer, longer passages with different redundancy and
coverage. A gate trained against a fixed-1200 index is therefore out of distribution under the
new default even though its feature names still load. **Retrain it**
(`tools/answerability_train.py`) after changing `CHUNK_CHARS_AUTO` or `CHUNK_CHARS`; nothing in
the loader can detect the drift, which is why it is written down here.

That prediction was then **measured** (see
[findings-gate-shape-features.md](findings-gate-shape-features.md)): re-collecting the gate's
300 labelled queries through a corpus-aware index drops the nine-score-feature gate from AUC
0.8146 to 0.7862 and raises its risk@80 from 0.4069 to 0.4295, while the shipped twelve-feature
gate holds (0.8290 → 0.8323). The evidence-shape features are what carries it across the
change.

### The percentile is not critical (but p90 is the best of those tried)

SciFact's derivation by quantile, and the engine score each produces:

| quantile | derived size | single-chunk coverage | nDCG@10 | vs fixed 1200 |
| --- | --- | --- | --- | --- |
| — (fixed) | 1200 (+400 overlap) | — | 0.7264 | — |
| p50 | 1426 | 50% | — | not run |
| p75 | 1811 | 75% | 0.7376 | +0.0112 [−0.0015, +0.0241], p=0.079 |
| **p90 (default)** | **2000** (clamped) | **86%** | **0.7417** | **+0.0153 [+0.0025, +0.0293]**, p=0.023 |
| p95 | 2000 | 86% | — | same as p90 by the clamp |

p90 beats p75 by **−0.0041 [−0.0121, +0.0031]** (p=0.28, not significant), so the exact
quantile is not a knife-edge: what matters is deriving a size that keeps most documents whole.
p90 is kept because it is the best of the two measured and its gain over the fixed default is
the significant one. The clamp makes p90 and p95 identical here, so there is nothing above p90
left to tune on this corpus.

## Honest limits and the next experiment

- Two corpora, both improved (SciFact +0.0153, p=0.023; the vendored docs MRR +0.0714), but
  the SciFact whole-document delta on its own was borderline (p=0.058) — the significant
  comparisons are the corpus-aware size against the fixed default, and whole-document against
  the merge rule (+0.0160, p=0.016).
- The bisection is unfinished. `merge_short` covered 1,200–1,600 characters and changed
  nothing, so the gain sits in documents **longer than 1,600**; `--chunk-chars 2000` would
  locate the band precisely. The shipped rule reaches the same place by a different route
  (it derives 2,000 for SciFact), so the bisection is now diagnostic rather than blocking.
- The **percentile and the 2,000-character clamp are heuristic**: 0.9 and 2,000 were chosen to
  keep most documents whole within a 512-token window. They were then swept on SciFact — p75
  is 0.0041 behind p90 (not significant) and p90/p95 are identical under the clamp — so the
  choice is robust to the quantile but still rests on one short-document corpus for the clamp
  itself.
- Long-document corpora now index at 2,000 rather than 1,200 characters. The vendored docs
  improved, but a corpus whose queries need *precise* local matches could prefer the smaller
  chunks; `CHUNK_CHARS_AUTO=false` is the escape hatch and the per-corpus answer is measurable
  with the same harnesses.
- The docs-corpus table above changes the baseline used by
  [findings-context-prefix.md](findings-context-prefix.md) (0.6429 → 0.8333 was measured at
  1,200; at 2,000 it is 0.7143 → 0.8929). The prefix still helps by a similar margin.
