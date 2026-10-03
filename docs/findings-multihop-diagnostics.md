# Findings: the multi-hop MAP@10 gap is recall *and* selection

This records the Tier 0.1 diagnostic from [`RESEARCH.md`](../RESEARCH.md) §3.1, run over
the full MultiHop-RAG answerable set (2,255 queries) on the shipped `bge-small` + BM25
fused pipeline at the official `overlap=200` chunking. Deterministic and offline; the raw
report is the gitignored `eval/benchmarks/multihop-rag/diagnostics.json`, and the numbers
below are pinned to it by `make claim-check`.

Reproduce:

```sh
make multihop-diagnostics
```

## Headline

| depth | pool fact recall | ceiling MAP@10 | hop-2 recall | hop-2 retention |
| --- | --- | --- | --- | --- |
| 10 | 0.4747 | **0.3970** | 0.2407 | 0.2407 |
| 100 | 0.8666 | **0.6088** | 0.7189 | 0.7189 |
| 500 | 1.0000 | **0.6592** | 1.0000 | 1.0000 |

Achieved MAP@10 of the fused order: **0.2557** (Hits@10 0.8129, Hits@4 0.6847).

**Widening the pool raises the ceiling, and part of the gap is recall.** Paired over queries,
depth 10 → 100 is **+0.2118 [+0.2037, +0.2203]** and 100 → 500 is **+0.0505 [+0.0468,
+0.0544]**, both significant. **31.9% of depth-100 pools (719 of 2,255) are missing at least
one gold fact**; those queries have a depth-100 ceiling of **0.4469** against **0.6845** for
the complete ones, and the fused order reaches 0.1498 on them against 0.3052 on the rest.
Making every pool complete would raise the depth-100 ceiling from **0.6088** to about
**0.6845** — **+0.076**, roughly a quarter of the 0.2896 between the best achieved rule
(0.3192) and that ceiling. The practical reading is therefore two-sided: ordering is the
larger lever, and recall is a live one as well.

The headroom decomposes into four measured parts. The first three come from
[`ceiling-decomp.md`](../../eval/benchmarks/multihop-rag/ceiling-decomp.md), which scores the
cross-encoder's pool rather than the fused pool (0.6072 against this report's 0.6088 — close,
but a different pool, so the parts are complementary and must not be added naively):

| headroom | size | what reaches it |
| --- | --- | --- |
| ordering the same CE top-10 | **+0.1254** | a set-aware ordering rule |
| selecting a better subset of a deeper list | **+0.0634** | a better selector over the CE's top-20 |
| new signal (full pool vs CE top-20) | **+0.0992** | a better scorer |
| completing the depth-100 pool | **+0.076** | better recall |

> **Correction, 2026-10-03 — this diagnostic's conclusion was sign-flipped, and the fix is now
> pinned.** The ceiling used to be reported as 0.7321 / 0.7304 / 0.6741, apparently flat and
> then falling, and the headline read *"the gap is selection, not recall"*. The metric divided
> the oracle AP by the number of gold facts **the pool happened to match**, while every rule it
> is compared against is scored by the official metric, which divides by the query's true gold
> count. The two agree only when the pool is complete, so the error inflated the ceiling on
> exactly the queries whose recall was failing. `pool_ceiling_map` now takes `gold_count` as a
> **required keyword with no default**, because the only obvious default is the wrong one, and a
> test pins the difference. Two further figures from that correction commit were wrong and are
> corrected here: the depth-500 value is **0.6592**, not the 0.6741 that the commit message
> called "already correct"; and "811 queries / 36%" was never derived from a committed report —
> the regenerated `diagnostics.json` gives **719 of 2,255 (31.9%)**, with ceilings 0.4469 and
> 0.6845. `tools/claim_check.py` now resolves every one of these against the report, so a
> hand-copied figure cannot drift silently again.

## The second hop carries the tax

The diagnostic splits each query's gold facts by *grounding* — the maximum
stopword-filtered stemmed query-term overlap over the pool candidates containing the fact.
Facts at the strict minimum grounding are the second hop (linked to the query only through
a bridge entity); 2,082 of 2,255 queries have a real relevance gap. Their recall:

- first hop: **0.6411** at depth 10, 0.9498 at 100;
- second hop: **0.2407** at depth 10, 0.7189 at 100.

The top-10 budget keeps only **24%** of the reachable second hop against 64% of the first.
The paired jump from depth 10 to 100 is +0.4781 [+0.4569, +0.4995] for the second hop
(and it continues to +0.2811 [+0.2622, +0.3004] from 100 to 500), while the first hop is
already at 0.95 by depth 100. This is the "second-hop tax" of
[redhop 2025](https://github.com/vysakh0/redhop/blob/main/docs/findings/SECOND_HOP_TAX.md)
reproduced on this pipeline, with the same shape: the low-relevance hop is what the
rank budget discards.

## What this implies for the programme

1. **Ordering and selection are the larger levers; recall is live but not cheap.** The
   decomposition puts **+0.1254** in ordering the *same* cross-encoder top-10 and **+0.0634**
   in selecting a better subset, against **+0.076** from completing every fused pool. A
   set-aware ordering rule remains the first target; "retrieve more" is no longer dismissed,
   but the measured hop-2 term-mining proxy is not the way to do it (next point).
2. **Depth expansion is back on the table; the entity proxy for it is exhausted.** The ceiling
   rises with depth, yet `tools/multihop_hop2.py` — mining rare terms of the best-ranked
   candidates as extra BM25 probes — adds only **+0.0044** pool fact recall, from **32** of
   2,255 queries (none worsened). Significant and marginal: the direction is supported, but the
   term proxy recovers about one incomplete pool in twenty-two, which is an argument for a
   model-based hop-2 rather than for a bigger candidate pool alone.
3. **The second hop is the target.** Any selection rule worth shipping should raise hop-2
   retention at a fixed top-10. The bridge-linked rescue in `RESEARCH.md` §3.1 Tier 1.1
   is aimed exactly there, and this diagnostic gives the metric to falsify it with
   (`hop2_fact_recall@10`, `hop2_retention@10`).
4. **The cross-encoder is not obviously the answer.** It raised MAP@10 to 0.2986 on the
   400-query subset — real, but far below the 0.61 ceiling. The experiments in
   `RESEARCH.md` §4 test whether blending, a rank-1 guardrail, or CE-plus-coverage closes more
   of it; the coverage-plus-quota rule is the best measured so far at **0.3192** on the full
   protocol.

## Partial credit: fact recall, which MAP hides

MAP@10 punishes a set that found two of three facts at `2/3`; a caller cares more about *how
much* of the answer set arrived. The coverage harness now reports `fact_recall@10` — the
fraction of a query's gold facts present in the returned top-10 — alongside MAP. On the full
2,255 queries:

| strategy | fact recall@10 | MAP@10 | Hits@4 |
| --- | --- | --- | --- |
| fused baseline | 0.4644 | 0.2546 | 0.6825 |
| cluster-dedup-0.9 | 0.5458 | 0.2790 | 0.7118 |
| facility-0.5 | 0.5314 | 0.2835 | 0.7197 |
| **facility-nored-0.5** (shipped) | **0.5524** | **0.2850** | **0.7326** |

The shipped coverage configuration covers **55.2%** of gold facts against the fused
baseline's 46.4% — **+0.0880 [+0.0797, +0.0964], p < 0.0001** — which is a larger relative
statement than MAP's +0.0304 and is the one that matches what a reader experiences: the
evidence set is more complete, not merely better ordered. It also shows the coverage pass
beats plain near-duplicate suppression on partial credit too (0.5524 vs 0.5458).

## Method note

Grounding and hop labels use lexical query-term overlap, not embeddings, so a second hop
that shares no surface tokens with the first (a pure paraphrase bridge) is labelled
no-gap and excluded — the same honest limitation redhop reports. Retention is
reachability, not answer quality.
