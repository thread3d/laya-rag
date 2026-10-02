# Findings: the multi-hop MAP@10 gap is selection, not recall

This records the Tier 0.1 diagnostic from [`RESEARCH.md`](../RESEARCH.md) §3.1, run over
the full MultiHop-RAG answerable set (2,255 queries) on the shipped `bge-small` + BM25
fused pipeline at the official `overlap=200` chunking. Deterministic and offline; raw
report in the gitignored `eval/benchmarks/multihop-rag/diagnostics.json`.

Reproduce:

```sh
make multihop-diagnostics
```

## Headline

| depth | pool fact recall | oracle ceiling MAP@10 | hop-2 recall | hop-2 retention |
| --- | --- | --- | --- | --- |
| 10 | 0.4747 | **0.7321** | 0.2407 | 0.2407 |
| 100 | 0.8666 | 0.7304 | 0.7189 | 0.7189 |
| 500 | 1.0000 | 0.6741 | 1.0000 | 1.0000 |

Achieved MAP@10 of the fused order: **0.2557** (Hits@10 0.8129, Hits@4 0.6847).

**Widening the pool does not raise the ceiling.** The best MAP@10 any reordering of the
fused top-100 can reach (0.7304) is statistically indistinguishable from the top-10's
(0.7321): Δ −0.0017 [−0.0174, +0.0156]. Going to depth 500 *lowers* the greedy oracle
(−0.0563 [−0.0613, −0.0513]) even as pool fact recall rises to 1.000, because the oracle
is a greedy heuristic and a larger candidate set dilutes it. The practical reading is
unambiguous: **from MAP@10's point of view, more recall is not the lever.** The gap from
0.2557 to ~0.73 lives entirely in which candidates are kept and in what order.

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

1. **Selection and ordering are the whole game.** The fused top-10 already admits an
   oracle ordering at MAP@10 0.73; the coverage pass captures only +0.0288 of that. The
   headroom is in a set-aware ordering rule, not in recall expansion.
2. **A `depth=500` expansion programme is de-prioritised.** Pool fact recall becomes
   1.000 at 500 and the oracle ceiling does not improve, so query decomposition, hop-2
   expansion and larger candidate pools buy recall that the MAP metric cannot spend.
3. **The second hop is the target.** Any selection rule worth shipping should raise hop-2
   retention at a fixed top-10. The bridge-linked rescue in `RESEARCH.md` §3.1 Tier 1.1
   is aimed exactly there, and this diagnostic gives the metric to falsify it with
   (`hop2_fact_recall@10`, `hop2_retention@10`).
4. **The cross-encoder is not obviously the answer.** It raised MAP@10 to 0.2986 on the
   400-query subset — real, but far below the 0.73 ceiling. The next experiments
   (`make multihop-ce-cache && make multihop-rerank-experiment`) test whether blending,
   a rank-1 guardrail, or CE-plus-coverage closes more of it.

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
