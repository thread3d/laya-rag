# Findings: the engine's MultiHop-RAG win does not travel to MuSiQue

Phase 1.2 measured the real `RagEngine` on MultiHop-RAG under its own corpus-aware chunking and
found it **+0.0535 MAP@10** above the 256-token harness, level with the harness's reranked row
([findings-engine-multihop.md](findings-engine-multihop.md)). This is the replication: the same
measurement on MuSiQue, where the coverage objective was already falsified once.

Reproduce (the corpus is always the full split; `--limit` narrows only the queries):

```sh
make engine-musique ENGINE_MUSIQUE_ARGS="--limit 200 --embed-model BAAI/bge-small-en-v1.5"
make engine-musique ENGINE_MUSIQUE_ARGS="--limit 200 --embed-model BAAI/bge-small-en-v1.5 \
    --retrieve-k 100 --out eval/benchmarks/musique/engine-hybrid-200-depth100"
.venv/bin/python tools/engine_musique.py --compare --out eval/benchmarks/musique/engine-hybrid-200
```

## Result

21,098 paragraphs (the full validation corpus), 21,738 chunks, `bge-small` dense + stemmed
lexical, no cross-encoder, thresholds 0. The first 200 validation queries, paired against the
harness's per-query MAP@10 on the same ids:

| system | MAP@10 (same 200) | paired vs harness `fused` |
| --- | --- | --- |
| harness `fused` | 0.4811 | — |
| harness `blend-0.75` | 0.4865 | — |
| harness `ce` | 0.4634 | — |
| harness `covquota-3-0.7` | 0.4529 | — |
| **engine, `retrieve_k=100`** | **0.4714** | **−0.0097 [−0.0190, −0.0007], p=0.034** |
| **engine, default `retrieve_k=20`** | **0.4695** | **−0.0116 [−0.0228, −0.0012], p=0.028** |

Engine metrics over the same 200: Hits@10 **0.9600/0.9650**, Hits@4 **0.9150/0.9100**, MRR@10
**0.8417/0.8444** (depth 20 / depth 100).

Three readings:

1. **The MultiHop-RAG engine win is corpus-specific.** On MultiHop-RAG the engine beat the
   harness by +0.0535; on MuSiQue it is *below* the harness's fused first stage by a small but
   significant margin. "A replica understates the engine" is a statement about one corpus, not a
   property of the harness.
2. **The obvious explanation is falsified.** The engine's default pool (`retrieve_k=20`) is much
   shallower than the harness's depth 100, so the first run was not like for like. Raising the
   engine to depth 100 moved MAP@10 by **+0.0019** and left the deficit significant. Pool depth is
   not the cause.
3. **The engine still beats the harness's second-stage rules.** It is above the harness's `ce`
   (0.4634) and `covquota-3.7` (0.4529) on this subset, and below `fused` and `blend-0.75`. So
   the second stage is not what the harness is winning with — its first stage is.

## The dense channel helps here and hurts on MultiHop-RAG

The same 200 queries with **no `--embed-model`** — Laya's encoder resolving to a dense weight of
zero, so recall is lexical only, which is what the engine ships:

| metric | engine hybrid | **engine default** | delta |
| --- | --- | --- | --- |
| MAP@10 | 0.4695 | 0.4156 | **−0.0540 [−0.0718, −0.0363]** |
| Hits@4 | 0.9150 | 0.8650 | **−0.0500 [−0.0950, −0.0050]** |
| Hits@10 | 0.9600 | 0.9050 | **−0.0550 [−0.0900, −0.0250]** |
| MRR@10 | 0.8417 | 0.7687 | **−0.0730 [−0.1059, −0.0415]** |

Every interval excludes zero, and the default is also **−0.0656 [−0.0863, −0.0456]** MAP@10 below
the harness's fused first stage on the same queries.

So the dense channel's contribution **reverses between corpora**, measured at the engine level on
the same protocol:

| corpus | leaving the dense channel off | reading |
| --- | --- | --- |
| MultiHop-RAG (2,255) | **+0.0119 MAP@10** [+0.0061, +0.0180] | dense **hurts** — the shipped default is the better configuration |
| MuSiQue (200) | **−0.0540 MAP@10** [−0.0718, −0.0363] | dense **helps**, substantially |

That is not a contradiction; it is the same shape as the project's per-corpus results for coverage
and for the cross-encoder — an instrument that helps one corpus and hurts another — and it is the
argument for **routing by corpus** rather than picking one global configuration. It also means the
engine's shipped default is the right choice for its MultiHop-RAG behaviour and the wrong one for
MuSiQue-style multi-hop, where roughly 0.054 MAP@10 is left on the table.

## What this does not show

- **Not a mechanism.** Chunk-level retrieval can spend several of the top-10 slots on different
  chunks of one paragraph, which would lower distinct-paragraph coverage relative to a harness
  that ranks whole paragraphs. That is a hypothesis, consistent with the numbers, and **untested**.
- **Not the full split.** The first 200 of 2,417 queries, not a random sample (MuSiQue has no
  seed-17 convention here), so this is a prefix comparison, not a headline number.
- **MAP@10 only is paired.** The harness report persists per-query MAP@10 alone; the other
  metrics are the engine's own and are not tested against the harness here.
- **One encoder, one chunking.** `bge-small` only, and the engine derived 1,200 characters for
  this corpus. The shipped default path (Laya's encoder, dense weight 0) is a separate run.

## Method notes

- **Do not reuse `tools/musique.py --limit-queries` for a subset comparison.** It passes the limit
  into `load_split`, so the corpus shrinks with the query set and a limited run retrieves over an
  easier paragraph set. `engine_musique.load_task` keeps the full corpus and narrows the queries,
  which is what makes the pairing honest; a test pins it.
- **Record the pool depth.** The two engine runs differ only in `retrieve_k`, and reporting only
  one of them would have made a shallow-pool artefact look like a capability gap — or, as here,
  hidden that the gap is real.
