# The reranker-fairness pack: R@k on a reranker-excluded pool

`docs/research/07` §4 (RESEARCH.md Tier 0.5) asks for **R@10/100/1000 and a reranker-excluded union
pool** alongside MAP@10/Hits@k. The objection it answers is real: the cross-encoder's gain is
measured by re-scoring a pool **the reranker did not build**, and a single pooled judgment set at a
fixed scoring depth can inflate a reranker's measured gain. `tools/reranker_fairness.py`
(`make reranker-fairness`) computes the pack from cached runs — no model, no network, seconds of CPU.

## Pool provenance, checked rather than asserted

The cache (`ce-scores-full.json`) is produced by `tools/multihop_ce_cache.py`, which builds the fused
first stage and **then** scores it, so the cross-encoder contributes no candidate. The pack echoes
the cache's own settings and **refuses to report** if any of them names a reranker in pool
construction, so the fairness property cannot be assumed away:

- `model` `BAAI/bge-small-en-v1.5`, `overlap` 200, `corpus_chunks` 17259, `rrf_k` 5, `depth` 100,
  `limit_queries` 2255, `rerank_model` `BAAI/bge-reranker-base`.
- **2,255 queries**; the pool is the fused first stage (BM25 + dense, RRF).

## Gold-fact recall@k, both arms on that one pool

| depth | fused | cross-encoder | reading |
| --- | --- | --- | --- |
| 10 | **0.4644** | **0.5452** | the CE lifts whole gold facts into the top-10 |
| 100 | **0.8480** | **0.8480** | identical — the pool's number, not the reranker's |

**Membership is the reason.** At the pool's own depth the two arms carry the **same candidate set
for 2255/2255 queries**; at depth 10 the reordering changes the top-10 set for **2255/2255**. That
difference *is* the ordering gain, and it is the whole of it: a second stage that only reorders
cannot add a candidate, so **R@k for k at or beyond the pool depth is the first stage's number**.
A claim that a reranker "improves R@1000" is therefore a claim about a retriever, not a reranker —
which is what this pack exists to make visible.

## The pool's own depth curve

From the depth-stratified diagnostics (`diagnostics.json`, max depth 500):

| depth | pool gold-fact recall | queries |
| --- | --- | --- |
| 10 | 0.4647 | 2255 |
| 100 | 0.8509 | 2255 |
| 500 | **0.9811** | 2255 |

## The ordering gain, isolated

From `rerank-experiment-full.json`, the same pool for both arms:

| arm | MAP@10 | Hits@4 | Hits@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| fused | 0.2546 | 0.6825 | 0.8120 | 0.5482 |
| cross-encoder | **0.2948** | **0.7348** | **0.8576** | **0.6106** |

Paired bootstrap over the 2,255 queries: **+0.0401 [+0.0321, +0.0484]**, p<0.001 (the report's own
machinery). The reranker's gain is real and it is entirely inside the pool.

## R@1000

**Not measurable from cache, and the pack says so.** The deepest cached pool is depth 500 and the
cross-encoder pool is depth 100, so R@1000 is reported as a bound with its reason rather than
invented. This is the Tier 0.5 falsifier handled honestly: *if gold support is not exhaustive at
depth 1000 the ceiling is still a floor* — at depth 500 the pool already holds **0.9811** of gold
facts, and the reorder oracle over the pool is the already-measured ceiling (0.6088 at depth 100,
rising to 0.6592 at depth 500). A depth-1000 diagnostics run turns the bound into a number without
changing the reading; `--deep <report>` folds one in when it exists.

## What this does not show

- **"Union" means the reranker-excluded fused pool, not a reconstruction from per-channel lists.**
  The cache retains the fused ranking, not the BM25 / dense / sparse rankings that produced it, so
  the pack cannot re-derive the union from its parts; what it *can* do — and does — is verify the
  pool was not built with the reranker and refuse it if it was.
- **No cross-encoder arm at depth 500.** The CE scores are cached only over the depth-100 pool. That
  does not affect any R@k at or beyond depth 100 (membership), but a MAP@10 measured on a deeper
  pool would be a different number.
- **MAP@10 here is the harness pool's, not the engine's.** These are the cached comparison rows
  (fused 0.2546 / CE 0.2948), not the engine's own 0.3009; the pack's subject is the reranker
  comparison on one pool, which is what Tier 0.5 asks about.
- **Gold-fact recall uses the benchmark's normalised substring test**, so a paraphrase that answers
  the question without repeating the gold sentence is counted as missed. Same convention as the rest
  of the repository's fact-recall work.

## Reproduction

```sh
make reranker-fairness              # writes eval/reranker-fairness.{json,md}
.venv/bin/python tools/reranker_fairness.py --check   # offline invariants, reads no cache
.venv/bin/python -m pytest tests/test_reranker_fairness.py -q
```
