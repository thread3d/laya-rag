# Findings: a document quota changes the passages, not (here) the ranking

Round 29's engine-level BEIR run showed the engine's top-10 *passages* collapse to fewer
documents (kept=10 but distinct=5 on some SciFact queries), and concluded that a document
quota was the natural next improvement. `RERANK_MIN_DOCS` now exists: the first N kept slots
are the best passage of N distinct documents, with the rest following in rank order.

## On SciFact it changes nothing measurable

The engine-level A/B is **exactly zero**: nDCG@10 0.7264 both ways, per-query series identical
for all 300 queries. A diagnostic on the real index explains why — the top-10 already held
5–10 distinct documents for every sampled query (candidates 20, distinct 12–19), so promoting
distinct documents into the head rearranged passages without changing which documents were
present or the order they first appeared in:

| query | base kept (docs) | quota kept (docs) |
| --- | --- | --- |
| 13 | 10 passages, 6 documents | 10 passages, first 5 documents distinct |
| 42 | 10 passages, 5 documents | 10 passages, first 5 documents distinct |

This is weaker than a proof of invariance and should be read as such. Promotion keeps each
document in its existing relevance order, so documents that are *present* keep their relative
order — but for a short `top_k` it can **add** documents that the base list pushed past the
cut, because the promoted documents are drawn from the whole sorted list, not just from the
first `top_k`. With `top_k=3`, `A,A,A,B,C` becomes `A,B,C`; with `top_k=10` on this corpus,
enough documents were already present that nothing moved. The exact zero is therefore an
empirical result for this corpus and this `top_k`, not a general law.

## What it does guarantee

**Prefix diversity**: the first N things a reader or a composer sees are N different
documents, so a duplicate chunk cannot occupy a leading slot. That is a presentation control
for an evidence bundle.

The harness's MultiHop result rests on a different mechanism: there the metric counts
**facts** across a query's gold support, and spending a slot on a new document buys a new fact
(`doc-quota` +0.0173 MAP@10). Document-level nDCG cannot see that, which is why the two
measurements disagree and why the feature is documented as a presentation control rather than
a retrieval gain.

## Honest limits

- One dataset, one `top_k` (10), one quota (5), 12 sampled queries for the diagnostic. A
  shorter `top_k` or a more duplicate-heavy list would move more, and the direction there is
  toward delivering *more* distinct documents, not fewer.
- The quota is left off by default: it has no measured retrieval benefit here, and where it
  helps is the fact-level and presentation cases above.
- This corrects round 29's own hypothesis — the engine's passage-level `top_k` looked like a
  ranking bug and is not one on this corpus.
