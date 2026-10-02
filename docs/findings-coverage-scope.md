# Findings: the coverage objective is a multi-fact instrument, not a default

`RECALL_COVERAGE_LAMBDA` is off by default even though it is the best measured pass on
MultiHop-RAG (MAP@10 0.2546 → 0.2850). That looked like a shipped win left switched off, so
this measures the same objective on BEIR — where a query is largely answered by **one**
abstract — to decide with evidence on both sides.

Reproduce:

```sh
make beir-coverage BEIR_COVERAGE_ARGS="--datasets scifact,nfcorpus,arguana"
```

## BEIR result: it hurts, significantly, everywhere

nDCG@10, `fused` is the shipped dense+BM25 hybrid (RRF k=5), cached `bge-small`:

| dataset | fused | coverage-0.5 | coverage-nored-0.5 | coverage-0.7 | coverage-nored-0.7 |
| --- | --- | --- | --- | --- | --- |
| scifact | **0.7408** | 0.6616 | 0.6763 | 0.7194 | 0.7224 |
| nfcorpus | **0.3673** | 0.3052 | 0.3190 | 0.3518 | 0.3559 |
| arguana | **0.5596** | 0.0870 | 0.1908 | 0.4525 | 0.4834 |

Every cell is a **significant** loss (paired bootstrap, n=300–1,406); λ=0.5 is catastrophic on
ArguAna, where the fused 0.5596 collapses to 0.0870. The redundancy-free variant is less bad
but still worse than no coverage at all.

## Why the two benchmarks disagree

- **MultiHop-RAG** questions need **several** distinct facts, and MAP@10 credits each one, so
  spending a slot on a *new* sentence atom is exactly the right objective.
- **BEIR** queries are answered by one relevant abstract. The objective then rewards "a new
  atom" over "the relevant document", so it trades the answer for diversity. ArguAna is the
  extreme: its gold is the **counter-argument**, and atom-diversity actively prefers an
  unrelated argument that shares no sentences with the top hit — the same relational-relevance
  failure that defeated the pointwise cross-encoder there.

## Verdict

**Keep it off by default.** The honest claim is now narrower and stronger: the coverage
objective is a **multi-fact-corpus** instrument — it is worth +0.0304 MAP@10 on MultiHop-RAG
and costs 0.02–0.08 nDCG@10 on every BEIR dataset measured. It is not a general improvement,
and the engine should not enable it automatically.

## Honest limits

- BEIR documents are single short abstracts (2–4 atoms each), so this tests "one document per
  query" rather than long multi-fact documents; a long-document corpus might sit in between.
- Three of the five BEIR datasets were run (scifact, nfcorpus, arguana) — the two largest,
  scidocs and fiqa, were not, and the loss on the three is already decisive.
- The objective itself is unchanged by this; only the default and the claim are.
