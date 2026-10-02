# Findings: the fused-pool rewriters are multi-fact instruments, not defaults

`RECALL_COVERAGE_LAMBDA` and `RECALL_DEDUP_THRESHOLD` are both off by default even though
each is a measured win on MultiHop-RAG (coverage +0.0304 MAP@10, near-duplicate suppression
+0.0244). That looked like a shipped win left switched off, so this measures both on BEIR —
where a query is largely answered by **one** abstract, and where near-duplicate abstracts are
rare — to decide with evidence on both sides.

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

**Near-duplicate suppression is worse still**, at every threshold:

| dataset | fused | dedup-0.8 | dedup-0.9 | dedup-0.95 |
| --- | --- | --- | --- | --- |
| scifact | **0.7408** | 0.6532 | 0.7246 | 0.7388 |
| nfcorpus | **0.3673** | 0.2578 | 0.3433 | 0.3644 |
| arguana | **0.5596** | 0.0521 | 0.4842 | 0.5354 |

All nine cells are significant losses; the loosest threshold (0.95) is the least harmful and
still never better than doing nothing.

## Why the two benchmarks disagree

- **MultiHop-RAG** questions need **several** distinct facts, and MAP@10 credits each one, so
  spending a slot on a *new* sentence atom is exactly the right objective.
- **BEIR** queries are answered by one relevant abstract. The objective then rewards "a new
  atom" over "the relevant document", so it trades the answer for diversity. ArguAna is the
  extreme: its gold is the **counter-argument**, and atom-diversity actively prefers an
  unrelated argument that shares no sentences with the top hit — the same relational-relevance
  failure that defeated the pointwise cross-encoder there.

## Verdict

**Keep both off by default.** The honest claim is now narrower and stronger: coverage and
near-duplicate suppression are **corpus-shape** instruments — they are worth +0.0304 and
+0.0244 MAP@10 on MultiHop-RAG and cost 0.002–0.51 nDCG@10 on every BEIR dataset measured.
They target a set of facts and duplicated chunks, and a single-abstract corpus has neither.
The engine should not enable either automatically on a corpus whose shape it does not know.

## Honest limits

- BEIR documents are single short abstracts (2–4 atoms each), so this tests "one document per
  query" rather than long multi-fact or heavily-duplicated documents; a corpus of long,
  overlapping documents might sit in between.
- Three of the five BEIR datasets were run (scifact, nfcorpus, arguana) — the two largest,
  scidocs and fiqa, were not, and the loss on the three is already decisive.
- The objectives themselves are unchanged by this; only the defaults and the claims are.
