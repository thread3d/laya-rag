# Findings: atomic lexical units help Hits@4, not MAP@10

This records the Tier 1.5 experiment from [`RESEARCH.md`](../RESEARCH.md): index
sentence-like atoms and return their parent chunk, versus indexing whole chunks. Model-free
and lexical, so it isolates the retrieval-unit question from representation quality.

Reproduce:

```sh
make multihop-atom-index ATOM_ARGS="--limit-queries 400"
```

## Setup

17,259 chunks at the official `overlap=200` become **199,872 atoms** (a fact atom is a
sentence-like span of at least 40 characters). Atoms are retrieved with the same BM25 the
engine ships, then mapped to parents, keeping the best atom per parent; the delivered
evidence is still a whole chunk. First 400 answerable queries.

| strategy | Hits@10 | Hits@4 | MAP@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| chunk-bm25 | 0.8075 | 0.6575 | 0.2644 | 0.5715 |
| atom-bm25 | 0.7425 | 0.5700 | 0.2254 | 0.5079 |
| **chunk+atom RRF** | 0.8075 | **0.7050** | 0.2688 | 0.5748 |

Paired over queries:

| comparison | delta | 95% CI | permutation p |
| --- | --- | --- | --- |
| atom-bm25 vs chunk Hits@10 | **−0.0650** | [−0.1100, −0.0225] | 0.0030 |
| atom-bm25 vs chunk Hits@4 | **−0.0875** | [−0.1425, −0.0350] | 0.0020 |
| atom-bm25 vs chunk MAP@10 | **−0.0390** | [−0.0601, −0.0193] | 0.0010 |
| chunk+atom vs chunk Hits@4 | **+0.0475** | [+0.0100, +0.0825] | 0.0120 |
| chunk+atom vs chunk Hits@10 | +0.0000 | [−0.0275, +0.0250] | 1.00 (ns) |
| chunk+atom vs chunk MAP@10 | +0.0044 | [−0.0091, +0.0172] | 0.49 (ns) |

## Reading

- **Atom-only retrieval is a clear negative.** Short units wreck BM25's term statistics and
  length normalisation: every metric falls significantly. This is the concern the research
  brief raised, measured rather than assumed, and it is the same failure shape as the
  rejected 600-character passage indexing.
- **Fusing atom and chunk rankings is a real, cheap gain on Hits@4** (+0.0475
  [+0.0100, +0.0825]) with Hits@10 unchanged and MAP a wash. Hits@4 asks "was a piece of
  evidence found early", and the atom channel's finer matching helps exactly that. It costs
  no model and no API.
- **It is not a MAP lever**, which is where the project's headline gap is, so it is
  recorded rather than shipped. If breadth at rank 4 matters for a deployment, a third
  RRF channel over atom rankings is the cheap way to get it; it needs an atom postings
  store alongside the chunk index, which is why it is not enabled by default here.

## Method note

This experiment exposed a harness contract bug worth recording: `official_metrics` takes a
list of *per-query ranked lists*. Handing it one flat list of strings slices a string into
characters and silently scores everything zero. The first run returned 0.0000 on every
metric, including the control row; the metric path is now a single tested helper
(`score_strategies`), and all three existing callers were audited and are correct.
