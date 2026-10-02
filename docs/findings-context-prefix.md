# Findings: the structural context prefix raises MRR on local docs

This records the Tier 1.6 work from [`RESEARCH.md`](../RESEARCH.md). A chunk that says
"revenue grew 3%" carries no company, period or document; the research version of this idea
generates that context with an LLM at ingest, which is not affordable locally. This is the
deterministic 80%: a breadcrumb the retrieval channels index but the reader never sees.

## Design

`Chunk` gained a `context` field and an `index_text` property (``context + "\n" + text``
when a context exists). The lexical and dense channels index `index_text`; the evidence
returned, the composer prompt, and the citation checker still use `text` alone. The
breadcrumb is the file name, the first Markdown heading and the first line
(`document_context`). It is **off by default** (`LAYA_RAG_CONTEXT_PREFIX=true` turns it on)
and requires a re-index, so existing indexes and rankings are unchanged.

## Measured (lexical, model-free; `make context-prefix`)

Vendored Laya documentation, 35 documents, 7 answerable labelled queries, BM25 with
stemming and stopword filtering:

| metric | without prefix | with prefix | delta |
| --- | --- | --- | --- |
| recall | 1.0000 | 1.0000 | +0.0000 |
| precision@10 | 1.0000 | 1.0000 | +0.0000 |
| MRR | 0.6429 | **0.8333** | **+0.1904** |
| median rank | 2 | 1 | — |
| worst rank | 6 | 3 | — |

Per query (gold rank): three queries improved — `d-langchain` 6 → 3, `d-evals` 2 → 1,
`d-adoption` 3 → 1 — and four were unchanged (all already at rank 1). **No query got
worse.** Recall and precision were already saturated, so the only room was ordering, which
is exactly what the breadcrumb supplies: `langchain.md` and `staged-adoption.md` now match
their own file names and headings, which the body text alone did not.

## Honest limits

- **Seven queries.** The movement is consistent and one-directional, but this is
  suggestive, not conclusive; it is a local docs corpus, not a benchmark.
- The gain is concentrated in a query class the prefix is built for — questions naming the
  document or its topic. A corpus whose bodies already repeat their titles would see less.
- The engine's relevance gate reads the chunk **text**, so a chunk retrieved purely on its
  breadcrumb can still be filtered by Laya's decision. The prefix improves the channel
  ranking; it does not override the gate, and that interaction is visible in the engine
  test (the context-only match is kept only at a zero threshold).
- It is a **capability, off by default**, not a changed default: +0.19 MRR on seven queries
  does not justify changing what every existing index produces.
