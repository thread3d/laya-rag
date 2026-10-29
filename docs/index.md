# laya_rag

A retrieval-augmented generation engine that uses [Laya](https://github.com/thread3d/laya) as its
retrieval and decision engine, with a command line, an HTTP API, an MCP server and a web interface.

It returns **evidence with calibrated probabilities, not prose**: the engine reports which passages
support an answer and how confident it is that they do, and abstains when they do not. Whoever
consumes that bundle — a reader looking at the interface, or the optional compose step that delegates
to an external generator — writes the final wording from exactly those passages.

This page is the way in. `README.md` stays the single copy of installation, configuration and the
honest case, so none of that is repeated here.

## Start here

Install it and run a first query with the README's [Setup](../README.md#setup) and
[Using it](../README.md#using-it). They stay in the README so there is one copy to keep current.

## Find a guide

| To | Read |
|---|---|
| install it, index a corpus, read the configuration table | [README](../README.md) |
| see every result the measurement programme produced, wins and negatives alike | [Findings index](FINDINGS.md) |
| **re-derive any number yourself**, with the command, its runtime and what it writes | [Reproducing a result](REPRODUCING.md) |
| compare the engine with published retrieval systems | [Comparisons across systems](COMPARISONS.md) |
| pick the work up: what landed, what was refuted, what is left and what it costs | [Programme status](PROGRAMME-STATUS.md) |
| ask what "best RAG system in the world" would take, and which of it this can win | [The best in the world, October 2026](best-in-the-world-2026-10.md) |
| read the HTTP API and MCP server design record | [An industry-standard API and an MCP server](api-mcp-design.md) |
| choose between the extractive answer path and an external generator | [Generation and placement](generation-placement.md) |
| decide whether fine-tuning the encoder is worth it | [Fine-tuning study](study-laya-finetuning.md) |
| avoid the measurement mistakes this programme has already paid for | [Measurement failures](findings-measurement-failures.md) |
| read the background research notes | [Research notes](research/) |
| read the dated 2026 market digests the October assessment rests on | [Retrieval market, 2026-10](research/09-retrieval-market-2026-10.md) · [RAG systems, 2026-10](research/10-rag-systems-2026-10.md) |
| look at the generated comparison charts | [Charts](charts/) |

## Two rules that make the pages readable

Numbers here are pinned **where it matters, and not everywhere**. `make claim-check` resolves **54**
registered figures against the report that produced them, and `make consistency-check` pins documents to
each other at registered sites and computes the BEIR coverage count from disk. The tracked prose contains
several thousand decimal figures in total, so the great majority are covered by neither gate: a number no
checker reaches is a claim on the author's care, and
[`findings-measurement-failures.md`](findings-measurement-failures.md) is the record of what care alone
has cost here. Where a number *is* registered, the generated artifact under `eval/` is the source of
truth, and a page that drifts from it is a bug in the page.

Anything this repository has not measured is labelled unmeasured, and text produced by an external
generator is labelled `engine + <model>`.
