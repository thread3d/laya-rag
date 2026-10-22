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
| compare the engine with published retrieval systems | [Comparisons across systems](COMPARISONS.md) |
| pick the work up: what landed, what was refuted, what is left and what it costs | [Programme status](PROGRAMME-STATUS.md) |
| read the HTTP API and MCP server design record | [An industry-standard API and an MCP server](api-mcp-design.md) |
| choose between the extractive answer path and an external generator | [Generation and placement](generation-placement.md) |
| decide whether fine-tuning the encoder is worth it | [Fine-tuning study](study-laya-finetuning.md) |
| avoid the measurement mistakes this programme has already paid for | [Measurement failures](findings-measurement-failures.md) |
| read the background research notes | [Research notes](research/) |
| look at the generated comparison charts | [Charts](charts/) |

## Two rules that make the pages readable

Numbers in these pages are pinned rather than typed: `make claim-check` resolves every quoted figure
against the report that produced it, and `make consistency-check` catches two documents disagreeing
about the same quantity. Where a number matters, the generated artifact under `eval/` is the source of
truth, and a page that has drifted from it is a bug in the page.

Anything this repository has not measured is labelled unmeasured, and text produced by an external
generator is labelled `engine + <model>`.
