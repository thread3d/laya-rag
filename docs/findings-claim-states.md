# Findings: three-state claim adjudication

ROADMAP Phase 5.2, closing a documented limit. The engine's adjudicator reported a claim about a
topic the corpus never mentions as `not_measured`, because the relevance gate never reached the
answerability question. `README.md` stated the consequence plainly: *"a fabricated claim about a
topic the corpus never mentions is not refuted"*. That conflates two different situations — the
corpus **says nothing**, and the corpus **says the opposite** — and this separates them with the
NLI hook the engine already ships.

## What it does

With `LAYA_RAG_ANSWERABILITY_NLI_MODEL` configured, a claim the relevance gate rejected is asked
about anyway, over the candidates the gate rejected, and its verdict becomes one of:

| state | meaning |
| --- | --- |
| `supported` | the relevance gate missed it, but the evidence entails the claim |
| `contradicted` | the evidence entails the negation (contradiction beats entailment) |
| `no_coverage` | neither — the corpus is silent about this claim |
| `not_measured` | no NLI scorer is configured, or nothing was retrieved for the claim |

`NLIAnswerability.verdict()` returns both NLI columns, and the contradiction index is read from the
checkpoint's `id2label` like the entailment one. A **single-label head reports no contradiction
column** rather than inventing one (`None`, not `0`): guessing there would turn "silent" into
"refutes", which is the opposite claim. The default path is untouched — with no NLI configured the
verdict is `not_measured`, exactly as before, and the existing adjudication tests pass unchanged.

## Demonstration with the real model

`cross-encoder/nli-deberta-v3-base` against one planted passage — *"Refunds are processed within
five business days of the request."* — over three constructed claims:

| claim | entailment | contradiction | state |
| --- | --- | --- | --- |
| Refunds are processed within five business days. | **0.994** | 0.0001 | `supported` |
| Refunds take ninety days to process. | 0.00001 | **0.99996** | `contradicted` |
| The Lima office offers free shipping on bulk orders. | 0.0002 | 0.0012 | `no_coverage` |

The separation is as clean as the task allows: the supported claim's entailment is 0.994, the
contradicted claim's contradiction is 0.99996, and the unmentioned topic sits near zero on both.

## What this does not show

- **Not a no-coverage recall figure.** The roadmap asks for ~300 hand-labelled claim/corpus pairs
  from `eval/corpus-planted` and a no-coverage recall. The table above is **three constructed
  claims proving the plumbing**, not a labelled evaluation; no recall number is claimed, and
  hand-labelling 300 pairs is the remaining work.
- **Not a new model.** It reuses the shipped `NLIAnswerability` hook and the checkpoint the engine
  already documents; nothing is trained here.
- **Not immune to truncation.** The NLI premise is truncated at 512 tokens, so a long evidence set
  can lose the part that contradicts the claim — the same limitation Phase 2.3 hit with ArguAna.
- **Not precedence over Laya.** The NLI speaks only for claims the relevance gate never reached; a
  claim the gate *did* reach is still adjudicated by Laya's calibrated decision, and a test pins
  that.

## Method notes

- **A silent zero is not a measurement.** `_passage_text` accepts a `ScoredPassage`, a `Hit` or a
  mapping — not a bare string. Passing strings made every verdict return `0.0` for both columns
  because the premise was empty, which reads exactly like a model that finds nothing; the check is
  that the premise is non-empty before scoring.
- **Both columns or neither.** An entailment-only seam cannot produce a contradiction score, and
  `1 - entailment` is not one; `verdict` takes its own seam rather than misusing the existing one.
