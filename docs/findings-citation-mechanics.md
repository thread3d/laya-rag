# Findings: citation mechanics over the composed answers

ROADMAP Phase 3.3. An end-to-end row is `engine + <generator>`, and most of what it measures belongs
to the model. This measures the half that belongs to the **engine**: whether the prose's citations
and quoted spans actually exist in the evidence the engine handed over. It is deterministic,
model-free — `verify.py` uses no weights and no thresholds — and it is a different question from
whether the prose is right.

The end-to-end report does not store evidence, so it is re-retrieved here. Retrieval is
deterministic at fixed settings, so these are the passages the generator saw.

Reproduce:

```sh
make citation-report            # re-retrieves and verifies the pilot's answers
```

## Result — the 14 composed answers from `endtoend-pilot.json`

| group | answers | citation coverage | quote validity | clean | invalid citations | injection flags |
| --- | --- | --- | --- | --- | --- | --- |
| all | 14 | 0.5185 | 0.8000 | **0 (0.0000)** | 1 | 0 |
| answerable | 9 | 0.5769 | **0.8696** | 0 (0.0000) | 0 | 0 |
| null | 5 | 0.4138 | **0.0000** | 0 (0.0000) | 1 | 0 |

Sentence-level detail: **39 of 81 sentences carry no citation** — 22 of 52 on the answerable records
(42%) and 17 of 29 on the null ones (59%). Quotes: 25 checked, 20 found.

## Three readings

**1. No composed answer meets the deterministic citation bar.** All fourteen fail, and the dominant
cause is uncited sentences, not wrong ones: nearly half the sentences in the generator's prose cite
nothing. This is the engine telling the truth about a pipeline it is only half of — the prose is not
citation-compliant as written, and a caller relying on `clean` would reject every answer here.

**2. The verifier catches fabricated quotes exactly where the corpus had nothing to quote.** On the
five `null_query` records — the ones whose gold answer is literally *"Insufficient information"* —
**both** quotes the generator offered were rejected: one was not found in the evidence, and one came
from the question rather than the corpus. On the nine answerable records **20 of 23 quotes were
found** (0.870). That is the deterministic verifier doing the job it exists for: a model asked to
answer an unanswerable question invents a quotation, and no weights are needed to catch it.

**3. The null side is a direction, not a rate.** Only **2 quotes** were checked on the null group, so
0/2 is not a rate anyone should quote; the answerable side's 20/23 is the better-supported number.

## What this does not show

- **Not answer correctness.** `clean` is citation *mechanics* — does the cited span exist — and says
  nothing about whether the answer is right. That is the generator's number and is not measured here.
- **Not a general rate.** 14 answers, one generator (`ornith-1.5:35b`), one prompt, one corpus. The
  uncited-sentence share is a property of *this* prose, not of RAG prose in general.
- **Not stored evidence, re-retrieved evidence.** Deterministic retrieval makes the re-run faithful
  at fixed settings, but a change to chunking or depth would change the evidence and therefore the
  verdict; the settings are recorded in the report.
- **Not a generator defect claim.** The engine's own prompt does ask for citations; whether the model
  ignores it, or the token budget truncates before the citations, is not separated here.

## Method notes

- **Split the row before quoting any of it.** The whole point of routing this through `verify.py` is
  that the citation half is the engine's and the prose half is the model's; reporting a single
  end-to-end accuracy would have merged them and attributed both to whichever was convenient.
- **Group by answerability.** Aggregating the null records with the answerable ones would have
  hidden the one result that matters most here — that fabricated quotes appear where the corpus had
  no answer.
