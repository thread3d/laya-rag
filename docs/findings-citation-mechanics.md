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

---

## What "cite-clean 1.0000" actually certifies — audited 2026-10-30

Every finding in this programme quotes `citation_clean_all`, and the number is 1.0000 nearly everywhere. **Nobody
had audited what it certifies.** Eight adversarial cases, each with an expected verdict:

| case | `clean` | expected | verdict |
| --- | --- | --- | --- |
| verbatim, cited correctly | True | True | correct |
| **verbatim, one number changed** (0.4799 → 0.4979) | **True** | False | **not caught by `clean`** |
| **verbatim, wrong citation index** (`[2]` for text in `[1]`) | **True** | False | **not caught by `clean`** |
| **polarity flipped** ("above" → "below") | **True** | False | **not caught by anything** |
| quote not present in the evidence | False | False | caught |
| a sentence with no citation | False | False | caught |
| two sentences, one fabricated | **True** | False | **not caught by `clean`** |

So `clean` is **structural**: every citation is in range, every quote it contains was found in the evidence, and no
sentence is uncited. It is a real property and it is the one that matters most for a quoting engine — a span that
does not exist cannot be emitted at all — but **it is not a claim about the numbers or the sense**.

**The code knows this and says so.** `verify.py`'s docstring: the content layer *"is reported in a separate
`content` block and never change[s] the `clean` verdict. A noisy check is one a reader learns to ignore."* The
content layer exists, works (it flagged the changed number and the fabricated figure), and is **advisory by
design**, because a numeric mismatch has legitimate exceptions — rounding, "five" versus 5, a range's endpoint.

**What was wrong was not the checker but the reporting.** No finding in this programme ever quoted the content
layer, so `citation_clean_all 1.0000` sat alone in every table as if it meant *the answers are faithful*. A reader
can be forgiven for concluding that the numbers had been checked; the checker checked them and nothing published
the result.

### Fixed going forward, and stated here for what is already recorded

The extractive row now records, beside the structural figure:

| field | meaning |
| --- | --- |
| `citation_clean_all` | structural: citations valid, quotes found, nothing uncited |
| `content_clean_all` | **advisory**: every cited claim's numbers, dates and units occur in the evidence it cites |
| `content_issue_total` | how many advisory findings, summed |
| `negation_flags` | the `possible_negation_flip` count |

**Rows measured before today do not carry these fields**, and their `1.0000` should be read as the structural
figure alone. The published 1.0000s are therefore true and narrower than they look — which is the same shape as
the chapter-versus-document problem in [`findings-multihop-placement.md`](findings-multihop-placement.md), and the
second time in one night that a number was right and its *name* was the problem.

### Measured on the engine's own output: the content layer cannot fail, and that is the point

Eight real answers from the extractive path, each through both layers:

| layer | result |
| --- | --- |
| structural (`clean`) | **7 of 8** |
| advisory content (`content.clean`) | **8 of 8**, zero issues in total |

Two things follow, and the first is a *structural* fact rather than an empirical one.

**On a verbatim-span engine the content layer has almost no discriminating power.** Every emitted sentence is a
substring of a cited passage, so its numbers are in the evidence *by construction* — a verbatim span cannot contain
a number its own source lacks. What the layer can still catch is **attribution**: a span quoted from passage 1 but
cited to passage 2 has numbers that are absent from passage 2, and the check fires. So its value here is "the
citation points at the right passage", not "the claim is honest", and recording `content_clean_all` on extractive
rows will read 1.0000 for a structural reason rather than an evidential one. **That is worth knowing before the
number is quoted** — the same class as everything else in this document.

Its real audience is the `engine + <model>` row, where the text is generated and can therefore say anything.

**The one structural failure is also structural.** The seventh answer was an **abstention**, whose fixed sentence —
*"The retrieved evidence does not contain a sentence that answers the question."* — counts as one uncited claim.
The harness documents that interaction rather than smoothing it, and it is why `citation_clean_all` is 1.0000 over
an answerable-only set: abstentions are near zero there, so the sentence never appears. My probe happened to include
a query the engine abstained on, which is how a documented interaction turned up in eight samples.

### The negation check's coverage, stated as a limit

`_NEGATION` matches explicit negators — *not, no, never, cannot, without, none, nor*. It does **not** see antonym
pairs: *above/below*, *higher/lower*, *more/fewer* all pass silently, which is why the fourth case above is flagged
by nothing at all. That is a deliberate heuristic at a high bar ("two shared content tokens"), and it is worth
being explicit that **a polarity inversion is outside what this engine's checks can detect** — the defence against
it is structural rather than semantic: the engine emits *verbatim spans*, so an inverted claim cannot be produced
by the extractive path in the first place. A generator would change that, which is why the `engine + <model>` row
is labelled separately.
