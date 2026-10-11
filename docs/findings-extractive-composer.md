# A citation-first extractive composer

The audit's Bet 5 and the generation brief's A5 both point at the same hole. The engine's measured
citation mechanics are a **negative**: over the pilot's 14 answers no answer met the deterministic
citation bar, dominated by **39 of 81 sentences carrying no citation**. A generative composer buys
fluency with a *probabilistic* attribution guarantee. `src/laya_rag/extractive.py` buys a
**structural** one instead: a sentence that is not present verbatim in a retrieved passage is never
emitted, and with nothing to emit the composer abstains.

## The guarantee and the contract

- Every emitted sentence is a **verbatim substring** of a retrieved passage. The composer checks
  this explicitly: a sentence split from the merged document text is kept only if it is found in one
  of the passage texts, so overlap trimming or a merge cannot invent a span.
- Documents are numbered in first-appearance order — the same order `build_messages` presents
  evidence in — so `[n]` means the same thing in the composed answer, the prompt and
  `verify_citations`.
- Each sentence also carries its **source path and chunk id**, which is the structured citation
  contract: `verify_citations` can be run over the result without parsing prose.
- No generator, no API, no model: scoring is the passage's own relevance (falling back to recall),
  sentences are ordered best-first, duplicates are emitted once, and an empty or fully-filtered
  evidence set abstains with a fixed sentence.

## Four surfaces, one of which is the point

`RagEngine.query_extractive(query, ...)` returns the answer plus the structured sentences and, with
`include_evidence=True`, the full `query` result so the citations travel with the evidence they
index into; `laya-rag extract QUERY [--max-sentences N] [--tenant T] [--json]` prints it;
**`POST /v1/extract`** and the MCP **`extract_answer`** tool expose it to callers that never touch
this repository. The route is the point: it is the one surface whose output is *entirely* the
engine's, because no model is in the path at all, and it is what makes a reproducible, externally
runnable end-to-end row possible (§P12 of `SCORE-LIFT.md`).

This deletes the "39 of 81 uncited" failure mode **by construction** rather than by prompting — the
fourth time this repository replaces a statistical fix with a structural one.

## Building the surface found the format bug

The marker used to **trail** its sentence — `"Refunds take five days. [1]"` — which is how a reader
expects citations and how the composed prose read. `verify_citations` splits an answer into claims
*after* the terminal punctuation, so that `[1]` was orphaned onto the **next** claim: the sentence
it supported read as uncited and the final marker dangled as a citation to nothing. Every extractive
answer failed the project's own deterministic check, which is precisely the failure the composer was
built to remove. The round trip through `/v1/extract` → `/v1/verify` is what exposed it; the unit
tests only asserted that each sentence's number appeared *somewhere* in the answer, which the
trailing form satisfies.

The marker now **leads** the sentence it supports — `"[1] Refunds take five days."` — which matches
how `build_messages` numbers the evidence the model is shown, and is asserted by a test that runs
`verify_citations` over the composer's own output and requires `clean: true` with zero uncited
claims.

## What this does not show

- **No answer-quality measurement.** Extractive prose reads worse than generated prose, and the
  end-to-end row (accuracy, sentence-level attribution precision, the abstention curve) that Bet 5
  asks for is not run here. What is guaranteed is attribution, not usefulness. The surfaces for it
  now exist; the run is the next item.
- **Sentence scoring is the document's relevance**, not a per-sentence scorer, and citations are
  document-level (passages are merged per document), so two sentences from one file share a number.
- **No contradiction check.** `adjudicate_claims` can mark a claim `contradicted`; this composer
  emits any verbatim sentence above the score floor, so a contradicted span is still emittable.
  Gating emission on adjudication is the next step.
- **The trailing-vs-leading format was never measured for readability.** Leading markers are
  machine-checkable and match the prompt's numbering; a reader may prefer them trailing. Nothing
  here measures which reads better, and a future surface presenting these answers to a human could
  re-render them.

## Verification

Twenty-three tests. Composer: sentence splitting; the verbatim guarantee with citations present in
the answer; citation numbering by document first appearance; duplicate suppression; empty input
abstains; `max_sentences` and best-first order; `min_score` filtering; the engine method end-to-end
with a stub model; the CLI parser; and — added with the format fix — a round trip through
`verify_citations` requiring zero uncited claims, plus a check that every marker leads its sentence.
API: the verbatim guarantee against the returned passages, the `/v1/extract` → `/v1/verify` round
trip, abstention, the pre-index conflict, request validation and the OpenAPI contract. MCP: the tool
is registered, quotes the corpus, round-trips through `verify_answer`, abstains, and rejects bad
budgets. `make test`: **1,567 passed, 5 deselected**.
