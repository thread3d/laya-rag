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

Two surfaces: `RagEngine.query_extractive(query, ...)` returns the answer plus the structured
sentences, and `laya-rag extract QUERY [--max-sentences N] [--tenant T] [--json]` prints it.

This deletes the "39 of 81 uncited" failure mode **by construction** rather than by prompting — the
fourth time this repository replaces a statistical fix with a structural one.

## What this does not show

- **No answer-quality measurement.** Extractive prose reads worse than generated prose, and the
  end-to-end row (accuracy, sentence-level attribution precision, the abstention curve) that Bet 5
  asks for is not run here. What is guaranteed is attribution, not usefulness.
- **No API or MCP route yet**; only the engine method and the CLI.
- **Sentence scoring is the document's relevance**, not a per-sentence scorer, and citations are
  document-level (passages are merged per document), so two sentences from one file share a number.
- **No contradiction check.** `adjudicate_claims` can mark a claim `contradicted`; this composer
  emits any verbatim sentence above the score floor, so a contradicted span is still emittable.
  Gating emission on adjudication is the next step.

## Verification

Nine tests: sentence splitting; the verbatim guarantee with citations present in the answer;
citation numbering by document first appearance; duplicate suppression; empty input abstains;
`max_sentences` and best-first order; `min_score` filtering; the engine method end-to-end with a
stub model (verbatim against the engine's own passages); and the CLI parser. `make test`:
**1346 passed, 5 deselected**.
