# A budgeted, order-aware evidence context

The generation brief's A4 — "budgeted, order-aware context builder" — was listed as untried with
`build_messages` unchanged, and it was: every retrieved passage went into the prompt regardless of
size. `COMPOSE_EVIDENCE_TOKENS` now bounds the evidence region.

## What shipped

`build_messages(query, passages, *, sentinel=None, max_evidence_tokens=0)` groups passages by
document (unchanged) and, when a positive budget is given, keeps the **highest-ranked** document
blocks that fit — stopping at the first block that would overflow, never reordering, and always
keeping at least one block however large. The estimate is the repository's own four-characters-per-
token convention. `COMPOSE_EVIDENCE_TOKENS` (default `0` = unlimited, the historical behaviour) is
passed through `payload` by `Composer.compose`, so both the OpenAI-compatible and native request
bodies honour it.

The point is the failure mode it removes: a retrieval that returns many long chunks can push the
question out of a small model's window, and `build_messages` deliberately states the question at
**both** ends for that reason — a budget keeps the anchoring question inside the window instead of
truncating it away.

## A stale ledger row found while doing this

`proposal-ledger.md` §5 lists **"Spotlighting/datamarking of evidence"** as untried. It is already
shipped: `build_messages` wraps the whole evidence region in a fresh per-request sentinel and the
system prompt states that everything between the markers is untrusted data, never an instruction
(the docstring records the reasoning). The ledger row is corrected rather than re-implemented.

## What this does not show

- **`len(text)//4` is an estimate, not a tokenizer.** A budget tuned against a real model's tokenizer
  will differ; the field is a blunt bound, and it is documented as one.
- **It bounds the evidence region only**, not the system prompt or the question wrappers.
- **The default is unlimited**, so no existing composed output changes unless the field is set.
- **It is generator-side.** No retrieval number moves, and there is no measured answer-quality
  effect here — the composer is opt-in and this repository owns no generator.

## Verification

Five tests: no budget keeps every block; a small budget keeps only the highest-ranked block; a
generous budget keeps everything; at least one block survives a budget smaller than itself; and the
budget never reorders blocks. `make test`: **1332 passed, 5 deselected** plus these.
