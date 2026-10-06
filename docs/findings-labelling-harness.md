# A harness for the two owed hand-labelled sets

The roadmap owes two human-labelled evaluations, and they are the items this pass deliberately did
**not** fake:

- **~300 claim/corpus pairs** for the three-state adjudication's no-coverage recall (Phase 5.2); the
  shipped demonstration is three constructed claims.
- **~200 composed answers** for the verifier's precision (Phase 5.3); the shipped set is
  constructed, not human-labelled.

A harness cannot label them — a model-generated label would defeat the point of a hand-labelled set
— but it can make the human work bounded and its result scorable. `tools/labelling_queue.py`
(`make labelling-queue`) does exactly that and nothing more:

- **`build`** turns records into a JSONL review queue. Each item keeps a **`proposed_label`** (from
  the NLI head or the verifier, supplied by whoever generated the records) that the reviewer accepts
  or corrects, and starts with `label: null`, `labelled_by: null`, `status: "unlabelled"`. The
  proposal is **never** copied into the label.
- **`score`** reads the labelled queue and reports the reviewer counts, the label distribution,
  **agreement** between the human and the proposal, and the metric the roadmap actually owes —
  **recall of the `no_coverage` class** among human-labelled items — plus the verifier's per-label
  counts.

```sh
.venv/bin/python tools/labelling_queue.py build --input pairs.json --out queue.jsonl
# a human edits `label` and `labelled_by` in the JSONL
.venv/bin/python tools/labelling_queue.py score --input queue.jsonl
```

## What this does not show

- **No labels exist.** Nothing has been labelled; no no-coverage recall or verifier precision is
  computed here. The harness makes the work executable, it does not perform it.
- **The proposal must come from somewhere.** `build` does not run the NLI head or the verifier; the
  records must already carry a `proposed_label` (or `null`).
- **The queue must be generated from a run that persists claims and evidence.** The existing pilot
  reports store aggregate counts and answers but not the per-claim evidence text, so producing the
  ~300 pairs needs a run (or manual construction) that persists `(query, claim, evidence)`.
- **Agreement is not correctness.** A proposal and a reviewer can agree and both be wrong; the
  reported agreement is a reviewer-efficiency number, and the human label is the measurement.

## Verification

Eight tests: JSON-list and JSONL loading; the queue starts unlabelled with the proposal kept
separate; missing ids are assigned; `score` reports agreement, per-label counts and no-coverage
recall (2 items, 1 caught → 0.5) and does not invent labels when none exist; reviewers are counted;
and the CLI `build`→label→`score` round trip. `make test`: **1364 passed, 5 deselected** plus these.
