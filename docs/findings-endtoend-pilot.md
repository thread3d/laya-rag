# Findings: an end-to-end pilot, and what the full row would cost

ROADMAP Phase 3. `BENCHMARKS.md` records that the project has **no end-to-end ranking** because no
end-to-end benchmark has been run to completion. This produces the first row, under the project's
own attribution rule: the answer quality is `engine + <model>` and is labelled with the generator,
while the engine's half — what it retrieved, whether it chose to answer — is scored separately.

Reproduce:

```sh
make endtoend ENDTOEND_ARGS="--answerable 12 --null 8"
```

## Protocol

The engine runs at its **shipped defaults** — the relevance and answerability gates are active, so
abstention is a decision rather than a threshold switched off for measurement. If it abstains, no
prose is written. Otherwise the host's local `ornith-1.5:35b` writes an answer from the retrieved
evidence, and the answer is scored three ways:

* **EM and F1** — SQuAD-style string scores against the benchmark's gold `answer`;
* **gold contained** — whether the normalized gold string appears in the prose. EM and F1 compare
  token multisets, which a *prose* answer almost never matches against a short entity gold
  ("Sam Bankman-Fried was the founder…" against "Sam Bankman-Fried"); containment is the
  deterministic proxy for a reader's judgement;
* **abstention**, scored separately on the answerable records (where abstaining is an **error**) and
  on the benchmark's `null_query` records (where abstaining is **correct**).

## Result — 20 queries, `ornith-1.5:35b`, retrieval index `engine-hybrid-400`

| reading | value |
| --- | --- |
| answerable records | 12 |
| abstained on answerable (error) | 3 (**25.0%**) |
| answered | 9 |
| exact match | **0.0000** |
| F1 | **0.0262** |
| **gold contained** | **0.6667** |
| null records | 8 |
| rejected on null (correct) | 3 (**37.5%**) |
| generator errors | 0 |
| seconds per query | **~36** |

Two readings, and the first is the one to take seriously:

1. **EM 0.000 against a 0.667 containment rate is the metric, not the system.** Every composed
   answer is *prose*; the golds are short strings. An answer that contains the right entity scores
   EM 0 and F1 0.026. Any end-to-end row for this engine has to report containment (or a judge),
   and saying "EM 0.00" without the containment figure beside it would be misleading.
2. **The abstention gap is +12.5 points on 20 queries and means nothing.** The project has already
   measured this properly on 601 queries: 30.9% abstention on nulls against 28.7% on answerables,
   AUC 0.509. The pilot is consistent with "the gate behaves like a fixed policy out of domain",
   and is not evidence either way.

## The cost finding, which decides what is possible here

The pilot ran **20 queries in 715 s — about 36 s/query**, nearly all of it generation (the engine's
retrieval is ~3 s). Extrapolated, the full MultiHop-RAG answerable set (2,255) plus the nulls (301)
is roughly **22–25 hours** on this host. That is not a blocker on the harness — it is a blocker on
treating a full end-to-end benchmark as a local job, and it is why Phase 3's remainder is listed as
external: the engine half is measured, the generator half is the cost.

## What this does not show

- **Not a benchmark row.** 12 answerable records is a pilot that proves the harness and produces a
  labelled number; it is not a ranking and no interval is quoted.
- **Not the benchmark's own assessment.** MultiHop-RAG scores answers with a judge; EM/F1 and
  containment are string measures. `verify.py` separately checks citation mechanics, which is a
  different and narrower question than whether the answer is right.
- **Not the engine's number.** The prose is `ornith-1.5:35b`'s; what is the engine's is the
  retrieval, the abstention decision, and the evidence bundle handed to the generator.
- **Neither the full cost axis nor FRAMES.** Per-query seconds are recorded, but peak RSS and index
  bytes per query are not; and the FRAMES adapter (Phase 3.1) was not built.

## Method notes

- **The index is reused, not rebuilt.** `--index-dir` points at the index from the Phase 1.2
  measurement, so the retrieval half is the *same* system those numbers describe, and the encoder
  must match the one the index was built with (`--embed-model`).
- **A generator is a per-query cost, not a per-benchmark one.** The report records `seconds` per
  query, which is what makes the extrapolation possible rather than a guess.
