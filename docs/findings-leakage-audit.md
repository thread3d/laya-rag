# Leakage audit: the dilution signature and empty held-out blocks

The evaluation brief (§9) asks for a contamination/leakage audit, and this repository has already
been bitten by the exact trap it needs. `--limit-queries` passed the **full qrels** alongside a
capped ranking set, so the evaluator macro-averaged over queries that were never scored and divided
by the wrong denominator — found in **four separate tools**. What made it fixable was that a report
persists per-query arrays, which makes the failure **decidable from the report alone**: a limited run
that scored the full qrels has a per-query array *longer than the limit it recorded*.

`tools/leakage_audit.py` (`make leakage-audit`) formalizes the check that was done by hand — "no
recorded report used a limited run, which was checked rather than assumed" — and adds one weaker
check.

1. **Dilution.** A report that records a query limit must not have any `per_query*` array (at any
   shallow nesting) with more entries than that limit.
2. **Held-out reading present.** A report that carries a `holdout` / `held_out` / `transfer` block
   must carry the fields that make the held-out claim checkable (`selected_on_a` +
   `selected_b_score`, a `gate_*` metric, or `splits`) — catching a tuning block recorded without
   its reading.

## Result

**392 JSON reports under `eval/` scanned; no dilution signature and no empty held-out block
found.** The hand-verification is now a gate: `make leakage-audit` exits `1` on any finding.

## What this does not show

- **It is not the encoder-contamination audit the brief's §9 names.** "Zero-shot claims assume the
  encoder has not seen the test queries" cannot be checked from a report; this tool checks the
  repository's own evaluation plumbing, not a model's pretraining corpus.
- **It only covers reports that persist per-query arrays and settings.** A report with scalars only
  is invisible to the dilution check; the coverage is the reports' own discipline.
- **The held-out check is presence-only.** A block can name its fields and still be computed
  in-sample; it does not verify the split.
- **`per_query` discovery is shallow and name-based** (depth ≤ 3, keys starting with `per_query`);
  a differently named structure is not inspected.

## Verification

Seven tests: nested per-query discovery; a diluted report is flagged; a consistent limited report
passes; a held-out block without a checkable field is flagged and one with it passes; `audit_tree`
over a synthetic `eval/`; and `--check` exits `1` on a finding. `make test`: **1350 passed, 5
deselected** plus these.
