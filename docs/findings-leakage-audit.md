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

**590 JSON reports under `eval/` scanned; no dilution signature and no empty held-out block
found.** The hand-verification is now a gate: `make leakage-audit --check` exits `1` on any finding.

Two corrections to the first version of this finding, both from running it against the `eval/`
tree as it now stands (166 GB, 27 GB of it JSON):

- **It had started to crash, so it had stopped being a gate.** `eval/bioasq-src/allMeSH_2020.json`
  is a 24 GB latin-1 MeSH dump; `path.read_text(encoding="utf-8")` raises `UnicodeDecodeError`,
  which is not an `OSError` and so was not caught. The tool died on the first file it could not
  decode, and the "392 reports scanned" recorded here was measured before BioASQ's source data
  landed. A gate that crashes is worse than no gate: it reads as green in a log.
- **Its coverage claim was unstated, and is now stated.** A 24 GB corpus takes minutes to read and
  is not a report, so files above a **32 MiB** cap are skipped — and *named*, with a reason, in the
  output, as are empty files (HuggingFace's `.no_exist/` sentinels) and list-shaped id caches. The
  count reads **590** scanned with **142** exclusions listed rather than dropped. An audit's
  coverage is itself a claim; the tool now makes it checkable.

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
- **The size cap is a threshold, not a type test.** A report larger than 32 MiB would be excluded
  and named rather than audited. None exists here — the largest thing the audit excludes that a
  tool *wrote* is a 49 MB cross-encoder score cache — but the cap is why "590 scanned" is not
  "every JSON file under `eval/`".

## Verification

Twelve tests: nested per-query discovery; a diluted report is flagged; a consistent limited report
passes; a held-out block without a checkable field is flagged and one with it passes; `audit_tree`
over a synthetic `eval/`; `--check` exits `1` on a finding; a non-UTF-8 file is named rather than
fatal; a malformed report, an oversized file and an empty sentinel are each named; and the
markdown states both its coverage and its holes. `make test`: **1,543 passed, 5 deselected**.

