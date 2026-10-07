# The cross-document consistency gate

`claim_check` pins a number in prose to the *report* that produced it. It does not pin two
documents to each other, and that is a different failure: the audit of this planning corpus found
the same quantity stated two ways in two files, each individually consistent with its own context
and neither compared to the other. This is the gate for that class, and the six drifts it was built
to catch.

## What was wrong

| quantity | where it disagreed | authoritative value | why the disagreement happened |
| --- | --- | --- | --- |
| BEIR coverage | `BENCHMARKS.md` "5 of 18"; `ROADMAP.md` "7 of 18" and later; `PROGRAMME-STATUS.md` "10 of 18" | **18 of 18** (counted from `eval/beir/*.json`; it read **13 of 18** when this gate was built, and the registry site moved with the reports) | the count is a moving target written into three documents by hand at three times, and never recomputed |
| published `voyage-02` + reranker | `README.md` 0.662 / 0.479; `BENCHMARKS.md` 0.663 / 0.480 | **0.6625 / 0.4795** (paper Table 5, pinned in `tools/compare.py` `PUBLISHED_MH`) | one table truncated to three decimals, the other rounded half-up |
| recalibration half-splits | `recalibration.md` 300; `README.md` and `BENCHMARKS.md` 500 | **300** (`recalibration.json` `transfer.splits`) | the two prose copies were written from memory of an earlier draft |
| lexical-control agreement | `findings-beir-breadth.md` pinned at 5/10 and 8/10; the `docs/FINDINGS.md` index said "8 of 9 within ±0.008" | **5 of 10** within ±0.008, **8 of 10** within ±0.02 | the index carried its own hand-typed copy of a sentence the breadth document had already had corrected once |
| peak RSS, 500–40,000-chunk sweep | `bench.md` 4,437.9–5,953.5 MB; `bench-scale.md` 64.9–2,038.5 MB | neither is wrong | `ru_maxrss` is a **process** high-water mark; `bench.md` ran after the Laya head was resident, `bench-scale.md` from a fresh process |

## What the gate does

`tools/consistency_check.py` (`make consistency-check`) holds a registry of
quantity → authority → every document site that states it, and fails when:

- an anchor matches no line (a prose edit moved the number off its site);
- a site's stated value is not the authority **rounded half-up** to that site's precision — the
  rule that makes 0.6625 render as 0.663 rather than 0.662;
- a superseded report is present without a quarantine marker, names a different replacement, or is
  cited by a document or by a `claim_check` claim;
- the computed BEIR coverage count disagrees with any document that states it;
- a document in the audit set above the repository breaks one of the two conventions in the next
  section.

Three design choices matter:

1. **The coverage count is computed, not typed.** `measured_beir()` reads the `datasets` keys of
   every report in `eval/beir/`, excludes superseded stems and bounded runs, and reports the
   membership with the count. A bounded run (TREC-COVID, a 40,000-document subsample) is listed
   but does not count toward "of 18".
2. **The external authority is read from the one place it is pinned.** The `voyage-02` row comes
   from `tools/compare.py`'s `PUBLISHED_MH`, so the docs and the chart cannot disagree about the
   same transcribed number. Corpus rows the audit quotes are read the same way, from the report's
   own `retrievers.bm25-engine` block.
3. **Superseded is machine-readable.** Ten pre-fix BEIR reports (the ArguAna BM25 row before the
   flat-control fix: 0.295 against 0.3935) sit beside their corrected replacements by design. Each
   now carries a `<stem>.superseded.json` marker; `--fix` writes them, and the check refuses to let
   one be read as a result.

## The audit set above the repository

`best-in-the-world/` (`measurement-audit.md`, `proposal-ledger.md` and the two briefs) reads this
repository's numbers from outside it and is **not committed to this repository**, so it has no
registry entry and cannot have required sites. It drifted exactly as the repository documents did:
after FEVER and the full HotpotQA run landed, its coverage count stayed at 10 and 11 of 18 and its
corpus table still called FEVER "queued". `check_audit_documents` holds it to two conventions:

- **A `> **Currency:**` banner** in every audit document, saying which sections are pinned and
  which are dated findings. Without it a reader cannot tell a current number from a snapshot.
- **The computed coverage count on every unfenced "of 18".** A dated finding keeps the number it was
  written with, but only inside a fence:

  ```markdown
  <!-- consistency: snapshot 2026-10-06 -->
  the audit found the count stated as 5, 7 and 10 of 18
  <!-- consistency: /snapshot -->
  ```

  The opening marker must carry a date; an undated or unclosed fence fails, and the fenced lines are
  removed before the count is read, so history is exempt only while it is declared as history.

A superseded report cited from an audit document is a failure too: `check_references` scans the
audit set alongside the repository. Because the directory is a sibling of the repository and not
part of it, an absent `best-in-the-world/` skips every audit check — a checkout of `laya-rag` alone
still passes, exactly as it does with `eval/` absent.

## Reproduction

```sh
make consistency-check
.venv/bin/python tools/consistency_check.py --fix   # write markers + eval/beir/coverage.json
```

`make consistency-check` skips (exit 0) when `eval/beir/` holds no reports, as `claim-check` does:
`eval/` is gitignored and regenerable.

## What this does not show

- **Agreement is not correctness.** The gate proves two documents state the same value and that the
  value matches its declared authority. If the authority is wrong — the transcribed paper row, or a
  report on disk generated before a bug fix — every document agreeing on it is still wrong. The
  superseded-report registry is the mitigation for the second case, and it is a human-maintained
  list, not a detection rule.
- **It does not scan every prose claim, only registered sites.** A quantity that is stated once and
  never registered is out of scope; `claim-check` is the tool for report-backed prose. The audit set
  above the repository is the one exception: it is scanned generically for "of 18", because that
  count is the quantity its documents drifted on, and it carries no registry of its own.
- **The coverage count can over-count** if a stale report not in the superseded registry is left on
  disk, because membership is a union over every report present. The registry is the guard, and it
  needs the same discipline as `SUPERSEDED` above.
