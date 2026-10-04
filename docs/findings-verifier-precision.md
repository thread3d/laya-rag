# Findings: the deterministic verifier's precision, and where it is weak

ROADMAP Phase 5.3. `verify.content_issues` raises four **advisory** findings — a number, a date or a
unit that disagrees with the cited evidence, and a possible negation flip — and the roadmap's
shipping bar is precision ≥ ~0.9. Precision is the bar that matters for an advisory check: one that
flags everything is worse than none, because a caller learns to ignore it.

Reproduce:

```sh
make verify-precision                      # writes eval/verify-precision.{json,md}
make verify-precision VERIFY_ARGS="--check" # exit 1 below the bar
```

## Result — constructed labelled set, 400 cases

| kind | positives | recall | precision | false positives |
| --- | --- | --- | --- | --- |
| `number_not_in_evidence` | 50 | **1.0000** | **1.0000** | 0 |
| `date_not_in_evidence` | 50 | **1.0000** | **1.0000** | 0 |
| `unit_mismatch` | 50 | **1.0000** | **1.0000** | 0 |
| `possible_negation_flip` | 50 | **1.0000** | **1.0000** | 0 |

**False-alarm rate on the 200 faithful controls: 0.0000.** Every detector clears the ≥0.9 bar on
clear-cut mutations, over two unit families (size and time) for the unit detector.

## Where it is actually weak: the near-misses

A perfect score on unambiguous mutations proves the plumbing, not the precision. The harness
therefore also runs six **known-hard** cases — each a documented *legitimate exception* that a
reader would call faithful, and that should therefore be quiet:

| case | flagged kinds | quiet |
| --- | --- | --- |
| same date, another format (`2024-03-12` vs `March 12, 2024`) | `date_not_in_evidence`, `number_not_in_evidence` | **no** |
| rounding (`1205` vs "about 1200") | `number_not_in_evidence` | **no** |
| range endpoint (`10 days` vs "between 5 and 10 days") | — | yes |
| number spelled out (`5` vs "five") | — | yes |
| percent as decimal (`5.0%` vs `5%`) | — | yes |
| negation in both texts | — | yes |

**Hard-case false-alarm rate: 0.3333 (2 of 6).** Both failures are the classes `verify.py` already
documents, and neither changes the `clean` verdict — but they are worth naming precisely:

- **A date in another format is the worst case**, because it produces *two* spurious findings: the
  date is unrecognised as equal, and its digits are then read as numbers absent from the evidence.
  This is the one worth fixing if precision on real answers matters, since composed prose routinely
  reformats dates.
- **Rounding** is inherent to comparing literals: `1205` is genuinely not `1200`.

## What this does not show

- **Not hand-labelled.** The roadmap asks for precision on ~200 hand-labelled composed answers.
  This set is **constructed** from templates with varied slots — reproducible and adversarially
  clear about each detector's blind spots, but a proxy. The hand-labelled evaluation is still owed,
  and nothing here should be quoted as if it were that.
- **Not real prose.** Template sentences are cleaner than composed answers: no pronouns, no
  multi-sentence claims, no hedging. A real answer is harder, and the hard-case table is the honest
  hint of which way.
- **Not a `clean` verdict claim.** These findings have severity `advisory` by design and do not
  affect the citation-mechanics verdict; this measures them on their own terms.

## Method notes

- **A perfect score on an easy set is not evidence.** The value of this harness is the hard-case
  table and the gate, not the 1.0000s; a test asserts the two known failures stay known, so a future
  fix must update the write-up in the same change.
- **Precision needs a negative that *should* be quiet.** Every positive is paired with a faithful
  control built from the same template, which is what makes a false positive countable at all.
