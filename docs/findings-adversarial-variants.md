# Seeded noise and injection variants

The adversarial audit shipped with a poison dose response and nine checks, and the roadmap left
one item owed: **noise and injection variants with frozen seeds**. Both are now in
`tools/adversarial_audit.py` and gated by `make adversarial-audit --check`.

## Seeded noise

`noise_probe` generates distractor documents from a fixed RNG seed (`NOISE_SEED = 20261006`) and
builds the retrieval index at doses 0, 1, 5, 20 and 100. The vocabulary is the corpus's **own**
content words, so the noise competes on the retrieval channel rather than being trivially ignored;
dose 0 is the control and must reproduce the clean rate.

| noise docs | gold hit rate | negative reject rate | queries with noise in top-k |
| --- | --- | --- | --- |
| 0 | 1.0000 | 1.0000 | 0.0000 |
| 1 | 1.0000 | 1.0000 | 1.0000 |
| 5 | 1.0000 | 1.0000 | 1.0000 |
| 20 | 1.0000 | 1.0000 | 1.0000 |
| 100 | 0.6000 | 1.0000 | 1.0000 |

Two readings, and the first is the useful one. **Negative rejection is not disturbed at any dose**:
`BM25Index.search` returns only positive-scoring documents, and a generated document that shares no
query term scores zero, so it cannot make a negative query answerable. **Gold retention is a
density effect**: 20 vocabulary-sharing distractors still leave every gold document in the top-5,
and 100 drops it to 0.6. That is not a threshold the suite enforces — it is a curve, and a change
that makes the engine more noise-robust shows up as a higher curve. The only gated property is the
dose-0 control (`noise_control_clean`).

## Injection variants

`injection_probe` runs nine named payloads through `verify.injection_flags`: seven covering each of
the six narrow markers (with a case-variant for `ignore_instructions`) and **two controls**, one
plain and one a near-miss (`"Ignore the noise and read the refund policy."`) that must not fire.

| variant | expected | found | ok |
| --- | --- | --- | --- |
| ignore_instructions | ignore_instructions | ignore_instructions | yes |
| ignore_case | ignore_instructions | ignore_instructions | yes |
| disregard_instructions | disregard_instructions | disregard_instructions | yes |
| role_reassignment | role_reassignment | role_reassignment | yes |
| chat_role_marker | chat_role_marker | chat_role_marker | yes |
| new_instructions | new_instructions | new_instructions | yes |
| secrecy_directive | secrecy_directive | secrecy_directive | yes |
| benign_policy | — (control) | — | yes |
| benign_near_miss | — (control) | — | yes |

**Marker recall 1.0000, control false positives 0.** Both are gated. The control arm is the one that
matters: the injection signal is *advisory*, so a marker that fires on a benign sentence would train
a reader to ignore the signal, which is worse than a miss.

## Reproduction

```sh
make adversarial-audit --check
```

`build_report` now carries `noise` and `injection_variants` sections and three new checks
(`injection_variants_flagged`, `injection_controls_clean`, `noise_control_clean`), for **12 checks**
total. `tests/test_adversarial_audit.py` pins determinism, the dose-0 control, the high-density gold
drop, full marker recall and the near-miss.

## What this does not show

- **The corpus is synthetic and tiny** — 13 documents, 6 queries. The noise curve is a property of
  this corpus and BM25's settings, not a general web-spam result.
- **The noise is one distribution.** Drawn from the corpus vocabulary with a uniform RNG; real spam,
  near-duplicate boilerplate and adversarial paraphrases are not sampled.
- **Marker recall is not resistance.** A payload being flagged says the deterministic checker saw an
  instruction-like marker; it says nothing about whether a generator would obey it — the engine
  cannot generate, so the flag is the whole of the defence here.
