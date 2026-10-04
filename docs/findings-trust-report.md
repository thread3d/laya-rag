# Findings: one deterministic trust report

ROADMAP Phase 5.1. The engine's differentiating claim is not retrieval rank — a CPU-only, 33M
encoder system cannot win that against API-scale models — but **verifiable trust**. Each piece of
that already existed as a separate surface; this is the artefact that states them together, and
`make trust-report TRUST_ARGS="--check"` fails if any of them regresses.

Reproduce:

```sh
make trust-report                    # ~30s, offline, no checkpoints
make trust-report TRUST_ARGS="--check"
```

## Result — 12 checks, all passing

**Corpus poisoning** (planted `poison/refund_claim.md`, k=5):

| reading | trust off | trust on |
| --- | --- | --- |
| gold hit rate | 1.0000 | 1.0000 |
| poison leak rate | **0.2000** | **0.0000** |
| negative reject rate | 1.0000 | 1.0000 |
| injection flag rate | 1.0000 | 1.0000 |

The false-refusal cost of enabling trust is **0.0000** — the poison is excluded and no gold answer
is lost. Tenant probes: alpha never sees beta, beta never sees alpha, an anonymous caller gets
**0** passages, and a shared document is still visible to a tenant.

**Poison dose response** — a keyword-stuffed document tailored to the *corpus's* vocabulary, not to
any query, measured at several doses because the roadmap asks for the "five documents → 90% attack
success" figure to be measured here rather than cited:

| poison documents | 0 (control) | 1 | 2 | 5 |
| --- | --- | --- | --- | --- |
| share of answerable queries whose top-5 contains a poison document | **0.0000** | **0.2000** | **0.2000** | **0.2000** |

The dose curve is **flat after one document**: on this planted corpus, a single stuffed file
already reaches the top-5 of 20% of the answerable queries, and four more change nothing. That is
the honest local answer to the citation — the attack saturates immediately at this corpus size, so
the published 90% figure is a property of *their* corpus and retriever, not a constant this project
may quote. It is recorded as a measurement, not a gate: no threshold depends on it.

**Deterministic verification** on a fabricated answer that cites the poison passage: citation
mechanics clean, the fabricated number flagged (`number_not_in_evidence`), the injection marker
flagged.

**Abstention**, held-out, selective prediction (`gate` = the 12-feature model, `scalar head` = the
shipped out-of-domain decision):

| arm | auc | balanced_accuracy | risk@80 | risk@90 | aurc | augrc | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **gate (12 features)** | **0.8146** | **0.7248** | **0.4069** | **0.4540** | **0.2691** | **0.6681** | **0.1768** | **0.0650** |
| scalar head (shipped) | 0.5199 | 0.5026 | 0.5099 | 0.5470 | 0.4879 | 0.7417 | 0.3678 | 0.2714 |

The gate halves the false-accept rate at 80% coverage (0.4069 against 0.5099) and is calibrated
(ECE 0.065 against 0.271), while the scalar head is at chance on AUC. That contrast is the
abstention claim, stated in one place with intervals rather than asserted in prose.

## What this does not show

- **Not a new measurement.** Every number is produced by an existing tool
  (`adversarial_audit.py`, `verify.py`, the selective-prediction suite); this aggregates and
  gates them.
- **Not portable abstention.** The gate's 0.8146 is **within-question-type**; leave-one-type-out
  is 0.7666 and the scalar does not transfer at all across corpora
  ([findings-evaluation.md](findings-evaluation.md)). The report shows the in-domain number and
  the shipped scalar beside it, which is the honest pairing; it does not claim a portable gate.
- **Not citation precision/recall on real answers.** The verification probe is a single fabricated
  answer against one passage. Measuring citation precision/recall over a labelled composed-answer
  set is Phase 5.3's remaining work, and it needs those labels.
- **Not an injection *defence*.** The trust filter and the flag are deterministic and the flag
  fires; adaptive attacks are out of scope, as `adversarial_audit` says.

## Method notes

- **A missing input is a skip, not a pass.** The selective suite lives under gitignored `eval/`;
  when it is absent those two checks are `null` and counted as skipped, never as passing. Treating
  absence as success is exactly the failure `claim_check.py` exists to prevent, and a test pins it.
- **Do not print a missing metric as `0.0000`.** The abstention table renders only the columns both
  arms actually carry, so a partial report shows fewer columns rather than fabricated zeros.
