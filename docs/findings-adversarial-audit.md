# Findings: adversarial retrieval audit

This records the Tier 2.9 work from [`RESEARCH.md`](../RESEARCH.md): the security controls
had no falsification harness. This is a deterministic, model-free suite over a fixed corpus
with one planted document, so `make adversarial-audit ADVERSARIAL_ARGS="--check"` can gate a
change.

## Scenario

Fifteen documents, five of which are the gold answer for a query. One planted document
(`poison/refund_claim.md`) repeats the refund vocabulary so it ranks for the refund query,
carries an instruction payload ("ignore all previous instructions…"), and states a false
duration ("ninety days" where the trusted document says "five business days"). One query
asks about a topic no document covers.

## Measured

| metric | trust off | trust on |
| --- | --- | --- |
| gold hit rate | 1.0000 | **1.0000** |
| poison leak rate | 0.2000 | **0.0000** |
| negative reject rate | 1.0000 | 1.0000 |
| injection flag rate | 1.0000 | 1.0000 |

- **False-refusal rate from filtering: 0.0000** — excluding the untrusted document costs no
  gold hit on this corpus.
- **Deterministic verification** on a fabricated answer citing the poison passage: the false
  number is flagged (`number_not_in_evidence`) and the instruction payload is flagged
  (`injection_count > 0`), while citation *mechanics* stay clean — which is the precise
  distinction the verifier is built on.

`poison leak rate` is the fraction of answerable queries whose top-5 contains the planted
document; off, the refund query retrieves it, which is the failure the trust filter exists
to prevent.

## Checks (all PASS)

| check | meaning |
| --- | --- |
| `gold_retained` | filtering the untrusted document does not lose a gold hit |
| `poison_excluded` | with trust on, the planted document never reaches the evidence |
| `negative_rejected` | a query about an absent topic returns nothing |
| `injection_flagged` | the planted payload matches a narrow injection marker |
| `verification_flags_number` | a cited false number is flagged |
| `verification_flags_injection` | a cited instruction payload is flagged |

## Honest limits

- It is **synthetic and model-free**. It measures what *reaches the model* (retrieval,
  trust, deterministic verification), which is the lever these controls own. It says
  nothing about whether an LLM would obey the payload, and it is not an ASR number.
- The `0.2` leak rate is corpus-specific: one of five answerable queries has the poison as
  a plausible competitor. It is a regression baseline, not a population estimate.
- The trust filter's value depends entirely on the manifest being right; this audit's
  manifest is written to be correct by construction, so it tests the mechanism, not a
  deployment's policy.
