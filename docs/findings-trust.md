# Findings: retrieval-time trust tiers

This records the Tier 2.8 work from [`RESEARCH.md`](../RESEARCH.md): the index had no notion
of authority, so a document added through the self-serve `/v1/index` route, pulled in from a
dependency, or edited on disk could rank for a query and be spliced verbatim into the
composer prompt. This adds a deterministic, model-free filter.

## Design

A **manifest** maps path globs to one of three tiers and a query sets a minimum:

| tier | rank |
| --- | --- |
| `quarantined` | 0 |
| `untrusted` | 1 |
| `trusted` | 2 |

```json
{ "default": "untrusted", "rules": { "*/docs/*": "trusted", "*/vendor/*": "untrusted" } }
```

- **First matching rule wins**, so a manifest can put a broad rule first and exceptions
  after it in file order.
- When a manifest is configured, anything it does not name falls to `default`, which is
  **`untrusted`** by default: an allow-list is the only safe reading of "these are the
  documents I vouch for".
- An unknown tier name is a load error, not a silent pass.
- With **no manifest configured nothing is filtered**, so the shipped behaviour is
  unchanged; the filter only runs when a manifest *and* a query minimum are both set.

Configured through `LAYA_RAG_TRUST_MANIFEST`, `LAYA_RAG_TRUST_DEFAULT_TIER` and
`LAYA_RAG_QUERY_MIN_TRUST`. `stats()` reports whether a manifest is configured, whether it
loaded, the default tier, the minimum and the rule count.

## Measured behaviour (deterministic, no model)

A two-document corpus — one trusted, one carrying a distinctive "secret escalation code"
— indexed into the engine with a stub model, `min_trust=trusted`, and the poison path
marked untrusted:

| query | `trust_filtered` | passages returned |
| --- | --- | --- |
| the poison document's content | > 0 | **0** (100% excluded) |
| the trusted document's content | ≥ 0 | trusted passage only; **0** poison paths leak |

With the manifest present but no query minimum, `trust_filtered` is 0 — the filter is
opt-in per query, as designed. The whole behaviour is pinned by 13 offline tests in
`tests/test_trust.py`.

## What it does and does not do

- It is a **filter, not a detector.** It enforces an ingest policy that a human or an
  upstream process wrote; it does not decide whether a document is malicious. A wrong
  manifest produces wrong filtering.
- It does **not** defend against prompt injection from a *trusted* document — that is what
  the spotlighting and injection flags address. The two are complementary: trust decides
  what may be retrieved, spotlighting decides how retrieved text is presented.
- Its cost is one `fnmatch` per candidate; the recall depth is widened (`recall_k * 3`)
  when it is active so excluding a document still leaves deeper candidates to refill.
- The honest caveat: it is a **capability**, and its quality is exactly the quality of the
  manifest. The value measured here is correctness (an untrusted document is never
  returned), not retrieval quality.
