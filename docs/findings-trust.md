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

## Per-tenant ACL

The same manifest can carry a `tenants` block mapping a caller identity to the path globs it
may retrieve:

```json
{ "default": "trusted",
  "tenants": { "alpha": ["alpha/*", "shared/*"], "beta": ["beta/*", "shared/*"] } }
```

`RagEngine.query(..., tenant="alpha")` (and the `tenant` field on `POST /v1/query`) applies
it. The semantics are deliberately **fail-closed**: once a tenant block exists, a request
with no tenant, or an unknown tenant, retrieves nothing. With no `tenants` block every caller
is allowed, so single-tenant deployments are unchanged. `acl_filtered` on the response counts
the exclusions separately from `trust_filtered`, and `stats()` reports the tenant names and
whether the ACL is enabled.

Measured by `make adversarial-audit` on a three-document corpus (two tenants plus a shared
document):

| check | result |
| --- | --- |
| alpha retrieves a beta document | **false** |
| beta retrieves an alpha document | **false** |
| a request with no identity retrieves anything | **false** (0 passages) |
| a shared document is visible to a listed tenant | **true** |

The isolation probe and all the trust/injection checks pass together
(`ADVERSARIAL_ARGS="--check"`).

### Limits

- It is a **filter keyed on a caller-supplied identity**, not authentication. The tenant
  string must come from a trusted place — the service's own auth layer — or a caller can
  simply claim another tenant's name. The API's single optional bearer token does not
  distinguish tenants today; a real deployment maps token → tenant outside this engine.
- Path globs are only as correct as the ingestion layout; a document filed under the wrong
  directory is isolated by that mistake, in both directions.
