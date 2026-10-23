# Changelog

All notable changes to `laya_rag`. Versions follow [semantic versioning](https://semver.org/).

Earlier development predates this file, so the history below begins at the first tagged release;
the measurement programme's own record is `SCORE-LIFT.md` §10 and `docs/FINDINGS.md`, which are
kept per-result rather than per-release.

## 1.1.1 — 2026-10-21

The first tagged release. Its substance is measurement, not new features: the BEIR-18 coverage
claim closed, two benchmark deviations explained or refuted, and the parallel scoring path that
made the largest of those runs affordable.

### Coverage

- **BEIR-18 is complete: 18 of 18 datasets carry a measured row.** `bioasq` was the last
  (14,914,604 documents, 500 queries, `bm25-engine` nDCG@10 **0.5193** against a published 0.465),
  so the macro is now computed over the full 18 rather than a subset.
- **MS MARCO remains an extra, not coverage** — it is the collection the BEIR encoders were trained
  on, and the gate counts over an explicit membership set.

### Two deviations explained, one of them against the engine's favour

- **FEVER's −0.121 was a settings artefact.** A per-field BM25F control on the same 6,666 queries and
  5,416,568 documents reads **0.7288**, against the flat control's 0.6323 and a published 0.753: the
  gap narrows to **−0.0242**, so ~80% of it was a flat-versus-multi-field difference rather than a
  retrieval deficit. This also means the headline summed deficit (**0.6154**) is an **upper bound**
  on the engineering gap, bounded in size by the measured title-weight sweep.
- **P11, the pool bet, converts but does not transfer.** A coverage-quota blend reaches **0.3147
  MAP@10**, significantly above `ce` (**+0.0266 [+0.0230, +0.0303]**) and the learned `ltr`
  (**+0.0208 [+0.0130, +0.0287]**); off-corpus it falls **below `ce`** (MuSiQue held-out 0.3984 vs
  0.4032), while a plain **`blend-0.75` transfers** (**+0.0120 [+0.0088, +0.0150]**). Hard-negative
  restriction collapses the ranker (**0.2939 → 0.0257**) through a training/test distribution shift.
- **Passage-level retrieval is disproven on genuinely long documents**, not merely untested:
  Robust04 **0.4077 → 0.3522**, TREC-NEWS **0.3351 → 0.2456** at a 600-character budget.

### Performance

- **`--jobs N` for the on-disk scoring passes**, pooled over queries with
  `mp.get_context("fork")` — macOS defaults to `spawn`, which re-imported and re-opened the index per
  worker and could show zero pool workers as if hung. Verified equality-gated on `scifact`
  (`--jobs 1` vs `4`: identical metric dicts, only `seconds` differs; **3.25×**).
- **Progress is now printed during scoring**, in the function that actually runs. A phase counter in
  the wrong (in-memory) function had produced seven hours of silence that looked like a stall.
- The on-disk BM25F `build()` accepts an **injected vocabulary**, and run files carry the pid — the
  two preconditions for generating index runs in parallel.
- The query deepening loop is **bounded** (8 widenings), so a filter that rejects most candidates can
  no longer re-rank exponentially without terminating.

### Provenance

- **Every BEIR and engine report now records the tool that wrote it and the arguments it was given**
  (`settings.producer`, `settings.argv`), with source-level guards in the test suite. Four static
  screens had failed to identify a report's producer from its settings alone.

### Known limits

- **SPLADE-v3 and doc2query cannot run on this host.** Both ship `pytorch_model.bin`, and torch 2.2.2
  — the newest macOS x86_64 wheel — is refused by `transformers` 4.57 under CVE-2025-32434. This is a
  platform limit, not a tuning choice; bypassing the guard was declined deliberately.
- **Six cross-encoder reports cannot be re-taken**: their writer is not in the tree, and an identical
  settings block proved not to be evidence of authorship. Their numbers stand with that caveat.
- **MTEB evaluation does not complete here** (`mteb.evaluate` forks ~33 workers and stalls); no MTEB
  score is claimed.
