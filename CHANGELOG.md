# Changelog

All notable changes to `laya_rag`. Versions follow [semantic versioning](https://semver.org/).

Earlier development predates this file, so the history below begins at the first tagged release;
the measurement programme's own record is `SCORE-LIFT.md` §10 and `docs/FINDINGS.md`, which are
kept per-result rather than per-release.

## 1.2.0 — 2026-10-24

Measurement-led again: the headline deficit is now known to be mostly configuration, two research bets are closed with
negatives, and two capabilities landed that were blocked on the security guard rather than on technique.

### The deficit was an upper bound, and now it has a number

A settings-matched per-field BM25F control was run against each cell's registered anchor, seven cells in all:

| cell | flat gap | per-field gap |
| --- | ---: | ---: |
| FEVER | −0.1207 | −0.0242 |
| TREC-COVID | −0.0372 | −0.0134 |
| DBPedia-Entity | −0.0077 | **+0.0144** |
| NQ | −0.0172 | **+0.0055** |
| Climate-FEVER | −0.0520 | −0.0351 |
| HotpotQA | +0.0322 | **+0.0623** |
| TREC-NEWS | −0.0629 | **−0.0653** |

**Mean −0.0379 → −0.0080**: about four fifths of the average deviation was configuration, three cells end at or above their
published anchor, and **TREC-NEWS is the counter-example** — no weighting recovers it. **Robust04 cannot be tested**: its corpus
carries no titles, so there is no second field to weight.

### The two residual cells fail for opposite reasons

- **TREC-NEWS is recall-bound with a favourable ranking** — 53 gold documents per query, only 45% found in the top 100, yet its
  first gold is usually rank 1–2 (`MRR@10` 0.7373).
- **Climate-FEVER is retrieval-bound on a sparse target** — three golds per query, 44% recalled, `MRR@10` 0.2362 (≈ rank 4).

Also recorded: `recall@100` is capped at `100/gold`, so TREC-COVID's 0.1212 is 60% of its achievable 0.203 while NQ's 0.79 is
79% of 1.0. Cross-cell recall comparisons need the ceiling.

### Closed with negatives

- **The pool bet (P11) does not survive training.** Coverage-quota converts on MultiHop-RAG (+0.0266 over `ce`, significant) but
  reverses off-corpus (MuSiQue held-out 0.3984 vs 0.4032), while a plain `blend-0.75` transfers (+0.0120 [+0.0088, +0.0150]).
  All three hard-negative variants fail: restriction collapses the ranker (0.2939 → 0.0257), weighting degrades it monotonically
  (0.2939 → 0.2780), and a **listwise** softmax is **+0.0016 [−0.0017, +0.0052]** — inside the noise — and **−0.0191
  [−0.0259, −0.0126] against the hand rule**.
- **Passage-level retrieval is disproven on genuinely long documents**: Robust04 0.4077 → 0.3522, TREC-NEWS 0.3351 → 0.2456 at a
  600-character budget.
- **The cross-encoder reports once called un-re-takeable do re-take**: their writer is `multihop_rerank_experiment.py` and all
  **18 rows of `rerank-experiment-full` reproduce to four decimals**.

### Two capabilities that were blocked on the security guard

- **`tools/dotbin_to_safetensors.py`** converts legacy `.bin` checkpoints **without `torch.load`** — a restricted unpickler whose
  `find_class` admits only tensor-rebuild callables, so a pickle naming `os.system` is refused rather than executed. Validated by
  behaviour rather than shape: the doc2query model generates *"who was convicted of fraud and conspiracy"* and
  `splade-v3-distilbert` expands *"arrested"* for a passage that never says it.
- **What it unlocked is only partly worth having.** The sparse channel pays **+0.0338** on scifact — but through the encoder
  already in the tree; the converted SPLADE is +0.0004. doc2query pays **+0.0095** (0.6762 → 0.6857) at **3.0 docs/s**, which is
  ~21 days for a 5.4M-document corpus. `naver/splade-v3` proper is **gated**, which is an access grant rather than a format
  problem.

### Harness and engineering

- **`--jobs N` pools every scoring phase** (`mp.get_context("fork")`, since macOS defaults to `spawn` and re-opens the index per
  worker); equality-gated on scifact, 3.25× on the large builds, 13 workers observed on a 5.4M-document corpus.
- **Sharded runs are safe**: `--dump-rankings` writes the per-query rankings, `--shard I/N` ranks a slice, and
  `tools/beir_merge_rankings.py` unions the sidecars and scores the union **once** against the full qrels. Both refusals are
  tested — `--shard` without a sidecar exits 2, and merging an incomplete union exits 2 — because a shard's report must never be
  mistaken for a dataset score. Verified end to end: two 150-query scifact shards merge to the same **0.6762** a single run reads.
- **Reports record their producer**: `settings.producer` and `settings.argv` on every BEIR and engine report, guarded by tests.
- **Progress is printed from the function that actually runs**; a phase counter in the wrong (in-memory) path had produced seven
  hours of silence that looked like a stall.

### Known limits

- **SPLADE-v3 proper and doc2query remain unpromoted**: v3 is gated, and doc2query's gain does not justify its cost at corpus
  scale on this host. Both would need MPS expansion or a smaller student model.
- **MTEB does not complete here**; no MTEB score is claimed.
- **Six cross-encoder reports** were re-taken and reproduce, but their writer's flags beyond the cache are not recorded, so only
  the cache-plus-defaults path is reproducible.

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
