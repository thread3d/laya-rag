# What could be added next: an external survey mapped to this repo's measured gaps

This is the **outward-facing** companion to [`IMPROVEMENTS.md`](IMPROVEMENTS.md).
`IMPROVEMENTS.md` diagnoses the system from the inside — it reads the code, names the file
and line, and ranks the fixes. This document looks **outward**: it surveys 2024–2026 RAG
research and tooling, then maps each idea onto a gap this repository has actually measured,
and states what it would cost here, what it should move, and how to falsify it.

The seven per-theme research briefs it summarises are in [`docs/research/`](docs/research/).
Each is self-contained, with dated citations, costs, and falsification tests.

It deliberately does **not** repeat anything already measured and rejected locally — the
ledger is §2 — and it labels every proposal by what it needs:

| tag | meaning |
| --- | --- |
| `[offline]` | runs today on this host with the current venv, no network |
| `[dep]` | needs a package install (network once) |
| `[data]` | needs a dataset download (network, disk) |
| `[model]` | needs a model download or a training run |
| `[gpu]` | realistically needs a GPU or many hours; not this host |
| `[llm]` | needs a generator; the resulting number belongs to the model, not the engine |

Host facts behind the cost estimates: Intel Mac (`x86_64`), 56 logical cores, Metal/MPS,
28 GB free disk (per `BENCHMARKS.md`), Python 3.11, `torch 2.2.2`, `transformers 4.57.6`,
`sentence-transformers 4.1.0`, `faiss-cpu 1.15.1`, `numpy 1.26`, `scipy`/`sklearn`, SQLite
with FTS5. **Not installed:** `onnxruntime`, `tantivy`, `lancedb`, `rank_bm25`.

---

## 1. The measured state a proposal has to beat

Everything below is quoted from `README.md`, `RANKINGS.md` and `BENCHMARKS.md`; none of it is
re-derived here.

| axis | measured | the gap that matters |
| --- | --- | --- |
| BEIR retrieval (5 datasets) | 0.452 single `bge-small`+BM25; **0.495** 3-model ensemble | already above published BGE-large; headroom is cost, not quality |
| MultiHop-RAG Hits@10 / Hits@4 | **0.864 / 0.712** with coverage dedup | already above every published row including reranked ones |
| MultiHop-RAG MAP@10 | **0.2835**; **0.2986** with the reranker on the 400-query subset | the headline retrieval gap |
| MultiHop pool-reorder oracle | **0.7307** at depth 100 | **the evidence is already retrieved; ordering / set-selection is the loss** |
| Answerability, out of domain | AUC **0.51** scalar; in-domain feature gate **0.81** | no portable calibrated abstention |
| Cross-encoder (BEIR vs MultiHop) | −0.027…−0.318 on BEIR; **+0.0464 MAP@10** on MultiHop | a second stage must be *routed*, not global |
| Scale | 5 of 18 BEIR datasets buildable; BioASQ 14.9M blocked | ingestion is the wall, not query time |
| Generation | opt-in Ollama only; no entity linker, no judge, no detector | cannot be scored end-to-end |
| Security | no tenancy, no ACL, no injection defense | a poisoned file is model-visible |
| ArguAna | pointwise CE 0.568 → 0.250 | relational (opposition) relevance is unmodelled |

**Two corrections to the usual framing, both important.**

1. **The published 0.479 is a *reranked* number; the internal bar is now 0.2986.** The
   repository's own `bge-reranker-base` row on the 400-query subset already reaches MAP@10
   0.2986, above the 0.2835 that the README compares against 0.479. Any new proposal should
   be judged against **0.2986**, not 0.2835 — otherwise a real regression can read as a gain.
2. **The single most important number is 0.7307.** The fused top-100 already contains enough
   gold evidence for an oracle reordering to reach MAP@10 0.7307. So the highest-value work
   is not "retrieve more" and not "embed better" — it is **deciding which candidates to keep,
   and how many distinct facts the kept set covers**. Most of the ranked list below points
   there.

---

## 2. The do-not-retry ledger (already paid for)

Proposing any of these again would waste a week. They are recorded here so the survey stays
honest about what has already been falsified **on this system**:

- BM25 pseudo-relevance feedback (SciFact 0.682 → 0.424).
- Fixed-size passage-level indexing + max-pool at abstract scale.
- CombSUM / max-normalised score fusion (RRF beats it because the scales do not normalise).
- A larger encoder (`bge-large` over `bge-base`): fusion absorbs it, measured twice.
- **Pointwise** cross-encoder as a full re-sort (all five BEIR datasets; ArguAna catastrophic).
- Laya's decision head as a ranker (it is calibrated as a filter).
- Naive full-split fine-tuning with mined hard negatives (below zero-shot).
- SPECTER v1 as the SciDocs fix (pooling bug fixed, still a negative).
- Recalibrating the single answerability scalar (monotone maps cannot create ranking signal).
- `bge-base`/`bge-large` under the MultiHop protocol: the fused hybrid does not improve.

The pattern is a useful prior: **the failures are almost all "trust a stronger pointwise
model more", and the wins are almost all "change the objective or the routing".**

---

## 3. The two mechanisms that explain the headline gaps

The research converges on two causes that the repository has not yet instrumented. Both are
testable in hours.

### 3.1 The second-hop tax: relevance selection discards the bridge

Multi-hop questions need a *second* passage linked to the query only through a bridge entity.
That passage is, by construction, **low relevance to the query string**. 2025 work
("The Second-Hop Tax", [findings](https://github.com/vysakh0/redhop/blob/main/docs/findings/SECOND_HOP_TAX.md))
measured directly, with bootstrap CIs, that any operation ranking or filtering by query
relevance — cross-encoder reranking, threshold filtering, density pruning — drops it.
**Verified against the primary source:** on the distractor-filtered strategy, second-hop
retention falls from **0.968 [0.958, 0.977]** at τ=0.05 to **0.439 [0.411, 0.468]** at τ=0.30
(HotpotQA, n=1327), and from **0.751** to **0.043 [0.034, 0.053]** (MuSiQue, n=1484). A
`reasoning_preserving` strategy that readmits low-relevance chunks **lexically** linked to a
kept seed recovers **+2.1 / +8.4 / +23.2 pts at τ=0.10/0.20/0.30** (HotpotQA) and **+28.8 pts at
τ=0.20** (MuSiQue), with non-overlapping CIs from τ=0.10.

This matters because **Laya's relevance filter and the cross-encoder are both exactly that
operation.** It also explains the local numbers without any contradiction: the CE helps
MultiHop on average (it promotes the first hop and some co-relevant passages) while still
taxing the bridge; the coverage pass recovers some of the loss by spending slots on distinct
facts. The oracle ceiling of 0.7307 is what remains after the tax.

The source's own honest limits, which bound the proposal: the rescue signal is **lexical**, so
a second hop sharing no surface tokens with its first hop (pure paraphrase linkage) is not
rescued; **retention is reachability, not answer quality**; and the density-pruning and MuSiQue
sub-results did not all survive a sharper grounding signal, so the filtering tax is the robust
two-dataset core. The document is a project findings note, not a peer-reviewed paper.

**Cheapest high-value experiment in this document:** instrument *second-hop retention* — for
each query, the fraction of gold facts whose chunk was in the fused pool but was dropped by
the relevance/CE filter — and split the depth-100 vs depth-500 ceiling. If retention is low
and depth-500 ≫ depth-100, expansion/rescue is the programme; if depth-500 ≈ depth-100,
it is purely a selection problem. `[offline]`, hours.

### 3.2 Topical single-vector matching cannot represent relational relevance

A single densely pooled query vector matches the query's *topic*; it cannot simultaneously
match a bridge entity and several distinct answer facts. The same defect explains ArguAna:
its gold is the **counter**-argument, so an *agreeing* passage is topically near-identical
and a pointwise MS-MARCO ranker scores it high
([Wachsmuth et al. 2018](https://aclanthology.org/P18-1023.pdf);
[Shi et al. 2023](https://arxiv.org/abs/2304.08807)). The fixes split by relation:

- **Multi-hop (topical but distributed):** late-interaction MaxSim — different query tokens
  can match different evidence facts.
- **Relational (opposition/contradiction):** comparative/listwise scoring, pairwise
  preferences, or an NLI/stance head. Late interaction does **not** fix ArguAna — the 33M
  `answerai-colbert-small-v1` scores 50.09 there but 76.11 on HotpotQA
  ([answer.ai, 2024](https://www.answer.ai/posts/2024-08-13-small-but-mighty-colbert.html)).

These are different instruments for different failure modes, and conflating them is how a
"general reranker" gets adopted and then loses on five of six corpora.

---

## 4. The ranked programme

Ranked by expected value per unit of cost and risk. Every Tier 0 and Tier 1 item is
`[offline]` except the two marked otherwise (0.6 needs `[data]`, 1.7 needs `[model]`), and
none needs a GPU or an API. The number to beat is **MAP@10 0.2986** on MultiHop — the
repository's own reranked row, not the 0.2835 no-rerank baseline — plus the published BEIR
rows in §1.

### Tier 0 — diagnostics and claim-hardening (hours, `[offline]`)

| # | addition | what it settles | falsify / signal |
| --- | --- | --- | --- |
| 0.1 | **Depth-500 ceiling + per-hop retention instrumentation** | is the loss recall (pool) or selection (ordering)? | if ceiling(500) ≫ ceiling(100), expansion is justified; if flat, selection dominates |
| 0.2 | **Paired permutation test + Benjamini–Hochberg FDR** across the comparison grid in `tools/evalkit.py` | the many "all significant" cells currently carry no family-wise control | pre-register the cell family; queries (not documents) are resampled |
| 0.3 | **Selective-prediction suite for the gate** (risk–coverage, AURC, risk@80% coverage, Brier/NLL, debiased ECE) | "AUC 0.815" says nothing about deployable risk | audit provenance of the "unanswerable" labels first |
| 0.4 | **Power / minimum-detectable-effect statement** on each query set | separates "no effect" from "underpowered" for the +0.005-scale deltas | report empirical SD as a range; post-hoc framing only |
| 0.5 | **Reranker-fairness pack** (R@10/100/1000, reranker-excluded union pool) | the +0.0464 CE gain is measured on a pool the reranker did not build | if it does not hold on a reranker-free pool, say so |
| 0.6 | **Replicate the coverage gain on MuSiQue + 2WikiMultiHopQA** `[data]` | the multi-hop headline rests on one dataset | the single highest-value falsification available; ~100 MB downloads |

### Tier 1 — cheap, high-certainty, model-free (days, `[offline]`)

| # | addition | gap attacked | expected | falsify |
| --- | --- | --- | --- | --- |
| 1.1 | **Bridge-linked second-hop rescue** — after relevance/CE filtering, readmit low-relevance candidates lexically linked to a kept seed (bridge-entity term/n-gram overlap); drop only unlinked junk | §3.1, the measured cause of the MAP gap; composes with the CE and the coverage pass | MAP@10 0.2986 → **0.31–0.33** hypothesis | log retention before/after; if retention rises without MAP, the tax is not the bottleneck |
| 1.2 | **CE × set-coverage joint selection** — feed CE scores into `facility_location_select` as the relevance term, plus bridge-rescue and a per-doc quota | both parts already measured positive; they have never been combined | +0.01–0.03 over CE alone | held-out A/B; if CE+coverage equals CE alone, coverage is subsumed |
| 1.3 | **Rank-1 guardrail + calibrated CE blending** — isotonic/temperature-calibrate CE logits on a held-out half; forbid demoting first-stage rank 1 unless the calibrated margin is large | bounds top-heavy flips (the ArguAna collapse lives in rank-1 swaps); `blend_scores` exists | recovers 30–70% of CE-induced BEIR loss at no MultiHop cost | guardrail on/off at fixed blend across five BEIR sets; if gain just mimics blend<1 it is redundant |
| 1.4 | **Query-type routing gate** — tiny logistic gate over cheap features (query length, query/doc length ratio, lexical overlap, first-stage margin) deciding *whether* to run the CE and at what blend | CE helps MultiHop, hurts five topical corpora; global on/off wastes half | ~half the CE loss recovered at no MultiHop cost | leave-one-dataset-out; if it cannot beat global "CE off", drop it |
| 1.5 | **Atomic units + parent-window return** — index sentence atoms (dense side only), map atom → parent chunk, return the window; keep chunk-level BM25 | the measured fact-loss class (fact > overlap and > 1 sentence) and fact-density per MAP slot | MAP@10 +0.01–0.04; removes the fact-loss class | atom→top-100→parents vs fused baseline; keep BM25 at chunk granularity — 600-char passage indexing already lost here |
| 1.6 | **Deterministic structural context prefix** — heading breadcrumb + title + path + first line, prepended before *both* dense encode and BM25 | context starvation (a chunk that says "revenue grew 3%" with no company/period) | +0.005–0.02 on title-dependent corpora | A/B on five BEIR + MultiHop; repeated context inflates `df`, so it may be noise outside SciDocs |
| 1.7 | **Three-state claim adjudication** — extend `adjudicate_claims` with a local NLI pass so a claim about an unmentioned topic becomes `contradicted`/`no_coverage`, not `not_measured` | the stated "fabricated claim is not refuted" limit; reuses the shipped `NLIAnswerability` hook | closes the limit with a local small model `[model]` | hand-label ~300 claim/corpus pairs from `eval/corpus-planted`; report no-coverage recall |
| 1.8 | **Deterministic numeric/negation/unit/date verification** in `verify.py` | citation *mechanics* are checked; a cited span with the wrong number or a flipped negation is not | CPU-only, no model | 200 labelled composed answers; only ship at precision ≥ ~0.9 |

### Tier 2 — medium bets (1–3 weeks, some dependencies)

| # | addition | gap attacked | expected | falsify / skeptic |
| --- | --- | --- | --- | --- |
| 2.1 | **Late-interaction MaxSim channel** (`answerai-colbert-small-v1`, 33M) over the fused top-100; CPU-viable via [MUVERA](https://arxiv.org/abs/2405.19504) / [SLIM](https://arxiv.org/abs/2302.06587) rather than full PLAID | §3.2 distributed matching; the second hop | MAP@10 +0.01–0.03; near-zero on ArguAna | if on-the-fly MaxSim does not beat the pointwise CE on MultiHop, skip it |
| 2.2 | **Relational stance reranker** (DeBERTa-v3-base-MNLI; score by contradiction−entailment) `[model]` | ArguAna's opposition relation — the one corpus a pointwise ranker cannot do | 0.568 → toward/above the dense 0.596 | if contradiction-only ordering loses to BM25 there, the hypothesis is wrong; 300–500-token docs may truncate the signal |
| 2.3 | **Listwise reranker** (`jina-reranker-v3`, 0.6B, "last but not late"; **CC BY-NC 4.0, non-commercial**) `[model]` | set-aware comparative scoring; most likely single model to beat RRF on ArguAna while keeping multi-hop (its ArguAna nDCG@10 is 73.43) | hypothesis, not a citation: its self-reported 61.85–61.94 is a 13-dataset average, not comparable to the 5-set 0.452; no CPU/MPS benchmark is published | reproduce locally on the five sets + MultiHop before believing it |
| 2.4 | **Generator-free hop-2 expansion from bridge entities** — mine salient entity phrases from top chunks that are absent from the query, issue as extra dense+BM25 queries, RRF-fuse | raises the candidate-pool ceiling itself | ceiling 0.7307 → 0.78–0.82 hypothesis | use *entity* terms, not raw top-doc terms, and fuse not replace — otherwise it reproduces the rejected BM25 PRF failure |
| 2.5 | **Non-LLM entity graph + personalized PageRank** (spaCy NER/noun-chunk co-occurrence, scipy PPR) `[dep]` | associative hops; structurally different from RRF | ceiling +0.03–0.06 | if PPR over the depth-100 pool does not beat RRF on gold-fact coverage, the graph is noise; alias resolution is the real failure point |
| 2.6 | **BM25S mmapped sparse lexical channel** `[dep]` | Python-dict postings; the documented 10M-doc lexical wall | published NQ 2M: 4.36 GB → **0.49 GB**; MSMARCO 8M: 10.21 → 1.14 GB ([repo](https://github.com/xhluca/bm25s), [arXiv 2407.03618](https://arxiv.org/html/2407.03618v1)) | top-100 diff — expect tie noise, not byte-identical; verify stemmer/stopword parity |
| 2.7 | **On-disk term dictionary + tombstones via SQLite FTS5 (stdlib, no dep) or Tantivy** `[offline]`/`[dep]` | the term-dict + chunk-id list is the residual 10M-doc wall | near-zero resident postings; incremental delete | FTS5's own BM25/tokenizer changes rankings unless you keep your scoring over its postings |
| 2.8 | **Trust-tiered provenance + retrieval-time trust filter + spotlighting** `[offline]` | corpus poisoning and indirect prompt injection into the Ollama composer | cheap; deterministic; testable offline | with an open `/v1/index` the honest default is "untrusted"; spotlighting is partial against adaptive attacks |
| 2.9 | **Adversarial retrieval regression suite** (poison / noise / negative rejection) in `eval/` | makes 2.8 and the gate falsifiable | enables the security claims | freeze seeds; distrust single-number ASR |

### Tier 3 — strategic (weeks; changes the character of the system)

| # | addition | why | needs |
| --- | --- | --- | --- |
| 3.1 | **Distil the 3-model ensemble into one packaged encoder**, then submit to MTEB | converts "retrieval quality comes from stock components" into an owned, submittable model; the projection-head path already exists | `[model]` + Hub/network |
| 3.2 | **Streaming ingestion + sharded builds + Parquet/SQLite artifacts + FAISS RaBitQ / binary quantization with rescore** | makes the 13 unbuildable BEIR datasets a schedule, not a wall; [RaBitQ](https://arxiv.org/abs/2405.12497) (SIGMOD 2024) is in FAISS ≥1.11 and the installed `faiss-cpu` is 1.15.1 | `[dep]` + CPU-hours; binary + rescore retains **up to ~96% quality at 32× reduction, but this is model-specific** (74.77–96.45% across encoders; [HF, 2024](https://huggingface.co/blog/embedding-quantization)) |
| 3.3 | **RGB + BRIGHT + LegalBench-RAG adapters** | external replication of negative rejection and reasoning-intensive retrieval; all retrieval-only | `[data]` |
| 3.4 | **Agentic decomposition / adaptive routing with a local small model** | legitimate, but the headline score belongs to the generator under the existing `engine + <model>` rule | `[llm]`; document the label |
| 3.5 | **ACL / tenant filtering at query time** (add `acl`/`tenant` to `Chunk`, predicate on `VectorIndex.search`, over-fetch) | the index today has no metadata predicate, so no ACL is possible; do it *before* multi-user, not instead of the deterministic defenses | `[offline]`; near-zero value until a second principal exists |

---

## 5. Theme verdicts

### 5.1 Multi-hop and agentic retrieval — [brief 03](docs/research/03-multihop-agentic.md)

The theme is **not** "add an agent". Two mechanisms dominate: the second-hop tax (§3.1) and
distributed single-vector matching (§3.2). The ordering the brief recommends — depth-500
retention instrumentation → bridge-linked rescue → CE × coverage → generator-free hop-2
expansion → MaxSim → entity graph → document-level aggregation → RAPTOR-lite → *then* a
generator — is the spine of Tier 0/Tier 1 above. The one thing to resist is reaching for a
local LLM decomposer first: it is the most expensive item and the only one whose number is
not the engine's.

### 5.2 Second-stage ranking, fusion, routing — [brief 02](docs/research/02-second-stage-ranking-fusion.md)

The decisive insight is *why* the pointwise CE failed on ArguAna: it scores **topical**
query–passage compatibility and has no supervision for the **relation** between two texts.
So "add a bigger reranker" is the wrong lesson. The right ones are (a) route the second
stage per query/corpus, (b) blend and guard the top rank, and (c) use the correct class for
the relation — listwise, pairwise, or an NLI/stance head. Normalised convex-combination
fusion is a documented improvement over RRF ([Bruch et al. 2023](https://arxiv.org/abs/2210.11934)),
and it is cheap, but per-corpus tuning must be held out. There is published external support
for the relational diagnosis: SPLADE-v3's own paper reports that a DeBERTaV3 pointwise
re-ranker fails to beat it on ArguAna because the "counter-argument task might be more
intricate for a re-ranker", and the listwise `jina-reranker-v3` scores 73.43 nDCG@10 on
ArguAna — the corpus where the pointwise CE collapsed.

### 5.3 Chunking, representation, index architecture — [brief 01](docs/research/01-chunking-representation-index.md)

The governing constraint is that **the oracle ceiling already sits inside the pool**, so
recall-oriented representation bets (late chunking, Dense-X propositions, RAPTOR) rank below
the cheap model-free fixes. Two exceptions are worth the days: atomic units + parent-window
return (attacks the measured fact-loss class) and a deterministic structural context prefix
(context without an LLM). Anthropic's [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
is real but vendor-run, and its LLM-at-ingest cost is prohibitive locally.

### 5.4 Generation and faithfulness — [brief 04](docs/research/04-generation-faithfulness.md)

Most of this is generator-owned and should stay labelled `engine + <model>`. The engine-side
wins are cheap and extend existing plumbing: three-state adjudication on the shipped
`NLIAnswerability` hook, deterministic numeric/negation verification in `verify.py`, and
nugget/aspect coverage accounting (a programmatic AutoNuggetizer proxy) that also supplies a
stopping rule for any agentic loop. Local faithfulness filters (MiniCheck, HHEM-2.1-Open)
are advisory and must not be presented as engine scores.

### 5.5 Evaluation and benchmarks — [brief 07](docs/research/07-evaluation-benchmarks.md)

The harness is already unusually strong (paired bootstrap, ECE, held-out gate fitting,
pool-oracle). What is missing is *claim discipline*: a permutation test with FDR control, a
selective-prediction view of abstention, a power/MDE statement, reranker-fairness controls,
and **external replication** on MuSiQue/2Wiki (the multi-hop claim currently rests on one
dataset). BRIGHT and RGB are the two new benchmarks worth adding, both retrieval-only.

### 5.6 Scale, storage and vector search — [brief 06](docs/research/06-scale-storage-vector-search.md)

Ranking-preserving infrastructure first: BM25S (mmapped sparse lexical), an on-disk term
dictionary (SQLite FTS5 stdlib — which has `contentless_delete=1` tombstones with zero
dependencies — or Tantivy), FAISS IVF-RaBitQ + OPQ + mmap, streaming ingestion and sharded
builds. **Verified locally:** the installed `faiss-cpu 1.15.1` already exposes
`IndexIVFRaBitQFastScan` and `IndexHNSWRaBitQ`, and macOS 26.6.2 satisfies the x86_64 wheel
tag, so this is a code change, not an upgrade; `faiss-cpu` exposes no Metal classes, so drop
the "Metal IVF" idea. Quality-changing quantization (binary/int8 + rescore, Matryoshka) is
measured elsewhere but must be re-measured here, since the encoder is `bge-small`, which is not
MRL-trained, and the "~96% retained" figure is 74.77–96.45% depending on the encoder. The
ONNX Runtime + CoreML ANE plan is **not executable on this Intel host** (no Neural Engine).
Note the repo's own finding that exhaustive scan is ~11.6 ms at 1M vectors, so ANN is a scale
play, not a latency emergency.

### 5.7 Security, privacy and robustness — [brief 05](docs/research/05-security-privacy-robustness.md)

Grounded in the actual code: `build_messages` splices retrieved text verbatim into the user
turn, `VectorIndex.search` takes **no metadata predicate** (so no ACL is possible today), and
`adjudicate_claims` re-retrieves over the whole index (so a ranking poison can flip the
*second* verification layer too). Cheap deterministic defenses first: trust-tiered
provenance with a retrieval-time trust filter, spotlighting/datamarking, provenance-bound
`verify.py`, an adversarial regression suite, and constant-time bearer comparison
(`hmac.compare_digest`). ML injection detectors are last and labelled signal-only.

---

## 6. A concrete sequence

If the goal is the largest honest movement in the next few weeks, in order:

1. **Instrument (hours).** Depth-500 ceiling + second-hop retention in
   `tools/multihop_official.py`; permutation/FDR in `tools/evalkit.py`. This decides whether
   the rest of the multi-hop programme is recall or selection, and hardens existing claims.
2. **Rescue and combine (days).** Bridge-linked second-hop rescue (1.1) and CE × coverage
   joint selection (1.2). Target: **0.2986 → 0.31–0.33**, measured against the honest bar.
3. **De-risk the second stage (days).** Rank-1 guardrail (1.3) and the query-type routing
   gate (1.4), so the CE stops costing BEIR while keeping MultiHop.
4. **Cheap quality (days).** Atomic units + parent window (1.5) and the structural prefix
   (1.6), with a re-index and a held-out paired bootstrap each.
5. **Replicate (days, `[data]`).** MuSiQue + 2Wiki (0.6); if the coverage gain does not
   replicate, relabel the headline as dataset-specific — which is itself a result.
6. **Infrastructure in parallel (weeks).** BM25S and the FTS5/Tantivy term dictionary; then
   FAISS RaBitQ + quantized-first-pass-with-fp32-rescore.
7. **Strategic, when training time exists.** Distil the ensemble into one encoder and submit
   it; that is the item that changes the character of the system.
8. **Security baseline (days, can run anywhere).** Trust tiers, spotlighting, provenance-bound
   verification, adversarial regression, constant-time auth.

Items 1–5 and 8 need no GPU and no external API. Item 6's BM25S/FTS5 parts need a one-time
`pip install` (the installed `faiss-cpu 1.15.1` already carries RaBitQ, so the ANN part is a
code change). Item 7 is the only one that needs a training run.

---

## 7. Verification status and caveats

The first pass ran its seven research agents concurrently and the web-search backend
rate-limited them (HTTP 429), which left several citations resting on search snippets. This
version was **re-verified serially, one theme at a time, against primary sources** (arXiv
abstracts, ACL Anthology, official documentation, raw changelogs, model cards, and — for the
second-hop tax — the primary findings document itself). Per-theme memos are in
[`docs/research/verification/`](docs/research/verification/). The material corrections:

| theme | correction |
| --- | --- |
| second-hop tax | retention is **0.968→0.439** (HotpotQA) / **0.751→0.043** (MuSiQue) on the *distractor-filtered* strategy; rescue is **+23.2 pts** (HotpotQA τ=0.30) and **+28.8 pts** (MuSiQue τ=0.20); the source's MuSiQue density-pruning sub-result **did not survive** a sharper signal; the rescue is **lexical only** |
| ranking | `2304.08807` is **Shi et al.**, not Reimers; jina-reranker-v3 is **13 datasets / 61.85–61.94 / CC BY-NC 4.0**; ColBERT-small's 72.6 is bge-base **HotpotQA**, not BEIR; SPLADE-v3 is BERT-base/CoCondenser, and its paper independently reports pointwise rerankers failing on ArguAna |
| chunking | CodeRAG-Bench tests only **fixed token windows** and is not evidence for AST chunking; doc2query is a **from-scratch 6-layer transformer**, not T5; late chunking's own Appendix finds **overlap gave no clear advantage**; HF's "~96%" is 74.77–96.45% model-dependent; brief 01's late-interaction item had wrongly claimed an ArguAna fix |
| scale | `faiss-cpu 1.15.1` locally **already exposes `IndexIVFRaBitQFastScan`** and macOS 26.6.2 clears the wheel gate (verified by executing it), but **`faiss-cpu` has no Metal classes**; the ONNX/CoreML **ANE plan cannot run on this Intel host**; USearch's "10×" is 96-d only; FTS5 has `contentless_delete=1` tombstones built in |
| security | the prompt-injection detector is **~184M**, not 435M; OWASP LLM01 is the #1 risk with indirect injection as a subtype; spotlighting's encoding ablations are **base64/ROT13**, not zero-width/bidi; PoisonedRAG is **5 documents → 90% ASR** |
| evaluation | the ICTIR 2025 paper is about **training-set** construction and does **not** support pool-bias; BRIGHT's CPU-cheap sets are Pony/TheoremQA-T/StackExchange, not LeetCode; use **AUGRC** and equal-mass `ECE_sweep` |
| generation | the numeric/entity-mismatch claim is supported by **AttributionBench only** (fine-grained-information insensitivity >66%), not ALCE; venues corrected (LLMLingua-2 → Findings of ACL 2024; Judging the Judges → GEM² 2025) |

Still read as **hypotheses, not results**, because they are vendor-run or single-source and no
one has reproduced them on this host: Anthropic's 35–49% contextual-retrieval figures, Jina's
BEIR average, the BM25S/Faiss memory numbers, USearch's build claims, HHEM's accuracy, and the
second-hop-tax percentages (a project findings note with honest limits, not a peer-reviewed
paper — treat it as a hypothesis with a clear test in Tier 0.1).

- **Nothing here was implemented.** No source file was changed; this is a survey. The briefs
  in `docs/research/` and the verification memos in `docs/research/verification/` are kept
  separate from `IMPROVEMENTS.md` (in-repo diagnosis) and `PLAN.md` (execution plan).
- **The prior to respect.** Every failure in §2 came from trusting a stronger pointwise model
  more. Every win came from changing the objective or the routing. Proposals that amount to
  "use a bigger model on the same objective" should be the most heavily discounted.

---

## Appendix — research briefs

| file | theme | headline recommendation |
| --- | --- | --- |
| [01](docs/research/01-chunking-representation-index.md) | chunking, representation, index | atomic units + parent window; deterministic context prefix |
| [02](docs/research/02-second-stage-ranking-fusion.md) | second-stage ranking, fusion, routing | rank-1 guardrail + calibrated blending; NLI stance for relations |
| [03](docs/research/03-multihop-agentic.md) | multi-hop, agentic, graph | bridge-linked second-hop rescue; MaxSim as a multi-hop instrument |
| [04](docs/research/04-generation-faithfulness.md) | generation, faithfulness | three-state adjudication; deterministic numeric verification |
| [05](docs/research/05-security-privacy-robustness.md) | security, privacy, robustness | trust tiers + spotlighting + provenance-bound verification |
| [06](docs/research/06-scale-storage-vector-search.md) | scale, storage, vector search | BM25S + on-disk term dict; FAISS RaBitQ + quantization |
| [07](docs/research/07-evaluation-benchmarks.md) | evaluation, benchmarks, rigor | permutation + FDR; selective prediction; MuSiQue/2Wiki replication |

**Verification memos** (the serial primary-source pass behind §7) live beside the briefs:
[01 chunking](docs/research/verification/01-chunking-verification.md),
[02 ranking](docs/research/verification/02-ranking-fusion-verification.md),
[04 generation](docs/research/verification/04-generation-verification.md),
[05 security](docs/research/verification/05-security-verification.md),
[06 scale](docs/research/verification/06-scale-verification.md),
[07 evaluation](docs/research/verification/07-evaluation-verification.md). Brief 03
(multi-hop) was verified by fetching its load-bearing primary source directly (the redhop
findings document), so it has no separate memo.
