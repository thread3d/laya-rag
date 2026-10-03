# Frontier review, 2026-10-03 — what would make laya-rag the best RAG system

Companion to [`RESEARCH.md`](../../RESEARCH.md) (the seven-theme external survey, 2026-10-02)
and [`IMPROVEMENTS.md`](../../IMPROVEMENTS.md) (the in-repo diagnosis). This is a **one-day-later
delta pass**: a deliberately small search budget aimed only at material that is *new since the
survey*, plus the three things the survey left structurally open. It does not restate the
survey's ledger; where an item here overlaps one there, it says so and says what changed.

Search discipline: **3 batched `web_search` calls (12 queries) and 3 full-text fetches.** Every
claim is labelled by how it was verified — `[full text]` (fetched the primary source),
`[snippet]` (search surface only, treat as a lead), or `[repo]` (a number or file in this
repository). Nothing here was implemented.

---

## 0. The honest bar, and the one number that decides everything

Quoted from `RESEARCH.md` §1 and `IMPROVEMENTS.md`; not re-derived:

| axis | bar to beat | why |
| --- | --- | --- |
| MultiHop-RAG MAP@10 | **0.2986** (the repo's own reranked 400-query subset) | the 0.479 comparison is a *reranked* published number; 0.2835 is the un-reranked baseline |
| MultiHop-RAG pool-reorder oracle | **0.7307** at depth 100 `[repo]` | the gold evidence is *already in the pool*; the surviving loss is selection + set-construction |
| BEIR retrieval (5 datasets) | 0.452 single / **0.495** 3-model ensemble | already above published BGE-large; cost, not quality |
| Answerability out of domain | scalar AUC 0.51; in-domain feature gate **0.815** | no portable calibrated abstention |
| Cross-encoder | −0.027…−0.318 BEIR; **+0.0464** MultiHop | route it, never global |

**Two structural facts the survey states and this pass sharpens:**

1. **The 0.7307 ceiling is a *top-100 reordering* ceiling, not a corpus ceiling.** Every
   objective shipped so far — `facility_location_select`, `doc_quota_select`,
   `cluster_dedup_select` `[repo]` — consumes the fused pool and reorders it. None of them can
   admit a passage the *topical* first stage never retrieved. That is why "the gap is selection,
   not recall" has been true so far without being the end of the story: selection is the gap
   *within this pool*.
2. **Laya's relevance question is a 2-way question about the wrong thing for multi-hop.**
   `RELEVANCE_QUESTION` is `"Does \`passage\` help answer \`query\`?"` and the state is
   `{"query": …, "passage": …}` (`src/laya_rag/rerank.py:154`) `[repo]`. A second-hop passage is
   low-relevance *to the query string by construction*. The head was asked a question that
   cannot separate the bridge from the wrong chain. This is the same defect BridgeRAG (below)
   names, and — importantly — it is fixable **inside the shipped model**, not by importing one.

---

## 1. What "best" means: four axes, scored

"The best RAG system" is not one number. Best-in-class in 2026 means all four of these, and the
system is graded separately on each because the fixes do not overlap.

| axis | what it means | where laya-rag stands | the binding constraint |
| --- | --- | --- | --- |
| **A. Retrieval quality** | recall *and* set-construction on hard queries | strong hybrid; above published rows on BEIR and MultiHop Hits@10; MAP@10 is the open gap | set-level selection; the second-hop tax |
| **B. Decision quality** | calibrated abstention, per-claim support, refusal | the differentiator: Laya's calibrated `noul` is a real, rare asset; feature gate 0.815 in domain | out-of-domain transfer; one scalar per blob |
| **C. Scope completeness** | the capabilities a 2026 stack is expected to have | text-only, single-hop-shaped, English, no graph, no multimodal, no structured/table path, no temporal awareness | each is an *absent whole capability*, not a tuning gap |
| **D. Operational integrity** | freshness, deletion, dedup, provenance, tenancy, injection resistance | deletion pruning shipped (`plan` Phase 6.1) `[repo]`; injection and tenancy named but unimplemented | no metadata predicate in `VectorIndex.search`, so no ACL and no freshness filter is possible today |

The survey's ranked programme is almost entirely **A** and **B**. This pass adds the **C** and
**D** rows as first-class, because a system cannot be "the best" if whole capability classes are
absent, however good its MAP@10 is.

---

## 2. New primary sources since the survey, and what each one changes

### 2.1 BridgeRAG — the survey's own §3.1 hypothesis, now a published method `[full text]`

[Bacellar, *BridgeRAG: Training-Free Bridge-Conditioned Retrieval for Multi-Hop QA*,
arXiv:2604.03384v2](https://arxiv.org/abs/2604.03384) (Apr 2026, CC BY 4.0).

The survey's Tier 1.1 ("bridge-linked second-hop rescue") was inferred from a project findings
note. BridgeRAG is the peer-adjacent version of the same idea, and it is sharper than the note:

- **The scoring target is wrong, not just the filter.** Second-hop relevance is not
  `s(q, c)` but **`s(q, b, c)`** — the conditional utility of candidate `c` given the query `q`
  **and the retrieved bridge `b`**. Its motivating example is exact: a 2-way judge promotes the
  film passage for *"Who is the spouse of the actor who played the Terminator?"*; conditioning on
  the Schwarzenegger passage promotes Maria Shriver.
- **Coverage and scoring are separable, and it says so in the ablation.** Dual-entity ANN
  expansion broadens the candidate pool *independently of* the judge. That is the same
  separation the repo already has between `_coverage_select` and the relevance filter — and it
  suggests they should be wired as two stages rather than two alternatives.
- **Fusion.** It fuses SVO-expansion scores with judge scores by **percentile-rank (PIT)
  normalisation**, explicitly claiming this is more robust than min-max on multi-hop. The repo
  fuses with RRF `[repo]`; PIT is a cheap, third fusion rule worth a held-out A/B.
- **Mechanism metrics worth stealing.** "Productive flip rate" — 18.7% on parallel-chain vs
  0.6% on single-chain queries — is a *diagnostic for ranking churn*, and it is exactly the
  control the repo lacks when it measures "all cells significant" (survey Tier 0.2).
- **Headline, training-free R@5:** 0.8146 MuSiQue, 0.9527 2WikiMultiHopQA, 0.9875 HotpotQA,
  beating PropRAG and HippoRAG2. **Not comparable to the repo's numbers** — different metric
  (R@5), different datasets, and it uses **Llama 3.3 70B** with 3 LLM calls/query plus
  `NV-Embed-v2` for ANN. Treat as a hypothesis with a named mechanism, not a transferable row.

**The novel move this enables here.** BridgeRAG buys the tripartite judge with a 70B LLM. This
repository already owns a **calibrated decision head that answers arbitrary typed questions over
a body of text** — which is what a tripartite judge is. Adding a bridge to Laya's question is a
state-key change and a question string:

```
state    {"query": q, "bridge": b, "passage": c}
question {"type": "noul",
          "instructions": "Does `passage` help answer `query`, given that `query` is about `bridge`?"}
```

No LLM, no graph, no training run, no network — `[offline]`, days. This is the single highest
expected-value item in this document, because it uses the system's *differentiator* to attack
its *headline gap*, and because it is falsifiable in one harness run.

The counter-prior it must respect: Laya's 2-way relevance used as a *ranker* lost on both BEIR
datasets (SciFact 0.654 → 0.470) `[repo]`. The hypothesis is not "Laya ranks well"; it is "Laya
ranks well *when asked a question that is not structurally blind to the bridge*." Ship it behind
`RERANK_BLEND` and the proposed rank-1 guardrail so the downside is bounded.

### 2.2 DISCo — coverage as an *index*, which is how you escape the 0.7307 ceiling `[full text]`

[Nair et al., *DISCo: A Dense Subset Index for Collective Query Coverage*, ICLR 2026](https://k-rt-k.github.io/DISCo/).

The repo's `facility_location_select` `[repo]` and DISCo share one idea — **coverage is a
facility-location submodular objective, greedy gives a (1−1/e) guarantee** — but DISCo makes the
step the repo has not: it rewrites the marginal gain as an inner product in a *lifted* space
(`q̂_S = [q; F(S,q)]`, `x̂ = [x; −1]`), so all set-dependence moves into the **query** vector and
the corpus vector stays static and indexable. With random hyperplane features the hinge becomes
MaxSim-like, so **each greedy step is an ANN probe in the lifted space**, sublinear in corpus
size. On MS-MARCO (8.8M) it matches greedy coverage **>100× faster**; on HotpotQA it beats PLAID
and exact greedy on coverage error and MAP-on-gold.

Why this matters specifically here: the repo's best coverage result (0.2850) is a *re-ranking* of
a topically-built top-100, so it is capped at 0.7307. DISCo's formulation lets the coverage
objective **build** the pool, which is the only measured route past that cap. Two honest
constraints:

- DISCo is **token-level**: it needs contextual token vectors, i.e. a late-interaction
  representation. The shipped `LayaModel.embed` is mean-pooled `[repo]`, and the survey already
  noted the bge path is single-vector. So DISCo is *downstream* of a MaxSim-capable channel.
- The repo's own `facility_location_select` operates on **sentence atoms**, not query tokens —
  a coarser objective that is already the best measured pass. DISCo is a re-formulation of that
  objective, not a different objective.

The synthesis that changes priorities: **the late-interaction channel is not merely a
second-stage reranker (survey Tier 2.1). It is the substrate that lets coverage selection escape
the top-100 reorder ceiling.** That raises it from a Tier-2 "might help" to the enabling
dependency of the highest-ceiling item in the programme. The cheap first probe is not ColBERT —
it is to ask whether Laya's own checkpoint exposes token states that a MaxSim head could use
(the repo's own §4 open item), which needs no download.

### 2.3 RRK / compressed listwise reranking `[full text: abstract]`

[Déjean & Clinchant, *Efficient Listwise Reranking with Compressed Document Representations*,
arXiv:2604.26483, AAAI 2026](https://arxiv.org/abs/2604.26483) (see also the
[AAAI proceedings entry](https://ojs.aaai.org/index.php/AAAI/article/view/40811)).

Listwise reranking (survey Tier 2.3) has moved from "a bigger model reads the whole list" to
**compress each document into a fixed-size multi-token embedding, then judge the list over those
compressed representations**, distilled. Its 8B model runs **3–18× faster than 0.6–4B rerankers
while matching or beating them**, with the advantage widening on long documents.

What it changes: the survey's listwise item was costed as `[model]` and risky on this Intel host
(an 8B, CC BY-NC). The *idea* is what is portable and cheap here: **the repo already computes a
dense vector per chunk; a listwise judge over compressed vectors is a small head on top of
representations it has, not a new 8B download.** Combined with §2.1, the shape is: Laya scores
`(q, b, c)` pointwise; a small listwise head over compressed vectors handles the *set*; the
coverage objective picks the set. Three instruments, three failure modes, no GPU.

### 2.4 MUSS, and a second reason to keep relevance *and* diversity explicit `[snippet]`

[*MUSS: Multilevel Subset Selection for Relevance and Diversity*, PMLR v337 (2026)](https://proceedings.mlr.press/v337/nguyen26d.html).
Snippet-level only: multilevel subset selection trading relevance against diversity. Relevant
as a structured alternative to the repo's single-λ greedy `facility_location_select`, and as
independent 2026 evidence that "relevance + diversity as an explicit subset objective" is the
active line rather than a local idiosyncrasy. Not load-bearing for any recommendation here.

### 2.5 Operational integrity — the axis the survey under-weighted `[snippet]`

Four independent 2026 sources converge on **index lifecycle as a retrieval-quality problem**, not
a housekeeping one. Snippet-level; treat each as a lead:

| source | claim |
| --- | --- |
| [Oracle: *How to Detect RAG Index Drift*](https://blogs.oracle.com/developers/how-to-detect-rag-index-drift-deleted-docs-stale-chunks-and-duplicate-embeddings) | deleted docs, **stale chunks**, and **duplicate embeddings** are three distinct drift classes needing three distinct detectors |
| [*Measuring Retrieval Freshness and Accuracy Degradation in Continuous ETL-Driven RAG*, JISEM 2026](https://jisem-journal.com/index.php/journal/article/view/14454) | retrieval accuracy degrades measurably as an ETL-driven index ages |
| [*Authorization-First Retrieval: Enforcing Least Privilege in Multi-Agent RAG*, TrustNLP 2026](https://aclanthology.org/2026.trustnlp-main.15.pdf) | authorisation must be a *retrieval-time predicate*, not a post-filter — which is exactly the `VectorIndex.search` metadata gap `[repo]` |
| [*RAGBench*](https://arxiv.org/pdf/2407.11005), [*Bencharking LLM Faithfulness … Evolving Leaderboards*, EMNLP 2025 Industry](https://aclanthology.org/2025.emnlp-industry.54.pdf) | end-to-end RAG scoring is now a labelled, explainable harness with evolving leaderboards — the repo's `engine + <model>` labelling rule is the right response, but a *shared* harness would make it comparable |

The repo is ahead of most stacks on one of these already: Phase 6.1 prunes deleted/unreadable
paths and reports `removed_chunks` `[repo]`. What it has never done is **detect the other two
drift classes** — a chunk whose source changed under a stable path (incremental replace handles
the path, but not a *content* change detection signal), and near-duplicate embeddings
(`RECALL_DEDUP_THRESHOLD` suppresses duplicates *at query time* `[repo]`, which is a different
thing from detecting them in the index).

### 2.6 GraphRAG vs vector RAG, measured `[snippet]`

The [superluminar 2026 AWS comparison](https://superluminar.io/2026/05/27/optimising-and-evaluating-llm-knowledge-retrieval-graphrag-vs-vector-rag-on-aws/)
and an [open reproduction repo](https://github.com/Dima806/graphrag_vs_vectorrag) put a *measured*
answer where the survey had a cost argument: GraphRAG wins on global/multi-hop summarisation at
a large ingest and query cost, and loses or ties on single-fact lookup. This is consistent with
BridgeRAG's stronger result — **you can get the multi-hop win without the graph**, by
conditioning rather than by building an offline entity graph. It weakens the case for the
survey's Tier 2.5 (PPR over a non-LLM entity graph) and strengthens §2.1.

---

## 3. The capability matrix

Where a best-in-class 2026 text-and-beyond RAG stack sits, against this engine. `absent` means
the capability does not exist, not that it is weak.

| capability | best-in-class 2026 | laya-rag | cheapest route |
| --- | --- | --- | --- |
| hybrid lexical + dense recall | table stakes | **present**, measured above published rows | — |
| learned sparse (SPLADE-class) | common | **present** (shipped third channel) | — |
| pointwise reranker | common, usually net-positive | present, opt-in, **routed** (correct call) | keep |
| listwise / set-aware reranker | the 2026 frontier | **absent** | §2.1 + §2.3, no GPU |
| explicit set-coverage objective | rare, active research | **present and best-measured pass** | promote to pool *builder* (§2.2) |
| calibrated abstention | rare; usually uncalibrated | **present — the differentiator** | portability (§B) |
| per-claim support / adjudication | common in 2026 stacks | **present, three-state gap** | reuses `NLIAnswerability` `[repo]` |
| provenance & citation mechanics | table stakes | **present** (`verify.py`, deterministic) | extend to numeric/negation |
| index lifecycle: delete/prune | table stakes | **present** (Phase 6.1) | — |
| index lifecycle: staleness & dup detection | common | **absent** | §2.5, offline |
| query-time ACL / tenancy | required in any multi-user deployment | **absent structurally** (no metadata predicate) | `acl` on `Chunk` + over-fetch filter |
| injection defence | table stakes | **absent** (verbatim splice into composer) | survey Tier 2.8, offline |
| temporal / recency-aware retrieval | common | **absent** | metadata + freshness-weighted fusion |
| multimodal (image/table) | common | **absent** | a whole capability; out of scope for this host |
| structured / text-to-SQL | common in enterprise | **absent** | out of scope |
| knowledge graph | common, expensive | **absent — and §2.6 suggests that is defensible** | no |
| end-to-end generation scoring | required for leaderboards | **deliberately labelled out** | shared harness, `engine + <model>` |

The pattern is favourable and worth stating plainly: this system is **already at or beyond the
frontier on the rare things** (calibrated abstention, explicit coverage, deterministic
provenance, honest measurement) and **behind on the commodity things** (listwise ranker,
lifecycle drift detection, ACL, injection defence). Commodity gaps are cheap to close;
the rare strengths are not cheaply copied. That asymmetry is the strategy.

---

## 4. The programme, updated

Ranked by expected value per unit cost, against the honest bar **0.2986**, with tags:
`[offline]` runs today on this host · `[dep]` one-time package install · `[data]` download ·
`[model]` model download/training · `[gpu]` not this host · `[llm]` needs a generator (number is
the model's, not the engine's).

### Tier 0 — instrument and harden (hours, all `[offline]`) — unchanged, do first

| # | item | settles |
| --- | --- | --- |
| 0.1 | depth-500 pool ceiling + **second-hop retention** instrumentation | recall vs selection; the precondition for everything in Tier 1 |
| 0.2 | paired permutation test + Benjamini–Hochberg FDR in `tools/evalkit.py`; add BridgeRAG's **productive-flip rate** as a churn control | the "all significant" grid gets family-wise control and a churn diagnostic |
| 0.3 | selective-prediction suite (risk–coverage, AURC, risk@80%, Brier/NLL, debiased ECE) | turns AUC 0.815 into deployable risk |
| 0.4 | power / minimum-detectable-effect statement per query set | separates "no effect" from "underpowered" |

### Tier 1 — the Laya-native judge and the set objective (days–2 weeks)

| # | item | tag | expected | falsify |
| --- | --- | --- | --- | --- |
| **1.1** | **Laya tripartite judge `s(q,b,c)`** — add `bridge` to the state and a bridge-conditioned `noul` question; bridge = top fused candidate (then top-k multi-bridge); ship behind `RERANK_BLEND` + rank-1 guardrail; sweep ≥3 wordings on a held-out half | `[offline]` | the repo's headline lever; MAP@10 0.2986 → **0.31–0.33** hypothesis | if 3-way does not beat 2-way at 3-way's best blend on the held-out half, the question is wrong and this closes permanently |
| **1.2** | **Joint CE × coverage × rescue** — feed CE scores into `facility_location_select` as its `rel` term, run bridge rescue as readmission, keep the per-doc quota; the three best-measured passes have never been combined | `[offline]` | +0.01–0.03 over CE alone | if joint == CE alone, coverage is subsumed by a good ranker |
| 1.3 | **Rank-1 guardrail + calibrated CE blending** (isotonic/temperature on a held-out half; never demote first-stage rank 1 without a large calibrated margin) | `[offline]` | recovers much of the CE-induced BEIR loss at no MultiHop cost | guardrail on/off at fixed blend on 5 BEIR sets |
| 1.4 | **Query-type routing gate** (logistic over cheap features: query length, query/doc ratio, lexical overlap, first-stage margin) → CE on/off and blend | `[offline]` | stops the global CE wasting half its decisions | leave-one-dataset-out; must beat global "CE off" |
| 1.5 | **PIT percentile-rank fusion** as a third fusion rule beside RRF, A/B'd on the held-out half | `[offline]` | BridgeRAG reports it more robust than min-max on multi-hop; free to test | if PIT ≤ RRF on held-out, close it |
| 1.6 | **Dual-entity pool expansion, model-free** — top-k noun-chunks/NER entities absent from the query, each issued as a dense+BM25 probe, union-max into the pool; fuse, never replace | `[offline]` | raises the *pool* ceiling itself (BridgeRAG's coverage leg, without its LLM) | if gold-fact coverage of the depth-100 pool does not rise, the entity extractor is the failure |
| 1.7 | **Atomic units + parent-window return** (sentence atoms on the dense side, map to parent chunk, return the window) | `[offline]` | attacks the measured fact-loss class; MAP +0.01–0.04 | atom→top-100→parents vs fused baseline |
| 1.8 | **Deterministic structural context prefix** (heading breadcrumb + title + path + first line, before both encode and BM25) | `[offline]` | +0.005–0.02 on title-dependent corpora | repeated context inflates `df`; may be noise outside SciDocs |
| 1.9 | **Three-state adjudication** via the shipped `NLIAnswerability` hook | `[model]` | closes the stated "fabricated claim is not refuted" limit | hand-label ~300 claim/corpus pairs |
| 1.10 | **Numeric/negation/unit/date verification** in `verify.py` | `[offline]` | citation mechanics exist; correctness of a cited span does not | 200 labelled answers; ship only at precision ≥ ~0.9 |

### Tier 2 — lifecycle and integrity (days, `[offline]`)

| # | item | tag | why now |
| --- | --- | --- | --- |
| 2.1 | **Staleness detector**: content hash per `(path, chunk)`; on re-index, flag chunks whose source content changed but whose embedding is unchanged/near-unchanged | `[offline]` | §2.5 class 2; the append-only on-disk backend cannot do this yet (needs a rebuild path) |
| 2.2 | **Duplicate/drift detector**: cluster near-duplicate embeddings offline; report the cluster count in `stats()` | `[offline]` | §2.5 class 3; `RECALL_DEDUP_THRESHOLD` hides duplicates at query time, it does not measure them |
| 2.3 | **ACL/tenant predicate on `VectorIndex.search`** + `acl`/`tenant` on `Chunk`, over-fetch then filter | `[offline]` | §2.5 Authorization-First: retrieval-time, not post-hoc; the survey's Tier 3.5, and it is a prerequisite for any second principal |
| 2.4 | **Trust-tiered provenance + retrieval-time trust filter + spotlighting** | `[offline]` | already in the survey (Tier 2.8); unchanged and still cheap |
| 2.5 | **Adversarial retrieval regression suite** (poison / noise / negative rejection) | `[offline]` | makes 2.4 and the gate falsifiable and freezes them |

### Tier 3 — the ceiling (weeks; this is the strategic bet)

| # | item | tag | why |
| --- | --- | --- | --- |
| **3.1** | **MaxSim / late-interaction channel from Laya hidden states** (token-level embeddings, no mean pooling) — first *inspect whether the checkpoint exposes usable token states*, then a MaxSim head over the fused pool | `[offline]` probe → `[model]` | it is the enabling substrate for **3.2**, and the repo's own §4 open item; probe is hours |
| **3.2** | **DISCo-style coverage index** — lifted-space marginal gain + random-projection feature map so each greedy coverage step is an ANN probe; pool *built* by coverage, not reordered | `[gpu]`/`[dep]` at scale; the objective itself is a code change | the only measured route past the **0.7307** reorder ceiling |
| 3.3 | **Compressed listwise head** over existing chunk vectors | `[model]`, small | set-aware scoring without an 8B reranker |
| 3.4 | **Distil the 3-model ensemble into one packaged encoder**; submit to MTEB | `[model]` | converts "quality comes from stock components" into an owned, submittable model |
| 3.5 | MuSiQue + 2WikiMultiHopQA replication of the coverage gain | `[data]` | the multi-hop headline rests on one dataset; highest-value falsification available |
| 3.6 | Agentic decomposition with a local LLM | `[llm]` | legitimate, but its headline score is the generator's — label it `engine + <model>` |

### The do-not-retry ledger still stands

Unchanged from `RESEARCH.md` §2, and worth one addition: **do not build a knowledge graph to
fix multi-hop.** §2.1 and §2.6 both indicate the conditioning signal, not the graph, carries the
gain — and the graph is the expensive version. The general prior the ledger names still holds:
failures come from trusting a stronger pointwise model on the *same objective*; wins come from
changing the objective or the routing. §2.1 is a change of objective; §2.2 is a change of where
the objective runs.

---

## 5. The concrete two-week sequence

1. **Instrument (hours).** Tier 0.1 retention + depth-500 ceiling; 0.2 FDR + productive-flip.
2. **The Laya tripartite judge (days).** Tier 1.1. This is the one item that could move the
   headline number, uses no external resource, and directly tests whether the system's own
   differentiator can supply what the field buys with a 70B model. Log retention and flip rate
   alongside MAP so the *mechanism* is measured, not just the score.
3. **Combine (days).** Tier 1.2 joint CE × coverage × rescue, with 1.3's guardrail as insurance.
4. **Free points (days).** 1.5 PIT fusion, 1.6 model-free dual-entity expansion, 1.7 atomic units.
5. **Integrity (days, parallel).** Tier 2.1–2.3: staleness, duplicate drift, ACL predicate.
6. **The ceiling (weeks).** Probe Laya token states (3.1); if usable, MaxSim then DISCo (3.2).

Steps 1–5 need no GPU, no API, no download. Step 6's probe is hours; its payoff is the only
measured path past 0.7307.

---

## 6. Verification status and caveats

| source | verification | what is load-bearing |
| --- | --- | --- |
| BridgeRAG, arXiv:2604.03384 | **`[full text]`** — abstract, intro, task formulation, method, related work fetched | the `s(q,b,c)` formulation; coverage/scoring separability; PIT fusion; the R@5 headline and its metric mismatch with this repo |
| DISCo, ICLR 2026 | **`[full text]`** — project page incl. objective, lifted-space derivation, results summary | coverage as a submodular objective; the lifted-space indexability trick; the >100× MS-MARCO figure |
| RRK, arXiv:2604.26483 / AAAI 2026 | **`[full text: abstract]`** — abstract page only, not the results tables | compressed multi-token doc representations + listwise distillation; the 3–18× claim is the authors' |
| MUSS (PMLR v337) | `[snippet]` only | a lead; not load-bearing |
| Oracle index-drift; JISEM freshness; TrustNLP 2026 authorization | `[snippet]` only | the *existence and convergence* of the lifecycle view; the specific numbers are not quoted |
| RAGBench; EMNLP 2025 Industry faithfulness leaderboard | `[snippet]` only | that end-to-end RAG harnesses are standardised and evolving |
| GraphRAG vs vector RAG (superluminar 2026; Dima806 reproduction) | `[snippet]` only | weakens the graph argument; not load-bearing |

**Read as hypotheses, not results.** No item here was run on this host. BridgeRAG's R@5 numbers
are on different datasets and a different metric from this repo's MAP@10, and its pipeline uses
an LLM this host cannot run — the **mechanism** transfers, the **row** does not. DISCo is a
token-level method and is blocked on a late-interaction representation the repo does not yet
ship. RRK's model is 8B and non-commercial.

**The claims that are this document's own, and are testable in hours each:** that Laya can
implement `s(q,b,c)` without an LLM; that the 0.7307 figure is a reorder ceiling rather than a
corpus ceiling and can only be moved by changing what the first stage *builds*; and that the
system's strategic position is rare strengths plus cheap commodity gaps.

**Nothing here was implemented.** The survey discipline stands: `docs/research/` briefs and
verification memos stay separate from `IMPROVEMENTS.md` (diagnosis) and `PLAN.md` (execution).
