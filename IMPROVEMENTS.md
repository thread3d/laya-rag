# Where this RAG is worse, and what would actually move each point

This is an engineering response to the seven weaknesses recorded in
[`README.md` §"Where this RAG is worse"](README.md) and [`BENCHMARKS.md`](BENCHMARKS.md).
Each section states **what the code does** (with the file/line that decides it), **why the
measured number is what it is**, and **the ordered set of things that could change it** —
cheap first, with the expected effect and the way to falsify it. It deliberately does not
re-propose anything already measured and rejected.

Numbers already in the repo are quoted as measured; proposals are labelled by confidence.

**On the line numbers.** They point at the tree as it stood when each section was written, and
they drift as the code moves — a review pass found several pointing at unrelated code, including
in `config.py`, which no one had edited since the analysis. The **symbol or flag name is the
durable reference**; treat `file.py:NNN` as a hint and confirm against the name. Where a citation
was corrected the symbol was checked at the line it now names, not just the arithmetic.

---

## 0. The two structural facts that explain most of the list

1. **The engine is retrieval + decisions, not generation.** `compose.py` is an opt-in
   single-turn chat call to a local Ollama; `verify.py` checks citation *mechanics* and
   explicitly makes no truth claim (`src/laya_rag/verify.py:5-6`). So every generation-scored
   benchmark is somebody else's number, and the system cannot be entered where a stack is
   required.
2. **The differentiating model (Laya) is a filter/gate, not a retriever or an orderer.**
   `LayaModel.embed` mean-pools the checkpoint encoder (`src/laya_rag/model.py:124-136`); the
   shipped dense weight auto-resolves to `0` for it because it scores nDCG@10 ≈ 0.014 where
   BM25 scores 0.654 (`src/laya_rag/config.py:238`). Laya's relevance probability is kept
   as a *filter* because reordering by it loses (`src/laya_rag/config.py:324`).

So the weaknesses split cleanly into **retrieval-quality gaps the stock channels carry**,
**decision-head gaps**, and **scope/scale gaps**. Improvements should be aimed accordingly
rather than all at the checkpoint.

---

## 1. Coverage on multi-hop queries — MAP@10 0.279 vs 0.479 published

### What the code does

- Retrieval is a **single query pass**: one query embedding + one BM25 query, fused by RRF
  (`src/laya_rag/engine.py:950`; the benchmark harness uses exactly the same shape,
  `tools/multihop_official.py:190`). There is no decomposition, no second hop, and no
  query expansion anywhere (`grep` for `decompos|subquery|HyDE|step-back` across `src/` and
  `tools/` returns nothing).
- The coverage pass reorders **only** the fused top-100 (`tools/multihop_coverage.py:625`).
  Its four families are embedding MMR, article-level MMR, cluster dedup, and *query-term*
  coverage (`:95-144`). Gold evidence facts are used **only at scoring time** (`:170`,
  `:236-238`) — never to select. `query-coverage` therefore covers the query's own terms,
  which dense and BM25 have already matched; it is diversity-to-the-query, not coverage of
  the answer set.
- Chunks are `url#j` windows; the only document awareness is `doc-mmr`'s binary
  same-article penalty (`:104`, `:213`).
- The answer path keeps `rerank_k = 5` (`src/laya_rag/config.py:190`,
  `src/laya_rag/rerank.py:436`), and the relevance filter runs *before* any set-level
  judgment (`src/laya_rag/rerank.py:397`).

### Why the number is what it is

The metric is set-aware — `MAP@10 = ap / min(len(gold), 10)` with several facts per chunk
each contributing (`tools/multihop_official.py:74-84`) — while the retriever is a single
topical query. A multi-hop question's one embedding matches the *bridge* entity and the
lexical channel locks onto the one article that term-overlaps most, so the top-10 can be one
article's overlapping chunks. Every coverage objective is capped by **candidate-pool recall**:
a gold fact not in the fused top-100 is unreachable, so the held-out +0.024 is a re-ordering
gain, not a recall gain. That is exactly why it is a small fraction of the 0.20 gap.

### Improvements, cheapest first

| # | Change | Effort | Expected | How to falsify |
| --- | --- | --- | --- | --- |
| 1 | **Run the cross-encoder under the official protocol.** `--rerank` exists (`multihop_official.py:111`) but `eval/benchmarks/multihop-rag/official.json` contains **no reranked rows** — the published 0.479 is a *reranked* number and the like-for-like was never run. Try `bge-reranker-base`/`large` and `bge-reranker-v2-m3` at depth 100 on the fused stage. | hours | Unknown; this is the one direct comparison missing | If it reproduces the BEIR loss, close it permanently |
| 2 | **Measure the pool-recall ceiling.** Score "best achievable MAP@10 by reordering the fused top-100" and again at depth 500. | hours | Diagnostic — **done, and it settled both**: ceiling 0.609 at depth 100 and 0.659 at 500 (each step significant), with 35.4% of depth-100 pools (799 of 2,255) missing a gold fact | selection is the larger lever; recall is not solved |
| 3 | **Query decomposition / second hop (model-free first).** After pass 1, take the top documents' salient entities/phrases (not in the query), issue hop-2 dense+BM25 queries, fuse with RRF. Laya's relevance head scores hop-2 candidates as a filter. This is IRCoT without a generator. | days | The largest MAP lever, because it is the only one that raises pool recall | If pool-recall ceiling (2) was already high, this adds little |
| 4 | **Fact-level set coverage (submodular max-coverage).** Split candidates into sentence "fact atoms", score atoms for query relevance (Laya relevance head), and greedily select chunks maximising *distinct query-relevant atom* coverage with a redundancy penalty — a facility-location objective that matches MAP's definition. | 1–2 wks | Attacks the objective directly rather than by proxy | Compare against (1)/(3) on a held-out half |
| 5 | **Two-level document→passage selection with a document quota.** Force ≥k distinct documents into the top-10 (best chunk per doc), then fill. `doc-mmr@0.7` already reaches 0.274 from the same pool, so a hard quota should beat it. | hours–days | +0.005–0.02 MAP over dedup | Paired bootstrap on the held-out half |
| 6 | **Chunking/aggregation for the protocol.** At overlap 200 a single article's overlapping windows can dominate; dedup fixes part of it (`RECALL_DEDUP_THRESHOLD`, `config.py:279`). Consider non-overlapping sentence windows for the multi-hop harness, and fact-substring dedup. `query-coverage` already wins at overlap 0 (`coverage-ov0.json`). | days | Small, but removes a confound | Re-run `coverage-ov0` with the new objective |
| 7 | **Listwise / late-interaction second stage** (the repo's own open item): MaxSim or a listwise cross-encoder trained on the benchmark's evidence labels can score the *set*. | weeks | Potentially large | Only worth it if (3)–(5) plateau |

**Do not retry:** a bigger encoder (fusion absorbs it — measured twice), another pointwise CE
as a full re-sort (lost on all five BEIR corpora), or tuning coverage hyperparameters on the
evaluated queries (already held-out).

**Note on the honest comparison.** The published 0.479 is a reranked pipeline; the repo's
0.279 is not reranked. Until (1) is run, "MAP@10 loses badly" compares a no-rerank system
against a reranked one, which the document does not currently say.

---

## 2. Answerability out of domain — AUC 0.51, recalibration at chance

### What the code does

- The head is asked one `noul` question, verbatim: `"Can \`query\` be answered using only
  \`evidence\`?"` (`src/laya_rag/rerank.py:32-35`), with the evidence assembled as one
  numbered concatenation of the kept passages (`:144`) and judged in a single call (`:145-148`).
  That yields **one scalar** P(the blob answers).
- The relevance filter runs first and truncates to `top_k=5` before the head sees anything
  (`:255-262`). If nothing clears relevance, it short-circuits to `answerability = 0.0`
  (`:270-273`) — a structural zero, not a head output.
- The recalibration test consumes exactly that scalar and fits a threshold / monotone Platt
  map (`tools/multihop_recalibrate.py:169-173`).

### Why the number is what it is

A monotone rescaling cannot create ranking signal — its slope (−0.18) says the scalar ranks
the classes at chance, so "no transferable signal" is the correct conclusion *for this
scalar*. Two things the test therefore does not rule out:

- **The head may have signal that one scalar throws away.** Retrieval-confidence features
  (dense cosine, BM25 score, query-term coverage of the evidence, number of distinct
  documents, entity overlap) are known to separate hard negatives and are not in the scalar.
- **The score vector mixes mechanisms.** `p = 0` rows are relevance-gate short-circuits, so
  the class-conditional distributions are not from one decision (`multihop_recalibrate.py:182-189`).

### Improvements

| # | Change | Effort | Expected |
| --- | --- | --- | --- |
| 1 | **Feature-based gate with grouped cross-validation.** Build a small logistic model over the existing free signals (max/mean dense cosine, BM25, top relevance probability, distinct-doc count, query-term coverage, entity overlap, evidence length) and fit on the domain's null/answerable labels. This is the literal form of the repo's own framing: *the calibration procedure transfers, not the threshold*. | days | Best single shot at beating 0.53 AUC |
| 2 | **Add a dedicated answerability / entailment model.** An NLI cross-encoder ("does the evidence entail an answer?") or a SQuAD2.0/NQ-unanswerable-trained QA no-answer score. The `CrossEncoderModel` plumbing already exists (`src/laya_rag/model.py:386-441`); add an answerability mode. | days–weeks | Portable, out-of-domain signal the head lacks |
| 3 | **Per-passage support instead of one blob.** Score whether each passage supports a specific answer/claim, aggregate by noisy-OR, then combine support with coverage. Shared machinery with §1(4), so it pays twice. | 1–2 wks | Fixes both the gate and multi-hop set evidence |
| 4 | **Two explicit stages: retrieval confidence ∧ answerability.** Report both; abstain when either is low. Calibrate each separately instead of letting the relevance gate silently do most of the abstaining (already noted in *Known limits*). | days | Cleaner selective-prediction curves |
| 5 | **LLM-judge fallback (opt-in).** Ask the composer endpoint "can this be answered from this evidence?" — honest and out-of-domain-capable, but attribute it to the LLM. | hours | A working baseline / label source |
| 6 | **Stronger splits.** Test transfer across *question types, time, and articles* (grouped), not a random stratified half; report risk–coverage rather than one balanced-accuracy point. Random halves are the weakest transfer test. | days | Turns a weak negative into a real one |
| 7 | **Propagate the short-circuit distinction into the engine.** Do not present `0.0` as a calibrated head output; emit "not measured" (as `adjudicate_claims` already does, `engine.py:1369-1375`). | hours | Correct calibration inputs |

**Do not retry:** refitting the threshold on the head's probability (measured no-op), or
citing the shipped 0.5 threshold out of domain.

---

## 3. No owned generation → each external target is blocked for its own reason

### What the code does

`compose.py` provides a real but opt-in generator: local Ollama by default
(`ornith-1.5:35b`, `config.py:26,33`), OpenAI-compatible or native Ollama routes
(`compose.py:78-112`), streamed, with a citation-forcing system prompt (`:52-63`). The API
layer runs `verify_citations` (`api.py:456`) and `adjudicate_claims` (`api.py:477`). There is
**no entity linker, no batch `generate()` with logprobs, no nuggetizer, and no
factual-consistency scorer.** MTEB is *model*-centric, so an engine cannot be entered as-is.

### Improvements — split retrieval-side from generation-side

**Retrieval-side (no generator, highest credibility per unit effort):**

| # | Change | Effort | Why it matters |
| --- | --- | --- | --- |
| 1 | **Submit a model, not the engine, to MTEB.** Distil the 3-model ensemble (§4) into one HF `SentenceTransformer`-compatible encoder and submit it. Its retrieval tasks are largely BEIR's. | the distillation project | Converts 5/18 into a leaderboard row |
| 2 | **Run the benchmarks the report already calls runnable:** BRIGHT, LoTTE, LegalBench-RAG, FRAMES retrieval, RGB negative rejection. | days each | Broadens measured coverage with *no* generator |
| 3 | **Build a local TREC-RAG-shaped scorer.** `verify.py` + adjudication are the structural match; add nugget extraction from gold, nugget recall, and a support classifier. | weeks | Removes "no run exists" if the corpus/gold are obtainable |

**Generation-side (report as `engine + <model>`):**

| # | Change | Effort | Why it matters |
| --- | --- | --- | --- |
| 4 | **Offline batch generation adapter + a harness that records the generator id** alongside retrieval metrics (BERGEN/FlashRAG style), so retrievers can be compared holding the LLM fixed. | days | Makes generation-side numbers attributable instead of banned |
| 5 | **KILT:** add an entity linker (BLINK/`genre`) and a seq2seq generator behind the existing composer. | weeks | Full-stack row; the score is the generator's |
| 6 | **HHEM:** do not chase the leaderboard — it ranks detectors. Instead load a Vectara HHEM checkpoint as an optional faithfulness filter over composed prose and report agreement. | days | Honest use of the artifact |
| 7 | **Enforce attribution in the schema:** every composed-output metric carries `generator=<model>`. | hours | Prevents relabelling someone else's number |

**Do not:** claim any end-to-end accuracy as the engine's. The docs already say this; make it
structural.

---

## 4. Laya's own encoder is close to useless for retrieval

### What the code does

`LayaModel.embed` mean-pools the checkpoint encoder via `laya.embed_fn_from_agent`
(`src/laya_rag/model.py:114-136`); the checkpoint is a decision/agent model, not
retrieval-trained. Dense weight auto-resolves to `0` for Laya (`config.py:302`), so the
shipped default is lexical-only and `EMBED_MODEL` (stock bge) is what supplies semantic
recall. The engine's differentiator is therefore a *decision* model.

### Improvements

| # | Change | Effort | Expected |
| --- | --- | --- | --- |
| 1 | **Distil the 3-model ensemble into one owned encoder** (the repo's stated open item). Teacher = ensemble RRF ranks / `bge-large` scores; student = small encoder with MarginMSE or rank-distribution KL; optionally distil the multi-hop coverage objective too. | weeks | Keep most of +0.043 ensemble gain at ~¼ params; enables §3(1) and cuts §6 cost |
| 2 | **Give Laya's encoder a trained projection head before writing it off.** A linear/2-layer contrastive adapter on top of the mean-pooled embedding, trained on MS MARCO/in-domain pairs, tests whether the representation is unusable or merely unadapted. Also sweep pooling (CLS / last-token / layer) and task prefixes — the bge path showed pooling alone is worth ~0.014. | hours–days | Cheap test with real upside |
| 3 | **Late interaction (ColBERT-style MaxSim) from Laya hidden states.** Replace mean pooling with token-level embeddings + MaxSim — the architecture lever the repo itself identified, at near-zero new parameters. Needs a multi-vector store, so pair with §6. | weeks | Potentially large over mean pooling |
| 4 | **Ship a working retrieval encoder as the default.** Even stock `bge-small`, or the distilled student, should be the default rather than an env opt-in; today a query with no lexical overlap retrieves nothing (*Known limits*). | hours | Product-grade recall floor |
| 5 | **Keep Laya where it is strong:** the gate/filter. Consider `typed-decisions` for richer decisions. Do not re-attempt it as a ranker (measured negative, `config.py:327`). | — | Avoids wasted work |

---

## 5. Cross-encoder reranking did not pay off

### What the code does

`CrossEncoderModel.score_pairs` is a pointwise MS-MARCO sequence classifier with sigmoid
(`model.py:386-441`). In the harness, `crossencoder_rankings` slices the first-stage top-`depth`
and then **discards the first-stage order entirely**: `order = sorted(..., key=-block[index])`
(`tools/beir.py:545-576`, esp. `:561`, `:574`). It re-scores raw `corpus[doc_id]`, paying the
512-token truncation a second time. Measured: −0.027 to −0.318 nDCG@10 on all five corpora,
with ArguAna the catastrophe (relevance there is an *opposition* relation a pointwise ranker
does not represent).

### Improvements

| # | Change | Effort | Expected |
| --- | --- | --- | --- |
| 1 | **Blend, don't replace.** Score = first-stage rank score + λ·CE, or RRF over the first-stage list and the CE list, with λ chosen on a calibration half. This bounds the downside and answers whether CE has *any* value here. | hours–days | Likely the highest-value change in this section |
| 2 | **Use CE as a filter, not an orderer** (`RERANK_ORDER=recall` already exists, `config.py:324`): drop low-CE candidates, keep the fused order. | hours | Safe, cheap |
| 3 | **Route the second stage per corpus.** `tools/beir_routing.py` already selects channels on a calibration half; extend it to CE on/off and λ. | hours–days | Recovers CE where it helps, no loss elsewhere |
| 4 | **Right model class for the relation.** Listwise (RankZephyr-style) or pairwise CE; for ArguAna, train/pick on the opposition convention. | weeks | Addresses the actual failure mode |
| 5 | **Late-interaction second stage** (ColBERT / `answerai-colbert-small-v1`): term-level interaction handles negation/opposition better than a pooled pointwise CE. | days–weeks | Strong for relational relevance |
| 6 | **Guardrail: never reorder the top 2.** CE only from rank 2–3 down, on a smaller deduped pool. | hours | Cheap insurance |
| 7 | **Diagnose by relation type** (topical vs relational) and use it as the routing signal. | days | Makes routing principled |

**Do not retry:** `bge-reranker-base` as a full re-sort (measured on all five), or assuming
depth is the problem — depth 100 is already the literature setting.

---

## 6. Scale — in-memory index and Python-dict postings

### What the code does

- `VectorIndex` holds `_chunks` (full chunk text) and `_vectors` (fp32) in RAM
  (`src/laya_rag/index_store.py:55-63`, `:101-130`); `save` writes every chunk's text into
  `index.json` (`:216-232`). `RagEngine.index_paths` accumulates every embedding in a Python
  list and then `np.vstack`s it (`src/laya_rag/engine.py:739-751`) — two to three live copies.
- `BM25Index` keeps `_term_freqs` (a `Counter` per chunk), `_doc_freq`, and
  `_postings: dict[str, list[tuple[int,int]]]` (`src/laya_rag/lexical.py:106-126`) — heavy
  Python objects.
- `OnDiskBM25` and `StreamingVectorIndex` are **implemented and validated** (128 MB packed
  postings for 171k TREC-COVID docs, byte-identical ranking; flat 1,249 MB streaming build)
  but live only in `tools/ondisk.py` — a grep confirms **neither is imported anywhere in
  `src/`**. The engine's build path is purely in-memory.
- ANN is in-RAM faiss IVF, optional (`index_store.py:161-176`, `engine.py:761`); there is
  no on-disk ANN. The BEIR harness is fully eager: whole `corpus.jsonl` as a string + dict
  (`tools/beir.py:131-152`), full `.npy` matrix (`:362-388`), and the CE pair list
  materialised for every query × depth at once (`:563-566`).
- Dense cost: 384-dim fp32 = 1.5 KB/doc → BioASQ ≈ 25 GB, times the live copies.

### Improvements, ordered by leverage

| # | Change | Effort | Why |
| --- | --- | --- | --- |
| 1 | **Move chunk text out of RAM/metadata.** `Chunk` already carries `path`, `start_char`, `end_char` (`index_store.py:269-279`), so text can live in a compressed/mmap sidecar or be re-read lazily from source. Biggest, simplest memory win; ranking unchanged. | days | Removes the largest resident object |
| 2 | **Wire the on-disk stores into the engine** behind an index-backend setting, and make the build path stream instead of `vstack` (`engine.py:739-751`). Add an incremental update path (the tool has none — `build()` writes fresh). | days–weeks | Makes >RAM corpora buildable/queryable |
| 3 | **Replace/augment pure-Python BM25 with an on-disk IR engine.** SQLite FTS5 (stdlib, offline) or Tantivy gives on-disk postings, compression, and incremental update, and deletes the 2.4 GB Python-dict peak. Keep `BM25Index` for small corpora/tests. | days–weeks | Directly answers "postings are Python dicts" |
| 4 | **Bounded external k-way merge for the term dictionary** (the documented open item) via `heapq` over sorted runs. | days | Flat-memory 10M+ lexical build |
| 5 | **On-disk + quantized ANN.** faiss `IVFPQ`/`OPQ` (or `IVFFlat` with `OnDiskInvertedLists`), or `usearch`/`hnswlib` with mmap; scalar/binary quantization cuts 384-d to ~48–96 B/vector (16M docs ≈ 0.8–1.5 GB). Consider Matryoshka 128-d for pass 1. | weeks | Turns ANN from toy to deployable |
| 6 | **Stream the BEIR harness.** Iterate `corpus.jsonl` line by line, embed → positional memmap write, postings per block → merge. Never materialise the dict or the `.npy`. Add `--shard` for parallel build. | days–weeks | Removes the 2–3× peak |
| 7 | **Cap CE pair materialisation** — score per query in streaming batches instead of building one global list (`beir.py:563-566`). | hours | Removes a large transient |
| 8 | **Store per-query scores in SQLite/Parquet**, not giant JSON. | hours | Makes scale runs analysable |

After (1)–(5) the "five of eighteen datasets" limit becomes a build schedule rather than a
capability wall — the same conclusion `RANKINGS.md:548-551` already reaches.

---

## 7. SciDocs — 0.199 vs a published 0.225

### What the code does

The dense path embeds `title + text` as **one unsplit chunk** with `max_length=512` and CLS
pooling (`tools/beir.py:196-208`, `:250`, `:288-302`). Measured on the cached corpus:
title+text averages 177 words, p99 is 548 words, and ~2.4% of documents exceed ~380 words
(≈512 tokens). So truncation is a real but **small tail effect**, not the main cause. The
repo itself calls the shortfall "a single-encoder effect"; the 3-model ensemble reaches 0.225.

### Improvements

| # | Change | Effort | Expected |
| --- | --- | --- | --- |
| 1 | **Use a scientific/citation-domain encoder.** SciDocs relevance is a citation relation between a paper title and a cited abstract; `allenai/specter2` (or SciNCL) is trained for exactly that. Add it as a per-corpus embedder. | hours–days | Likely clears 0.225, not just reaches it |
| 2 | **Distil the ensemble into one encoder** (§4) — preserves the single-model claim at lower cost. | weeks | 0.225-class at ~¼ params |
| 3 | **Representation sweep for SciDocs only:** `--dense-text-only`, query instruction on/off (queries are titles), pooling, title weighting. | hours | Free points |
| 4 | **Dense passage splitting + max-pool** for the ~2% long abstracts. Only BM25 has `passage_rankings` today (`beir.py:426-478`); add the dense equivalent. | days | Small here, unlocks Robust04/TREC-NEWS later |
| 5 | **Per-corpus model routing.** Extend `beir_routing.py` from channel selection to embedder selection (SciDocs→SPECTER2, ArguAna→bge-small text-only, …). | days | General form of the fix |
| 6 | **Citation-aware training pairs** if distilling/adapting: mine cited pairs as positives. | part of §4 | Domain alignment |

---

## 8. Prioritised cross-cutting plan

| Rank | Action | Addresses | Effort | Risk |
| --- | --- | --- | --- | --- |
| 1 | Run the CE under the MultiHop protocol + measure the pool-recall ceiling | §1 | hours | none (diagnostic) |
| 2 | Wire on-disk stores into the engine; text out-of-line; SQLite/Tantivy lexical | §6 | weeks | low (ranking-preserving) |
| 3 | Blend CE with the first stage + per-corpus second-stage routing | §5 | days | low |
| 4 | Feature-based answerability gate with grouped CV (+ NLI option) | §2 | days–weeks | medium |
| 5 | Query decomposition + fact-level submodular coverage | §1 | weeks | medium |
| 6 | Distil the ensemble into one encoder; submit it to MTEB | §3, §4, §6, §7 | weeks | medium |
| 7 | Run BRIGHT / LoTTE / LegalBench-RAG / FRAMES / RGB | §3 | days–weeks | low |
| 8 | SPECTER2 (or similar) for SciDocs via routing | §7 | hours–days | low |

Items 1, 2, 3 and 8 are low-risk and cheap and should not wait on the research items. Item 6
is the one that changes the *character* of the system — from "retrieval quality comes from
stock components" to an owned, submittable model.

### Measurement discipline to carry into all of it

- **Pin the chunking constant** in every cross-system comparison (up to +0.06 Hits@10 from an
  undeclared overlap — `BENCHMARKS.md:119-132`).
- **Report the candidate-pool ceiling** alongside any re-ranking result, so a selection gain
  is never confused with a recall gain.
- **Use grouped splits** (question type / time / article) for transfer claims, not random
  halves.
- **Label every generator-attributed number** with its model id.

---

## Implementation status (this pass)

See [`PLAN.md`](PLAN.md) for the phased plan and checkboxes. What landed, and what it
measures:

### Code

| Area | Change | Where |
| --- | --- | --- |
| Multi-hop set coverage (§1) | `doc_quota_select`, `facility_location_select` (sentence-atom submodular max-coverage), `pool_ceiling_map` diagnostic; new held-out families | `tools/multihop_coverage.py`, `tests/test_coverage_select.py` |
| Multi-hop coverage in the engine (§1) | `RECALL_COVERAGE_LAMBDA` + `_coverage_select` on the fused candidate pool, so the measured best objective is in the shipped retrieval path and not only the harness | `src/laya_rag/engine.py`, `src/laya_rag/config.py`, `tests/test_recall_coverage.py` |
| Answerability (§2) | Pure-NumPy multi-feature logistic gate (9 retrieval/evidence signals), fit/save/load/AUC, plus a trainer that scores it out of sample | `src/laya_rag/answerability.py`, `tools/answerability_train.py`, `tests/test_answerability*.py` |
| Answerability wiring (§2) | Optional gate on `LayaReranker.rerank`, `ANSWERABILITY_GATE_PATH`, `answerability_measured` so a structural zero is not fed to calibration, gate status in `stats()` | `src/laya_rag/rerank.py`, `src/laya_rag/config.py`, `src/laya_rag/engine.py` |
| Cross-encoder blending (§5) | `blend_scores` + `blend` (`RERANK_BLEND`) and `--rerank-blend`: a second stage nudges via RRF instead of replacing the first-stage order | `src/laya_rag/rerank.py`, `tools/beir.py` |
| Scale (§6) | Build path preallocates the embedding matrix instead of list + `np.vstack` | `src/laya_rag/engine.py` |
| SciDocs / routing (§7) | `--dataset-models` (per-corpus encoder) and `--dense-text-only-datasets` | `tools/beir.py` |
| On-disk engine backend (§6) | `INDEX_BACKEND=ondisk`: `OnDiskBM25` (packed mmap postings) + `OnDiskVectorIndex` (mmapped vectors, deferred chunk text via `os.pread`), with engine build/query wiring and ranking-parity tests | `src/laya_rag/ondisk.py`, `src/laya_rag/engine.py`, `tests/test_ondisk_engine.py` |
| Quantized ANN (§6) | `OnDiskVectorIndex.build_ann` (IVF+PQ, mmapped fallback) and engine `ANN` wiring | `src/laya_rag/ondisk.py`, `tests/test_ondisk_engine.py` |
| NLI answerability (§7) | `NLIAnswerability` (lazy cross-encoder, label resolution, injectable scorer) with precedence over the gate | `src/laya_rag/answerability.py`, `src/laya_rag/engine.py`, `tests/test_nli_answerability.py` |
| Encoder projection head (§4) | `LinearProjection` ridge distillation + `evaluate_projected` on cached Laya/teacher embeddings | `tools/projection_head.py`, `tests/test_projection_head.py` |
| Generation/placement (§3, prep only) | Target table, generator-attribution enforcement, citation-mechanics report; no submissions run | `docs/generation-placement.md`, `tools/generation_eval.py`, `tests/test_generation_eval.py` |

Offline suite: **472 passed, 4 deselected** (baseline 344).

### Measured results

- **Multi-hop coverage (§1).** The new submodular evidence-set objective is the best pass
  measured: `facility-0.5` reaches **MAP@10 0.2835** (held-out **+0.0288** over the 0.2546
  fused baseline) with **Hits@4 0.7197**, beating the previous best `cluster-dedup-0.9`
  (0.2790, +0.0244). `doc-quota-4` adds +0.0173. A later full-protocol run showed the
  dense-gram redundancy term is not needed at the recommended λ: `facility-nored-0.5`
  reaches **MAP@10 0.2850** (+0.0015 [+0.0002, +0.0029], p=0.032), **Hits@4 0.7326** and
  **Hits@10 0.8723**, so `RECALL_COVERAGE_REDUNDANCY=false` is the shipped default (at λ=0.7
  the penalty does help, +0.0036, which is why it stays exposed). Reports:
  `eval/benchmarks/multihop-rag/coverage-set.json`, `coverage-nored.json`.
- **The gap is selection *and* recall (corrected 2026-10-03).** The pool-recall ceiling — the best
  MAP@10 any reordering of the fused top-100 could reach — is **0.609** corrected (0.7307 as
  originally reported, mis-normalised), against the 0.2850 achieved
  and the 0.479 published. The evidence is already in the candidate pool; ordering is where
  the remaining headroom is.
- **SciDocs (§7).** Query instruction lifts the single `bge-small` encoder from **0.1973 to
  0.2052** nDCG@10 on cached embeddings, with no re-embedding. Reports:
  `eval/beir/scidocs-title2*.json`.
- **Answerability (§2).** The feature gate is a real fix, not another negative. On a fresh
  300-query MultiHop-RAG run (150 null / 150 answerable), Laya's shipped scalar abstains at
  32.7% on nulls and 32.0% on answerable questions — the same chance behaviour as before —
  but the gate fit on one half scores the other at **ROC-AUC 0.8146 [0.7628, 0.8570]** and
  **balanced accuracy 0.7248**, against the scalar's **0.5199 / 0.5026** on the same
  queries. That is a **within-question-type** figure: holding out a whole question type
  (`tools/answerability_train.py --group-by question_type`) gives **0.7666 [0.6712,
  0.8267]**, worst on comparison questions (0.6640) and strongest on temporal ones (0.8276),
  still far above the scalar's 0.5235 on the same folds. So the signal was always in the
  retrieval/evidence features; the single scalar simply discarded it. Artefacts:
  `eval/benchmarks/multihop-rag/gate-features.json`, `answerability-gate.json`,
  `answerability-gate.json.gate`, `answerability-grouped.json`.

- **Cross-encoder on MultiHop-RAG (§5/§1) — it pays here, unlike on BEIR.** A deterministic
  400-query subset (seed 17, overlap 200) reranked with `bge-reranker-base` at depth 100:
  the hybrid improves MAP@10 0.2522 → **0.2986** (+0.0464 [+0.0275, +0.0655]), Hits@4
  0.6625 → **0.7225** (+0.0600), Hits@10 0.7975 → **0.8600**, MRR@10 0.5342 → **0.6010** —
  every paired interval excludes zero. The earlier "reranking did not pay off" result was
  BEIR-specific; on the multi-hop corpus it is the largest single lever measured, above the
  coverage pass. Report: `eval/benchmarks/multihop-rag/official-rerank-subset.json`.
  This is a subset, not the full 2,255-query protocol: the full run is hours on this host.
- **On-disk backend (§6).** Tests pin byte-identical rankings against the in-memory backend
  for both the mmap postings and the streamed vectors, with chunk text read on demand; the
  quantized IVF+PQ ANN is opt-in with an exact mmap fallback. Full BioASQ-scale builds still
  need streaming ingestion (the chunker materialises text) and a bounded term-dictionary
  merge.
- **Laya's encoder is unadapted, not useless (§4).** On SciFact, a closed-form ridge
  projection on the frozen Laya document embeddings, distilled from cached bge-small
  embeddings, lifts nDCG@10 from **0.0141 to 0.0810** — 5.7× — with a teacher-document-only
  control capping the doc-side ceiling at 0.1327. So a cheap affine adaptation recovers a
  large part of the representation's signal; the remaining gap is the *query* encoder, which
  is what a packaged distilled encoder would fix. Report: `eval/beir/projection-scifact.json`.
- **SPECTER on SciDocs — an honest negative, and a pooling bug fixed (§7).**
  `sentence-transformers/allenai-specter` needs CLS pooling; the harness's `auto` gave it mean
  (0.0943 dense / 0.1599 hybrid), and with `--biencoder-pooling cls` it improves to 0.1475 /
  0.1801 — still below `bge-small`'s 0.1973 / 0.1929 and the ensemble's 0.225. SPECTER v1 is
  not the SciDocs fix; the `--biencoder-pooling` flag and the non-BGE pooling caveat are the
  durable outcomes. Reports: `eval/beir/scidocs-specter*.json`.

### Still external (Phase 5)

- **Encoder distillation** — the projection-head path (`tools/projection_head.py`) exists and
  is unit-tested; packaging a submittable encoder is a training run.
- **Full-protocol MultiHop rerank and the full BEIR 18** — hours on MPS / disk-limited
  (BioASQ 14.9M, FEVER 5.4M, NQ 2.7M documents).
- **Runnable retrieval-only benchmarks** (BRIGHT, LoTTE, LegalBench-RAG, FRAMES, RGB) — format
  adapters plus downloads; the run order is in `docs/generation-placement.md`.
- **§3 submissions** — prepared only (scaffold + doc), never submitted.


One honest caveat on the gate: it was fit and scored on a 300-query split of one domain
(MultiHop-RAG news) and uses features derived from the engine's own retrieval, so it is
"recalibrate on the target domain", which is exactly the procedure the earlier negative
recommended — not proof that a gate trained here transfers elsewhere untouched.

---

## Known-limits pass

The README's [`Known limits`](README.md#known-limits) are measured probes, not prose. Three of them
were closed or moved this pass; each is covered by offline tests and the `make bench` probe named
beside it.

| Limit | Change | Where | Measured |
| --- | --- | --- | --- |
| A fact longer than the overlap is lost across a chunk boundary | default `CHUNK_OVERLAP_CHARS` 200 → 400, plus a sentence-aware overlap: a cut inside a sentence backs the next chunk up to the sentence start | `src/laya_rag/chunking.py`, `config.py` | the 290-character fact that was in no chunk at shift 250 is intact and retrieved at shifts 250/300/350/450 (`make bench BENCH_ARGS="--probes boundary"`) |
| Deleting a file leaves retrievable chunks | `index_paths` prunes indexed paths under a re-indexed directory that are gone or were refused by the loader; `--no-prune` / `prune=false` opts out, single-file roots are never pruned | `src/laya_rag/engine.py`, `cli.py`, `api.py` | `tests/test_engine.py` (five tests) |
| On-disk BM25 build peak RSS dominated by one block | a block flushes on `--block-tokens` as well as `--block-docs`; the run merge is a heap holding one row per run | `src/laya_rag/ondisk.py`, `tools/ondisk.py` | 30,000 × 200-token synthetic corpus: **1,045 MB → 453 MB** peak RSS at the same build time, identical ranking across block sizes |

The remaining limits are design positions or need resources this host does not have, and stay in the
README as stated: the dense channel is off by default because Laya's encoder is not retrieval-trained
(a bi-encoder opt-in is the documented fix); a pointwise cross-encoder helps a weak first stage but
not a strong one; Laya's probability is a filter, not a ranker; a claim about a topic the corpus
never mentions is `not_measured` rather than refuted; and the state budget is soft because Laya
truncates the state and the best passage is read first.

**Correction (2026-10-03).** The ceiling quoted as 0.7307 was mis-normalised — it divided by the gold
facts the pool matched, not the query's true gold count — and is **0.609** corrected. It also *rises*
with pool depth (0.659 at 500) rather than falling, and 35.4% of depth-100 pools are missing a gold
fact. The conclusion below that the gap is purely selection therefore overstates it: recall is a live
lever. See the README diagnostics subsection.
