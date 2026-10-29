> **Provenance.** A research digest compiled on **2026-10-29** for
> [`docs/best-in-the-world-2026-10.md`](../best-in-the-world-2026-10.md). Every figure was read from the
> cited source on that date; the raw page and PDF caches, the MTEB API dumps and the per-section working
> files are **not** committed (they live under the session's `.scratch/research/`), so the paths in the
> appendix below are the session's, not this checkout's. Provenance tags — `[MTEB-official]`,
> `[paper Table N]`, `[model card]`, `[vendor]`, `[3rd-party bench]`, `[blog measurement]` — are the
> digest's own and are kept because the difference between them is the point. Nothing here is a claim of
> this repository's; the claims it supports are in the assessment, and each carries its own falsifier.

---

# Retrieval & Ranking: State of the Art, late October 2026

**Scope:** retrieval and ranking only (first-stage retrieval, reranking, sparse/lexical, index infrastructure,
quantization, benchmarks). Written for a fully-local, CPU-only, no-API-key RAG engine whose current numbers are
BEIR-18 best-per-dataset macro nDCG@10 = **0.4799** (SPLADE published 0.4740 as its comparison row).

**Compiled:** 2026-10-29. Every number below carries a date and a provenance tag:

| tag | meaning |
|---|---|
| `[MTEB-official]` | read out of the official MTEB leaderboard API (`mteb-leaderboard-backend.hf.space`), i.e. re-measured by the benchmark, not the authors |
| `[paper Table N]` | self-reported by the authors in their own table (protocol stated) |
| `[model card]` | the authors' own model card / repo README |
| `[vendor]` | vendor blog or commercial page — treat as marketing until reproduced |
| `[3rd-party bench]` | an independent, reproducible benchmark run by someone other than the model's author |
| `[blog measurement]` | an individual's blog benchmark; single machine, method visible, not peer reviewed |

The state of the art moves fast enough that **my own memory of this field is stale**: the 2026 material
(Section 1's leaders, RTEB, the LightOn/"On" family, KURE/v-splade, PipeANN, Doc2Query++) is all from
primary pages fetched today, and where a search came up empty I say so explicitly.

---

## 1. What actually leads MTEB v2 retrieval today

### 1.1 Source and method

The official leaderboard is now a public API. I read the raw score tables rather than a screenshot:

```
GET https://mteb-leaderboard-backend.hf.space/v1/benchmarks/MTEB%28eng%2C%20v2%29/scores
```

Snapshot fetched **2026-10-29**. The benchmark contains 41 tasks and **349 model rows: 221 have a retrieval
score, 191 have all 41 tasks.** The **retrieval** sub-type is a mean nDCG@10 over exactly these 10 tasks:

`ArguAna, CQADupstackGamingRetrieval, CQADupstackUnixRetrieval, ClimateFEVERHardNegatives,
FEVERHardNegatives, FiQA2018, HotpotQAHardNegatives, SCIDOCS, TRECCOVID, Touche2020Retrieval.v3`

The newest entries in the snapshot are dated **2026-10-21** (`hotchpotch/bekko-static-embedding-v1`) and
**2026-10-10** (`dmis-lab/UltraBERT-Embed-150M`, `DataScience-UIBK/uibk-embedding-v1`), so this is a current
board, not an archived one.

Cross-check: the independent aggregator ModelMenu's "MTEB English · Retrieval" board, last updated
**Oct 27, 2026**, lists the same top-10 in the same order with the same scores (70.7 / 70.0 / 69.4 / 68.8 / 68.5 /
67.5 / 67.1 / 66.2 / 66.2 / 64.4) — <https://modelmenu.dev/benchmarks/mteb>. Two independent readings agree,
including on the 68.8 row below that is easy to miss.

Caveat worth knowing: the API flags `MTEB(eng, v2)` with `displayOnLeaderboard: false`, and `MTEB(eng, v1)`
declares v2 as its `newVersion`. The live front-end at <https://leaderboard.mteb.org/> still serves the numbers.

### 1.2 The top of the board (retrieval nDCG@10, MTEB(eng,v2), 10 tasks)

`P` = total parameters (billions), `D` = embedding dim, `zs` = % of tasks the model was *not* trained on
(MTEB's own metadata; lower = more benchmark overlap).

| # | model | nDCG@10 | P | D | released | open weights | license | zs | CPU-feasible? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **IEITYuan/Yuan-embedding-2.0-en** | **0.7069** | 0.596 | 1024 | 2025-11-27 | yes | apache-2.0 | 95 | query-time yes (~1 s/q), bulk-index no |
| 2 | jcorners/ingot-8b-r3 | 0.7000 | 7.567 | 4096 | 2026-05-23 | **no** | — | 82 | no |
| 3 | **Qwen/Qwen3-Embedding-8B** | 0.6944 | 7.567 | 4096 | 2025-06-05 | yes | apache-2.0 | 95 | no (GPU) |
| 4 | geevec-ai/geevec-embeddings-1.0 | 0.6876 | n/a | 4096 | 2026-04-02 | **no** | — | n/a | no |
| 5 | Qwen/Qwen3-Embedding-4B | 0.6846 | 4.022 | 2560 | 2025-06-05 | yes | apache-2.0 | 95 | borderline |
| 6 | ByteDance-Seed/Seed1.5-Embedding | 0.6745 | n/a | 2048 | 2025-04-25 | no | — | 51 | no |
| 7 | Kingsoft-LLM/QZhou-Embedding | 0.6712 | 7.071 | 3584 | 2025-08-24 | yes | apache-2.0 | 48 | no |
| 8 | infgrad/Jasper-Token-Compression-600M | 0.6619 | 0.607 | 2048 | 2025-11-14 | yes | mit | 48 | plausible |
| 9 | annamodels/LGAI-Embedding-Preview | 0.6618 | 7.11 | 4096 | 2025-06-11 | yes | apache-2.0 | 51 | no |
| 10 | Bytedance/Seed1.6-embedding | 0.6490 | n/a | 2048 | 2025-06-18 | no | — | 48 | no |
| 11 | google/gemini-embedding-001 | 0.6435 | n/a | 3072 | 2025-07-14 | no | — | 95 | no (API) |
| 12 | infly/inf-retriever-v1 | 0.6407 | 7.069 | 3584 | 2024-12-24 | yes | apache-2.0 | 80 | no |
| 13 | nvidia/NV-Embed-v2 | 0.6284 | 7.85 | 4096 | 2024-09-09 | yes | **cc-by-nc-4.0** | 51 | no |
| 14 | **DataScience-UIBK/uibk-embedding-v1** | **0.6248** | 0.179 | 128 | 2026-10-10 | yes | apache-2.0 | 87 | **yes** |
| 15 | geevec-ai/geevec-embeddings-1.0-lite | 0.6223 | 0.366 | 4096 | 2026-04-02 | yes | apache-2.0 | n/a | borderline |
| 16 | **Qwen/Qwen3-Embedding-0.6B** | **0.6183** | 0.596 | 1024 | 2025-06-05 | yes | apache-2.0 | 95 | **yes** |
| 17 | GeoGPT/GeoEmbedding | 0.6091 | 7.111 | 4096 | 2025-04-22 | yes | apache-2.0 | 58 | no |
| 18 | infly/inf-retriever-v1-1.5b | 0.6083 | 1.543 | 1536 | 2025-02-08 | yes | apache-2.0 | 80 | borderline |
| 19 | codefuse-ai/F2LLM-v2-14B | 0.6063 | 13.99 | 5120 | 2026-03-09 | yes | apache-2.0 | 43 | no |
| 20 | Linq-Embed-Mistral | 0.6014 | 7.111 | 4096 | 2024-05-29 | yes | cc-by-nc-4.0 | 95 | no |
| 21 | nvidia/NV-Embed-v1 | 0.6013 | 7.85 | 4096 | 2024-09-13 | yes | cc-by-nc-4.0 | 51 | no |
| 22 | jinaai/jina-embeddings-v5-omni-small | 0.6007 | 1.626 | 1024 | 2026-04-01 | yes | **cc-by-nc-4.0** | n/a | no |
| 33 | Snowflake/snowflake-arctic-embed-l | 0.5904 | 0.335 | 1024 | 2024-04-12 | yes | apache-2.0 | 95 | **yes** |
| 35 | **jinaai/jina-embeddings-v5-text-nano** | 0.5880 | 0.212 | 768 | 2026-02-17 | yes | **cc-by-nc-4.0** | n/a | **yes** |
| 42 | gte-Qwen2-7B-instruct | 0.5809 | 7.069 | 3584 | 2024-06-15 | yes | apache-2.0 | n/a | no |
| 47 | e5-mistral-7b-instruct | 0.5762 | 7.111 | 4096 | 2024-02-08 | yes | mit | 95 | no |
| 49 | **dmis-lab/UltraBERT-Embed-150M** | 0.5720 | 0.151 | 768 | 2026-10-10 | yes | apache-2.0 | 95 | **yes** |
| 60 | google/embeddinggemma-300m | 0.5569 | 0.308 | 768 | 2025-09-04 | yes | gemma | 95 | **yes** |
| 78 | microsoft/harrier-oss-v1-270m | 0.5390 | 0.268 | 640 | 2026-03-27 | yes | mit | 87 | **yes** |
| 108 | google/embeddinggemma-2 | 0.5133 | 0.271 | 768 | 2026-10-06 | yes | gemma | 95 | **yes** |
| 56 | jinaai/jina-embeddings-v4 | 0.5615 | 3.935 | 2048 | 2025-06-24 | yes | cc-by-nc-4.0 | 86 | no |
| 64 | bge-large-en-v1.5 | 0.5544 | 0.335 | 1024 | 2023-09-12 | yes | mit | 93 | **yes** |
| 154 | all-MiniLM-L6-v2 | 0.4292 | 0.023 | 384 | 2021-08-30 | yes | apache-2.0 | 86 | **yes** |
| 156 | mteb/baseline-bm25s | 0.4207 | 0 (lexical) | — | 2026-05-06 | yes | mit | n/a | **yes** |

All rows `[MTEB-official]`, snapshot 2026-10-29 (models to 2026-10-20).

### 1.3 Reading of it

* **(a) CPU-feasible and (c) <1B and (b) openly licensed at once:** the honest shortlist is
  **Yuan-embedding-2.0-en (0.596B, apache-2.0, 0.7069)** — the board leader — plus
  **Qwen3-Embedding-0.6B (0.596B, apache-2.0, 0.6183)**, **snowflake-arctic-embed-l (0.335B, apache-2.0, 0.5904)**,
  and below them a 2026 tier at 200–600M: **jina-v5-text-nano (0.212B, 0.5880, but CC-BY-NC — non-commercial)**,
  **Jasper-Token-Compression-600M (0.607B, mit, 0.6619)**.
* **The one-line summary of the leaderboard is "small won".** The top model is 0.6B and is a *fine-tune of
  Qwen3-Embedding-0.6B*, not a bigger model (<https://huggingface.co/IEITYuan/Yuan-embedding-2.0-en> `[model card]`).
  It beats every 7–14B open model on the board and beats Gemini's embedding API by 6.3 points.
* **Licensing is now the sharpest filter, not accuracy.** jina's whole v5 line and NV-Embed-v1/v2 are
  CC-BY-NC-4.0. If the product is commercial, the usable high tier is Yuan / Qwen3 / Jasper / Snowflake
  (apache-2.0 / mit).
* **`zs` matters.** Rows with `zs` ≤ 51 (Seed, QZhou, F2LLM, LGAI) trained on a large share of the evaluation
  tasks; compare them with that in mind. Yuan and Qwen3-8B both sit at `zs=95` on this board, so Yuan's win is
  not a contamination artefact *on MTEB's own accounting* — but it is a leaderboard submission, not an
  independent re-run.
* **Not on the board at all:** SPLADE/SPLADE++ (no submission with full BEIR coverage), ColBERTv2 (only SCIDOCS),
  BGE-M3 (only 5 of 15 BEIR tasks). Their numbers have to come from papers — Section 2.

---

## 2. BEIR rows, 2024–2026, with the provenance of each

Two different things get called "BEIR average". Keep them apart:

* **BEIR-13** = the 13 datasets papers tabulate (ArguAna, ClimateFEVER, DBPedia, FEVER, FiQA, HotpotQA,
  NFCorpus, NQ, Quora, SCIDOCS, SciFact, TRECCOVID, Touché-2020). BM25 = **44.0** `[paper Table 2, Contriever]`.
* **BEIR-15** = MTEB's BEIR board = those 13 **plus CQADupstackRetrieval and MSMARCO**. BM25s = **0.3984**
  `[MTEB-official]`.

The two are not interchangeable, and I measured the gap rather than guessing it: recomputing the 13-subset from
the same MTEB rows gives Qwen3-Embedding-0.6B **0.5760** vs its 15-task **0.5552**, Qwen3-8B **0.6499** vs
**0.6276**, NV-Embed-v2 **0.6493** vs **0.6321**, bm25s **0.4183** vs **0.3984**. **The 13-task subset runs
~1.7–2.2 points higher than the 15-task mean for every model.** So a paper's "BEIR 51.7" and a leaderboard's
"BEIR-15 0.5552" differ by both the dataset set and the harness; **any comparison to Laya's 0.4799 must first
establish which dataset set, which mean (task-mean vs best-per-dataset), and which run it uses.** Laya says
"BEIR-18", MTEB's board is 15, papers use 13 — three denominators.

### 2.1 Official-protocol numbers (MTEB re-measured, mean nDCG@10)

All `[MTEB-official]`, snapshot 2026-10-29. The BEIR board holds **319 model rows, of which 201 have all 15
tasks**; `n` below is the number of BEIR tasks the model has a score for, and **only n=15 rows are directly
comparable to each other.**

| model | P | BEIR-15 nDCG@10 | n | zs |
|---|---|---|---|---|
| nvidia/NV-Embed-v2 | 7.85B | **0.6321** | 15 | 60 |
| Qwen3-Embedding-8B | 7.567B | **0.6276** | 15 | 73 |
| BAAI/bge-en-icl | 7.11B | 0.6216 | 15 | 60 |
| infly/inf-retriever-v1 | 7.069B | 0.6171 | 15 | 0 (⚠ trained on BEIR) |
| Qwen3-Embedding-4B | 4.022B | 0.6158 | 15 | 73 |
| Linq-Embed-Mistral | 7.111B | 0.6024 | 15 | 73 |
| DataScience-UIBK/uibk-embedding-v1 (2026-10-10) | 0.179B | 0.5979 | 15 | **26** |
| SFR-Embedding-Mistral | 7.111B | 0.5911 | 15 | 66 |
| gte-Qwen2-7B-instruct | 7.069B | 0.5886 | 15 | n/a |
| lightonai/mLateOn (2026-07-30) | 0.307B | 0.5756 | 15 | 26 |
| lightonai/LateOn (2026-04-21) | 0.149B | 0.5722 | 15 | 26 |
| e5-mistral-7b-instruct | 7.111B | 0.5707 | 15 | 73 |
| lightonai/SlideOn | 0.174B | 0.5668 | 15 | 26 |
| lightonai/DenseOn | 0.149B | 0.5620 | 15 | 26 |
| snowflake-arctic-embed-l | 0.335B | 0.5598 | 15 | 80 |
| **Qwen3-Embedding-0.6B** | 0.596B | **0.5552** | 15 | 73 |
| snowflake-arctic-embed-m-v2.0 | 0.305B | 0.5549 | 15 | 73 |
| lightonai/ColBERT-Zero | 0.149B | 0.5539 | 15 | 66 |
| bge-large-en-v1.5 | 0.335B | 0.5434 | 15 | 93 |
| jina-embeddings-v3 | 0.572B | 0.5317 | 15 | 86 |
| jina-embeddings-v4 | 3.935B | 0.5193 | 11 | 86 |
| jina-embeddings-v5-text-nano | 0.212B | 0.5687 | 14 | n/a |
| **mteb/baseline-bm25s** | 0 (lexical) | **0.3984** | 15 | n/a |
| all-MiniLM-L6-v2 | 0.023B | 0.4195 | 15 | 86 |
| BAAI/bge-m3 | 0.568B | 0.4417 | **5** ⚠ not comparable | 80 |
| colbert-ir/colbertv2.0 | 0.109B | 0.1562 | **1** ⚠ not comparable | 93 |
| google/gemini-embedding-001 | n/a | 0.6492 | **4** ⚠ not comparable | 80 |
| IEITYuan/Yuan-embedding-2.0-en | 0.596B | 0.7169 | **4** ⚠ not comparable | 73 |

⚠ = the leaderboard only holds a partial task set for that model, so the mean is over an easier/different
subset. **Do not quote the ⚠ rows as BEIR averages.** BGE-M3 and ColBERTv2 in particular have *never* been
run on full BEIR-15 by MTEB; anyone quoting a BEIR number for them is quoting a paper table.

### 2.2 Paper tables (self-reported, protocol in the caption)

**BEIR-13, nDCG@10, "Avg. w/o CQA" row — Contriever's table, still the cleanest single cross-model grid**
`[paper Table 2, arXiv:2112.09118]`, data 2021–2022 but universally cited:

| method | BEIR-13 avg |
|---|---|
| BM25 | 44.0 |
| BM25 + cross-encoder | 49.5 |
| DPR | 26.3 |
| ANCE | 41.3 |
| TAS-B | 43.7 |
| Gen-Q | 43.1 |
| ColBERT | 45.1 |
| **SPLADE v2** | **50.6** |
| **Contriever** (unsupervised) | **47.5** |
| Contriever + CE | 51.2 |

**SPLADE-v3 technical report** `[paper Table 1/2, arXiv:2403.06789, 2024-03-11]` — BEIR-13 mean nDCG@10:

| model | BEIR-13 | TREC19 | TREC20 | FLOPS (rel.) |
|---|---|---|---|---|
| SPLADE++SD | **50.7** | 73.0 | 71.8 | 1.4 |
| **SPLADE-v3** | **51.7** | 72.3 | 75.4 | 1.2 |
| SPLADE-v3-DistilBERT | 50.0 | 75.2 | 74.4 | 1.4 |
| SPLADE-v3-Lexical (no query expansion) | 49.1 | 71.2 | 73.6 | 0.6 |
| SPLADE-v3-Doc | 47.0 | 71.5 | 70.3 | 1.4 |

Per-dataset, SPLADE++→v3 moves: NQ 53.3→58.6, FiQA 34.9→37.4, TREC-COVID 72.5→74.8, Touché 24.2→29.3 — but
**SciFact stays at 71.0 → 71.0**. That is *exactly* the project's own negative ("SPLADE-v3 weights bought
+0.0004 on SciFact"); the published per-dataset table says the same thing, so this is not an implementation
failure on Laya's side. It is the model.

**Gemini Embedding's own table** `[paper Table, arXiv:2503.07891, 2025-03]` reports MTEB(Eng,v2) *type*-level
Retrieval as **67.71** for Gemini Embedding vs 59.68 (NV-Embed-v2), 60.08 (gte-Qwen2-7B), 59.16
(Cohere-multilingual-v3), 59.27 (text-embedding-3-large). **MTEB's current board disagrees with the ordering**:
gemini-embedding-001 0.6435 vs NV-Embed-v2 0.6284 vs gte-Qwen2-7B 0.5809 `[MTEB-official 2026-10-29]`. The
retrieval task set was revised (hard-negative variants, Touche2020Retrieval.v3), so 2025 self-reported and
2026 leaderboard retrieval numbers are ~3–6 points apart. **Never mix the two columns in one table.**

**ColBERTv2 / PLAID.** I could not extract ColBERTv2's own BEIR average from the two-column PDF cleanly, so I
am not going to quote a remembered figure. What is verified: ColBERT (v1) = 45.1 on BEIR-13 `[Contriever Table
2]`; the 2026 late-interaction revival measures ColBERT-family models on MTEB's BEIR-15 at 0.5132–0.5539
(ColBERT-Zero-unsupervised 0.5132, ColBERT-Zero-supervised 0.5190, ColBERT-Zero 0.5539, GTE-ModernColBERT-v1
0.5475) `[MTEB-official]`. PLAID is an *engine*, not a quality claim, and it matters for the CPU argument: its own
abstract claims **2.5–7× lower latency on GPU and 9–45× on CPU vs vanilla ColBERTv2**, reaching "tens or just a
few hundreds of milliseconds" per query **on CPU at 140M passages** `[paper, arXiv:2205.09707]`. So the reason
Laya's late-interaction experiment lost is **storage (337×) and quality (−0.1548 MAP@10), not CPU latency** —
worth stating precisely, because "ColBERT is too slow on CPU" would be the wrong lesson.

**E5-Mistral / NV-Embed-v2 / gte-Qwen2 / Qwen3 / jina.** For these the 2026 leaderboard numbers in §2.1 are
the better citation: they are on the same task set, same metric, same harness. The papers' own BEIR numbers
for these models mostly do not exist in the "BEIR average" form (E5-Mistral reports MTEB 56-task averages;
NV-Embed-v2 reports MTEB 15-retrieval averages; BGE-M3 reports MIRACL/MKQA/NQ because it targets multilingual).

---

## 3. Reranking in 2026: listwise has won on quality-per-parameter, not on throughput

### 3.1 The single most useful controlled grid

jina's paper evaluates every reranker on **the same top-100 list from one first stage** (jina-embeddings-v3,
0.5B, BEIR-13 nDCG@10 = **55.81**) — `[paper Table 2/3, arXiv:2509.25085v4, 2025-10-06]`:

| reranker | params | BEIR-13 nDCG@10 | Δ over first stage |
|---|---|---|---|
| *first stage only* | 0.5B | 55.81 | — |
| jina-reranker-v2 | 0.3B | 57.06 | **+1.25** |
| **Qwen3-Reranker-0.6B** | 0.6B | 56.28 | **+0.47** |
| **bge-reranker-v2-m3** | 0.6B | 56.51 | **+0.70** |
| jina-colbert-v2 (late interaction) | 0.6B | 54.49 | −1.32 |
| jina-reranker-m0 (2.4B, multimodal) | 2.4B | 58.95 | +3.14 |
| mxbai-rerank-base-v2 | 0.5B | 58.40 | +2.59 |
| jina-reranker-v3 (listwise, LBNL) | 0.6B | **61.85** | **+6.04** |
| Qwen3-Reranker-4B | 4.0B | 61.16 | +5.35 |
| mxbai-rerank-large-v2 | 1.5B | 61.44 | +5.63 |

**Three things fall out of this, and they matter for Laya:**

1. **A 0.6B listwise model matches a 4B cross-encoder** (61.85 vs 61.16). Parameter efficiency of listwise
   ranking is real.

   **Addendum, 2026-10-29, when this was acted on.** Reading the row was not the same as being able to use
   it, so it was checked directly: `jinaai/jina-reranker-v3` loads and runs here — 596M parameters,
   `AutoModel` with `trust_remote_code`, and a `rerank(query=…, documents=[…])` that scores a whole candidate
   list in one pass — which is the interface the pointwise models this programme had measured do not have.
   Its licence is **CC-BY-NC-4.0**, so it can be measured as a research instrument and cannot be shipped
   inside the engine's dependency path; the Apache-2.0 alternative, `bge-reranker-v2-gemma`, is 2.5B with a
   generative interface. **The market fact and the deployable fact are therefore different**, which is the
   distinction this digest exists to keep: a 0.6B listwise model that a commercial deployment cannot use is a
   quality-per-parameter result, not a procurement recommendation. The measurement it enabled is registered in
   [`../findings-reranker-generation.md`](../findings-reranker-generation.md).
2. **Small cross-encoders are nearly worthless on a strong first stage** (+0.5 to +0.7 nDCG@10 for
   bge-reranker-v2-m3 / Qwen3-Reranker-0.6B). This reproduces Laya's finding that cross-encoders "helped weak
   first stages but lost to strong bi-encoders" — with a number on it.
3. **Late interaction as a reranker is a loser** (jina-colbert-v2 −1.32 below the plain first stage).
   Consistent with the project's −0.1548 MAP@10 ColBERT negative.

**The 2026 refresh** `[model card, jinaai/jina-reranker-v3.5, released 2026-07-20]`: BEIR **63.20**,
MIRACL 74.11, RTEB 70.95, Struct-IR 48.3, on a Qwen3-0.6B backbone with 3L2G hybrid attention,
131K-token context, "1.22×–1.56× faster listwise inference on short → long contexts (A100, FlashAttention-2)
than v3". License **CC-BY-NC-4.0** — non-commercial. This is the current open-weights quality leader at 0.6B.

### 3.2 Reasoning rerankers (Rank-R1 / Rank1 / test-time compute)

`[paper, arXiv:2502.18418, Feb 2025]` — BRIGHT average nDCG@10, all models reranking the same BM25+CoT top-100:

| model | BRIGHT avg nDCG@10 |
|---|---|
| BM25 (no rerank) | **0.0789** on MTEB's BRIGHT board `[MTEB-official]` (the BRIGHT paper's own BM25 number is a different implementation and higher; the reranker rows below start from BM25+GPT-4o-CoT top-100, so they are not directly comparable) |
| RankLLaMA-7B | 13.9 |
| Rank1-0.5B | ≈13.9 ("comparable to RankLLaMA 13B with 25× fewer params") |
| Rank1 non-reasoning ablation (7B) | 17.5 |
| **Rank1-7B** | **27.5** |
| Rank1-14B | 28.7 |
| Rank1-32B | 29.4 |
| GPT-4o listwise (upper bound, not comparable) | much higher; the paper explicitly excludes listwise from the comparison as "an order of magnitude more time at inference" |

Read this as: **on reasoning-intensive retrieval the reasoning chain is worth +10 nDCG@10 (17.5 → 27.5) at
7B**, and returns flatten immediately after (28.7, 29.4 at 2× and 4.5× the parameters). Cost: a CoT per
candidate window, 100–1000 generated tokens; `<1000` output tokens was enough per candidate in their MS MARCO
setup. **That is not CPU-feasible** at any useful rate.

Cross-reference: MTEB's BRIGHT board (only 29 models, 2026-10-29 `[MTEB-official]`) tops out at **0.2720**
(Seed1.5-Embedding, closed) and **0.2289** (gte-Qwen2-7B) for *retrieval*, versus **0.0789** for BM25s. So
Rank1-7B's 27.5 as a reranker is genuinely at the frontier on this benchmark — the one place where a
reasoning LLM still buys a lot.

### 3.3 The "is LLM listwise still 2–4× better than a cross-encoder?" question

**No, not in 2026, not at equal size.** On the controlled BEIR-13 grid the listwise 0.6B model is +6.0 over the
first stage and the 4B cross-encoder is +5.35 — a gap of ~0.7 nDCG@10 points, i.e. ~1%. The old 2–4× gap was a
comparison between a 3B listwise LLM and a 110M MiniLM cross-encoder; at matched parameter count, listwise
wins by a point, not by a multiple. Where a large multiplier still exists is **BRIGHT-style reasoning
retrieval** (§3.2), and there it comes from test-time *reasoning*, not from the listwise architecture.

### 3.4 Latency and the numbers I could not verify

* Listwise inference cost scales with list length in one context window; the v3.5 card's only speed claim is
  relative (1.22–1.56× faster than v3) and is vendor-reported `[vendor]`.
* The LightOn family `[model card, 2026-07-08]` states the tradeoff honestly: at rerank-100 on ViDoRe, the
  larger *pointwise* model (PW-2B) is "both faster and slightly stronger" than the listwise LW-0.8B, and
  listwise costs "≈ n/2 windows ... ~18 decoded tokens per window" — i.e. **pointwise cross-encoders remain the
  throughput choice**; listwise is the memory/latency-elastic quality choice.
* **monoT5**: still listed on MTEB as a model (castorini/monot5-{base,large,3b}-msmarco-10k, all 2022) but has
  no BEIR score on any board I could read, and no 2025–26 paper I found benchmarks it head-to-head. Treat it as
  historical. Do not cite a BEIR number for monoT5 without a table.
* **Cohere Rerank / Voyage rerank (2026 versions): I could not verify any current BEIR-style number.** Both are
  closed; their pages report proprietary eval sets. Flagging this as a **gap** rather than guessing.

### 3.5 What reranking is worth on the *official* board

MTEB(eng,v2) also carries a Reranking sub-score (MindSmallReranking and friends) `[MTEB-official]`. Top of it:
Yuan-embedding-2.0-en **0.5327**, QZhou 0.5177, Qwen3-8B 0.5156, ingot-8b-r3 0.5093, Qwen3-4B 0.5076,
Jasper-600M 0.5060, e5-mistral 0.4978. Note these are *embedding models used as rerankers* — a 0.6B dense model
tops a board where 2026's dedicated 0.6B listwise rerankers are not entered. That is a board-coverage artefact,
not a result.

---

## 4. Sparse and lexical: learned sparse still wins, by less than it used to

### 4.1 The official sparse rows (BEIR-15, MTEB-measured)

`[MTEB-official, 2026-10-29]` — 23 models are classified `modelType: sparse`:

| model | P | released | BEIR-15 | license |
|---|---|---|---|---|
| nlpai-lab/KURE-Sparse-base | 0.593B | **2026-10-12** | **0.5782** | apache-2.0 |
| lightonai/SlideOn (32k-dim) | 0.174B | 2026 | 0.5668 | apache-2.0 |
| Linkup-Platform/linkup-sparseup-embed-v1 | 0.188B | 2026-09-16 | 0.5522 | apache-2.0 |
| opensearch neural-sparse-distil (best of 7 sizes) | 0.137B | 2025-06-18 | 0.5270 | apache-2.0 |
| **mteb/baseline-bm25s** | 0 | 2026-05-06 | **0.3984** | mit |
| naver/v-splade-efficient / -quality | 0.33B | **2026-07-01** | not yet scored on BEIR | apache-2.0 |
| naver/splade-code-8B | 8.365B | 2026-02-24 | not scored | cc-by-nc-sa |
| telepix/PIXIE-Splade-v1.5 | 0.149B | 2026-04-22 | not scored | apache-2.0 |

**The headline for a local system: learned sparse buys ~+13 to +18 points of BEIR-15 nDCG@10 over an untuned
BM25s baseline (0.3984 → 0.5270–0.5782).** Whatever else is true, "BM25 is enough" is false on out-of-domain
BEIR. Note also that the *best* sparse model on the board in late 2026 (KURE-Sparse-base, 0.5782) is a
**0.593B** model released two weeks ago — sparse models are no longer cheap BoW artefacts.

### 4.2 Per-dataset, SPLADE++ vs SPLADE-v3

See §2.2 table: BEIR-13 50.7 → 51.7, but SciFact flat at 71.0, Quora *down* 84.9 → 81.4, Touché up 24.2 → 29.3.
The newer weights are a redistribution, not a free win. That is exactly what the project measured (+0.0338 on
SciFact going from BM25 to SPLADE; +0.0004 from SPLADE to SPLADE-v3).

### 4.3 Doc-side vs query-side expansion in 2026

* **Doc2Query++** `[paper, arXiv:2510.09557v2, 2025-10-13]` — topic-coverage-guided document expansion +
  dual-index (sparse + dense) fusion. Per-dataset nDCG@10 with BM25 (k1=0.9, b=0.4) over expanded docs:
  NQ 0.3223 → 0.3415 (BM25 → Doc2Query++), SCIDOCS 0.1430 → 0.1568, **FiQA 0.2466 → 0.2972**,
  ArguAna 0.3437 → 0.3561, SciFact 0.6776 → 0.6945. Plain Doc2Query sits between baseline and ++.
  So doc expansion is worth **+0.01 to +0.05 nDCG@10** — real, corpus-dependent, and *smaller than the
  generation cost* if you are generating at 3 docs/s. This is the best available corroboration of Laya's
  +0.0095 at 3 docs/s: the ceiling for doc2query-style expansion is about +0.05 on the best case (FiQA), and
  +0.01 on the average case.
* **LLM query expansion on top of SPLADE-v3** `[paper, arXiv:2609.37911, 2026]` — a controlled study with a
  frozen document index, fixed 256-dim query budget, 4 expansion formats (term lists, pseudo-document,
  multiple pseudo-references, corpus-steered text): **all 12 method×collection comparisons improve aggregate
  nDCG@10, best relative gains +4.81% (NFCorpus), +8.92% (TREC-COVID), +9.47% (SciDocs)**; 11/12 significant
  after Holm; gain persists in 103/114 interpolation settings; shuffled-text and lexical-bag controls also
  beat baseline, i.e. the *vocabulary* carries most of the benefit. Converting those relative gains to points
  depends on each collection's headroom (NFCorpus and SciDocs sit near 0.16–0.38 nDCG@10, TREC-COVID near 0.7),
  so the absolute delta is roughly **+0.015 to +0.065** — I did not extract their absolute table, so treat that
  range as derived, not quoted. It is nonetheless **2–7× Laya's measured +0.0095 from doc-side expansion**, and
  it is paid at query time (one LLM call), not at index time. This is the single most actionable sparse finding in
  this report for a local system *if* an LLM is already resident for generation.
* **SAE-based learned sparse (SPLARE)** `[paper, arXiv:2603.13277, ICLR 2026]` — replaces the vocabulary
  projection with a pretrained sparse autoencoder over LLM residual streams; 7B and 2B variants; "consistently
  outperforms vocabulary-based LSR in multilingual and out-of-domain settings"; top MMTEB multilingual results
  for sparse. Also states that "as of the time of writing, no sparse retrieval model is listed on the MTEB
  (Multilingual, v2) leaderboard" — which is no longer true as of 2026-10 (23 sparse models are listed on the
  English board), so the space moved even during the paper's review.

### 4.4 BM25F / per-field weighting

Best evidence I found: **mFAR — Multi-Field Adaptive Retrieval** `[paper, arXiv:2410.20056v2, 2025-04-16]`,
which decomposes a document into fields, indexes each with dense and lexical methods independently, and learns
a query-conditioned field-weighting model — reported as beating BM25F and previous multi-field retrievers on
semi-structured corpora. **I did not extract its per-dataset numbers, and I found no 2026 head-to-head
"SPLADE vs tuned BM25F" study.** Flagging that as a gap: the claim "learned sparse beats *tuned BM25F*" is
weaker than "learned sparse beats *untuned* BM25" and I could not find a 2026 paper that settles it.

What can be bounded: on BEIR-15 the spread between untuned BM25s (0.3984) and the best sparse model (0.5782) is
0.18; BM25F tuning typically recovers part of the field/weight gap on structured corpora (titles, headers),
which is the same mechanism mFAR learns. If the project's corpora are semi-structured (docs with titles,
headings, code), **per-field BM25F is the cheapest remaining upgrade on the lexical side and is currently
under-measured in the public literature.**

---

## 5. Index infrastructure on CPU with mmap, 1M–100M vectors

### 5.1 Measured, single-machine, dated

The best dated source I found is **BitBudget** `[3rd-party bench, data pinned 2026-06-06,
<https://sjmoran.github.io/bitbudget/>, raw rows in site/data.json]`. Two of its boards bear directly on this:

**Organisation axis at scale — 90M vectors (SIFT1B slice, 128-d, exact ground truth, single thread):**

| index | bytes/vec | recall@10 | QPS (1 thread) |
|---|---|---|---|
| Exact flat scan | 512 | 1.000 | **6** |
| **HNSW** | 704 | **0.966** | **17,448** |
| **IVF-PQ** | 64 | 0.829 | 434 |
| ITQ + rerank | 16 | 0.430 | 120 |

**Same harness at 100k × 128 (identical data, all four indexes):**

| index | bytes/vec | recall@10 | QPS |
|---|---|---|---|
| flat | 512 | 1.000 | 27,059 |
| **HNSW** | 704 | 1.000 | 51,895 |
| **IVF-PQ** | 64 | 0.712 | 33,745 |
| bit-trie (8-byte codes) | 8 | 0.977 | 16,320 |

Reading: **HNSW costs +37% bytes over raw float32 (704 vs 512 at 128-d) and buys three orders of magnitude of
QPS at 90M.** IVF-PQ buys 8× the footprint reduction but loses **14 recall points** at 90M (0.966 → 0.829) —
that is the number the project's IVF-PQ ANN should be judged against. A flat scan at 90M is 6 QPS: unusable.

**On-disk graph search is now a solved-ish problem.** `[paper, OSDI'25, "Achieving Low-Latency Graph-Based
Vector Search via Aligning Best-First Search Algorithm with SSD", Guo & Lu, Tsinghua]`:
on-disk DiskANN costs **4.18× (at 0.9 recall) / 3.14× (at 0.99 recall)** the latency of in-memory Vamana;
their PipeANN gets to **1.14×–2.02× of in-memory Vamana** and **35.0% of DiskANN's latency** at billion scale,
with **1.71× DiskANN's throughput**, by allowing compute and I/O to overlap instead of enforcing strict
best-first ordering. The reason naive on-disk search is slow is measured in the same paper: an SSD read is
~**7.43×** the compute latency per step, and DiskANN wastes most of that I/O.

**mmap-specific advice that follows from the above:** for 1M–10M vectors on a single box, an mmap'd HNSW
(node-array + int8 vectors) is the right default — the graph is the thing that must live in RAM; the vectors
can be paged. For 10M–100M, the choice is between (a) mmap'd HNSW on NVMe where you accept page-fault latency,
(b) DiskANN-style Vamana (SSD-resident, RAM = PQ codes + graph), or (c) IVF-PQ when RAM is the binding
constraint and −14 recall@10 is acceptable. **PipeANN's result says (b) is much closer to (a) than DiskANN's
published behaviour suggested**, and that the fix is in the search loop, not the storage.

### 5.2 What I could not verify

* **ScaNN, FAISS-IVF, HNSWlib head-to-head in 2026**: I fetched ScaNN's 2019 paper and HNSW's 2016 paper, both of
  which are the originals, not 2026 measurements. The Milvus ANNS trade-off study
  (`doi:10.1142/s0218194026500130`) is behind a Cloudflare interstitial — **403, not read**. Anyone quoting it
  should read it directly.
* **"Not All Indexes Are Equal"** was referenced by a secondary blog as arXiv:2401.09350, but that ID resolves to
  *Foundations of Vector Retrieval* (Bruch) — the blog's citation is wrong and I could not locate the paper.
  **Do not cite it on my authority.**

---

## 6. Quantization and Matryoshka: what compression actually costs

### 6.1 At RAG scale, on a real corpus

`[3rd-party bench, BitBudget, 2026-06-06]` — MS MARCO **8.84M passages**, MiniLM-L6 (384-d), nDCG@10 against
graded judgements:

| representation | bytes/vec | nDCG@10 | % of float | lossless? |
|---|---|---|---|---|
| float32 | 1536 | 0.3653 | 100 | — |
| **int8** | **384** | **0.3661** | **100** | **yes** |
| **binary (Hamming)** | **48** | **0.3412** | **93** | no |
| **binary + float rerank** | **48** | **0.3652** | **100** | **yes** |
| RaBitQ (1-bit, unbiased estimator) | 48 | 0.3408 | 93 | no |

(int4 was not run at this scale by the benchmark; see §6.2 for its measured value on a 3072-d embedder.)

**int8 is free. Binary is free *if and only if* you rerank the shortlist at full precision — and then it is
lossless at 32×.** RaBitQ, despite the theoretical error bound, lands where plain binary lands at this scale
(93%).

### 6.2 MRL / truncation vs quantization

`[3rd-party bench, BitBudget 2026-06-06]`, text-embedding-3-large (3072-d, Matryoshka-trained):

| method | bytes/vec | nDCG@10 | % of float |
|---|---|---|---|
| float32 | 12288 | 0.5430 | 100 |
| float16 | 6144 | 0.5430 | 100 |
| int8 | 3072 | 0.5430 | 100 |
| **Matryoshka truncation (4×)** | 3072 | 0.5270 | **96.9** |
| int4 | 1536 | 0.5410 | 99.6 |
| **binary** | 384 | 0.5360 | **98.8** |
| **binary + rerank** | 384 | 0.5430 | **100** |

The benchmark's own headline is the right summary: **"bits beat dimensions."** At the *same* 3072 bytes/vector,
int8 keeps **100%** while Matryoshka truncation to 768-d keeps **96.9%** — i.e. 4× fewer dimensions loses 3.1%
while 4× fewer bits loses 0%. And binary at **384** bytes/vector — 8× smaller than the truncation row — still
keeps **98.8%**, and 100% once the shortlist is reranked in float. Truncation is the *worst* of the three
compression families at equal bytes.

### 6.3 Independent corroboration and a warning

`[blog measurement, 2026-08-30, <https://cstsolution.com/blog/quantising-embeddings/>]` — 100k × 384 random
L2-normalized vectors, single-core, Apple M3:

* int8: **3.96× smaller**, keeps **0.984** of the float32 top-10, **1.000 after re-scoring the top 100** in float32.
* **int8 was not faster than float32** (45.1 ms vs 46.5 ms) despite reading 4× fewer bytes.
* **float16 was 36% slower than float32.**
* **binary: 32× smaller and 13.7× faster, but recall@10 = 0.062**, and only **0.931 after re-ranking 10,000
  candidates** (a tenth of the corpus).

The blog's own caveat is the important one and it is correct: random vectors are the *hardest* case for a
quantizer (no cluster structure), so recall on real embeddings is better, not worse. But the 0.062 binary
figure on random data next to BitBudget's 93% on MS MARCO shows how much the answer depends on whether the
vector distribution is "well-behaved". **Measure on your own corpus; do not port a number.**

**The operational takeaway for a CPU system: quantization buys bytes, not time.** int8/float16 do not speed up
a CPU dot-product scan; binary is the only one that does (13.7× in the blog's scan, mostly by cutting memory
traffic), and it needs a rerank stage to be usable.

### 6.4 Model-side MRL claims

* jina-embeddings-v4 `[paper, arXiv:2506.18902]`: single-vector 2048-d **truncatable to 128 with "minimal"
  loss**; multi-vector mode is 128-d per token. Their own wording ("can be truncated to as little as 128 with
  minimal [loss]") is not backed by a per-dimension table in the sections I read — treat as a claim, and note
  the independent te-3-large measurement above puts a 4× truncation at −3.1%.
* Yuan-embedding-2.0-en `[model card]` uses MRL in training and 1024-d output.
* bekko (below) ships 384-d truncatable to 256/128/64.

---

## 7. Which retrieval benchmarks matter now, and which a text-only local system can enter

`[MTEB-official, benchmark list, 2026-10-29]` — the board now has 78 benchmarks. The retrieval-relevant ones:

| benchmark | tasks | what it tests | current leader (MTEB, 2026-10-29) | BM25s row | enterable by a text-only local system? |
|---|---|---|---|---|---|
| **BEIR** | 15 | classic zero-shot heterogeneous retrieval | voyage-3-m-exp 0.6812 (closed) / NV-Embed-v2 0.6321 (open) | 0.3984 | **yes** — the reference |
| **RTEB(eng, beta)** | 21 | *new in 2026*: finance, health, code, legal, SWE-bench code | **bekko-embedding-v1-a25m 0.5216** | 0.3411 | **yes — and this is the one to target** |
| **BRIGHT** | 1 | reasoning-intensive retrieval | Seed1.5-Embedding 0.2720 | 0.0789 | yes but you will lose |
| **FollowIR** | 3 | instruction-conditioned retrieval (relevance flips) | FollowIR-7B 0.1215 | **−0.029** | yes; small field (204 models) |
| **LongEmbed** | 6 | long-document retrieval | uibk-embedding-v1 0.9153 | **0.9087** | **yes — BM25 is within 0.7% of SOTA** |
| **CoIR / MTEB(Code,v1)** | 10 / 12 | code retrieval | papr-embed-code-1-4b 0.8013 (zs=20) / Qwen3-8B 0.7986 (zs=100) | 0.4231 | yes (text code) |
| **MTEB(Law,v1)** | 8 | legal retrieval | octen-law-8b 0.7677; **litil-embed-0.6b 0.7330** | 0.5557 | yes |
| **MTEB(Medical,v1)** | 12 | biomedical | gte-Qwen2-7B 0.6655; F2LLM-v2-4B 0.6448 | none | yes |
| **ViDoRe(v1&v2)/v3** | 14 / 10 | visual document retrieval | Argus-Colqwen3.5-9b 0.8598 | — | **no** (needs images) |
| **MIEB / MAEB / MVEB** | 51–130 | image/audio/video embedding | — | — | **no** |

### 7.1 What this table says

1. **RTEB is the successor benchmark to watch.** It is explicitly described by MTEB's own baseline repo as
   "the v2 English benchmark" (`mteb.get_benchmark("RTEB(beta)")` in the bm25s model card). Its task list is
   domain-heavy — `EnglishFinance1–4Retrieval, HC3Finance, FinQARetrieval, WikiSQLRetrieval, CUREv1,
   ChatDoctorRetrieval, AILACasedocs, AILAStatutes, LegalSummarization, AppsRetrieval, HumanEvalRetrieval,
   DS1000Retrieval, MBPPRetrieval, Code1Retrieval, SWEbenchCodeRetrieval, FreshStackRetrieval` — which is much
   closer to what a local RAG engine is actually pointed at than BEIR's academic corpora. **It is also where a
   tiny model currently leads** (§8).
2. **LongEmbed is a "free win" zone for a lexical system**: bm25s at 0.9087 vs the best model at 0.9153, and a
   0.149B late-interaction model (GTE-ModernColBERT-v1, 0.9058) sits *below* BM25. Long-context retrieval is
   still dominated by exact matching.
3. **FollowIR is the benchmark where fusion actively hurts.** BM25 scores −0.029 (i.e. worse than not
   ranking); the best model is 0.1215. A negative BM25 row is a strong independent signal for the project's
   Touché fusion negative: when relevance depends on an instruction, lexical evidence can be *anti*-correlated.
4. **ViDoRe/MIEB/MAEB are out of reach** without a vision tower; do not put them on a roadmap for a text-only
   local system.

---

## 8. "Win small": 2026 results where a small local model beats a big one

Five dated cases, in decreasing order of how much I trust them.

### 8.1 bekko-embedding-v1-a25m — 25M active params, #1 on RTEB, zero-shot-clean

`[MTEB-official 2026-10-29]` + `[model card + author's own blog]` (the model author reports the CPU numbers on the model's own hardware; treat the throughput figures as author-reported, the leaderboard row as MTEB-measured)

| model | active params | RTEB(eng,beta) mean | zs |
|---|---|---|---|
| **hotchpotch/bekko-embedding-v1-a25m** | **24.9M** | **0.5216** | **100** |
| hotchpotch/bekko-embedding-v1-a8m | 7.7M | 0.4852 | 100 |
| codefuse-ai/F2LLM-v2-80M | 80M | 0.4840 | 100 |
| intfloat/e5-base-v2 | 109M | 0.4669 | 100 |
| nomic-embed-text-v2-moe | 475M | 0.4474 | n/a |
| **mteb/baseline-bm25s** | 0 | **0.3411** | n/a |

The card's measured CPU throughput (its own machine, OpenVINO): **134 docs/s on a Ryzen 9 7950X; 10.5 docs/s on
a Raspberry Pi 5**; the 8M sibling does 364 docs/s and 33 docs/s respectively `[model card]`. MIT license,
384-d truncatable to 64, 8k context, 100+ languages.

A 25M-parameter MIT model is **+0.055 above e5-base-v2 and +0.18 above BM25 on a 21-task, fully zero-shot,
domain-heavy retrieval benchmark.** This is the cleanest "small wins" result in the entire dataset: `zs=100`
means MTEB records zero task overlap, so it is not a contamination artefact.

### 8.2 Yuan-embedding-2.0-en — 0.6B beats every 7–14B open model and Gemini

Covered in §1.2: 0.7069 vs Qwen3-8B 0.6944, ingot-8B 0.7000, gemini-embedding-001 0.6435, on the same board,
same 10 tasks, same zero-shot fraction (95). Apache-2.0. It is a fine-tune of Qwen3-Embedding-0.6B, so the
"win" is mostly **data + MRL + multi-task training, not scale** — which is the replicable part.
Caveat: leaderboard submission, not independently re-run by me.

### 8.3 uibk-embedding-v1 — 179M at 128 dimensions, ahead of 7B models on BEIR-15

`[MTEB-official]` 0.5979 (15 tasks) `[model card, 2026-10-10]`: a ModernBERT-base encoder with **both a dense
head and a ColBERT head writing into the same 128-d vector list**, so a single MaxSim index gets both signals
("the dense vector written into the token list so that MaxSim adds the two scores"). Beats e5-mistral-7B
(0.5707), gte-Qwen2-7B (0.5886), and LateOn (0.5722) at 1/40th the parameters.

**Read the caveat carefully**, because the authors state it themselves: `zs=26` on the BEIR board, and the card
says "on the BEIR datasets whose training splits are in the fine-tuning data (MS MARCO, NQ, HotpotQA, FEVER,
FiQA). Training queries that equal a BEIR test or development query were removed from every training set."
So: five of fifteen BEIR datasets had their train splits used, with dedup claimed. **Its 0.5979 is a
partially-in-domain number and should not be placed in the same column as Qwen3's 0.5552 or NV-Embed's 0.6321.**
The architectural idea (one index, two scores) is the transferable insight, not the number.

Worth noting separately: on **MTEB(eng,v2) retrieval** — a different board — the same model scores **0.6248 at
`zs=87`**: rank 14 of 221, within 0.4 points of NV-Embed-v2 (0.6284) at 179M parameters and 128 dimensions. That
number is much less exposed to the contamination caveat, and it is the strongest small-beats-large signal on the
Eng-v2 board after Yuan.

### 8.4 LongEmbed: a zero-parameter baseline within 1% of SOTA

bm25s 0.9087 vs uibk 0.9153 on LongEmbed `[MTEB-official]`. Not a "model win" but the strongest possible
statement that **on long-document retrieval, scale buys almost nothing.**

### 8.5 A fifth case, on code retrieval — and what small models do *not* win

Code retrieval, where I expected scale to matter most, is *also* small-friendly `[MTEB-official, CoIR board,
10 tasks]`: F2LLM-v2-0.6B **0.7766**, papr-embed-code-1-0.6b 0.7719, **embeddinggemma-2 (271M, `zs=100`)
0.7633**, F2LLM-v2-330M 0.7613, Qwen3-Embedding-0.6B 0.7496 (`zs=100`), SPLADE-code-0.6B 0.7479 — against a
board top of 0.8013 (papr-embed-code-1-4b) and 0.7986 (Qwen3-8B, `zs=100`). A 271M zero-shot model is within
3.8 points of the best code retriever in existence.

What small models do **not** win: BRIGHT (best dense retriever 0.2720, BM25 0.0789 — every model is far below
the reasoning-reranked numbers), the absolute level of RTEB (best is 0.52 — nobody has solved it), and FollowIR
(best 0.1215 overall — nobody wins there). The pattern is not "scale always wins": it is that **reasoning-heavy
and instruction-conditioned retrieval are the only two axes where a 25M–600M model is clearly outclassed.**

---

## 9. Confidence and gaps

### High confidence (read the primary table myself, dated, protocol stated)

* MTEB(eng,v2) retrieval ranking and the full BEIR-15 board — official API, 2026-10-29, cross-checked against an
  independent aggregator dated 2026-10-27.
* The reranker grid in §3.1 — one paper, one harness, one first stage, all baselines re-run.
* The index and compression numbers in §5.1 and §6.1/6.2 — one reproducible benchmark, one pinned date
  (2026-06-06), raw data on disk.
* SPLADE-v3's BEIR-13 table and its SciFact-flat result — primary PDF.
* Contriever's BEIR-13 grid (BM25 44.0 … SPLADE v2 50.6 … Contriever 47.5) — primary PDF.
* Rank1's BRIGHT numbers — primary PDF.
* Licenses and parameter counts — Hugging Face model API, queried per model.

### Medium confidence

* CPU feasibility annotations. They are my judgement plus two data points: bekko's own 134 docs/s and a
  blog-measured **~1.1 s/item for Qwen3-Embedding-0.6B on an i7-13700H with torch-CPU, 1.5–1.7 GB RAM**
  `[blog measurement, 2026, <https://lilting.ch/en/articles/qwen3-embedding-qdrant-cpu-benchmark>]`. That blog is
  one machine, default sentence-transformers settings, no ONNX/INT8; a well-optimized ONNX-int8 path could be
  2–5× faster, and I did not find a 2026 measurement of that for the 0.6B model. **Order of magnitude: ~1 s per
  query-time embed on CPU, ~0.1–0.5 s optimized; bulk indexing at 10²–10³ CPU-hours per million documents.**
* SPLARE's contribution, mFAR's contribution — abstract-level only; I did not extract their tables.
* RTEB's stability as a benchmark — it is `beta`, 21 tasks, and only 11 models have complete scores.

### Gaps — things I looked for and did not find

1. **Cohere Rerank and Voyage rerank 2026: no verifiable BEIR-style numbers.** Closed models, proprietary evals.
2. **No 2026 head-to-head "learned sparse vs tuned BM25F"** study. mFAR is the closest, and it is 2025 and
   multi-field-specific.
3. **ColBERTv2's own BEIR average** — the two-column PDF would not extract cleanly and I refused to quote from
   memory. The 2026 ColBERT-family numbers (0.51–0.55 on BEIR-15) are solid.
4. **BGE-M3 has never been run on full BEIR-15** by MTEB (5/15 tasks), and its own paper reports multilingual
   benchmarks instead. Any BEIR number for BGE-M3 in circulation is either partial or a paper table.
5. **monoT5: no 2025–26 benchmark.** Effectively superseded; treat any BEIR figure for it as 2022 folklore.
6. **The Milvus ANNS trade-off study** (`doi:10.1142/s0218194026500130`) returned HTTP 403 (Cloudflare); not read.
7. **"Not All Indexes Are Equal"** appears to be mis-cited by the blog that referenced it; the arXiv ID resolves
   to a different work. Not cited here.
8. **No measured CPU-throughput table at 2026** for: IVF-PQ build time, HNSW build time, mmap page-fault
   behaviour at 10M+ on NVMe. The QPS/recall numbers exist; the *build* and *cold-cache* numbers do not.
9. **FollowIR / BRIGHT / RTEB have very thin boards** (204 / 29 / 11 models), so "state of the art" there is a
   statement about who bothered to submit.

---

## 10. What a CPU-only local system can still win

Written against the project's measured state: BEIR-18 macro nDCG@10 0.4799, BM25 + optional dense + IVF-PQ,
322M decision model, mmap index, dense channel defaulted off.

**1. The cheapest available win is swapping the embedding model, not the architecture.**
MTEB's BEIR-15 rows put untuned BM25s at 0.3984 and Qwen3-Embedding-0.6B at 0.5552 `[MTEB-official]` — a 0.075
gap on the face of it. **Discount it, but not to zero:** Laya's statistic is a best-per-dataset macro over 18
datasets, which is a *more generous* statistic than MTEB's mean over 15 (adding datasets can only raise a
max-based macro, and the extra three are chosen by the project). Assume that flatters it by 0.02–0.04, and the
conservative claim is still **+0.03 to +0.05 nDCG@10 available from swapping in a 596M apache-2.0 encoder**, more
from Yuan (0.7069 on the eng-v2 retrieval board). The
project has already measured that *its own* mean-pooled 322M encoder is useless for dense retrieval (nDCG@10
0.014 on SciFact) — the conclusion should not be "dense is dead", it should be "a contrastively-trained 0.6B
encoder is a different object from a mean-pooled decision model". The prior is strong: **every** 200M–600M
open embedder on the board (arctic-l 0.5904, jina-v5-nano 0.5880, Jasper-600M 0.6619, Yuan 0.7069) crushes 0.4799.

**2. Query-time CPU embedding is affordable; bulk CPU embedding is not.**
~1.1 s per embedded item on CPU for a 0.6B encoder `[blog measurement]`, versus 134 docs/s for a 25M model
`[model card]`. So: **index with the small model (or with a GPU one-off), query with the same small model, and
never plan to re-embed a corpus on CPU at 0.6B scale.** The 25M bekko tier is the only one where CPU indexing
is a non-issue (134 docs/s on the author's Ryzen 9 with OpenVINO — core count not stated — i.e. roughly
**2 hours per million documents**, and 364 docs/s for the 8M sibling, ~45 min/million).

**3. A small listwise reranker is the one place a 0.6B model clearly beats a big one — but it must be measured
against a *strong* first stage, not BM25.** +6.04 nDCG@10 (55.81 → 61.85) over a 0.5B bi-encoder on BEIR-13
`[paper]`. Expect far less if the first stage is BM25-only, because a reranker's ceiling is the *recall* of the
candidate set. jina measures 61.85 on top-100 from a 0.5B bi-encoder; LightOn measures 47.30–47.98 for its
0.8B rerankers on top-100 from **BM25** — a 14-point separation between two runs of comparable reranker class,
which is partly the candidate list and partly different BEIR subsets (LightOn excludes NQ/MS MARCO as
contaminated). It is not a controlled experiment, but the direction is unambiguous. **So the project should
always report reranker deltas at two candidate depths (e.g. top-20 and top-100) and two first stages (BM25 and
the strongest available dense), or it will keep concluding "rerankers don't help" when what it measured was
"BM25's top-100 does not contain the answers".**

**4. Lexical is not the weak channel on the benchmarks that look like local RAG.**
LongEmbed: BM25 0.9087 vs SOTA 0.9153. RTEB: BM25 0.3411 vs SOTA 0.5216 (here it *is* weak). BEIR: 0.3984 vs
0.5782. So the honest position is "lexical is competitive on long-document/mostly-verbatim retrieval, and loses
badly on domain-shifted, semantically-paraphrased retrieval" — which is a *routing* problem, not a modelling one.

**5. Fusion should be a per-query decision, and there is now a 2026 paper that says exactly that.**
"The Fusion Tax" `[paper, arXiv:2610.30612, IEEE Access, published 2026-10-15]` measures, over 7 BEIR corpora,
5 dense encoders (22M–596M), BM25, SPLADE and a cross-encoder second stage: even the *best* fusion method
(dense-weighted score interpolation) ranks a relevant document below the better of its two components on
**13–48% of queries** and below the deployable dense-only retriever on **8–21%** (14–48% / 11–25% under
leave-one-corpus-out weight selection), while still winning in aggregate; the downside is significant under
loss-averse evaluation on 6 of 7 corpora. RRF is worse than weighted interpolation wherever the dense retriever
is clearly stronger. **This is the published, dated, multi-corpus version of the project's own −0.091 Touché
fusion loss.** A 322M decision model that gates fusion per query is the right shape of response — and the paper
also warns that fusion-failure predictors "change sign with the sparse–dense balance" across corpora, so a gate
trained on one corpus will not transfer. Calibrate per corpus, or abstain.

**6. Compression and index: the affordable configuration.**
int8 for vectors (lossless, no speed gain), binary + float rerank when bytes must go 32× (lossless at RAG scale
on 8.84M MS MARCO `[3rd-party bench 2026-06-06]`), HNSW for the graph at +37% bytes over raw, IVF-PQ only if RAM
is binding and −14 recall@10 at 90M is acceptable. Skip Matryoshka-without-quantization: at equal bytes,
1-bit codes retained 98.8–100% vs 96.9% for a 4× truncation.

**7. The entry points a text-only local system can actually defend.**
BEIR (baseline), **RTEB(eng,beta)** (the new domain-heavy board where a 25M MIT model leads — a realistic target
for a small local engine), LongEmbed (lexical near-parity), FollowIR (where selective/abstaining behaviour is
the win), MTEB(Law/Medical) (a 0.596B model is #2 on law). Not BRIGHT, not CoIR's top, not ViDoRe/MIEB/MAEB.

**8. Where small genuinely cannot win, and should not try:** reasoning-intensive retrieval (BRIGHT: 0.27 ceiling
for retrieval, and only reasoning rerankers at 27.5 vs 13.9 get near it), code retrieval at the top of CoIR
(0.80, 4–14B models), and any benchmark whose top row is an 8B+ LLM backbone with domain pretraining.

**9. One methodological recommendation that dominates all of the above:** before chasing any of these numbers,
publish the project's own BEIR row on **the same 15 datasets, same metric implementation, same tie-breaking**
that MTEB uses, next to `mteb/baseline-bm25s` (0.3984) and Qwen3-Embedding-0.6B (0.5552). Right now the project
compares a "BEIR-18 best-per-dataset macro" against SPLADE's BEIR-13 number; those are different quantities,
and the gap between them (13 vs 15 vs 18 datasets, mean-of-means vs best-per-dataset) is the same size as most
of the effects being argued about.

---

## Appendix — reproduction trail

Everything in this report came from files saved in this directory, so any number can be re-checked without
re-fetching:

| path | contents |
|---|---|
| `.scratch/research/api/*.json` | raw MTEB leaderboard API responses: `mteb_eng_v2_scores.json` (349 rows), `scores_BEIR_.json` (319), `scores_BRIGHT_.json`, `scores_FollowIR_.json`, `scores_LongEmbed_.json`, `scores_CoIR_.json`, `scores_RTEB_eng__beta__.json`, `scores_MTEB_Law__v1__.json`, `scores_MTEB_Medical__v1__.json`, `scores_ViDoRe_v1_v2__.json`, `scores_MTEB_Code__v1__.json`, `benchmarks.json`, `models_all.json` (970 models with params/licence/type) |
| `.scratch/research/api/bitbudget_data.json` | the pinned BitBudget results behind §5.1, §6.1, §6.2 (`"updated": "2026-06-06"`) |
| `.scratch/research/src/rs-pdf-*.txt` | pdftotext of the primary PDFs (SPLADE-v3, Contriever, ColBERTv2, PLAID, E5-Mistral, NV-Embed-v2, mGTE/gte-Qwen2, BGE-M3, jina-v4, Qwen3-Embedding, Gemini Embedding, jina-reranker-v3, Doc2Query++) |
| `.scratch/research/src/rs-*.txt` | HTML-to-text of the 2026 papers and pages (Fusion Tax, SPLARE, Wacky Weights, QE+SPLADE-v3, mFAR, Rank1, Rank-R1, OSDI'25 PipeANN, HNSW, ScaNN, DiskANN, BitBudget page, int8 blog, Qwen3-CPU blog) |
| `.scratch/research/cards/*.md` | raw Hugging Face model cards (Yuan, LateOn, jina-v5-text-nano, jina-reranker-v3.5, LightOn-rerank, Nemotron-3.5-Rerank, KURE-Sparse, bekko, harrier, bm25s baseline) |
| `.scratch/research/analyze_mteb.py` | the script that ranks/joins the API dumps (coverage-aware, so partial-task rows are not silently compared to full-task rows) |

The MTEB API is the only *live* source here; everything else is a saved snapshot, so a reader in a month should
re-run `/v1/benchmarks/.../scores` rather than trust the snapshot, and should treat the paper and model-card
numbers as fixed at the dates printed beside them.
