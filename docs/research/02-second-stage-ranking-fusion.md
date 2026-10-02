# Second-Stage Ranking, Fusion & Routing for laya-rag — Technical Brief

**Root cause for ArguAna.** An MS-MARCO pointwise cross-encoder scores *topical* query–passage compatibility. It has no supervision for the *relation between two texts*. ArguAna's gold document is the **counter**-argument, so an agreeing argument on the same topic is topically near-identical and scores high, while the opposing one shares fewer terms. The counter-argument retrieval literature has shown similarity alone is insufficient and that the contrast relation must be modeled explicitly ([Wachsmuth et al. 2018](https://aclanthology.org/P18-1023.pdf); [Shi et al. 2023](https://arxiv.org/abs/2304.08807)). Classes that fix relational relevance: **(i)** comparative/listwise scoring with both texts in one context, **(ii)** NLI/stance heads supervised on contradiction, **(iii)** pairwise/duel objectives. Late interaction (ColBERT MaxSim) is still topical *token* matching — it does not encode opposition, so expect it to help multi-hop, not ArguAna. Cohere/Voyage rerankers are API-only and out of scope for this host.

Ranked by expected value per unit of implementation cost.

## 1. Rank-1 guardrail + calibrated CE blending
**What:** Temperature-scale (or isotonic-fit) the cross-encoder logits on held-out queries, then forbid the second stage from demoting first-stage rank-1 unless its calibrated margin over rank-2 exceeds a threshold.
**Gap:** Attacks the CE damage (ArguAna 0.568→0.250) by bounding top-heavy flips; much nDCG@10 loss lives in rank-1 swaps. `blend_scores` already min-max mixes, so this is a margin rule on top.
**Maturity/evidence:** Fusion/calibration is established ([Bruch et al. 2023](https://arxiv.org/abs/2210.11934)); guardrails are engineering practice, not a single citable result — treat as a hypothesis, not a paper.
**Cost:** Trivial (numpy; no new model). Reuses existing `RERANK_BLEND`/`resolved_rerank_order`.
**Expected:** Recovers 30–70% of the CE-induced loss on the four hurt corpora while keeping most MultiHop gain.
**Falsify / skepticism:** Compare guardrail-on vs guardrail-off at fixed blend on all five BEIR sets; if the gain is just "behaves like blend<1", it is redundant. A hard guardrail also blocks correct fixes when rank-1 is genuinely wrong.

## 2. Normalized convex-combination / learned fusion beyond RRF
**What:** Normalize each channel (min-max or z-score per query), then convex-combine or fit a 3–5 feature logistic fusion over {BM25, dense, CE, SPLADE}. Keep RRF as fallback.
**Gap:** RRF discards score magnitude and its `k=60` is arbitrary; with a strong lexical and weak dense channel it already loses. Learned fusion can encode "trust CE only when lexical and dense disagree" — the oracle routing the config notes are recoverable.
**Maturity/evidence:** [Bruch et al. 2023](https://arxiv.org/abs/2210.11934) (ACM TOIS, DOI 10.1145/3596512) finds normalized convex combination **outperforms** RRF in- and out-of-domain, is agnostic to score normalization, and is sample-efficient (one parameter); RRF is parameter-sensitive.
**Cost:** Cheap numpy; no new model. The existing `reciprocal_rank_fusion(weights=...)` needs a score-carrying sibling.
**Expected:** +0.005–0.02 nDCG@10 single; more on the 3-model ensemble.
**Falsify / skepticism:** On the five corpora, does a tuned convex combo beat weight-tuned RRF out-of-fold? If not, RRF's rank-only robustness wins. Per-corpus tuning overfits; normalize only with query-local statistics.

## 3. Query-type / relational gate (cascade)
**What:** A tiny logistic gate over cheap features (query length, query/doc length ratio, lexical overlap, first-stage score margin) that decides per query whether to run the pointwise CE at all, and its blend weight.
**Gap:** Exactly the measured pattern — CE helps MultiHop but hurts five topical corpora; a global on/off wastes the good half.
**Maturity/evidence:** Routing is active ([RouterRetriever 2024](https://arxiv.org/html/2409.02685v1); [RAGRouter 2025](https://arxiv.org/html/2505.23052v2)). Be skeptical: [per-query routing often fails](https://www.researchsquare.com/article/rs-9132181/v1.pdf).
**Cost:** Negligible; reuses `blend_scores`.
**Expected:** Recovers roughly half the CE-induced loss at no MultiHop cost.
**Falsify / skepticism:** Leave-one-dataset-out: if the gate cannot beat a global "CE off", drop it. Risk of learning BEIR quirks rather than the relation.

## 4. Relational stance reranker (NLI contradiction head)
**What:** Score `(query_argument, candidate)` with an NLI cross-encoder and use contradiction probability, or contradiction−entailment margin, as the ranking signal for argumentative queries.
**Gap:** Directly targets ArguAna: NLI is supervised on the exact relation (one text contradicts another) that the pointwise MS-MARCO CE lacks, and is less fooled by topical agreement.
**Maturity/evidence:** NLI is mature; stance detection is benchmarked ([EZ-STANCE, ACL 2024](https://aclanthology.org/2024.acl-long.838.pdf)). Counter-argument work above supports the mechanism.
**Cost:** DeBERTa-v3-base-MNLI ≈184M; ~0.3–1.5 s per 100 pairs on 16 CPU threads. No API.
**Expected:** ArguAna nDCG@10 back toward or above the 0.568 first stage; near-neutral elsewhere.
**Falsify / skepticism:** If contradiction-only ordering on ArguAna dev does not beat BM25 or the fused stage, the hypothesis is wrong. NLI is trained on *sentence* pairs; ArguAna docs are 300–500 tokens, so truncation/windowing may destroy the signal.

## 5. jina-reranker-v3 — 0.6B listwise, "last but not late"
**What:** A 0.6B listwise reranker that puts query + N candidates in one context, applies causal attention across them, then extracts per-document last-token scores. This is listwise/relational without API access.
**Gap:** Attacks both gaps at once — set-aware comparative scoring (ArguAna) and smaller than 7B LLM rerankers (CPU/MPS feasible).
**Maturity/evidence:** [arXiv 2509.25085, 2025](https://arxiv.org/abs/2509.25085); [HF](https://huggingface.co/jinaai/jina-reranker-v3). Self-reported **61.85–61.94 BEIR nDCG@10 over 13 datasets** (paper body 61.85; abstract/card 61.94 — internally inconsistent), so not comparable to this project's 5-set 0.452; verify locally. Its **ArguAna nDCG@10 is 73.43**, i.e. listwise is strong exactly where the pointwise CE collapsed (0.568 → 0.250). **License: CC BY-NC 4.0 (non-commercial)** — a real constraint for a self-hosted production host.
**Cost:** ~1.2 GB fp16 (estimate — the card advertises GGUF/MLX but carries no CPU/MPS benchmark); fine on MPS, minutes/query on CPU for k=20. Candidate count limited by context length.
**Expected:** The single most likely model to beat fused RRF on ArguAna while preserving multi-hop.
**Falsify / skepticism:** Reproduce on the five BEIR sets before believing it; listwise models are often distilled to imitate GPT-4 orderings and may not transfer to a CPU-local setup.

## 6. Setwise / pairwise local LLM reranking (PRP)
**What:** Prompt a small local LLM (via the existing Ollama path) for binary A-vs-B preferences and aggregate by tournament/all-pairs, or use setwise argmax over triples.
**Gap:** Comparative judgments are easier than absolute relevance for a small model, so this is the cheap route to relational scoring without a full permutation.
**Maturity/evidence:** [Setwise, SIGIR 2024](https://arxiv.org/abs/2310.09497); [PRP, NAACL Findings 2024](https://aclanthology.org/2024.findings-naacl.97/); [PRP-Graph, ACL 2024](https://aclanthology.org/2024.acl-long.313/); listwise lineage [RankGPT 2023](https://aclanthology.org/2023.emnlp-main.923/), [RankZephyr 2023](https://arxiv.org/abs/2312.02724), [RankRAG, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/db93ccb6cf392f352570dd5af0a223d3-Paper-Conference.pdf).
**Cost:** O(k²) short calls; k=10 → 45 pairs. A 1–3B Ollama model on CPU is seconds to tens of seconds/query — offline eval only.
**Expected:** Modest ArguAna gain, neutral multi-hop.
**Falsify / skepticism:** Position bias and O(k²) latency; PRP sometimes trails pointwise on MS-MARCO-style data. Run same local model pointwise vs pairwise on ArguAna.

## 7. Late-interaction MaxSim channel — answerai-colbert-small-v1 (33M)
**What:** Add a MaxSim reranker/channel over the fused top-100 using a 33M ColBERT model.
**Gap:** Multi-hop: it scores **HotpotQA 76.11** vs bge-base **72.6 on HotpotQA** (bge-base BEIR AVG is 53.25) while being **3.3× smaller** — the same regime where CE helped. **It is not an ArguAna fix** (ArguAna 50.09, weak by design).
**Maturity/evidence:** [answer.ai, 2024](https://www.answer.ai/posts/2024-08-13-small-but-mighty-colbert.html); [model](https://huggingface.co/answerdotai/answerai-colbert-small-v1); [ColBERTv2](https://aclanthology.org/2022.naacl-main.272/); [Jina-ColBERT-v2](https://arxiv.org/html/2408.16672).
**Cost:** 33M, milliseconds/doc on CPU once doc token embeddings exist; on-the-fly MaxSim over 100 candidates is workable, persistent multi-vector index is not (storage).
**Expected:** MultiHop MAP@10 +0.01–0.03; ArguAna ≈0.
**Falsify / skepticism:** If on-the-fly MaxSim over top-100 does not beat the pointwise CE on MultiHop, skip. Multi-vector storage is real; do not index it.

## 8. SPLADE-v3 as a chunk-level sparse channel
**What:** Add SPLADE-v3 chunk expansion scores as a third RRF/learned-fusion channel.
**Gap:** Weak dense channel: learned sparse adds semantic term generalization while staying in the inverted index and on CPU. Score the *chunk's own* expansion vector (the previously rejected variant was passage-level max-pool, which conflates chunk vectors).
**Maturity/evidence:** [SPLADE-v3, 2024](https://arxiv.org/abs/2403.06789); [model](https://huggingface.co/naver/splade-v3).
**Cost:** ~110M params (BERT-base/CoCondenser; a separate DistilBERT variant exists at BEIR-13 50.0); query-time expansion ~50–150 ms CPU; document expansion is an index-time pass and adds memory. Exact gains over v2: BEIR-13 50.7 → 51.7, MS MARCO MRR@10 37.6 → 40.2, and **FLOPS fell 1.4 → 1.2** — cheaper, not costlier. Its own paper reports that a DeBERTaV3 pointwise re-ranker fails to beat SPLADE-v3 on ArguAna because the "counter-argument task might be more intricate for a re-ranker" — published external support for the relational-relevance diagnosis.
**Expected:** Small average gain, strongest on SciFact/NFCorpus; potentially larger on ArguAna.
**Falsify / skepticism:** v3's gain over v2 is modest; if the channel does not beat or complement BM25 out-of-fold on the five sets, drop it. Note BEIR set counts differ across papers (answer.ai 15, SPLADE-v3/jina 13, RouterRetriever 14), so never compare "BEIR averages" without checking the set list.

## Cross-cutting evaluation caveats
BEIR per-dataset scores are small-sample and tuned by many groups; ArguAna's counter-argument relation is unusual enough that one corpus should not drive architecture. Report all five plus MultiHop, use query-level bootstrap CIs, and hold out the tuning corpus for any router/guardrail. Reranker and late-interaction models are mostly MS-MARCO-distilled, so their "general" BEIR averages hide exactly the relational weakness seen here ([answer.ai caveats](https://www.answer.ai/posts/2024-08-13-small-but-mighty-colbert.html)).
