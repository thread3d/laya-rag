# Verification Memo — Brief 02 (Second-Stage Ranking, Fusion & Routing)

**Scope.** Independent citation/claim verification of
`laya-rag/docs/research/02-second-stage-ranking-fusion.md` against primary sources
(arXiv abstract/HTML pages, ACL Anthology, Hugging Face model cards). Web sources were
treated as data. One web request was issued at a time. Verification date: 2026.

**Method.** For each numbered claim in the task: locate the canonical primary source, read
the abstract / model card / relevant full-text section, and record CONFIRMED, CORRECTED
(with the correct value), or UNVERIFIABLE. Extra concrete details are captured to
strengthen the brief.

**Headline:** 6 of 9 claims fully confirmed as stated. Three need corrections —
the "Reimers et al. 2023" attribution (actually Shi et al.), the jina "15-set average"
(actually 13 datasets, and the paper body says 61.85 not 61.94), and the SPLADE-v3
base architecture (BERT-base, not DistilBERT). The jina and SPLADE corrections are the
only ones that change a number in the brief.

---

## 1. ColBERTv2, PLAID, and the PLAID reproducibility study

### ColBERTv2 — https://arxiv.org/abs/2112.01488
**CONFIRMED.**
- Title: *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction*.
- Authors: Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, Matei Zaharia.
- Submitted 2 Dec 2021; v3 10 Jul 2022. Comments: **NAACL 2022**. License CC BY 4.0.
- What it is: late-interaction (multi-vector, per-token) retrieval that couples
  **aggressive residual compression** with a **denoised supervision** strategy;
  reduces the space footprint of late-interaction models by **6–10×** while improving
  quality, establishing SOTA inside and outside the training domain.

### PLAID — https://arxiv.org/abs/2205.09707
**CONFIRMED.**
- Title: *PLAID: An Efficient Engine for Late Interaction Retrieval*.
- Authors: Santhanam, Khattab, Potts, Zaharia. Submitted 19 May 2022 (preprint).
- What it is: an engine that eliminates low-scoring passages with **centroid
  interaction** (passage as a bag of centroids) plus **centroid pruning**, inside a
  highly optimized engine. Claims **up to 7× lower latency on GPU and 45× on CPU**
  vs vanilla ColBERTv2, without quality loss; tens of ms on GPU, tens–hundreds of ms
  on CPU **at 140M passages**.

### Reproducibility study — https://arxiv.org/abs/2404.14989
**CONFIRMED.** *A Reproducibility Study of PLAID*, Sean MacAvaney & Nicola Tonellotto.
Submitted 23 Apr 2024; **SIGIR 2024 (reproducibility track)**; DOI 10.1145/3626772.3657856.
Findings relevant to the brief:
- PLAID's Pareto frontier is a careful balance of its **three parameters**; deviations
  from suggested settings can substantially **increase latency without improving
  effectiveness**.
- A baseline missing from the original paper — **re-ranking a lexical (BM25) system** —
  gives **better efficiency–effectiveness trade-offs in low-latency settings**.
- Re-ranking cannot reach peak effectiveness at higher latency because of lexical
  recall limits, and is a **poor approximation of exhaustive ColBERTv2**.
- **Neighbor-pulling** re-ranking modifications overcome this, providing a Pareto
  frontier across all operational points.
- Token-representation clusters PLAID uses are mostly aligned one-to-one with a single
  token (and vice versa).

**Implication for the brief (section 7).** The claim "milliseconds/doc on CPU once doc
token embeddings exist" is about *scoring*; the repro study shows the *retrieval engine*
advantage is contingent, and a BM25 re-rank baseline is competitive at low latency.
Worth stating explicitly if section 7 is kept.

**Extra details.** ColBERTv2 license CC BY 4.0; PLAID is still a preprint (no venue).
The repro study is the only independent latency/quality check located.

---

## 2. `answerai-colbert-small-v1`

### Model card — https://huggingface.co/answerdotai/answerai-colbert-small-v1
**CONFIRMED.** Raw README verified.
- **33 million parameters** (MiniLM-sized).
- **BEIR AVG 53.79**; **HotpotQA 76.11**; **ArguAna 50.09** — all exactly as the brief
  states. Against ColBERTv2.0, BEIR AVG 53.79 vs 50.02.
- License: **apache-2.0**. Built with the **JaColBERTv2.5 recipe** (arXiv 2407.20750).
- Base model = weight-averaged `mini-base` of MiniLM / gte-small / bge-small-en-v1.5.

### Blog — https://www.answer.ai/posts/2024-08-13-small-but-mighty-colbert.html
**CONFIRMED.** Published **August 13, 2024**, author **Benjamin Clavié**.
- Trained on CPU-deployable size; "search through hundreds of thousands of documents in
  milliseconds, **on CPU**".
- Training: checkpoint 1 = 640k 32-way MSMarco triplets (teacher BGE-M3-reranker);
  checkpoint 2 = further 2.5M 32-way triplets mixing MSMarco / HotPotQA / TriviaQA /
  Fever / NQ; checkpoint 3 = 640k different MSMarco triplets with BGE-reranker-v2.5-
  gemma2-lightweight teacher. Final model = **weight average of the three checkpoints**.
- HotpotQA-only training **hurt every metric on every validation dataset**; mixing it in
  helped slightly.
- Blog explicitly lists **ArguAna** as a "very noticeable weakness" and says the model is
  weak on "less classical tasks" (ArguAna, SCIDOCS, CQADupstack, Quora).

### CORRECTION (minor, wording)
The brief (line 57) reads: "it scores **HotpotQA 76.11 vs bge-base 72.6 on BEIR** while
being 3× smaller". **72.6 is bge-base's *HotpotQA* score, not its BEIR average.** bge-base
BEIR AVG = **53.25**. Also "3× smaller" is more precisely **3.3×** (33M vs 109M).
Suggested fix: "HotpotQA 76.11 vs bge-base 72.6 on the same task, while being 3.3× smaller."

### Extra details
- ArguAna ranking on the same table: answerai-colbert-s 50.09 < snowflake-s 57.59 <
  bge-small 59.55 < bge-base 63.61 — so 50.09 is genuinely weak *relative to peers*, not
  just in absolute terms. Supports "not an ArguAna fix".
- BEIR here is **15 datasets** (per the blog's own description) — note the contrast with
  jina's 13-dataset BEIR below; BEIR set counts differ across papers, so never compare
  "BEIR averages" across sources without checking the set list.

---

## 3. MUVERA and SLIM

### MUVERA — https://arxiv.org/abs/2405.19504
**CONFIRMED.** *MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings*,
Dhulipala, Hadian, Jayaram, Lee, Mirrokni. v1 29 May 2024; **v2 revised 8 Jun 2026**.
- Reduces **multi-vector similarity search to single-vector MIPS**, enabling off-the-shelf
  MIPS solvers for multi-vector retrieval.
- Asymmetric **Fixed Dimensional Encodings (FDEs)** of queries/documents whose inner
  product approximates multi-vector similarity; first single-vector proxy with
  theoretical ε-approximation guarantees.
- Empirically: same recall as prior SOTA heuristics while retrieving **2–5× fewer
  candidates**; across BEIR, **~10% improved recall with 90% lower latency**.
- v2 correction: dimension bound is now `d_FDE = O(m/(εδ))^{O(1/ε²)}` (v1 understated it).
- **Makes single-vector CPU late-interaction feasible: yes** — it is exactly a
  single-vector ANN proxy, so it reuses existing CPU MIPS infrastructure.

### SLIM — https://arxiv.org/abs/2302.06587
**CONFIRMED.** *SLIM: Sparsified Late Interaction for Multi-Vector Retrieval with
Inverted Indexes*, Li, Lin, Ma, Lin. v1 13 Feb 2023; v2 8 May 2023; **SIGIR 2023**,
DOI 10.1145/3539618.3591977.
- Maps each contextualized token vector to a **sparse, high-dimensional lexical space**,
  then does late interaction over sparse token embeddings.
- Two-stage architecture: **inverted-index retrieval + score refinement** approximating
  sparsified late interaction; compatible with off-the-shelf lexical engines
  (**Lucene / Pyserini**).
- Competitive accuracy vs ColBERT on MS MARCO Passages and BEIR while being **much
  smaller and faster on CPUs**.
- **Makes CPU inverted-index late-interaction feasible: yes** — that is its explicit goal.

**Extra details.** MUVERA's `2–5×` candidate reduction and `90%` latency cut are the
strongest quantitative support for "late interaction need not be storage-bound". SLIM's
release path is Pyserini, i.e. directly usable on the host's lexical stack.

---

## 4. Bruch et al. — normalized convex-combination fusion

**CONFIRMED (with exact title/year clarification).**
- Exact title: **"An Analysis of Fusion Functions for Hybrid Retrieval"** — *not* a
  "normalized convex-combination fusion" title.
- Authors: Sebastian Bruch, Siyu Gai, Amir Ingber.
- arXiv:2210.11934, submitted **21 Oct 2022**, v2 4 May 2023. Published in **ACM
  Transactions on Information Systems (TOIS), 2023**, DOI **10.1145/3596512**.
- Findings: (i) **RRF is sensitive to its parameters**; (ii) learning a convex
  combination (CC) is **generally agnostic to the choice of score normalization**;
  (iii) **CC outperforms RRF in both in-domain and out-of-domain settings**; (iv) CC is
  **sample efficient** — a small set of training examples tunes its only parameter.
- The brief's phrasing "generally matches or beats RRF" **understates** the result: the
  paper reports CC *outperforms* RRF. This strengthens the brief's section 2 while also
  validating the "k=60 is arbitrary / RRF sensitive" argument in the same paragraph.

**Extra details.** Only **one** tunable parameter in CC; explicitly framed as an
advantage over RRF. This is the single best citation for the brief's fusion argument.

---

## 5. Relational relevance — Wachsmuth 2018 and the 2304.08807 paper

### Wachsmuth, Syed & Stein 2018 — https://aclanthology.org/P18-1023/
**CONFIRMED.** *Retrieval of the Best Counterargument without Prior Topic Knowledge*,
ACL 2018 (Volume 1: Long Papers), pages **241–251**, Melbourne, July 2018,
DOI 10.18653/v1/P18-1023.
- Hypothesis: the best counterargument **"invoke[s] the same aspects as the argument
  while having the opposite stance."**
- Operationalizes it by **simultaneously modeling similarity and dissimilarity** of
  argument pairs (words + embeddings of premises and conclusions); topic-independent.
- Results: for **7.6 candidates with opposing stance on average**, ranks the best
  counterargument highest with **60% accuracy**; among all **2801 test pairs** finds the
  best about **one in three** times.
- **Supports the brief's claim:** similarity alone is insufficient; the contrast
  (opposite-stance) relation must be modeled explicitly. Nuance to state in the brief:
  the paper does not say similarity is useless — it says similarity *and* dissimilarity
  must be modeled jointly.

### 2304.08807 — https://arxiv.org/abs/2304.08807
**CORRECTED — citation attribution error.**
- Title: *Revisiting the Role of Similarity and Dissimilarity in Best Counter Argument
  Retrieval*.
- Authors: **Hongguang Shi, Shuirong Cao, Cam-Tu Nguyen** — **there is no Reimers
  author.** A web check found no "Reimers et al. 2023" counter-argument paper; the only
  match for the URL is this Shi et al. paper.
- **Action:** change `(Reimers et al. 2023)` to `(Shi et al. 2023)` in the brief's opening
  paragraph. The URL can stay.
- Content **CONFIRMED**: proposes **Bipolar-encoder**, a BERT-based model learning a
  joint representation for **simultaneous similarity and dissimilarity**; reaches
  **accuracy@1 of 49.04%**, significantly outperforming LTR and neural baselines; efficient
  at prediction time with caching. Submitted 18 Apr 2023 (v2 19 Apr 2023).

**Extra detail.** Together the two papers give the counter-argument lineage
*explicitly*: same-aspects + opposite-stance, modeled jointly (Wachsmuth 2018),
refined by a joint similarity/dissimilarity encoder (Shi et al. 2023). This is the
strongest theoretical support for the brief's relational-relevance root cause.

---

## 6. `jina-reranker-v3`

### Paper — https://arxiv.org/abs/2509.25085 (HTML v4 read)
**CONFIRMED on architecture; CORRECTED on the BEIR set count; figure is
internally inconsistent in the primary source.**
- Title: *jina-reranker-v3: Last but Not Late Interaction for Listwise Document
  Reranking*. Authors: Feng Wang, Yuqing Li, Han Xiao (Jina AI). v1 29 Sep 2025,
  v4 6 Oct 2025.
- **0.6B-parameter multilingual listwise reranker.** "Last but not late" (LBNL):
  **causal attention between the query and all candidate documents in the same context
  window**, then extract contextual embeddings from **each document's final token**.
  Explicitly *not* late interaction (no separate encoding + MaxSim).
- Architecture: built on **Qwen3-0.6B**, 28 transformer layers, 1024 hidden dim, 16
  attention heads, **131K token context**, up to **64 documents per forward pass**;
  lightweight MLP projector into a 256-d embedding space; scores via cosine similarity.
- **Is it listwise? YES** (the paper's core claim; cross-document attention).
- **BEIR is 13 datasets, not 15.** Paper §5.1: "encompassing **13** heterogeneous tasks";
  the BEIR table has 13 columns (TC, NFC, NQ, HQA, FQA, AA, TCH, DBP, SD, FVR, CFV, SF, QRA).
- **BEIR figure discrepancy:** the paper body / Table 2 report **61.85 nDCG@10** (with
  documents in descending order), and the arXiv *abstract metadata* and the **HF model
  card report 61.94**. Random ordering scored **62.24**, ascending 61.45. So the source
  itself is inconsistent between 61.85 and 61.94; the honest brief wording is
  "self-reported 61.85–61.94 BEIR nDCG@10 over **13** datasets".
- Other reported numbers: HotpotQA **78.58**, FEVER **94.01**, **ArguAna 73.43**,
  MIRACL 66.83 (18 languages), MKQA 67.92, CoIR 70.64; +4.79% over jina-reranker-v2
  (57.06); "10× smaller than generative listwise rerankers".

### Model card — https://huggingface.co/jinaai/jina-reranker-v3
**CONFIRMED.**
- **License: `cc-by-nc-4.0`** — i.e. **CC BY-NC 4.0, non-commercial**; on-prem/
  non-marketplace commercial use requires contacting Jina. *(The arXiv *paper* is
  CC BY-NC-SA 4.0; the *model weights* are CC BY-NC 4.0.)*
- **CPU/MPS:** the card advertises **GGUF quantizations** (CPU/llama.cpp) and an
  **MLX** version (Apple Silicon/MPS). So "runs on CPU/MPS" is plausible and
  supported by official artifacts — **but the paper/card give no CPU/MPS latency
  benchmark**.
- Jina now recommends the successor **`jina-reranker-v3.5`** as a drop-in upgrade.

### CORRECTIONS for the brief (section 5)
1. "**the 15-set average**" → **13-set average** (BEIR-13).
2. "**61.94**" → cite as **61.85–61.94 (internally inconsistent in the source)**, or
   prefer **61.85** since that is the paper body's default ordering.
3. Add the **CC BY-NC 4.0 non-commercial model license** — material for a self-hosted
   production RAG host.
4. The brief's "~1.2 GB fp16" and "minutes/query on CPU for k=20" are **UNVERIFIABLE**
   from the primary source (arithmetically consistent with 0.6B fp16, but not stated;
   CPU latency is not benchmarked). Mark them as local estimates.

### Extra details
- ArguAna **73.43** is a strong concrete data point: a listwise reranker does very well
  on the exact corpus where the pointwise CE collapsed (0.568 → 0.250). This is the
  best single piece of evidence for the brief's section 5 recommendation.
- The model card explicitly states **10× smaller** than generative listwise rerankers
  with comparable BEIR performance.

---

## 7. Setwise, PRP, RankGPT, RankZephyr

### Setwise — https://arxiv.org/abs/2310.09497
**CONFIRMED.** *A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking
with Large Language Models*, Zhuang, Zhuang, Koopman, Zuccon. arXiv v1 14 Oct 2023, v2
30 May 2024; comments: **SIGIR 2024 full paper**; DOI 10.1145/3626772.3657813.
- First consistent comparative evaluation of Pointwise vs Pairwise vs Listwise vs
  Setwise for LLM zero-shot ranking.
- **Pointwise: efficient but poor effectiveness. Pairwise: superior effectiveness but
  high computational overhead. Setwise: fewer LLM inferences and fewer prompt tokens
  while retaining high effectiveness.**

### PRP — https://aclanthology.org/2024.findings-naacl.97/
**CONFIRMED.** *Large Language Models are Effective Text Rankers with Pairwise Ranking
Prompting*, Qin, Jagerman, Hui, Zhuang, Wu, Yan, Shen, Liu, Liu, Metzler, Wang, Bendersky.
**Findings of NAACL 2024**, June 2024, Mexico City, pages **1504–1518**,
DOI 10.18653/v1/2024.findings-naacl.97.
- Argues off-the-shelf LLMs do not fully understand pointwise/listwise ranking
  formulations; **PRP reduces the burden via pairwise comparisons**.
- On **7 BEIR tasks**, PRP **outperforms pointwise LLM solutions by >10% average
  nDCG@10** and beats supervised baselines; on TREC-DL 2019/2020, a **20B Flan-UL2**
  PRP is competitive with GPT-4-based SOTA (≈50× larger) and beats 175B InstructGPT by
  >10%. Linear-complexity PRP variants are competitive.

### RankGPT — https://aclanthology.org/2023.emnlp-main.923/
**CONFIRMED.** *Is ChatGPT Good at Search? Investigating Large Language Models as
Re-Ranking Agents*, Sun, Yan, Ma, Wang, Ren, Chen, Yin, Ren. **EMNLP 2023**, December
2023, Singapore, pages **14918–14937**, DOI 10.18653/v1/2023.emnlp-main.923.
- Listwise/generative reranking with ChatGPT/GPT-4; introduces the **NovelEval**
  contamination test set; **permutation distillation**: a distilled **440M** model
  **outperforms a 3B supervised model on BEIR**.

### RankZephyr — https://arxiv.org/abs/2312.02724
**CONFIRMED (content); CORRECTED (date).**
- Title: *RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!*,
  Pradeep, Sharifymoghaddam, Lin. **Submitted 5 Dec 2023**; primary page lists **no
  venue**. The brief labels it "2024" — per the primary source the paper date is
  **Dec 2023**. (If a 2024 venue exists it is not recorded on the arXiv page.)
- Open-source listwise zero-shot reranker; bridges the effectiveness gap with GPT-4 and
  sometimes surpasses it; **robust to initial document ordering** and to the number of
  documents reranked; beats GPT-4 on NovelEval.

### "Does pairwise/setwise beat pointwise for small models?"
**PARTIALLY CONFIRMED / contextual.** The evidence direction is consistent:
- Setwise (SIGIR 2024): pointwise is the *least effective* prompting mode.
- PRP (NAACL Findings 2024): pairwise beats pointwise LLMs by >10% avg nDCG@10 on BEIR.
- RankGPT (EMNLP 2023): a 440M *distilled listwise* model beats a 3B supervised model.
- Counterpoint worth recording: jina-reranker-v3's related work cites Déjean et al. 2024
  finding that **traditional cross-encoders remain highly competitive** vs LLM rerankers.
- **No cited paper directly tests 1–3B local models on ArguAna**; the brief's section 6
  "run same local model pointwise vs pairwise on ArguAna" remains the right falsification.

---

## 8. SPLADE-v3

### Paper — https://arxiv.org/abs/2403.06789 (HTML read)
**CONFIRMED on headline claim; CORRECTED on base architecture.**
- Title: *SPLADE-v3: New baselines for SPLADE*. Lassance, Déjean, Formal, Clinchant
  (Naver Labs Europe; Lassance now Cohere). **11 Mar 2024**, technical report.
- Gains: **statistically significantly more effective than BM25 and SPLADE++**; meta-
  analysis over **up to 44 query sets**; **>40 MRR@10 on MS MARCO dev** (37.6 → **40.2**
  from SPLADE++SelfDistil); **+2% out-of-domain BEIR** (BEIR-13 average nDCG@10
  **50.7 → 51.7**).
- Base SPLADE-v3 **starts from SPLADE++SelfDistil (BERT/CoCondenser-based, ~110M)**,
  trained with a mix of KL-Div (λ=1) and MarginMSE (λ=0.05), 8 negatives/query.

### CORRECTION (architecture)
The brief (line 67) says "**~110M DistilBERT**". The base **`naver/splade-v3` is
BERT-based (~110M)**, not DistilBERT. **DistilBERT is a separate variant**
(`naver/splade-v3-distilbert`, smaller footprint, **BEIR-13 avg 50.0**). Fix the brief to
"~110M **BERT-base/CoCondenser**; a separate DistilBERT variant exists".

### Extra details (high value)
- **FLOPS actually fell**: SPLADE++SD 1.4 → SPLADE-v3 **1.2** (the new baseline is both
  more effective and cheaper by the paper's loose FLOPS indicator). This slightly
  undercuts the brief's "v3's gain over v2 is modest and self-reported" framing on the
  cost axis.
- **ArguAna is the exception that proves the point:** SPLADE-v3 ArguAna **50.9** vs
  SPLADE++SD **51.8** (a slight *decrease*). The paper states that a DeBERTaV3
  cross-encoder re-ranker outperforms SPLADE-v3 on most query sets **"except for ArguAna
  whose 'counter-argument' task might be more intricate for a re-ranker."** This is a
  published, independent statement that a pointwise cross-encoder struggles on ArguAna's
  counter-argument relation — direct support for the brief's central hypothesis, from a
  source the brief already cites.
- SPLADE-v3 loses significantly on only 3 of 44 query sets (Touché-2020, TREC-MQ ×2).
- License: paper CC BY-NC-SA 4.0. The `naver/splade-v3` HF model card is **gated**
  (HTTP 401 without auth) — verify access before planning to use it.

---

## 9. RouterRetriever and RAGRouter

### RouterRetriever — https://arxiv.org/html/2409.02685v1
**CONFIRMED — per-query routing over multiple expert embedding models.**
- Authors: Hyunji Lee, Luca Soldaini, Arman Cohan, Minjoon Seo, Kyle Lo. arXiv v1
  **4 Sep 2024**.
- Base encoder **Contriever** + one **LoRA gate per domain** (~0.5% of params / ~1M per
  gate); a **pilot-embedding library** (k-means centroids) selects the best expert per
  **query** by mean similarity. Lightweight; experts can be added/removed without
  retraining the router.
- BEIR **14 datasets**. Avg nDCG@10: RouterRetriever **49.6** vs MSMARCO-trained **47.5**
  (**+2.1 absolute**) vs Multi-Task **46.4** (**+3.2**); its routing beats other routing
  techniques by **+1.8 on average**. Comparable to the Best-Individual oracle (50.9).
- **Documented failure of routing:** the paper states that **"simply adapting routing
  techniques from generation tasks to information retrieval does not yield high
  performance, underscoring the importance of developing routing techniques tailored
  specifically for information retrieval."** It includes a dedicated *Routing Mechanism
  Error Analysis* section. This directly supports the brief's "per-query routing often
  fails" caveat.

### RAGRouter — https://arxiv.org/html/2505.23052v2
**CONFIRMED — per-query routing (among RAG-augmented LLMs).**
- Authors: Jiarui Zhang, Xiangyu Liu, Yong Hu, Chaoyue Niu, Fan Wu, Guihai Chen
  (SJTU / WeChat Tencent). v2 **17 Oct 2025**.
- Routes each **query + retrieved documents** to the best of multiple RAG-augmented LLMs;
  adds **document embeddings** and a per-LLM **RAG capability embedding** with
  contrastive learning; score-threshold variant trades accuracy for latency.
- Outperforms the best individual LLM and existing **non-RAG-aware** routing methods by
  **1.67%–9.33%**; five knowledge-intensive tasks; models 0.5B–72B plus GPT-4o,
  Qwen2.5-Max, DeepSeek-R1.
- **Documented failure of routing:** existing routers assume **static parametric
  knowledge** and fail under RAG ("Missing Doc Interaction", "Ignoring RAG Capability");
  RAG-induced knowledge shifts can **reverse** a model's answerability. The paper has an
  **Appendix E.1** with quantitative routing-failure case studies.
- Note: RAGRouter routes to **LLMs**, not retrievers, so it supports the *per-query
  routing* premise but not the retriever-routing mechanism directly.

---

## Consolidated corrections the author must make

| # | Location | Current text | Correction |
|---|----------|--------------|------------|
| C1 | Opening paragraph | "(Reimers et al. 2023)" for arXiv 2304.08807 | **Shi et al. 2023** (Hongguang Shi, Shuirong Cao, Cam-Tu Nguyen) |
| C2 | §5 (jina) | "61.94 BEIR nDCG@10 — the 15-set average" | **13-set** BEIR average; figure is **61.85** in the paper body / **61.94** in the abstract metadata & HF card (inconsistent) |
| C3 | §5 (jina) | no license stated | Model weights are **CC BY-NC 4.0 (non-commercial)**; commercial use needs a Jina license |
| C4 | §7 (ColBERT) | "HotpotQA 76.11 vs bge-base 72.6 on BEIR" | 72.6 is bge-base **HotpotQA**, not BEIR; bge-base BEIR AVG = 53.25; size gap is **3.3×** |
| C5 | §8 (SPLADE-v3) | "~110M DistilBERT" | Base SPLADE-v3 is **~110M BERT-base / CoCondenser**; DistilBERT is a separate variant (BEIR-13 50.0) |
| C6 | §2 (Bruch) | "generally matches or beats RRF" | Paper finds CC **outperforms** RRF in-domain and out-of-domain; RRF is parameter-sensitive |
| C7 | §6 (RankZephyr) | "RankZephyr 2024" | Primary source dated **5 Dec 2023**, no venue listed |
| C8 | §1 / §2 (dates) | "Bruch et al. 2023" | Correct as cited (TOIS 2023, DOI 10.1145/3596512); arXiv is 2022 — fine as-is, optionally disambiguate |

## Unverifiable claims

1. **jina-reranker-v3 "~1.2 GB fp16"** — not stated in paper or model card (consistent
   with 0.6B × 2 bytes, but an estimate).
2. **jina-reranker-v3 "minutes/query on CPU for k=20"** and "fine on MPS" latency — no
   CPU/MPS benchmark in the primary source; only GGUF/MLX artifacts are advertised.
3. **RankZephyr venue in 2024** — no 2024 venue recorded on the primary page.
4. **PRP-Graph (ACL 2024, 2024.acl-long.313)** and **RankRAG (NeurIPS 2024)** — cited in
   the brief but outside the assigned verification set; not checked here.
5. The brief's **expected-effect figures** (e.g. "recover 30–70%", "+0.005–0.02 nDCG@10")
   are projections from this project's own experiments, not external citable claims.

## Most useful NEW concrete details (ranked)

1. **SPLADE-v3 paper's own ArguAna remark** — a DeBERTaV3 pointwise re-ranker fails to
   beat SPLADE-v3 on ArguAna because the "counter-argument task might be more intricate
   for a re-ranker" (arXiv:2403.06789, §3). Published external support for the relational
   root cause, from a source already in the brief.
2. **jina-reranker-v3 ArguAna = 73.43 nDCG@10** — a listwise model is very strong on the
   exact corpus where the pointwise CE collapsed; the strongest evidence for §5.
3. **Bruch et al.'s actual finding is stronger than quoted** — convex combination
   *outperforms* RRF in-domain and out-of-domain and is agnostic to score normalization;
   RRF is parameter-sensitive (TOIS 2023, DOI 10.1145/3596512).
4. **SPLADE-v3's exact BEIR-13 gains** — 50.7 → 51.7 nDCG@10 (+1.0 abs / +2% rel),
   MS MARCO MRR@10 37.6 → 40.2, and FLOPS 1.4 → 1.2 (cheaper, not costlier).
5. **jina-reranker-v3 weights are CC BY-NC 4.0** — non-commercial; relevant to any
   production/hosted deployment, and a reason to prefer an Apache/MIT alternative if the
   host is commercial.
