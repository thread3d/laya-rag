# Citation verification — `04-generation-faithfulness.md`

Verification method: each entry checked against the primary source (arXiv abstract/HTML, ACL
Anthology, PMLR, Nature/PMC, official proceedings) one request at a time. Status legend:
**CONFIRMED**, **CORRECTED** (value/title/venue was wrong; correction given), **UNVERIFIABLE**
(no primary source found, or the source does not support the specific claim).
Date of check: 2026 (session date); arXiv versions as noted.

---

## 1. MiniCheck (item 1 & item 6)

- **CONFIRMED** — [arXiv:2404.10774](https://arxiv.org/abs/2404.10774), "MiniCheck: Efficient
  Fact-Checking of LLMs on Grounding Documents" (Tang, Laban, Durrett), **EMNLP 2024**.
  - All three claims hold: synthetic GPT-4-distilled training data; **MiniCheck-FT5 (770M
    parameters)** reaches **GPT-4 accuracy** on the new **LLM-AggreFact** benchmark at **400× lower
    cost**; the released model is a Flan-T5 (FT5) variant.
  - Extra: LLM-AggreFact unifies recent fact-checking/grounding datasets; training data synthesis
    teaches sentence-synthesis checking; MiniCheck-L (770M) is the model referenced in brief item 6.
  - Note: "matches GPT-4" means *accuracy parity*, not superiority; the paper says it "reaches GPT-4
    accuracy" and "outperforms all systems of comparable size".

## 2. RAGChecker

- **CONFIRMED** — [arXiv:2408.08067](https://arxiv.org/abs/2408.08067), Ru et al., **NeurIPS 2024
  Datasets & Benchmarks Track**
  ([proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract-Datasets_and_Benchmarks_Track.html)).
  The arXiv page still says "Under Review", so the venue is confirmed from the NeurIPS proceedings,
  not arXiv.
  - Claim-level entailment is central: RAGChecker's metric suite includes claim-level
    precision/recall, context utilization, noise sensitivity, hallucination, self-knowledge and
    faithfulness, checked by an entailment model; meta-evaluation shows better human correlation than
    prior RAG metrics. Extra: 8 RAG systems evaluated; code at `amazon-science/RAGChecker`.

## 3. ALCE + AttributionBench (the "numeric/entity mismatch dominates" claim)

- **CONFIRMED** — [ALCE, arXiv:2305.14627](https://arxiv.org/abs/2305.14627), Gao, Yen, Yu, Chen,
  **EMNLP 2023**. Automatic metrics along three dimensions — **fluency, correctness, citation
  quality** (citation precision/recall / recall of citation-worthy statements); on ELI5 the best
  models lack complete citation support 50% of the time. ALCE is the source for the citation
  precision/recall usage in brief item 5.
- **CONFIRMED (venue)** — [AttributionBench, arXiv:2402.15089](https://arxiv.org/abs/2402.15089), Li,
  Yue, Liao, Sun, **Findings of ACL 2024**
  ([2024.findings-acl.886](https://aclanthology.org/2024.findings-acl.886/)). Binary "Attributable /
  Not Attributable"; 7 source datasets (ExpertQA, Stanford-GenSearch, AttributedQA, LFQA, BEGIN,
  HAGRID, AttrEval-GenSearch); fine-tuned GPT-3.5 ≈ 80% macro-F1; 300+ error cases manually labeled.
- **PARTIALLY UNVERIFIABLE / OVERSTATED** — "numeric/entity mismatch is a large share of attribution
  failures in ALCE and AttributionBench". AttributionBench's error analysis supports the *spirit*
  (>66% of errors = insensitivity to fine-grained information, explicitly "facts, events, **numbers**";
  ~26.8% = human/model accessible-information mismatch), but the dominant named category is
  fine-grained-information insensitivity, not "numeric/entity mismatch" as a metric. ALCE does not
  decompose attribution failures into numeric/entity mismatch. Recommend rewording to: "AttributionBench
  attributes >66% of automatic-evaluator errors to insensitivity to fine-grained information
  (facts/events/numbers) and ~27% to human-vs-model evidence-access mismatch."
  - Extra: AttributionBench train/ID/OOD split totals 13,322 / 1,610 / 1,686.

## 4. RAGBench / TRACe

- **CONFIRMED** — [arXiv:2407.11005](https://arxiv.org/abs/2407.11005), Friel, Belyi, Sanyal (Galileo),
  "RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems".
  - TRACe = ***uTilization, Relevance, Adherence, Completeness*** (four metrics; Utilization/
    Adherence/Completeness score the generator, Relevance scores the retriever). Adherence is defined
    as **answer faithfulness / groundedness / attribution** and is a boolean at example level.
  - Extra: RAGBench = **100k samples**, 12 component datasets, **5 domains** (biomedical, general
    knowledge, legal, customer support, finance); context length 100–11k tokens; responses from
    GPT-3.5 and Claude-3-Haiku; a fine-tuned **DeBERTa-large (400M)** beats few-shot LLM judges.
    The brief's "TRACe completeness" is a real metric (fraction of relevant spans used).

## 5. Lost in the Middle, RECOMP, LLMLingua-2

- **CONFIRMED** — [arXiv:2307.03172](https://arxiv.org/abs/2307.03172), Liu et al., "Lost in the
  Middle: How Language Models Use Long Contexts". **Venue note:** the arXiv comment says "Accepted
  for publication in TACL, 2023"; it appeared in **TACL Vol. 12 (2024), pp. 157–173** — so the brief's
  "TACL 2024?" is correct as published. Finding: performance highest when relevant info is at the
  beginning or end, degrading in the middle, even for explicitly long-context models; tasks are
  multi-document QA and key-value retrieval. Supports the brief's best-first-and-last ordering.
- **CONFIRMED** — [RECOMP, arXiv:2310.04408](https://arxiv.org/abs/2310.04408), Xu, Shi, Choi,
  **ICLR 2024** ([proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/file/bda88ed2892f5e61c9a9bf215c566913-Paper-Conference.pdf)).
  Extractive + abstractive compressors; can return the empty string for selective augmentation;
  compression rate as low as **6%** with minimal performance loss; transferable across LMs.
- **CORRECTED (venue)** — [LLMLingua-2, arXiv:2403.12968](https://arxiv.org/abs/2403.12968), Pan et al.
  Brief says **EMNLP Findings 2024**; the arXiv comment says **"Accepted at Findings of ACL 2024"**.
  Method: data distillation + token-classification compression with XLM-RoBERTa-large/mBERT;
  3–6× faster than prior compression, 1.6–2.9× end-to-end speedup at 2–5× compression; evaluated on
  MeetingBank, LongBench, ZeroScrolls, GSM8K, BBH.

## 6. Adaptive-RAG, CRAG, Self-RAG

- **CONFIRMED** — [Adaptive-RAG, arXiv:2403.14403](https://arxiv.org/abs/2403.14403), Jeong, Baek, Cho,
  Hwang, Park, **NAACL 2024**. A smaller-LM complexity classifier routes between no-retrieval,
  single-step and iterative retrieval-augmented strategies; labels collected from actual model
  outcomes and dataset inductive biases.
- **CONFIRMED (venue is arXiv-only)** — [CRAG, arXiv:2401.15884](https://arxiv.org/abs/2401.15884),
  Yan, Gu, Zhu, Ling, "Corrective Retrieval Augmented Generation". arXiv lists **no conference
  venue** (the brief's "2024" is only the year). Lightweight retrieval evaluator returns a confidence
  degree triggering actions; **large-scale web searches** extend static corpora; decompose-then-
  recompose filtering. This confirms the brief's warning that CRAG's gains rely on web search and a
  fine-tuned evaluator.
- **CONFIRMED** — [Self-RAG, arXiv:2310.11511](https://arxiv.org/abs/2310.11511), Asai, Wu, Wang, Sil,
  Hajishirzi, **ICLR 2024** ([poster](https://iclr.cc/virtual/2024/poster/18095)). Adaptive on-demand
  retrieval with reflection tokens; 7B and 13B models beat ChatGPT and retrieval-augmented
  Llama2-chat on open-domain QA/reasoning/fact verification, with factuality/citation gains in
  long-form generation.

## 7. RAGTruth

- **CONFIRMED** — [arXiv:2401.00396](https://arxiv.org/abs/2401.00396), Niu et al., "RAGTruth: A
  Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models", **ACL 2024**
  ([2024.acl-long.585](https://aclanthology.org/2024.acl-long.585/)).
  - Extra: **~18,000** naturally generated responses from diverse LLMs under RAG, manually annotated
    at case and **word level** with hallucination intensity; a fine-tuned small LLM reaches
    GPT-4-prompt-based detection competitiveness. Correctly framed in the brief as a test of *the
    filter*, not the engine.

## 8. "Let Me Speak Freely?"

- **CONFIRMED** — [arXiv:2408.02442](https://arxiv.org/abs/2408.02442), Tam et al., "Let Me Speak
  Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models".
  - Finding: a **significant decline in reasoning** under structured-format (JSON/XML) restrictions,
    and **stricter format constraints generally cause greater degradation** in reasoning tasks. This
    directly supports the brief's counter-evidence to the structured-citation contract.
  - Extra/nuance: the effect is task-dependent — the paper reports domain-knowledge/classification-type
    tasks sometimes benefit from formats, so "formats hurt" should be scoped to reasoning tasks.

## 9. SelfCheckGPT, semantic entropy, Lynx (why not CPU/engine-feasible)

- **CONFIRMED** — [SelfCheckGPT, arXiv:2303.08896](https://arxiv.org/abs/2303.08896), Manakul, Liusie,
  Gales, **EMNLP 2023 main conference**. Sampling-based, zero-resource: it needs **multiple sampled
  generations** per question to measure consistency — exactly the dependency the engine lacks. Not
  CPU/engine-feasible. Extra: WikiBio GPT-3 passages; AUC-PR for sentence-level detection.
- **CONFIRMED** — Semantic entropy: Farquhar, Kossen, Kuhn, Gal, "Detecting hallucinations in large
  language models using semantic entropy", **Nature 630:625–630 (2024)**,
  DOI [10.1038/s41586-024-07421-0](https://doi.org/10.1038/s41586-024-07421-0)
  ([open access via PMC11186750](https://pmc.ncbi.nlm.nih.gov/articles/PMC11186750/)).
  - Needs **M sampled generations** (10 used) clustered by **bidirectional entailment** using an NLI
    model or LLM — repeated sampling plus logprobs/entailment, neither available to the engine. Not
    CPU/engine-feasible.
  - Extra: semantic entropy AUROC averaged **0.790** vs naive entropy 0.691, P(True) 0.698, embedding
    regression 0.687, across 30 task×model combinations; also proposes AURAC (rejection-accuracy
    curve); GPT-4 results use a discrete black-box variant.
- **CONFIRMED (size detail)** — [Lynx, arXiv:2407.08488](https://arxiv.org/abs/2407.08488), Ravi et al.,
  "Lynx: An Open Source Hallucination Evaluation Model"; **HaluBench = 15k samples**. The released
  models are built on Llama-3 at **8B and 70B**; the brief's "~8B: MPS-possible, CPU-impractical" is
  accurate for the 8B variant (and conservative for 70B).

## 10. NoLiMa

- **CONFIRMED** — [arXiv:2502.05167](https://arxiv.org/abs/2502.05167), Modarressi et al., "NoLiMa:
  Long-Context Evaluation Beyond Literal Matching", **ICML 2025** (arXiv comment "Accepted at ICML
  2025").
  - Claim: extends needle-in-a-haystack with needles that have **minimal lexical overlap** with the
    query, requiring latent association. 13 LLMs claiming ≥128K context; at **32K, 11 models drop
    below 50%** of their strong short-context baselines; GPT-4o falls from **99.3% to 69.7%**;
    reasoning/CoT does not rescue it. Supports the brief's "long context degrades at length".

---

## Summary for brief 04

| # | Citation | Status |
|---|----------|--------|
| 14 | MiniCheck (EMNLP 2024; 770M; 400× cheaper) | CONFIRMED |
| 15 | RAGChecker (NeurIPS 2024 D&B) | CONFIRMED |
| 16 | ALCE (EMNLP 2023) | CONFIRMED |
| 16 | AttributionBench (ACL Findings 2024) | CONFIRMED; the specific "numeric/entity mismatch dominates" wording is OVERSTATED (dominant category is fine-grained-information insensitivity incl. numbers) |
| 17 | RAGBench/TRACe | CONFIRMED (uTilization, Relevance, Adherence, Completeness; 100k samples) |
| 18 | Lost in the Middle | CONFIRMED (TACL Vol. 12, 2024) |
| 18 | RECOMP (ICLR 2024) | CONFIRMED |
| 18 | LLMLingua-2 | **CORRECTED — Findings of ACL 2024, not EMNLP Findings 2024** |
| 19 | Adaptive-RAG (NAACL 2024) | CONFIRMED |
| 19 | CRAG | CONFIRMED (arXiv only; no conference venue listed) |
| 19 | Self-RAG (ICLR 2024) | CONFIRMED |
| 20 | RAGTruth (ACL 2024) | CONFIRMED |
| 21 | Let Me Speak Freely? | CONFIRMED (scoped to reasoning tasks) |
| 22 | SelfCheckGPT (EMNLP 2023) | CONFIRMED (sampling-dependent) |
| 22 | Semantic entropy (Nature 2024) | CONFIRMED (sampling + entailment clustering) |
| 22 | Lynx (~8B / 70B) | CONFIRMED |
| 23 | NoLiMa (ICML 2025) | CONFIRMED |

Corrections in brief 04: **LLMLingua-2 venue** (Findings of ACL 2024) and the **AttributionBench
numeric/entity** framing. No citation was fully UNVERIFIABLE.
