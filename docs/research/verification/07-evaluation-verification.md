# Citation verification — `07-evaluation-benchmarks.md`

Verification method: each entry was checked against the primary source (publisher abstract/landing
page, ACL Anthology, PMLR, arXiv abstract/HTML, NIST/official site). PDFs were avoided where an HTML
abstract existed. Status legend: **CONFIRMED** (claim holds as written), **CORRECTED** (a value/title/
venue was wrong; correction given), **UNVERIFIABLE** (could not be checked against a primary source,
or the source does not support the specific claim).
Date of check: 2026 (session date); arXiv versions as noted.

---

## 1. Smucker et al., CIKM 2007 + Carterette, TOIS 2012

- **CONFIRMED** — [Smucker, Allan & Carterette, "A comparison of statistical significance tests for
  information retrieval evaluation", CIKM 2007, pp. 623–632](https://doi.org/10.1145/1321440.1321528).
  The DOI in the brief resolves to this paper (not to a differently titled one). It compares the
  bootstrap, the paired t-test, and randomization/permutation tests, and is the canonical IR
  reference for paired permutation testing.
  - Extra: pages 623–632; the finding is that the paired t-test is unreliable (liberal on small topic
    sets) while bootstrap and permutation tests behave better — consistent with the brief's
    "bootstrap intervals are mildly liberal on small query sets".
  - Note: the brief cites it without a title, so no title correction is owed.
- **CONFIRMED** — [Carterette, "Multiple testing in statistical analysis of systems-based information
  retrieval experiments", TOIS 2012](https://doi.org/10.1145/2094072.2094076). Argues for
  multiplicity control (e.g. FDR / Benjamini–Hochberg-style adjustment) when many system×topic-set
  comparisons are reported, exactly the use the brief makes of it.
  - Extra: this is TOIS Vol. 30, No. 1, Article 4 (2012); the companion line of work on
    multiplicity adjustment for IR experiments is Carterette, SIGIR 2012 ("Deciding on an adjustment
    for multiplicity in IR experiments").

## 2. NeurIPS 2024 — "Overcoming Common Flaws in the Evaluation of Selective Classification Systems"

- **CONFIRMED** — [arXiv:2407.01032](https://arxiv.org/abs/2407.01032); published at
  [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/047c84ec50bd8ea29349b996fc64af4b-Abstract.html)
  (Traub, Bungert, Lüth, Baumgartner, Maier-Hein, Maier-Hein, Jaeger).
  - Named flaws / contribution: defines **5 requirements** for multi-threshold selective-classification
    metrics (task alignment, interpretability, flexibility) and shows current approaches fail them;
    proposes **AUGRC** (Area under the Generalized Risk Coverage curve), interpretable as average risk
    of undetected failures; empirically changes metric rankings on **5 of 6** datasets over a benchmark
    of **6 datasets and 13 confidence scoring functions**.
  - Extra: v2 revised 19 Oct 2024; subjects cs.LG/cs.CV/stat.ME.

## 3. Roelofs et al., 2022 — debiased calibration

- **CONFIRMED** — [Roelofs, Cain, Shlens & Mozer, "Mitigating Bias in Calibration Error Estimation",
  AISTATS 2022, PMLR 151:4036–4054](https://proceedings.mlr.press/v151/roelofs22a.html).
  This is the primary source; the brief's URL was the PDF, abstract page is above.
  - Debiasing method: the **debiased estimator** of Bröcker (2012) / Ferro & Fricker (2012), plus the
    authors' own **ECE_sweep**, which uses **equal-mass (equal-count) bins** and takes the largest number
    of bins that preserves monotonicity of the calibration function. Finding: equal-mass bins have
    lower bias than equal-width bins.
  - Extra: pages 4036–4054; title is exactly "Mitigating Bias in Calibration Error Estimation"
    (the brief's "debiasing calibration error" is a paraphrase, not a title).

## 4. Nixon et al., 2019 — static vs adaptive ECE binning

- **CONFIRMED** — [arXiv:1904.01685](https://arxiv.org/abs/1904.01685), "Measuring Calibration in Deep
  Learning", CVPRW 2019 (Uncertainty & Robustness in Deep Visual Learning workshop).
  - The paper is a comprehensive empirical study of exactly the choice the brief cares about:
    all-class vs max-probability, thresholding, class conditionality, **number of bins, adaptive
    (density-aware) bins**, and L1 vs L2 norm. Conclusion relevant to the brief: **adaptive binning
    schemes give more stable metric rank-ordering** as bin count varies; class-conditional and L2
    variants also improve evaluations.
  - Extra: datasets MNIST, Fashion-MNIST, CIFAR-10/100, ImageNet; conclusions on recalibration method
    ranking are "drastically impacted" by metric choice.

## 5. Sakai, SIGIR Forum 2014

- **CONFIRMED** — [Sakai, "Statistical Reform in Information Retrieval?", SIGIR Forum 48(1), June
  2014, pp. 3–12](https://sigir.org/files/forum/2014J/2014J_sigirforum_Article_TetsuyaSakai.pdf)
  (issue TOC: <https://sigir.org/forum/issues/june-2014/>).
  - Supports the brief's use: it argues for statistical reform in IR, including power/test-collection
    size, effect sizes and the reporting of confidence intervals rather than bare significance.
  - Extra: volume 48, number 1, pages 3–12; follow-up book is Sakai, *Laboratory Experiments in
    Information Retrieval*, Springer 2018 (cited in the brief at item 3, not re-verified here).

## 6. ICTIR 2025 — "Impact of Shallow vs Deep Relevance Judgments on BERT Rerankers"

- **CORRECTED (title)** — exact title is [**"Impact of Shallow vs. Deep Relevance Judgments on
  BERT-based Reranking Models"**](https://arxiv.org/abs/2506.23191), Iturra-Bocaz, Vo & Galuščáková,
  ICTIR '25 (Padua, 18 July 2025), DOI
  [10.1145/3731120.3744602](https://doi.org/10.1145/3731120.3744602) — DOI matches the brief.
- **CORRECTED / MIS-CITED (finding direction)** — the brief uses this paper as evidence that
  "shallow judged pools structurally favour rerankers". The paper is about **training data**, not
  evaluation-pool bias, and its finding is that shallow-judged training sets *enhance* reranker
  generalization/effectiveness at matched instance counts (deep-trained models can even fall below
  BM25), while the deep sets' disadvantage is partly mitigated by more negatives (≈1:8 optimum on
  MS MARCO V1). The paper does not show shallow *evaluation* pools structurally favour rerankers.
  - Extra: MS MARCO V1/V2 + LongEval; both retrieval and reranking at depth 10; MAP/NDCG/MRR; ~1,000
    shallow instances suffice to beat BM25; 4.2k–5k instance grids.

## 7. "Drowning in Documents" 2024

- **CONFIRMED** — [arXiv:2411.11767](https://arxiv.org/abs/2411.11767), "Drowning in Documents:
  Consequences of Scaling Reranker Inference" (Jacob, Lindgren, Zaharia, Carbin, Khattab, Drozdov).
  - What it shows: rerankers improve initially as more documents are scored, but effectiveness
    **gradually declines and can degrade quality beyond a limit**; measured on full retrieval (not just
    re-scoring first-stage output), with strong dense first-stage retrieval and challenge sets incl.
    curated/out-of-domain data to avoid contamination.
  - Extra: v2 (11 Jul 2025) accepted to the **ReNeuIR 2025 workshop @ SIGIR 2025** — the brief's
    "2024" is the arXiv date; cite the workshop venue if a publication venue is needed.

## 8. AutoNuggetizer, TREC 2024 + "The Great Nugget Recall" SIGIR 2025

- **CONFIRMED** — [arXiv:2411.09607](https://arxiv.org/abs/2411.09607), "Initial Nugget Evaluation
  Results for the TREC 2024 RAG Track with the AutoNuggetizer Framework" (Pradeep, Thakur, Upadhyay,
  Campos, Craswell, Lin).
  - Extra: nugget methodology refactored from the TREC 2003 QA Track; LLM nugget creation + LLM
    assignment; calibrated against human/manual nuggets on **21 topics × 45 runs**; strong automatic
    vs manual correlation.
- **CONFIRMED** — [arXiv:2504.15068](https://arxiv.org/abs/2504.15068), "The Great Nugget Recall:
  Automating Fact Extraction and RAG Evaluation with Large Language Models", **SIGIR 2025** (arXiv
  comment: "To appear in SIGIR 2025"), described as a significant revision of 2411.09607.
  - Extra: full-auto vs manual agreement is strongest at the run level; per-topic agreement is still
    the weak point — directly relevant to the brief's "nugget redundancy / correlation" falsifier.

## 9. BRIGHT

- **CONFIRMED** — [arXiv:2407.12883](https://arxiv.org/abs/2407.12883), Su et al., **ICLR 2025**
  ([poster](https://www.iclr.cc/virtual/2025/poster/27702)). 1,384 real-world queries; **12 datasets**;
  best MTEB model (SFR-Embedding-Mistral, 59.0 MTEB) scores **18.3 nDCG@10** on BRIGHT; explicit query
  reasoning improves retrieval by up to 12.2 points.
  - **CORRECTED (subset characterization)** — the brief's "small sets (leetcode, pony, stackoverflow,
    theoremqa, aops)" is not the small end of BRIGHT. Corpus sizes (Table 1): Pony 7,894 (smallest);
    TheoremQA-T 23,839; Economics 50,220; Psychology 52,835; Biology 57,359; Sustainable Living
    60,792; Robotics 61,961; Stack Overflow 107,081; Earth Science 121,249; AoPS 188,002;
    TheoremQA-Q 188,002; **LeetCode 413,932 (largest)**. The genuinely CPU-cheap subsets are
    **Pony, TheoremQA-T, and the ~50–60k StackExchange sets**, not LeetCode.
  - Extra: 7 StackExchange datasets (relevance = cited in an accepted/highly-voted answer, unanimous
    annotator agreement) + 5 coding/theorem datasets (LeetCode, Pony, AoPS, TheoremQA-Q, TheoremQA-T);
    benchmark is robust against pretraining leakage by the authors' own test.

## 10. MuSiQue (TACL 2022) + 2WikiMultiHopQA (COLING 2020)

- **CONFIRMED** — [MuSiQue, TACL 2022, 10:539–554](https://aclanthology.org/2022.tacl-1.31/),
  DOI 10.1162/tacl_a_00475. 25K 2–4-hop questions (MuSiQue-Ans), plus MuSiQue-Full with unanswerable
  contrast questions; 3× human–machine gap; single-hop model drops 30 F1 points. Gold supporting
  paragraphs are part of the released data (`paragraphs` with `is_supporting`), so "gold paragraphs
  supplied" holds.
- **CONFIRMED** — [2WikiMultiHopQA, arXiv:2011.01060](https://arxiv.org/abs/2011.01060), accepted
  **COLING 2020** (title: "Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of
  Reasoning Steps"). Ships evidence/reasoning-path annotations (supporting facts/paragraphs) and
  context paragraphs, so gold support is provided.
  - Extra: the brief's venue "COLING 2020" is correct; both datasets are HuggingFace-downloadable and
    CPU-usable for gold-paragraph retrieval.

## 11. RGB

- **CONFIRMED** — [arXiv:2309.01431](https://arxiv.org/abs/2309.01431), "Benchmarking Large Language
  Models in Retrieval-Augmented Generation", **AAAI 2024**. Four abilities: noise robustness, negative
  rejection, information integration, counterfactual robustness; English + Chinese; 6 LLMs evaluated.
  - Extra: the brief's caveat is right — RGB is a *generation*-side benchmark; only its retrieval-side
    rates (e.g. noise-robustness retrieval) can be claimed by a retrieval/decision engine. Negative
    rejection is LLM-judged.

## 12. Contamination survey (arXiv:2411.03923)

- **CONFIRMED (with reframing)** — [arXiv:2411.03923](https://arxiv.org/abs/2411.03923), "Evaluation
  data contamination in LLMs: how do we measure it and (when) does it matter?" (Singh, Kocyigit,
  Poulton, Esiobu, Lomeli, Szilvasy, Hupkes).
  - It is not a neutral survey but a large-scale empirical analysis + a proposed method, **ConTAM**:
    a survey/comparison of n-gram contamination metrics across **13 benchmarks and 7 models from 2
    families**, grounding each metric in whether models benefit from the examples it flags.
  - Extra: finds contamination may have larger effects than recent LLM reports claim; the **longest
    contaminated substring** is a better signal than the union of all contaminated substrings; large
    *n* and ignoring pre-training-infrequent matches both cause false negatives. This supports the
    brief's n-gram overlap audit and its "closed corpora are unprovable" limitation.

## 13. Judge bias stack

- **CONFIRMED** — MT-Bench/Chatbot Arena: [arXiv:2306.05685](https://arxiv.org/abs/2306.05685),
  **NeurIPS 2023 Datasets & Benchmarks Track**. Documents position, verbosity and self-enhancement
  biases; GPT-4 judge >80% agreement with human preference.
- **CORRECTED (venue)** — "Judging the Judges": [arXiv:2406.12624](https://arxiv.org/abs/2406.12624),
  Thakur, Choudhary, Ramayapally, Vaidyanathan, Hupkes. Brief says **COLM 2024**; the paper's own
  journal reference/comment places it in the **GEM² 2025 workshop (ACL 2025)**, pp. 404–430. Finding:
  13 judge models × 9 exam-taker models; only the best/largest judges align reasonably; score gaps up
  to 5 points vs humans; sensitivity to prompt complexity/length; leniency bias.
- **CONFIRMED** — Self-preference: [arXiv:2410.21819](https://arxiv.org/abs/2410.21819), Wataoka,
  Takahashi, Ri, **NeurIPS 2024 Safe Generative AI Workshop**; new quantitative metric; GPT-4 shows
  significant self-preference; bias tracks **perplexity/familiarity** (LLMs prefer lower-perplexity text
  regardless of authorship).
- **CONFIRMED** — Position bias: [arXiv:2406.07791](https://arxiv.org/abs/2406.07791), Shi et al.,
  "Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge". Brief links v7; latest
  version v9, now **AACL-IJCNLP 2025**. Three metrics (repetition stability, position consistency,
  preference fairness); 15 judges × MTBench/DevBench, 22 tasks, ~40 solution models, >150,000
  instances; bias is not chance and is strongly affected by the quality gap between solutions.
- **CONFIRMED** — PPI: [Science 2023, DOI 10.1126/science.adi6000](https://www.science.org/doi/10.1126/science.adi6000);
  preprint [arXiv:2301.09633](https://arxiv.org/abs/2301.09633) (Angelopoulos, Bates, Fannjiang,
  Jordan, Zrnic). Valid CIs for means/quantiles/regression coefficients when an ML-predicted set is
  combined with a small labeled set; more accurate predictions → tighter CIs. Supports the brief's
  "~100–200 human labels once" framing.

---

## Summary for brief 07

| # | Citation | Status |
|---|----------|--------|
| 1 | Smucker CIKM 2007; Carterette TOIS 2012 | CONFIRMED / CONFIRMED |
| 2 | Selective-classification flaws (NeurIPS 2024) | CONFIRMED |
| 3 | Roelofs 2022 | CONFIRMED |
| 4 | Nixon 2019 | CONFIRMED |
| 5 | Sakai SIGIR Forum 2014 | CONFIRMED |
| 6 | ICTIR 2025 shallow/deep judgments | CORRECTED (title; finding is about training data, not pool bias) |
| 7 | Drowning in Documents | CONFIRMED (now ReNeuIR 2025 workshop) |
| 8 | AutoNuggetizer; Great Nugget Recall | CONFIRMED / CONFIRMED |
| 9 | BRIGHT ICLR 2025 | CONFIRMED (12 datasets) / CORRECTED (named "small" subsets are wrong) |
| 10 | MuSiQue TACL 2022; 2Wiki COLING 2020 | CONFIRMED / CONFIRMED |
| 11 | RGB AAAI 2024 | CONFIRMED |
| 12 | Contamination 2024 | CONFIRMED (ConTAM; 13 benchmarks, 7 models) |
| 13 | MT-Bench; Judging the Judges; self-preference; position bias; PPI | CONFIRMED / **CORRECTED (GEM² 2025, not COLM 2024)** / CONFIRMED / CONFIRMED / CONFIRMED |

No citation in brief 07 was fully UNVERIFIABLE. The two substantive corrections are (a) the ICTIR
2025 paper's finding direction, and (b) the "Judging the Judges" venue.
