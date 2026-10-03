# Evaluation-methodology brief — laya-rag (retrieval + decisions, no generation)

Scope: the engine returns evidence and calibrated decisions, not prose. Judge-based RAG metrics (RAGAS answer-relevance/faithfulness, ARES, TRACe groundedness, RAGChecker claim-entailment) score a component this system does not have; they matter only for the optional compose step and are the weakest things to claim. The ranking below is weighted to what is offline, CPU-feasible, already partly instrumented in `tools/evalkit.py`, and hardens or falsifies *existing* numbers.

Already present: paired bootstrap, ECE + reliability bins + threshold sweep, per-query metric arrays, grouped/held-out splits, a pool-reorder oracle ceiling. Missing: permutation tests, multiple-comparison control, power/MDE, risk–coverage/AURC, debiased calibration, pool-construction controls, cross-dataset replication.

---

**1. Permutation test + FDR control across the comparison grid**
1. *What:* paired randomization test beside the bootstrap, plus Benjamini–Hochberg FDR over all (retriever × dataset × metric) cells in one table.
2. *Gap:* `tools/beir.py` reports a per-cell 10k-resample bootstrap and asserts "all significant"; bootstrap intervals are mildly liberal on small query sets, and dozens of cells are tested with no family-wise control.
3. *Evidence:* [Smucker et al., CIKM 2007](https://dl.acm.org/doi/10.1145/1321440.1321528); [Carterette, TOIS 2012](https://dl.acm.org/doi/10.1145/2094072.2094076).
4. *Cost:* ~120 NumPy lines in `evalkit.py`, reusing `per_query_metric`; exact permutation for n≤12, else 10k draws; seconds. No APIs.
5. *Claim:* "every reported delta survives a paired permutation test and BH-FDR at q=0.05 across the declared family."
6. *Falsify:* permutation needs exchangeable paired query deltas; invalid if queries were reused to choose the configuration. Pre-register the cell family or the FDR denominator is arbitrary. State that queries, not documents, are resampled.

**2. Selective-prediction suite for the answerability gate (risk–coverage, AURC, risk@coverage, Brier/NLL, debiased ECE)**
1. *What:* evaluate abstention as selective prediction rather than AUC/balanced-accuracy at one threshold.
2. *Gap:* AUC ignores the operating point, balanced accuracy depends on one arbitrary cut, and ECE is binning-sensitive; the headline "gate AUC 0.815" says nothing about the risk accepted at deployable coverage.
3. *Evidence:* [NeurIPS 2024, Overcoming Common Flaws in the Evaluation of Selective Classification Systems](https://arxiv.org/html/2407.01032); [Roelofs et al., 2022, debiasing calibration error](https://proceedings.mlr.press/v151/roelofs22a/roelofs22a.pdf); [Nixon et al., 2019](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Uncertainty%20and%20Robustness%20in%20Deep%20Visual%20Learning/Nixon_Measuring_Calibration_in_Deep_Learning_CVPRW_2019_paper.pdf).
4. *Cost:* ~100 NumPy lines in `answerability_train.py`; risk–coverage curve, **AURC and AUGRC** (the NeurIPS 2024 paper defines 5 requirements and proposes AUGRC = average risk of undetected failures), risk@50/80/90% coverage, Brier, NLL, **ECE_sweep** (equal-mass bins with the largest count preserving monotonicity) plus the debiased estimator, bootstrapped over the existing 200 stratified splits. Seconds.
5. *Claim:* "at 80% coverage selective risk is X% [CI] versus random abstention; AURC = Y" — a decision-relevant statement.
6. *Falsify:* risk–coverage is only as good as the labels. If the "unanswerable" class in `gate.json` is an artifact of gold-support extraction, every metric inherits it — audit label provenance first. Also report coverage actually reached at the deployed threshold, since AURC credits untaken coverage.

**3. Power analysis / minimum detectable effect (topic-set-size design)**
1. *What:* state the smallest Δ each query set can resolve at 80% power.
2. *Gap:* deltas of 0.004–0.03 are reported on 300–6k queries with no power statement; "not significant" is currently indistinguishable from "underpowered".
3. *Evidence:* [Sakai, SIGIR Forum 2014, Statistical Reform in IR](https://sigir.org/files/forum/2014J/2014J_sigirforum_Article_TetsuyaSakai.pdf); [Sakai, Springer 2018, Laboratory Experiments in IR](https://link.springer.com/book/10.1007/978-981-13-1199-4).
4. *Cost:* analytic paired design + simulation on existing per-query arrays; ~80 lines; instant.
5. *Claim:* "on 300 queries the harness detects ΔnDCG@10 ≥ ~0.02 at 80% power; the +0.0288 MultiHop gain clears it, the +0.005 SciDocs shift does not."
6. *Falsify:* MDE depends on observed per-query variance — report the empirical SD and a range, not one number; computed after seeing the effect it is post-hoc, so frame it as design guidance for the next run.

**4. Reranker-fairness pack: depth-stratified recall, system-excluded union pools, judged-depth sensitivity**
1. *What:* report R@10/100/1000 and a reranker-excluded union pool alongside MAP@10/Hits@k.
2. *Gap:* MultiHop rerank +0.0464 is measured on a fused pool the reranker did not construct; a single pooled judgment set and a fixed scoring depth can inflate a reranker's measured gain (pool bias).
3. *Evidence:* [Buckley & Voorhees, 2004](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=150469) on pool bias; [Drowning in Documents, ReNeuIR 2025 @ SIGIR](https://arxiv.org/html/2411.11767) finds reranker quality **declines past an optimal scoring depth** on full retrieval. **Do not cite ICTIR 2025 "Shallow vs. Deep Relevance Judgments on BERT-based Reranking Models" here** — that paper is about *training-set* construction (shallow-judged training data can *improve* generalization), not evaluation-pool bias.
4. *Cost:* reuse cached run/candidate files; add depth curves and a second pool built without the reranker's candidates; CPU-only.
5. *Claim:* "the gain holds on a pool constructed without the reranker, and improves R@1000, not only MAP@10."
6. *Falsify:* if gold support is not exhaustive at depth 1000 the ceiling is still a floor; bound it with the reorder oracle (0.609 corrected, was 0.7307) and say so.

**5. LLM-free nugget / claim-recall protocol for multi-hop**
1. *What:* score partial support as retrieved-nugget coverage instead of single-gold MAP.
2. *Gap:* single-gold MAP punishes partially-correct evidence lists; the published 0.479 gap is metric-sensitive.
3. *Evidence:* [AutoNuggetizer, TREC 2024](https://arxiv.org/html/2411.09607v1); [TREC 2024 RAG evaluation overview](https://trec-rag.github.io/annoucements/evaluation/).
4. *Cost:* build nuggets programmatically from gold support sentences (entity/relation split, token-overlap dedupe) and score chunk coverage; ~250 lines; no API (the field's version is GPT-4-based — flagged dependency).
5. *Claim:* "nugget support recall@10 = X, so part of the MAP@10 gap is a single-gold artifact."
6. *Falsify:* programmatic nuggets are noisier than human ones; report nugget redundancy and correlation with MAP@10 — if they move together the item added nothing.

**6. BRIGHT (CPU-cheap subsets) as a hard, reasoning-intensive retrieval benchmark** *(new benchmark)*
1. *What:* add the genuinely small BRIGHT sets: **Pony (7,894 docs)**, **TheoremQA-T (23,839)**, and the ~50–60k StackExchange sets (Economics/Psychology/Biology/Sustainable Living/Robotics). Note **LeetCode is BRIGHT's largest corpus (413,932 docs)** and AoPS/TheoremQA-Q are 188,002 — those are not CPU-cheap.
2. *Gap:* BEIR's 5 buildable datasets are largely lexical/topical; nothing tests reasoning-intensive retrieval where BM25 is competitive and dense-only is weak.
3. *Evidence:* [BRIGHT, ICLR 2025](https://arxiv.org/pdf/2407.12883) — 12 datasets, 1,384 queries; the best model in the paper (SFR-Embedding-Mistral, 59.0 MTEB) scores only **18.3 nDCG@10**, which is the point. [site](https://brightbenchmark.github.io/).
4. *Cost:* BEIR-format JSONL, so it loads through `tools/beir.py`; hours of CPU indexing; subset the large corpora; no APIs.
5. *Claim:* "on reasoning-intensive queries hybrid+BM25 beats dense-only by ΔX" — even a negative is a legitimate published-style result.
6. *Falsify:* BRIGHT queries are instruction-formatted; a raw-query run understates. Confirm the encoder postdates the corpus (no contamination) and report which of the 12 sets are excluded and why.

**7. MuSiQue + 2WikiMultiHopQA gold-paragraph retrieval** *(new benchmark)*
1. *What:* rerun the coverage objective on two independent multi-hop sets with provided gold paragraphs.
2. *Gap:* the entire multi-hop claim rests on one dataset (MultiHop-RAG); no replication.
3. *Evidence:* [MuSiQue, TACL 2022](https://aclanthology.org/2022.tacl-1.31/); [2WikiMultiHopQA, COLING 2020](https://arxiv.org/abs/2011.01060).
4. *Cost:* HuggingFace parquet downloads (~100 MB), gold paragraphs supplied, CPU-hours; no APIs.
5. *Claim:* "the facility-location coverage gain replicates on two external multi-hop sets."
6. *Falsify:* this is the highest-value falsification in the list — if +0.0288 does not replicate, the headline is dataset-specific and should be labelled as such.

**8. RGB negative-rejection / noise-robustness retrieval subset** *(new benchmark)*
1. *What:* test the gate on an external benchmark built for noise robustness and negative rejection.
2. *Gap:* gate evidence is in-domain MultiHop only; abstraction over an external rejection corpus is unmeasured.
3. *Evidence:* [RGB, AAAI 2024](https://arxiv.org/html/2309.01431v2); [repo](https://github.com/chen700564/rgb).
4. *Cost:* documents are included; retrieval metrics and accept/reject rates only; CPU-hours. The official rejection metric is judge-based — flagged, not used.
5. *Claim:* "on an external noise/rejection set the gate's false-accept rate is X at Y coverage."
6. *Falsify:* Chinese+English, different domain; domain shift may swamp the gate. Only retrieval-side rates are claimable.

**9. Contamination / leakage audit**
1. *What:* n-gram overlap between eval queries and documented encoder corpora, and planted-probe/BEIR disjointness.
2. *Gap:* zero-shot claims assume the encoder has not seen the test queries; currently asserted, not shown.
3. *Evidence:* [Evaluation data contamination, 2024](https://arxiv.org/html/2411.03923); [lm-eval-harness decontamination doc](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/decontamination.md).
4. *Cost:* offline inverted n-gram index; hours; no APIs.
5. *Claim:* "no evaluated query shares a 13-gram with the documented pretraining corpora listed."
6. *Falsify:* absence of overlap in *available* corpora is not proof for closed ones — ship the corpus list as an explicit limitation.

**10. Judge reliability / prediction-powered inference (only if compose is ever scored)**
1. *What:* document judge bias and, if judging is unavoidable, attach valid CIs via a small hand-labeled set.
2. *Gap:* RAGAS/ARES/TRACe/RAGChecker all lean on LLM judges; self-preference and position bias are documented.
3. *Evidence:* [MT-Bench/Chatbot Arena, 2023](https://arxiv.org/html/2306.05685); [self-preference, 2024](https://arxiv.org/abs/2410.21819v1); [position bias, 2024](https://arxiv.org/html/2406.07791v7); [PPI, Science 2023](https://www.science.org/doi/10.1126/science.adi6000).
4. *Cost:* judge-free by default (no local judge is trustworthy on this host); PPI needs ~100–200 human labels once.
5. *Claim:* at most "judge-scored, uncalibrated"; with PPI, "judge estimate with human-anchored 95% CI".
6. *Falsify:* judge–human agreement κ < 0.5 voids the number entirely.

**Deprioritized:** "BEIR-hard" is not a canonical released artifact — the real successors are BRIGHT and FollowIR. LoTTE/LongEmbed test long-document and long-context embedding behaviour that 256-token chunking sidesteps. MTEB v2 and retrieval leaderboards compare checkpoint embeddings, not this pipeline ([MTEB v2](https://huggingface.co/blog/isaacchung/mteb-v2)). FRAMES ([2409.12941](https://arxiv.org/html/2409.12941v2)) and LegalBench-RAG ([2408.10343](https://arxiv.org/pdf/2408.10343)) are downloadable and CPU-feasible but rank below replication.

> **Correction, 2026-10-03.** The ceiling quoted here (0.7307 at depth 100) divided the oracle AP by the
> gold facts *the pool matched* rather than by the query's true gold count, which inflated it on exactly
> the queries whose recall was failing. Corrected it is **0.609**, and it **rises** with depth
> (0.397 / 0.609 / 0.659 at depth 10 / 100 / 500). 35.4% of depth-100 pools are missing a gold fact, so
> recall is a live lever, not a solved one. See the README diagnostics subsection.
