# Generation, faithfulness and agentic answering — ranked additions for laya-rag

Framing is taken from `docs/generation-placement.md` and `BENCHMARKS.md`: the engine owns retrieval quality, answerability and deterministic citation mechanics; everything after compose is `engine + <model>`. Ranked by engine-owned credibility × cheapness. Items 1–6 are engine-side; 7–9 are full-stack and their quality score belongs to the generator.

## A. Engine-side, cheap and honest

**1. Three-state claim adjudication: `supported` / `contradicted` / `no_coverage`.**
`adjudicate_claims` (`engine.py:845`) re-retrieves per claim and returns `supported|unsupported|not_measured`; a fabricated claim about a topic the corpus never mentions lands in `not_measured` — the stated gap. Add a local NLI entailment/contradiction pass over the claim and its re-retrieved top-k and emit the third state explicitly.
Evidence: MiniCheck ([arXiv:2404.10774](https://arxiv.org/abs/2404.10774), EMNLP 2024) — a 770M Flan-T5 matches GPT-4 on LLM-AggreFact at ~400× lower cost; RAGChecker ([arXiv:2408.08067](https://arxiv.org/abs/2408.08067), NeurIPS 2024 D&B) makes claim-level entailment its core metric.
Cost: **local small model, CPU/MPS**. The repo already carries an `NLIAnswerability` hook and the 9-feature NumPy gate (`answerability.py`), so this extends existing plumbing.
Hypothesis: replaces a degenerate `not_measured` bucket with a graded, falsifiable state.
Falsify: hand-label ~300 claim/corpus pairs from `eval/corpus-planted` (supported / contradicted / no-coverage), report per-class F1, especially no-coverage recall. Skepticism: MNLI-style NLI is weak when the premise is *insufficient* rather than contradictory; keep the NLI verdict separate from Laya's calibrated probability, which the repo's own docs note is wording-sensitive.

**2. Extend the deterministic verifier beyond quotes: numbers, negations, units, dates.**
`verify.py` checks citation range and verbatim quotes only. Add mechanical checks: every number/percent/date in a claim must occur in its cited span; negation/antonym flips; unit mismatch ("days" vs "business days", MB vs GB).
Evidence: AttributionBench ([arXiv:2402.15089](https://arxiv.org/abs/2402.15089), ACL Findings 2024) labels **fine-grained-information insensitivity** (facts, events, numbers) as its **dominant error category, >66%**, with ~26.8% human-vs-model evidence-access mismatch. ALCE ([arXiv:2305.14627](https://arxiv.org/abs/2305.14627), EMNLP 2023) supports the citation-precision/recall framing but does **not** decompose errors into numeric/entity mismatch — cite it for that, not for this.
Cost: **CPU-only**, days, no model.
Falsify: 200 composed answers from the local 35B, human-labelled; a flag is defensible only at precision ≥ ~0.9. Skepticism: regex numeric checks false-positive on ranges/rounding — keep them advisory and never silently delete text.

**3. Nugget/aspect coverage accounting (engine-side AutoNuggetizer proxy).**
Decompose the query into atomic aspects (lexical rules, optionally a local small model) and report per-aspect coverage of the returned bundle.
Evidence: AutoNuggetizer ([arXiv:2411.09607](https://arxiv.org/abs/2411.09607), TREC 2024), "The Great Nugget Recall" ([arXiv:2504.15068](https://arxiv.org/abs/2504.15068), SIGIR 2025), TRACe completeness in RAGBench ([arXiv:2407.11005](https://arxiv.org/abs/2407.11005)).
Cost: **CPU-only** if lexical; local-small if LLM-decomposed.
Hypothesis: predicts answer completeness with no generator, and supplies the stopping signal for item 8.
Falsify: correlate with the repo's `pool_ceiling_MAP@10` and `facility_location_select`; if ≈1, it is redundant. Skepticism: paraphrase over-credits overlap — report coverage, never correctness.

**4. Coverage-aware, budgeted, order-aware context builder.**
`build_messages` concatenates grouped documents with no budget, dedup, or ordering. Reuse the shipped submodular `facility_location_select` at compose time to pack to a token budget, dedupe, and order best-first-and-last — the same mechanism the repo already measured when it moved the question to both ends.
Evidence: lost-in-the-middle ([arXiv:2307.03172](https://arxiv.org/abs/2307.03172), TACL Vol. 12, 2024), RECOMP ([arXiv:2310.04408](https://arxiv.org/abs/2310.04408), ICLR 2024), LLMLingua-2 ([arXiv:2403.12968](https://arxiv.org/abs/2403.12968), Findings of ACL 2024).
Cost: **CPU-only**, days.
Falsify: fixed model, vary packer, measure post-packing recall of gold facts; any quality number is `engine + <model>`. Skepticism: a packer that drops a rare decisive fact is a net loss.

**5. Structured citation contract: JSON claim→evidence via constrained decoding.**
Emit `{claims:[{text, cites:[n]}]}` through an Ollama/llama.cpp grammar, then run `verify_citations` per claim.
Evidence: ALCE citation precision/recall ([arXiv:2305.14627](https://arxiv.org/abs/2305.14627)); counter-evidence — rigid formats can degrade reasoning ("Let Me Speak Freely?", [arXiv:2408.02442](https://arxiv.org/abs/2408.02442), 2024).
Cost: grammar-capable local runtime + **CPU-only** parser/verifier.
Falsify: same model, prose vs JSON prompt; if JSON costs measurable quality, keep prose and parse post hoc.

**6. Local small-model faithfulness filter, advisory and labelled.**
Per-sentence support probabilities from MiniCheck-Flan-T5-L (770M) or HHEM-2.1-Open ([HF card](https://huggingface.co/vectara/hallucination_evaluation_model)), reported as `engine + <model>` and never as an engine score.
Evidence: MiniCheck (above). **HHEM is a vendor benchmark — treat its ranking with suspicion.** The honest public test is RAGTruth ([arXiv:2401.00396](https://arxiv.org/abs/2401.00396), ACL 2024), which becomes applicable to *the filter*, not the engine.
Cost: **local small model, CPU-feasible** (seconds per sentence).
Falsify: RAGTruth span-level precision/recall; if HHEM's rank does not reproduce, drop it. Skepticism: detectors drift across generators and domains; a score is a flag, not a certificate.

## B. Full-stack, generator-owned

**7. Adaptive/corrective retrieval routing.** A CPU classifier chooses no-retrieval / single-shot / iterative re-retrieval, triggered by item 3's coverage; score the routing (retrieve-when-needed AUC, retrieval metrics), never answer accuracy. Adaptive-RAG ([arXiv:2403.14403](https://arxiv.org/abs/2403.14403), NAACL 2024), CRAG ([arXiv:2401.15884](https://arxiv.org/abs/2401.15884), 2024), Self-RAG ([arXiv:2310.11511](https://arxiv.org/abs/2310.11511), ICLR 2024). Falsify: routing labels on a held-out set; CRAG's gains rely on web search and a fine-tuned evaluator, neither available locally.

**8. Agentic decomposition with an engine-side stopping rule.** Decompose, retrieve per sub-question, stop when aspect coverage saturates. ReAct ([arXiv:2210.03629](https://arxiv.org/abs/2210.03629)), Self-Ask ([arXiv:2210.03350](https://arxiv.org/abs/2210.03350)), agentic-RAG survey ([arXiv:2501.09136](https://arxiv.org/abs/2501.09136), 2025), Search-R1 ([arXiv:2503.09516](https://arxiv.org/abs/2503.09516), 2025). Cost: local small/mid model on **MPS**; the answer is the LLM's. Falsify: if deterministic PRF/query expansion (measured in `eval/beir/prf-combsum.md`) matches it on `tools/multihop_*.py`, the LLM decomposer is not earning its latency. Agentic loops routinely hurt single-hop.

**9. Judge-based harness with validity controls — only for generator-labelled numbers.** Position-swapped pairwise judging, tie handling, a human meta-eval, reported agreement. MT-Bench ([arXiv:2306.05685](https://arxiv.org/abs/2306.05685)), "Judging the Judges" ([arXiv:2406.12624](https://arxiv.org/abs/2406.12624), GEM² 2025, an ACL workshop); metric definitions from RAGAS ([arXiv:2309.15217](https://arxiv.org/abs/2309.15217)) and ARES ([arXiv:2311.09476](https://arxiv.org/abs/2311.09476)). Cost: **hosted or the local 35B**, 10–15 s/query. Falsify: judge–human κ; publish nothing below an agreed bar.

## Not worth adopting engine-side
SelfCheckGPT ([arXiv:2303.08896](https://arxiv.org/abs/2303.08896)) and semantic entropy ([Nature 630:625–630, 2024](https://www.nature.com/articles/s41586-024-07421-0); AUROC 0.790 vs 0.691 naive entropy, but needs **10 sampled generations** plus bidirectional-entailment clustering) need repeated sampling or token logprobs the engine lacks (no batch `generate()` with logprobs) — generator-side only. Lynx ([arXiv:2407.08488](https://arxiv.org/abs/2407.08488)) is ~8B: MPS-possible, CPU-impractical. Long context is not an engine substitute: NoLiMa ([arXiv:2502.05167](https://arxiv.org/abs/2502.05167), ICML 2025) and RAG-vs-long-context ([arXiv:2407.16833](https://arxiv.org/abs/2407.16833), EMNLP 2024) show degradation at length, and any such comparison is generator-owned.

**Bottom line:** items 1–6 extend code already present (`verify.py`, `adjudicate_claims`, `facility_location_select`, the NLI answerability hook) and produce engine-owned numbers; 7–9 are legitimate features whose headline score belongs to whatever model writes the answer, and must ship under the `engine + <model>` label the repo already enforces.
