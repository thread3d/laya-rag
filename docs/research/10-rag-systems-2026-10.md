> **Provenance.** A research digest compiled on **2026-10-29** for
> [`docs/best-in-the-world-2026-10.md`](../best-in-the-world-2026-10.md). Every figure was read from the
> cited source on that date; the fetched pages and PDFs and the six per-section working files are **not**
> committed (they live under the session's `.scratch/research/`), so paths naming them below are the
> session's, not this checkout's. Vendor-reported numbers are labelled; anything unverified is marked
> **UNVERIFIED** with the attempt that failed. Nothing here is a claim of this repository's; the claims it
> supports are in the assessment, and each carries its own falsifier.

---

# RAG systems: the state of the art as of late 2026

**A reading of the field for the Laya RAG project** — a fully-local, CPU-only, no-API-key
# RAG systems: the state of the art as of late 2026

**A reading of the field for the Laya RAG project** — a fully-local, CPU-only, no-API-key
retrieval-augmented engine that does not generate text: it returns evidence passages with calibrated
probabilities, adjudicates claims as `supported` / `contradicted` / `no_coverage`, and abstains.

*Research cut-off: 29 October 2026. Every figure below was read from the cited source in this
session unless marked otherwise; the fetched pages and PDFs are cached as plain text under
`.scratch/research/src/`, and the per-section working files are `section1-benchmarks.md`,
`section2-architectures.md`, `section3-faithfulness.md`, `section4-protocol.md`,
`section5-landscape.md`, `section6-nongenerative.md`. Vendor-reported numbers are labelled as such.
Anything I could not verify is marked **UNVERIFIED** together with what I tried.*

---

## Executive summary — the twelve facts that change decisions

1. **The 2026 grading stack has three layers and only the first is generator-free.** TREC RAG 2025
   scored retrieval (nDCG@k / Recall@k), response quality (strict-vital nugget recall + sub-narrative
   coverage) and attribution (sentence-level weighted support precision/recall) on 105 narratives;
   the best RAG run under fully manual grading reached **strict vital 0.39 / coverage 0.70**, while
   the *attribution* table is topped by a run with near-zero nugget recall (strict vital **0.06**)
   because it says little — the precision/recall trap every abstaining system falls into
   ([TREC 2025 RAG overview](https://trec.nist.gov/pubs/trec34/papers/Overview%5Frag.pdf), 10 Mar 2026).
2. **The best retrieval runs at TREC 2025 used a generator in the loop** — "Generation-in-the-loop
   Pipeline" — and beat BM25 by ≈**+0.35 nDCG@30** (0.6934 vs 0.3468). That is the largest documented
   agentic-retrieval gain of the period and it is paid for in generator calls.
3. **TREC RAG 2026 is agent-first, has a generator-free Retrieval subtask, and has already closed
   its submission window** (8 Aug 2026; TREC 2026 conference Nov 2026). New corpus
   NVIDIA ClimbMix-400b, 119 narratives, AG task removed. Next realistic shot: TREC 2027.
4. **A 2026 shared task put sparse retrieval on top.** MTRAGEval (SemEval-2026 Task 8) Subtask A was
   won by **AILS-NTUA with a sparse retriever, nDCG@5 0.578 / Recall@5 0.622**, against BM25 0.354,
   BGE-base-1.5 0.390 and Elser 0.480 — on a 507-task multi-turn benchmark loaded with unanswerable,
   underspecified and non-standalone questions.
5. **Abstention is now a named, differentiating capability.** The same shared task's conclusion:
   "**abstention is a differentiating capability not correlated with overall generation quality**",
   and abstention-type questions are where the field's systems diverge most. AbstentionBench (Meta,
   2025) shows **reasoning fine-tuning degrades abstention by 24% on average**; RGB's best rejection
   rate is still **45% (EN)**; under controlled conflicting evidence, 2026 measurements find
   **65–91% over-answer rates**.
6. **No benchmark grades "correct evidence" and "honest abstention" as two separate scores over the
   same query set.** The closest are AbstentionBench (grades abstention separately from accuracy, but
   needs generated text), the 2026 Evidence Sufficiency Benchmark (CMC), RGB's negative-rejection
   testbed, and — conceptually — Asai & Choi's ACL 2021 decomposition into *paragraph selection* and
   *answerability prediction*. The gap is real and the raw material to fill it is public (see
   MultiHop-RAG's 301 `null_query` records below).
7. **The strongest counter-argument to a retrieval-first position is published and must be
   answered:** classical IR metrics correlate only moderately (≈0.55–0.70 Spearman for the best
   proposed alternative, worse for nDCG) with end-to-end answer accuracy, because LLM consumers have
   a different position discount and actively suffer from distractors
   ([Redefining Retrieval Evaluation in the Era of LLMs](https://arxiv.org/html/2510.21440), 2025).
8. **The retriever, not the generator, is the largest single lever — measured.** Same agent
   (gpt-5), same prompts, two retrievers: **55.90% → 70.12%** accuracy on BrowseComp-Plus, with
   calibration error falling **13.50% → 9.11%**. Retriever-only SOTA on that benchmark's own qrels
   is **nDCG@10 20.3** (Qwen3-Embed-8B) and **1.6 for BM25**
   ([arXiv:2508.06600](https://arxiv.org/html/2508.06600)).
9. **The evidence layer is demonstrably unsolved, which is where an evidence engine can matter:**
   BRIGHT best average nDCG@10 **24.3**, BrowseComp-Plus best 20.3, TREC 2025 deep-research retrieval
   nDCG@30 0.69 with BM25 at 0.35.
10. **Every adopted open-source "RAG system" is generator-bound and publishes nothing at the
    retrieval layer.** RAGFlow has 93,080 stars and no number; the five that publish anything
    (LightRAG, GraphRAG, Cognee, PageIndex, Khoj) all measure LLM-judged answer quality. Retrieval
    quality numbers live only in retrieval *infrastructure* (bm25s, Pyserini, Qdrant, Vespa) — none
    of which returns calibrated probabilities or abstains. The composite niche — local + CPU +
    key-free + rigorously measured — is **empty**.
11. **A small project can reach four no-generator leaderboards** (TREC RAG-R, MTRAGEval Subtask A,
    BrowseComp-Plus retriever table, BEIR), and the protocol bar is lower than it looks: paired
    intervals, contamination audit, per-query artefacts and a negatives register are ahead of most
    published RAG tables. MTEB is *not* a natural placement — it ranks encoders, not engines.
12. **The defensible claim is an instrument claim, not an answer-quality claim.** Details in
    §8 "The winnable claim in 2026".

---

## 1. End-to-end benchmarks and how they are graded in 2026

*All web sources accessed 2026-10-29 unless stated; local copies of every fetched page/PDF are in
`.scratch/research/src/` and each entry names the file it came from.*

### 1.0 The grading stack changed shape, and that matters more than any single number

Between TREC RAG 2024 and TREC RAG 2025 the field moved from "did the answer match the gold string"
to a **three-layer grading stack** that is now the de-facto template for a serious RAG evaluation:

1. **Retrieval quality** under the Cranfield paradigm — human (or LLM-pooled) qrels, `nDCG@k`,
   `Recall@k`. This layer needs **no generator**.
2. **Response quality** as *nugget recall* — key information units are extracted from judged-relevant
   documents, mapped to sub-narratives, and a response is scored by how many *vital* nuggets it
   fully supports. TREC reports the **strict vital score** and **sub-narrative coverage**
   ([TREC 2025 RAG overview](https://trec.nist.gov/pubs/trec34/papers/Overview%5Frag.pdf),
   arXiv:2603.09891v1, 10 Mar 2026, §3.2).
3. **Support / attribution** — every answer *sentence* is checked against its *cited* document and
   graded in three levels; the two reported quantities are **weighted precision** and **weighted
   recall** of supported sentences (ibid. §3.3; only the *first* cited document per sentence is
   assessed, a deliberate cost control carried over from 2024).

A fourth axis has appeared inside the graders themselves: **answerability conditioning**. MTRAG
introduced an "IDK judge" that first decides whether the response contains a full answer, a partial
answer or an explicit refusal, and *only then* computes the quality metrics (Katsis et al.,
[TACL 2025](https://aclanthology.org/2025.tacl-1.36.pdf), §6.3.3). The 2026 SemEval rendition turns
that conditioning into an explicit score table — for the *generation* subtasks, an unanswerable
question answered with an explicit IDK scores **1**, a partial scores **0**, a confident non-IDK
answer scores **0**, and a clarification request scores **0** (Rosenthal et al.,
[SemEval-2026 Task 8](https://aclanthology.org/2026.semeval-1.447.pdf), Table 3). Note the limit of
this: the *retrieval* subtask still scores only the **answerable** subset (MTRAG task definition,
[leaderboard page](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/)), so refusal is never graded
on the retrieval side.

**The practical consequence for a non-generative engine is sharp:** layer 1 is scorable on evidence
alone; layers 2–4 all take a *string of natural-language answer text* as input. A system that
returns passages plus calibrated probabilities can be fully ranked on layer 1 and cannot be entered
in layers 2–4 at all without emitting answer text.

### 1.1 Summary table

| Benchmark (date) | What is actually graded | Metric that is reported | Requires a generator? | Best published number I could verify |
|---|---|---|---|---|
| **TREC RAG 2025** (overview arXiv:2603.09891, 10 Mar 2026) | retrieval + end-to-end RAG + AG + relevance judging | retrieval `nDCG@30/@100`, `Recall@100`; response **strict vital** + **sub-narrative coverage**; support **weighted P/R** | **Both tasks exist**: *Retrieval* (R) needs no generator, *RAG/AG* do | R: **nDCG@30 0.6934** (`4method_merge`, UTokyo). RAG: **strict vital 0.39 / coverage 0.70** (`Kun-Third`, RMIT-IR). Support (manual): top **0.8435** P/R (`no-llm-refined`, AG) and **0.8098** (`cru-ablR`); best RAG run **0.7792 / 0.7635** (`bm25-rz7b-2025a`). Automatic grader inflates the same runs to **0.9522 / 0.9298** |
| **TREC RAG 2026** (site + track guidelines, checked 2026-10-29) | agent-first deep research; new corpus NVIDIA ClimbMix-400b; R and RAG tasks only (AG **removed**) | not yet published — conference Nov 2026 | R = no; RAG = yes (cited sentences required) | n/a (submissions closed 8 Aug 2026; judgements TBD) |
| **MTRAG** (TACL 2025) | 110 human conversations / 842 tasks, 4 domains; retrieval *and* generation subtasks | `Ans.Acc.` (IDK judge accuracy), `RLF` (RAGAS faithfulness judge), `RBalg`, `RBllm`; recall/nDCG@k for retrieval | Both; retrieval part no | best LLM `RBalg` **0.47** (Llama-3.1-405B) vs reference answer **0.88**; best faithfulness `RLF` **0.69–0.71** vs reference **0.86–0.87**; answerability accuracy **0.86–0.89** vs reference **0.98** |
| **MTRAGEval / SemEval-2026 Task 8** (2026) | multi-turn RAG on **MTRAG-UN**: unanswerable, underspecified, non-standalone, unclear questions; Subtasks A=retrieval, B=generation on gold passages, C=full RAG | A: **nDCG@5, Recall@5**; B/C: harmonic mean over `RBalg`, `RLF`, `RBllm` | A = **no generator required** | A: **nDCG@5 0.578 / Recall@5 0.622** (AILS-NTUA, **sparse** retriever); best baseline Elser 0.480, BGE-base-1.5 0.390, BM25 0.354. B: H-Mean 0.783 (RaguTeam); C: 0.586 (GenAIus) |
| **BrowseComp-Plus** (arXiv:2508.06600, Aug 2025; ACL 2026 version) | deep-research agents over a *fixed, released* corpus with human-verified evidence and gold docs | agent **Accuracy, Recall, Search Calls, Calibration Error**; retriever-only **Recall@k, nDCG@k** on a Cranfield qrel file | Both; retriever split no | agent: **70.12%** accuracy (gpt-5 + Qwen3-Embed-8B; 55.90% with BM25), calibration error **9.11%**. retriever-only: **nDCG@10 20.3**, Recall@1000 76.7 (Qwen3-Embed-8B); BM25 nDCG@10 **1.6** |
| **RAGTruth** (ACL 2024) | word/span-level hallucination in *naturally generated* RAG responses across 4 model families, 3 tasks | response-level and span-level hallucination-detection **P/R/F1** | **Yes** — the corpus is LLM output | detection: GPT-4-turbo **F1 63.4%**, SelfCheckGPT 58.8%, fine-tuned Llama-2-13B **78.7%** (response level); span-level best 58.2% |
| **RAGAS** (EACL 2024 demo) | a *metric suite*, not a leaderboard: faithfulness, answer relevance, context relevance/precision/recall | LLM-judged scores claimed to correlate with humans (faithfulness/context-relevance annotator agreement ≈95%, answer relevance ≈90%) | **Yes** — faithfulness is computed over generated statements | no leaderboard; 2024–2025 meta-evaluation shows the implementation fails unit tests — see GroUSE below |
| **GroUSE** (arXiv:2409.06595, Sep 2024) | meta-evaluation: do RAG evaluators detect known failure modes? | unit-test pass rate; correlation with GPT-4; calibration on reference cases | n/a (grades evaluators) | RAGAS and DeepEval "fail to cover all the presented cases"; strong GPT-4 correlation does **not** imply calibration |
| **FRAMES** (arXiv:2409.12941, Sep 2024) | 824 multi-hop questions, Wikipedia, accuracy under gold/retrieved/multi-step | **accuracy** (LLM auto-rater, reportedly 0.96 agreement / κ 0.889) | **Yes** | GPT-4 no-retrieval **0.408**, +BM25 top-2 0.452 / top-4 0.474, **multi-step pipeline 0.66**, oracle-context **0.729** |
| **MuSiQue** (TACL 2022; arXiv:2108.00573) | 2-hop→4-hop compositional questions over 20 paragraphs | **EM / F1** (QA answer strings) | **Yes** | best agentic-retriever row I verified: Search-R1-base **EM 0.196** on MuSiQue vs IRCoT 0.072, RAG 0.058 (Qwen2.5-7B) |
| **MultiHop-RAG** (arXiv:2401.15391, Jan 2024) | news KB; 2,556 records incl. 301 `null_query` ("Insufficient information") | retrieval **MAP@k, MRR@k, Hits@k**; separately **generation accuracy** | Both — retrieval split no, generation yes | retrieval: **MAP@10 0.4795 / Hits@10 0.7467** (voyage-02 + bge-reranker-large); no-reranker 0.3143. generation: GPT-4 **0.56** with retrieved chunks vs **0.89** with gold evidence |
| **RGB** (AAAI 2024, doi:10.1609/aaai.v38i16.29728) | noise robustness, **negative rejection**, information integration, counterfactual robustness | accuracy, **rejection rate**, error-detection rate, error-correction rate | **Yes** | highest rejection rate only **45%** (English) / **43.3%** (Chinese); integration accuracy ≤60%/67% clean, ≤43%/55% noisy |
| **CRUD-RAG** (ACM TOIS 2025, doi:10.1145/3701228) | Chinese news: continuation 10,728, 1-doc QA 3,199, multi-doc QA, hallucination modification, summarisation | ROUGE/BLEU/BERTScore + **RAGQuestEval**; 4 CRUD scenario buckets | **Yes** | no maintained public leaderboard; the paper reports its own ablated comparisons only |
| **LoTTE** (in ColBERTv2, arXiv:2112.01488, 2022) | 12 long-tail topic test sets, "search" and "forum" queries | **Success@5** (= Recall@5) | **No** — retrieval-only | ColBERTv2 beats SPLADEv2/RocketQAv2 by up to 3.7 / 8.1 points S@5; the *metric*, not a 2026 SOTA, is the useful part |
| **LegalBench-RAG** (arXiv:2408.10343, Aug 2024) | 6,889 expert-annotated query→minimal-snippet pairs over legal corpora | **Precision@k / Recall@k** at snippet level | **No** — retrieval-only by construction | their own sweep: best **Recall@64 84.19%**, **Precision@1 14.38%**; precision@1 "only 2.65%" on the hardest subset |

### 1.2 What the top of each leaderboard actually is

**TREC RAG 2024 is where the template came from, and it is worth knowing how small it started.**
The inaugural track ran **301 topics** over MS MARCO Segment v2.1 with **three** tasks — Retrieval
(R), Augmented Generation (AG, on the organisers' top-k segments) and RAG (own retrieval, mapped back
to MS MARCO segment IDs "for reproducibility and ease of evaluation"). The announced methodology was
already the modern one: score citation support first, then nuggetize, then assign nuggets to answer
sentences, then aggregate. Submissions closed **25 Aug 2024**, judgements were returned **Oct 2024**,
and the TREC 2024 conference was **Nov 2024**
([trec-rag.github.io/trec24](https://trec-rag.github.io/trec24/), accessed 2026-10-29). The 2024
overview paper itself was not retrievable (§1.5), so this report quotes **no 2024 winner numbers** —
a 2026 report that quotes 2024 TREC numbers without the overview PDF is quoting from memory.

**TREC RAG 2025 is the most complete public grading of RAG that exists, and it is a *retrieval*
story as much as a generation story.** 46 retrieval runs from 12 groups, 51 RAG runs from 16
groups, 25 AG runs from 9 groups, "over 150 submissions" (overview, §4.1–4.2). Three observations
that matter:

- The best retrieval run (`4method_merge`, UTokyo, nDCG@30 0.6934) and the second
  (`lucerank`, digsci, 0.6805) are both categorised **"Generation-in-the-loop Pipeline"** — i.e.
  the winning retrieval systems used an LLM to rewrite/plan/select. A pure lexical baseline
  (`anserini_bm25_top100`) sits at **0.3468**, and a 2021-vintage MiniLM dense baseline at
  **0.4029**. On a 105-narrative deep-research query set, the LLM-in-the-loop premium over BM25
  is roughly **+0.35 nDCG@30** — this is the single largest documented "agentic retrieval" gain
  in 2025–2026 and it is *not* free (it is a generator call per query, or several).
- Response evaluation collapses as you make the task realistic. Under **fully manual** grading
  (post-edited nuggets + manual assignment) the best RAG run reaches **strict vital 0.39** and
  **sub-narrative coverage 0.70**. Under the automated graders, the same systems' numbers move
  by ~0.02–0.03 and *re-order* — the overview explicitly notes strict vital is the metric most
  sensitive to upstream grader choices (Figure 7 rows a/b/c).
- Support/attribution behaves the way a precision/recall pair always behaves when a system says
  little. Under **NIST manual assessment** (Table 11, 17 narratives, only the *first* cited document
  per sentence is checked) the top of the table is **0.8435** weighted P/R (`no-llm-refined`, WING-II,
  an AG run) and **0.8098** (`cru-ablR`), with the best *RAG* run at **0.7792 / 0.7635**
  (`bm25-rz7b-2025a`); the bottom is 0.12/0.19. The **same runs re-scored by a GPT-OSS-120B grader**
  (Table 12) come out at **0.9522** and **0.9298 / 0.9189** — a consistent **+0.11 to +0.15**
  inflation from automatic attribution scoring. That is the number to quote at anyone presenting an
  LLM-judged citation score.
- **The precision/recall split is the sharpest lesson in the whole track.** `no-llm-refined` — a run
  whose name advertises that it does *not* use an LLM to write the answer — tops the support table at
  0.84 while scoring **strict vital 0.060 / coverage 0.160** on the response table (Table 7, last
  row), i.e. near-zero nugget recall. A system that emits little gets excellent attribution precision
  and no content score. Any abstaining or evidence-only engine will be scored exactly this way, and
  must report both halves or be quietly flattered by the support metric.

**TREC RAG 2026 is agent-first and has already closed.** The track switched corpora to NVIDIA
**ClimbMix-400b**, released **119 test narratives** on 6 July 2026, ran submissions until **8 Aug
2026**, and *removed the AG task* — leaving exactly two tasks, **Retrieval (R)** and **RAG**. The
RAG output format still requires per-sentence citations into a `references` list. Judgements are
"TBD" and the TREC 2026 conference is **November 2026**, so no 2026 leaderboard exists yet
([track site](https://trec-rag.github.io/); [RAG task spec](https://github.com/TREC-RAG/trec-rag-skills/blob/main/skills/trec-rag-2026-track-guidelines/references/rag-task.md);
[retrieval task spec](https://github.com/TREC-RAG/trec-rag-skills/blob/main/skills/trec-rag-2026-track-guidelines/references/retrieval-task.md)).
The **Retrieval task is generator-free and explicitly allows "a custom retrieval system"**, with
the submitted depth `k` chosen per narrative. That is the single most directly winnable placement
in the 2026 benchmark landscape for a CPU-only evidence engine — with the caveat that the 2026
deadline has passed and the next opportunity is TREC 2027.

**MTRAG → MTRAGEval is where abstention became a scored axis.** MTRAG (TACL 2025) established that
frontier LLMs sit around **0.86–0.89 answerability accuracy** and **0.63–0.71 faithfulness**, versus a
reference answer at **0.98 / 0.86–0.87**, and that performance drops on partially-answerable and
unanswerable turns. The 2026 SemEval task then put 92 submissions through MTRAG-UN and produced the finding that
should be printed on every non-generative system's front page:

> "top systems maintain competitive unanswerable scores, while many mid-tier teams drop sharply,
> suggesting that **abstention is a differentiating capability not correlated with overall
> generation quality**." — SemEval-2026 Task 8, §6.2.

It also produced the strongest 2026 datum for lexical retrieval: **the winning Subtask A system used
sparse retrieval** (nDCG@5 0.578), the runner-up also sparse, and the organisers' own baselines run
**BM25 0.354 < BGE-base-1.5 0.390 < Elser 0.480**. They attribute part of this to the benchmark
having been *built* with a sparse retriever, which is an honest and important caveat — but it is
still a 2026, 507-task, unanswerable-heavy, multi-turn benchmark where a lexical family reached the
top of the table.

**BrowseComp-Plus is the cleanest demonstration that the retriever, not the generator, is the
binding constraint** — and it is the one 2025–2026 benchmark that publishes *both* halves separately.
Same agent (gpt-5), same prompts, two retrievers: **55.90% (BM25) → 70.12% (Qwen3-Embed-8B)**
accuracy, with **calibration error 13.50% → 9.11%** and *fewer* search calls (23.23 → 21.74).
Retriever-only numbers on the same qrels are brutal: **BM25 nDCG@10 1.6**, Qwen3-Embed-8B **20.3**.
Two caveats a reviewer will raise: the corpus is fixed and released (good), the qrels are
human-verified evidence documents (good), and the agent numbers are all closed models except
gpt-oss-120B (42.89%) and Search-R1-32B (3.86%).

**MultiHop-RAG is the one mainstream "RAG benchmark" whose primary table is retrieval-only.** Its
Table 5 reports MAP@10 / MRR@10 / Hits@10 / Hits@4 for eight embedding models, with and without a
reranker, and contains **no lexical baseline at all**. The published peak is `voyage-02` +
`bge-reranker-large` at **MAP@10 0.4795 / Hits@10 0.7467**. The generation table (Table 6) is a
separate experiment: GPT-4 scores **0.56** on retrieved chunks and **0.89** on gold chunks — a
33-point gap that *is* the retrieval headroom, measured.

**MuSiQue is the hardest of these and is often mis-cited.** It is an **EM/F1 answer-string**
benchmark, not a retrieval benchmark. Strong 2025 agentic systems get **EM 0.196** (Search-R1 on
Qwen2.5-7B-base); IRCoT gets 0.072 and a plain RAG pipeline 0.058. Any report that quotes a
MuSiQue "0.37" without saying *which* metric is quoting a retrieval MAP/MRR-style number, not an
answer score, and mixing the two would be the kind of error a reviewer kills a paper over.

**RGB remains the canonical abstention benchmark of the LLM era**, and its numbers are unkind: the
best rejection rate in English is **45%**, in Chinese **43.3%**, and it separately measures
**negative rejection** — refusing when the answer *is* present — which is the failure mode every
"abstain more" system creates. RGB grades four abilities at once (noise robustness, negative
rejection, information integration, counterfactual robustness) and all four need generated text.

**FRAMES, CRUD-RAG, RAGTruth, RAGAS** are the four that are least useful as leaderboards in 2026 and
most useful as *metric definitions*: FRAMES for the gold/retrieved/oracle three-point decomposition
(0.408 → 0.474 → 0.729 → 0.66 multi-step, which cleanly separates retrieval headroom from reasoning
headroom); CRUD-RAG for the CRUD scenario taxonomy and RAGQuestEval; RAGTruth for span-level
hallucination labels; RAGAS for the three-metric vocabulary (faithfulness / answer relevance /
context relevance). RAGAS in particular should be cited as a *vocabulary*, not a verdict: GroUSE
found its implementations penalise faithful-but-verbose answers and that "strong correlation with
GPT-4 ... does not imply good pass rate on unit tests".

### 1.3 Does submission require running a generator?

| Benchmark | Generator required to submit? | Evidence |
|---|---|---|
| TREC RAG Retrieval (R) task, 2025 & 2026 | **No** | task spec accepts a custom retrieval system; outputs are ranked doc IDs |
| TREC RAG RAG/AG tasks | **Yes** — ≤400-word sentence-split answer with sentence-level citations | RAG task spec JSON schema |
| MTRAG / MTRAGEval Subtask A | **No** | Subtask A = ranked passages only |
| MTRAG Subtask B/C, MTRAGEval B/C | **Yes** | outputs are answer text |
| BrowseComp-Plus retriever table | **No** | Cranfield qrels, Recall@k / nDCG@k |
| BrowseComp-Plus agent table | **Yes** | agent emits a final answer + confidence |
| MultiHop-RAG retrieval table | **No** | MAP@k/MRR@k/Hits@k |
| MultiHop-RAG generation table | **Yes** | response accuracy |
| LoTTE, LegalBench-RAG | **No** | Success@5 / P@k, R@k |
| FRAMES, MuSiQue, RGB, RAGTruth, CRUD-RAG, RAGAS | **Yes** | all score generated text |

### 1.4 Numbers in this project that will be cross-examined, and what they are comparable to

These are stated here because a 2026 reviewer will go straight at them:

- **BEIR-18 macro nDCG@10 0.4799 "best per dataset"** is compared in the project's own README to
  SPLADE's published **0.4740** average. That comparison is *not* like-for-like: 0.4799 is a
  per-dataset maximum over different routing configurations, while the published number is one
  configuration evaluated everywhere. The README says so itself, which is the correct handling —
  but any external claim must repeat the qualifier or it will be read as a single-config result.
  I could not verify a 2026 SOTA BEIR-18 average in the time available; treat "above SPLADE" as
  "above the 2022 published systems the BEIR paper contains", not "above 2026 SOTA".
- **MultiHop-RAG MAP@10 0.3128 (engine) vs 0.2510 (harness)** is real, is measured under the
  benchmark's own metric code, and **loses to the paper's reranked rows (0.4795 and 0.4759)** while
  *beating* the paper's no-reranker rows. The README's Hits@10 0.896 vs the paper's 0.747 is the
  stronger claim and the one to lead with. A reviewer will also ask whether the comparison is
  fair given the paper's table has no lexical baseline — it does not, and the project's BM25 row
  (0.782 Hits@10) shows that most of the apparent win over published dense systems is lexical.
- **MuSiQue 0.3761 vs harness 0.3852** is a **retrieval MAP@10** figure over `is_supporting`
  paragraphs. It must always be labelled as such; the MuSiQue *answer* SOTA is EM ≈0.20 (Search-R1)
  and quoting 0.37 next to MuSiQue without the metric name would look like a fabricated SOTA.
- **Answerability gate AUC 0.815 (feature gate) vs 0.520 (shipped scalar)** must never be quoted
  bare: 0.815 is a **within-question-type** figure, falling to **0.767 [0.671, 0.827]** with a whole
  question type held out and to **ROC-AUC 0.509 [0.463, 0.555]** (chance) out of domain — see §3.4,
  which states all three. It also has no direct public comparator. The nearest published anchors are
  MTRAG's IDK-judge answerability accuracy (~0.86–0.89 for frontier LLMs, but that judge is itself an
  LLM reading a generated answer) and the "Sufficient Context" line of work (§3/§6). No benchmark I
  found grades a *retrieval-only* answerability predictor with a held-out AUC, which is both the gap
  and the opportunity.
- **The 20-record end-to-end pilot with EM 0.000** is correctly characterised as non-evidence.
  What makes it non-evidence is not the sample size alone: the gold answers are short entities and
  the pilot graded *prose* generations, so EM is structurally 0. Any end-to-end row needs an
  LLM-judge or nugget metric (TREC/MTRAG style) or a short-answer protocol (MuSiQue/Search-R1
  style), not EM on prose.

### 1.5 Things I could not verify

- The TREC RAG **2024** overview PDF: `trec.nist.gov/pubs/trec33/papers/Overview_rag.pdf` and the
  `%5F`-encoded variant both returned **404** on 2026-10-29, and I found no mirror. The 2024 track
  description here therefore comes from the 2024 track site (`https://trec-rag.github.io/trec24/`,
  fetched) — its task definitions and dates are reliable, but **no 2024 result numbers** are quoted
  anywhere in this report.
- No **FRAMES 2026 leaderboard** I could verify; the numbers above are the original paper's. Third-party
  aggregator sites (sota2.com, benchmarklist.com, theaggregate.ai) surface FRAMES/MuSiQue tables but I
  could not confirm their provenance or date, so I have not used their numbers.
- **CRUD-RAG** has no leaderboard I could find; "current best" is undefined.
- Whether the TREC RAG 2026 *judgements* will be released before the November 2026 conference.

---

## 2. Architectures: the state of the art as of late October 2026

**How to read the numbers.** Each claim gives the metric, the dataset, the baseline it beat, a link to
the primary source, and a date. Two labels recur:

- **[SHIPS A GENERATOR]** — the method trains, prompts, or owns an autoregressive generator, and its
  headline gain is produced by that generator's text. Most published "RAG gains" are in this class and
  are unavailable to an engine that does not generate.
- **retrieval-only** — a retrieval, routing, pruning or indexing component that can precede any reader.

Vendor numbers are marked **[vendor-reported]**; a vendor's own benchmark report **[lab-reported]**.
Anything whose number exists only as an image or marketing sentence is marked **UNVERIFIED** with what
was attempted.


---

### 2.1 Agentic and iterative retrieval

### 2.1.1 Interleaving reasoning with retrieval — IRCoT

IRCoT ([Trivedi et al., arXiv 2212.10509, posted 2022-12-20, ACL 2023](https://arxiv.org/abs/2212.10509))
uses the last sentence of a chain-of-thought step as the next query, alternating retrieve → reason until an
answer appears. **[SHIPS A GENERATOR]** On HotpotQA / 2WikiMultihopQA / MuSiQue / IIRC with Flan-T5-XXL it
raised gold-paragraph retrieval recall over one-shot retrieval by **+7.9, +14.3, +3.5 and +10.2 points**
(GPT-3: **+11.3, +22.6, +12.5, +21.2**), and answer F1 by **+9.4 / +15.3 / +5.0 / +2.5** (GPT-3: **+7.1 /
+13.2 / +7.1**, IIRC flat). A 3B Flan-T5 with IRCoT beat one-shot and zero-shot prompting of a 175B GPT-3 on
every dataset. Cost: 2–4 retrieval rounds per question, measured at **$1–3 and 20–40 minutes per 1,000
queries** versus **$0.1 and 3 minutes** for single-step retrieval on GPT-3.5-Turbo
([HippoRAG, arXiv 2405.14831, Table 17, 2024-05-23](https://arxiv.org/abs/2405.14831)). The dependency is
structural: the query *is* generated prose.

### 2.1.2 Self-critique loops — Self-RAG and CRAG

Self-RAG ([Asai et al., arXiv 2310.11511, 2023-10-17, ICLR 2024](https://arxiv.org/abs/2310.11511))
fine-tunes Llama2-7B/13B to emit reflection tokens deciding whether to retrieve, whether a passage is
relevant, and whether output is supported. **[SHIPS A GENERATOR]** Against a retrieval-augmented Llama2 of
the same size: **+16.7 PopQA (54.9 vs 38.2)**, **+23.9 TriviaQA (66.4 vs 42.5)**, **+42.4 PubHealth (72.4 vs
30.0)**, **+19.3 ARC-Challenge (67.3 vs 48.0)**; 13B reaches 55.8 / 69.3 / 74.5 / 73.1. Its own ablation
bounds retrieval's share: removing retrieval drops PopQA from 45.5 to 24.7.

CRAG ([Yan et al., arXiv 2401.15884, 2024-01-29](https://arxiv.org/abs/2401.15884)) inserts a lightweight
retrieval evaluator that labels evidence correct / incorrect / ambiguous and dispatches to
decompose-recompose, web search, or both. **[SHIPS A GENERATOR]** On the Self-RAG-Llama2-7B backbone it
beats plain RAG by **+7.0 PopQA (59.8 vs 52.8)**, **+14.9 Bio FactScore (74.1 vs 59.2)**, **+36.6 PubHealth
(75.6 vs 39.0)**, **+15.4 ARC (68.6 vs 53.2)**; stacked on Self-RAG it adds **+6.9 PopQA (61.8 vs 54.9)**
and **+5.0 Bio FactScore (86.2 vs 81.2)**. Two caveats: the "incorrect" branch calls the **Google Search
API** — precisely the hosted dependency a local engine refuses — and the evaluator is a fine-tuned
T5-large model sitting on top of a generator-centric pipeline.

### 2.1.3 Trained retrieval chains — CoRAG

CoRAG ([Wang et al., arXiv 2501.14342, 2025-01-24, NeurIPS 2025](https://arxiv.org/abs/2501.14342)) trains
an 8B model to emit sub-queries and intermediate answers as an explicit chain, with rejection sampling to
synthesise training chains. **[SHIPS A GENERATOR]** Its cleanest comparison is against a fine-tuned
Llama-3.1-8B sharing the *same* E5-large retriever but single-step: EM **+15.5 on 2WikiMultihopQA (70.6 vs
55.1)**, **+4.1 HotpotQA (54.4 vs 50.3)**, **+10.3 MuSiQue (27.7 vs 17.4)**, **+7.2 Bamboogle (48.0 vs
40.8)** at L=6; at L=10 with best-of-8, 72.5 / 56.3 / 30.9 / 54.4. Retrieval itself improves: **Recall@10
72.1 vs 59.1 (HotpotQA), 81.4 vs 54.9 (2Wiki)**. Cost is explicit — L retrieval steps per query, with the
paper's Figure 5 showing early stopping trades tokens back for EM.

### 2.1.4 RL-trained retrieval policies — the Search-R1 family

The dominant 2025–26 line puts the search call inside the reasoning trace and optimises it with
outcome-reward RL. All **ship a generator**; the policy *is* the generator.

| System | Source | Measured gain | Baseline |
|---|---|---|---|
| Search-R1 | [arXiv 2503.09516, COLM 2025](https://arxiv.org/abs/2503.09516) | Qwen2.5-7B-base 7-dataset average **0.431 vs 0.304**; 3B **0.303 vs 0.270** | plain RAG, same retriever |
| Search-R1 | same, Table 2 | 0.431 vs **0.348** (rejection-sampling SFT) and **0.239** (IRCoT) | strongest SFT baseline |
| R1-Searcher | [arXiv 2503.05592, 2025-03-18](https://arxiv.org/abs/2503.05592) | Llama-3.1-8B HotpotQA **ACCR 0.648 / ACCL 0.746 vs 0.468 / 0.506**; 2Wiki 0.594/0.628 vs 0.554/0.534 | ReARTeR, same backbone |
| ZeroSearch | [arXiv 2505.04588, 2025-05-07](https://arxiv.org/abs/2505.04588) | a **7B simulated** retriever matches a real search engine; **14B exceeds** it | real search engine |
| ReSearch / WebSailor | [arXiv 2503.19470, NeurIPS 2025](https://arxiv.org/abs/2503.19470) · [arXiv 2507.02592](https://arxiv.org/abs/2507.02592) | abstract-level claims only; no numeric delta in text | — |

Structural facts outlast the leaderboard: Search-R1 reports that masking retrieved tokens out of the loss
"consistently outperforms" the unmasked variant; R1-Searcher's RL uses only **8,148 training samples** yet
generalises out of domain; ZeroSearch exists because **hundreds of thousands of rollouts** made on-policy
training unaffordable. This family's central cost is *generator rollouts* — which a non-generative engine
never pays.

### 2.1.5 What "deep research" is actually measured on

All three flagship benchmarks grade reports or answers, never retrieval:

- **BrowseComp** ([Wei et al., arXiv 2504.12516, 2025-04-16](https://arxiv.org/abs/2504.12516)): 1,266
  hard-to-find web answers. Human trainers solved **29.2%**; GPT-4o **0.6%**, GPT-4o with browsing
  **1.9%**, o1 **9.9%**, OpenAI Deep Research **51.5%** — with **calibration error 91%** (GPT-4o: 69%).
  Browsing tools *increased* overconfidence. The strongest 2025 system was right about half the time and
  knew it almost never.
- **DeepResearch Bench** ([Du et al., arXiv 2506.11763, 2025-06-13](https://arxiv.org/abs/2506.11763)):
  RACE report-quality scores — Gemini-2.5-Pro Deep Research **48.88**, OpenAI Deep Research **46.98**,
  Perplexity Deep Research **42.25**, Claude-3.7-Sonnet w/ Search **40.67**.
- **2026 state**: [Gemini 3.1 Pro model card, results "as of February 2026", accessed
  2026-10-29](https://deepmind.google/models/model-cards/gemini-3-1-pro/) reports BrowseComp **85.9%**
  with search + Python + browsing **[vendor-reported]** — a large genuine advance, but every entry in
  that table is a generator with a live tool.


---

### 2.2 Query decomposition, expansion, and HyDE

**HyDE** ([Gao et al., arXiv 2212.10496, 2022-12-20, ACL 2023](https://arxiv.org/abs/2212.10496)) prompts an
instruct LLM for a *hypothetical answer document*, embeds it, and retrieves against the embedding.
**[SHIPS A GENERATOR — strictly]** It is the largest single retrieval-side gain in this section, and largest
exactly where lexical match fails: TREC DL19 nDCG@10 **44.5 → 61.3** for zero-shot Contriever (DL20 42.1 →
57.9; recall@1k 74.6 → 88.0), and on BEIR low-resource **SciFact 64.9 → 69.1, Arguana 37.9 → 46.6,
TREC-COVID 27.3 → 59.3, TREC-NEWS 34.8 → 44.0**. No retrieval-side equivalent exists: the hypothetical
document is a sampled completion.

Query *decomposition* proper has no clean standalone ablation: its measured effect appears inside CoRAG
(sub-queries, §2.1.3), IRCoT (sentence-level re-querying, §2.1.1) and CRAG's decompose-then-recompose
branch. Any claim that decomposition alone yields X points is unsupported.


---

### 2.3 GraphRAG and its descendants

**GraphRAG** ([Edge et al., arXiv 2404.16130, 2024-04-24](https://arxiv.org/abs/2404.16130)) builds an
LLM-extracted entity graph, partitions it into a Leiden community hierarchy, and pre-summarises every
community. **[SHIPS A GENERATOR]** Its quality claim is a win rate, not an accuracy: over ~1M-token podcast
transcripts and ~1.7M-token news, community summaries beat map-reduce summarisation of the raw source on
comprehensiveness at **57% (Podcast, intermediate level, p<.001)** and **64% (News, low-level, p<.001)**;
diversity **57% (p=.036)** and **60% (p<.001)**.

Cost cuts both ways. Query-side, root-level (C0) summaries need **26,657 tokens/query versus 1,014,611**
for map-reduce over source text on Podcast (**2.6% of max**; News 39,770 vs 1,709,694) — a **9×–43×**
reduction. Index-side, building the Podcast index took **281 minutes** on `gpt-4-turbo` against a 2M-TPM
endpoint for a ~1.1M-token corpus; Microsoft's own explainer puts vectorising *Wizard of Oz* at
**$0.0056** against a graph build orders of magnitude above it, with query latency **~20–24 s** (first
retrieval stage 10–15 s) ([Microsoft, "GraphRAG Costs Explained", accessed
2026-10-29](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/graphrag-costs-explained-what-you-need-to-know/4207978))
**[vendor-reported]**. Its per-model dollar tables are published as images and did not survive text
extraction, so **the exact indexing $ figures are UNVERIFIED**.

**LazyGraphRAG** ([Microsoft Research blog, 2024-11-25, accessed
2026-10-29](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/))
defers all LLM use to query time. **[vendor-reported]** Published claims: indexing cost identical to vector
RAG and **0.1% of full GraphRAG**; comparable global-query quality to GraphRAG global search at **>700×
lower query cost**; at **4% of GraphRAG global C2 query cost** it "significantly outperforms all competing
methods" on local and global queries. No per-dataset breakdown or replication exists — marketing-grade.

**LightRAG** ([arXiv 2410.05779, 2024-10-08](https://arxiv.org/abs/2410.05779)) replaces community
summarisation with dual-level entity/topic retrieval, reporting win rates of **84.8% vs 15.2% overall on
the Legal corpus** (67.6%/32.4% on Agriculture) and a cost edge: GraphRAG's retrieval used **610 level-2
communities × ~1,000 tokens = 610,000 tokens/query plus hundreds of API calls**, versus **<100 tokens and
one API call**. *Flag:* those are LLM-judge win rates on generated answers. An independent re-run inside
[HippoRAG 2 (arXiv 2502.14802, 2025-02-20, ICML 2025)](https://arxiv.org/abs/2502.14802), holding reader
(Llama-3.3-70B) and retriever (NV-Embed-v2) fixed, scored **LightRAG 6.6 average F1 vs GraphRAG 49.6** —
so LightRAG quality claims are judge- and benchmark-dependent.

**HippoRAG** ([Gutiérrez et al., arXiv 2405.14831, 2024-05-23, NeurIPS 2024](https://arxiv.org/abs/2405.14831))
is the most retrieval-only of the family: OpenIE triples + synonym edges + Personalized PageRank give
**single-step** multi-hop retrieval, **+7.3 average R@5 (72.9 vs 65.6)** over ColBERTv2, with 2Wiki at
**89.1 vs 68.2 R@5**. It matches multi-step IRCoT while being **10–30× cheaper and 6–13× faster online**
(§2.1.1), paying **≈10× slower and ~$15 more expensive offline indexing per 10,000 passages**. Critically
for local deployment, **Llama-3.1-8B-Instruct OpenIE is competitive with GPT-3.5 on all datasets except
2Wiki**, at $0 API cost and 120 min per 10k passages on vLLM.

**HippoRAG 2** ([arXiv 2502.14802](https://arxiv.org/abs/2502.14802)) fixes v1's regression on simple
factual recall by adding passage nodes to the PPR graph: **average F1 59.8 vs 57.0 (NV-Embed-v2, +2.8)**,
**average recall@5 78.2 vs 73.4 (+4.8)**, and an abstract headline of **+7% on associative-memory tasks**
over the prior state of the art.

**G-Retriever** ([arXiv 2402.07630, 2024-02-12, NeurIPS 2024](https://arxiv.org/abs/2402.07630)) matters for
cost: Steiner-tree subgraph retrieval cuts WebQSP input from **100,627 tokens / 1,371 nodes to 610 tokens /
18 nodes (↓99%)** and epoch time 18.7 → 6.2 min, while WebQSP accuracy still reaches **70.49 vs 57.05** for
GraphToken under a frozen LLM.


---

### 2.4 Adaptive and routed retrieval: when to retrieve at all

This branch transfers best to a non-generative engine, because the decision is a classifier, not a
completion.

- **SKR** ([Wang et al., arXiv 2310.05002, 2023-10-08, EMNLP 2023 Findings](https://arxiv.org/abs/2310.05002))
  decides per question whether to retrieve from one of four self-knowledge signals (prompt, in-context
  examples, trained classifier, kNN over seen questions). **Retrieval-only as a component.** On five QA
  sets with `gpt-3.5-turbo-0301`, SKR-kNN averaged **70.62 vs 66.52** for the best no-retrieval CoT baseline
  and **67.15** for IRCoT — beating both always-retrieve and never-retrieve.
- **Adaptive-RAG** ([Jeong et al., arXiv 2403.14403, 2024-03-21, NAACL 2024](https://arxiv.org/abs/2403.14403))
  gives the cleanest cost/quality curve, reporting steps and wall-clock beside accuracy on FLAN-T5-XL.
  HotpotQA: single-step **36.40 acc at 1.00 step / 1.00 time**, multi-step **47.00 at 5.53 / 9.38**,
  Adaptive-RAG **44.40 at 3.55 / 5.99**, oracle **54.80 at 1.59 / 2.77** — roughly three-quarters of the
  multi-step gain at 64% of its steps. On 2WikiMultihopQA it is much weaker (**46.40 vs 55.40**): router
  errors cost real accuracy on genuinely multi-hop queries.
- **DRAGIN** ([Su et al., arXiv 2403.10081, 2024-03-15, ACL 2024](https://arxiv.org/abs/2403.10081)) triggers
  retrieval on token-level uncertainty plus attention from the generator's own state. **[SHIPS A
  GENERATOR]** Llama2-13B-chat: HotpotQA EM **0.314 vs 0.267** (FS-RAG), 2Wiki **0.304 vs 0.270**.
  Strictly unavailable without a generator — it reads per-token probabilities.
- **Routing between retrieval and long context**: SELF-ROUTE ([arXiv 2407.16833,
  2024-07-23](https://arxiv.org/abs/2407.16833)) routes by self-reflection at **38.4%/61.4% of the tokens**
  (§2.5). Routing *between retrievers* (dense vs BM25 vs graph) is essentially unmeasured in the
  peer-reviewed literature — a gap worth naming.


---

### 2.5 Long context as an alternative to retrieval

**Claimed window ≠ usable window.** RULER ([Hsieh et al., arXiv 2404.06654, 2024-04-09,
ICLR 2024](https://arxiv.org/abs/2404.06654)) tested 17 models on 13 tasks: nearly all score near-perfect
on vanilla needle-in-a-haystack, yet **only half of the models claiming ≥32K maintain satisfactory
performance at 32K**. HELMET ([Yen et al., arXiv 2410.02694, 2024-10-03,
ICLR 2025](https://arxiv.org/abs/2410.02694)) ran **59 long-context models** to 128K and found NIAH does not
predict downstream performance. InfiniteBench ([arXiv 2402.13718, ACL
2024](https://arxiv.org/abs/2402.13718)): across 12 tasks averaging >100K tokens, **GPT-4 averaged 45.63**.

**Remove the lexical crutch and it collapses.** NoLiMa ([arXiv 2502.05167,
2025-02-07](https://arxiv.org/abs/2502.05167)) redesigned the needle so question and needle share almost no
tokens: at 32K, **11 of 13 models claiming ≥128K fell below 50% of their own short-context baseline**, and
**GPT-4o dropped from 99.3% to 69.7%**. Chroma's *Context Rot* ([2025-07, accessed
2026-10-29](https://research.trychroma.com/context-rot)) **[lab-reported]** extends this to 18 production
LLMs including GPT-4.1, Claude 4, Gemini 2.5 and Qwen3: performance degrades **non-uniformly** with input
length even on minimal tasks, distractors drive it, and on LongMemEval (~113K-token prompts) focused inputs
beat full inputs. Claude models abstained most under ambiguity; GPT models hallucinated most — abstention
under uncertainty was a measurable family difference, not a solved problem.

**The 2026 vendor datapoint is decisive.** Gemini 3.1 Pro's card advertises a **1M-token window** but
scores **84.9% on MRCR v2 (8-needle) at 128K versus 26.3% at 1M (pointwise)**, with the 1M column marked
"Not supported" for every competitor ([model card, results as of February 2026, accessed
2026-10-29](https://deepmind.google/models/model-cards/gemini-3-1-pro/)) **[vendor-reported]**. A 3×
collapse on the benchmark built to measure exactly this is the strongest published argument that stuffing
the window is not a retrieval substitute in 2026.

**Head-to-head, and cost per query.** Long context wins on quality when you can afford it and the corpus is
small. [Li et al., arXiv 2407.16833, 2024-07-23 (Google DeepMind)](https://arxiv.org/abs/2407.16833), 21
datasets: Gemini-1.5-Pro long-context **49.70 vs 37.33** for RAG, GPT-4o **48.67 vs 32.60**, GPT-3.5-Turbo
**32.07 vs 30.33** — yet SELF-ROUTE reached 46.41 (Gemini) and 48.89 (GPT-4o, statistically level with long
context) on **38.4% and 61.4% of the tokens**. [LaRA, arXiv 2502.09977,
2025-02-14](https://arxiv.org/abs/2502.09977) complicates it: at **32K** long context beat RAG for nearly all
models, at **128K** the ranking **reversed for weaker models**, there is "no universal winner", and RAG was
**more robust to answer position**. [Lost in the Middle, arXiv 2307.03172, 2023-07-06,
TACL 2024](https://arxiv.org/abs/2307.03172) is the origin: GPT-3.5-Turbo multi-document QA fell **>20%**
with the answer mid-context, and in the worst 20- and 30-document settings dropped **below its own
closed-book 56.1%**. [LongBench v2, arXiv 2412.15204, 2024-12-19](https://arxiv.org/abs/2412.15204) sets the
ceiling: 503 questions, 8K–2M words; human experts with search reach **53.7%** in 15 minutes, the best
direct-answer model **50.1%**, o1-preview **57.7%**.


---

### 2.6 Context compression and pruning

This is where a non-generative engine gains most, because the pruning signal can come from a small encoder
rather than a generator.

- **LLMLingua** ([arXiv 2310.05736, 2023-10-09, EMNLP 2023](https://arxiv.org/abs/2310.05736)) drops
  low-information tokens by small-LM perplexity: GSM8K full-shot **78.85 EM at 2,366 tokens** versus
  **77.33 EM at 20× compression** (117 tokens) — **−1.52 EM for 20× fewer tokens**. BBH is less free:
  70.07 → 56.85 (**−13.2 EM**) at 7×.
- **LongLLMLingua** ([arXiv 2310.06839, 2023-10-10, ACL 2024](https://arxiv.org/abs/2310.06839)) adds
  question-aware contrastive perplexity and reordering: **up to +21.4% on NaturalQuestions with ~4×
  fewer tokens**, **94.0% cost reduction** on LooGLE, **1.4×–2.6× latency** reduction at 2×–6× on
  ~10K-token prompts.
- **LLMLingua-2** ([arXiv 2403.12968, 2024-03-19, ACL 2024 Findings](https://arxiv.org/abs/2403.12968))
  reframes compression as token classification distilled from GPT-4, removing the generation loop:
  **3×–6× faster compression**, **1.6×–2.9× end-to-end latency** at **2×–5×** ratios.
- **Selective Context** ([arXiv 2310.06201, 2023-10-09, EMNLP 2023](https://arxiv.org/abs/2310.06201)):
  **50% context reduction** → **36% less inference memory**, **32% less time**, for **−0.023 BERTScore**
  and **−0.038 faithfulness** — the best quality-per-point-of-reduction here.
- **RECOMP** ([arXiv 2310.04408, 2023-10-06, ICLR 2024](https://arxiv.org/abs/2310.04408)) is widely
  misquoted: the famous "6%" is the **oracle** compressor. *Trained* compressors reach **25% for language
  modelling** and **5–10% of original tokens for QA with a <10% relative drop**; on HotpotQA the extractive
  compressor hit **11% compression losing 2.4 EM**. It also implements *selective augmentation* — returning
  an empty string when retrieval adds nothing — conceptually Laya's abstention.
- **Provence** ([Dejean et al., arXiv 2501.16214, 2025-01-27](https://arxiv.org/abs/2501.16214)) is the
  current practical state of the art and the most CPU-friendly: pruning is folded into a **DeBERTa-v3
  reranker** as sequence labelling, costing "almost nothing" in an existing pipeline. It prunes **49% of
  context with negligible to no drop** across domains, and its efficiency table is the useful part —
  **25 s / 7.9×10¹⁴ MFLOPS** standalone against **194 s / 5.4×10¹⁶** (LongLLMLingua), **38 s / 9.8×10¹⁵**
  (LLMLingua-2) and **499 s** (abstractive RECOMP). Generation speed-up is modest: **×1.2–1.4 at batch 1,
  ×1.9–2.0 at batch 256**.

**Counterweight.** Chroma's report is the empirical basis for claiming pruning *improves* quality, not just
cost: focused inputs beat full inputs at equal model. It is a lab report, but its methodology (aligned
GPT-4.1 judge, 194,480 calls) is transparent.


---

### 2.7 What this means for a non-generative engine

- **Available with no generator:** single-step multi-hop graph retrieval (HippoRAG's PPR-over-OpenIE,
  +7.3 avg R@5, matching IRCoT at 10–30× lower online cost, with a **local Llama-3.1-8B OpenIE indexer**);
  subgraph selection for cost (G-Retriever, −99% tokens); reranker-folded pruning (Provence, −49% context
  in ~25 s, 10–100× cheaper than LLMLingua-family pruners); small-classifier routing (SKR, Adaptive-RAG).
- **Available but degraded:** GraphRAG-style community summarisation — the index needs an LLM to write
  summaries, so a local generator is mandatory and 281 minutes per 1.1M tokens is the realistic order of
  magnitude. LazyGraphRAG's defer-to-query-time shape suits a CPU-only engine, but 0.1%/700× are
  vendor-reported and unreplicated.
- **Strictly off-limits:** Self-RAG reflection tokens, CRAG's trained evaluator, CoRAG chains, DRAGIN's
  uncertainty triggers, and the Search-R1 / R1-Searcher / ReSearch / ZeroSearch family. Each conditions on
  or optimises through a generator's next-token distribution; their headline numbers (0.431 vs 0.304;
  0.648 vs 0.468) are unattainable by construction.
- **HyDE is the sharpest loss:** the largest retrieval-side gain here (+16.8 nDCG@10 on DL19; +32 points on
  TREC-COVID), purely generator-dependent. The honest substitute — lexical+dense fusion plus reranking —
  gains less.
- **Abstention is an evidenced design choice, not a hedge:** BrowseComp's best 2025 system was right 51.5%
  of the time with 91% calibration error; LongBench v2's human ceiling is 53.7%; NoLiMa shows confident
  failure at 32K in 11 of 13 models.
- **Long context is settled on cost in retrieval's favour, unsettled on quality:** it wins when resourced
  (49.70 vs 37.33) at 1.6–2.6× the tokens even after routing, and the 1M window is not usable (MRCR v2:
  84.9% at 128K, 26.3% at 1M). A CPU-only engine can host neither, so the real comparison is retrieval
  versus a remote API — where retrieval's case is strong.
- **Iteration must be retriever-driven.** CoRAG (Recall@10 72.1 vs 59.1) and IRCoT (+7.9 to +22.6 recall)
  show extra rounds raise *retrieval* metrics directly. A non-generative engine can re-query on coverage
  gaps, entity linking and PPR expansion — it just cannot use generated prose as the query.


---

#### Verification notes

1. **GraphRAG per-dataset indexing dollars — UNVERIFIED.** Microsoft's "GraphRAG Costs Explained" publishes
   its token and cost tables as images; HTML fetch + BeautifulSoup extraction recovered only headings, one
   figure (38,371 tokens), the $0.0056 anchor and 20–24 s latency.
2. **ReSearch and WebSailor deltas.** Both abstracts make capability claims with no numeric delta in text
   ("generalises across benchmarks"; "matching proprietary agents"); result tables were not mined. Read as
   directional only.
3. **No 2026 peer-reviewed long-context-vs-RAG result was found.** Nothing superseding LaRA (2025) or Li et
   al. (2024) appears in the ACL/NeurIPS/ICLR pipeline; §2.5's 2026 evidence is a vendor model card plus a
   lab report. Searched arXiv title queries and web search for 2026 RAG surveys — the most recent
   ([arXiv 2610.01936, 2026-10-01](https://arxiv.org/abs/2610.01936)) is taxonomic, with no new
   measurements.
4. **Self-RAG's retrieval-frequency savings** appear only as a figure against threshold δ with no tabulated
   "X% fewer retrievals at Y% accuracy", so none is quoted.

---

## 3. Faithfulness, Attribution and Abstention — the measurement state of the art (late October 2026)

*Every figure below was read from the cited source in this session and is cached under `.scratch/research/src/faith-*.txt`. Vendor-reported numbers are marked inline. Unverified items carry an explicit marker.*


---

### 3.0 The crux, stated first

Almost every headline number in this literature is measured **on generated text**. A system that does not generate text cannot be entered into most of these benchmarks, and saying so is more honest than adapting a benchmark until it produces a number.

| Instrument | What it scores | System must emit | Runnable without a generator? |
|---|---|---|---|
| **HHEM-2.1-Open**, MiniCheck, AlignScore, Granite Guardian groundedness | Consistency of one `(evidence, claim)` pair | A **claim** — need not be generated | **Yes**, given claims. This is claim adjudication |
| **RAGAS Faithfulness** | Fraction of a *response's* claims entailed by context | A natural-language response | **No** |
| **Lynx / HaluBench**, **FaithJudge / FaithBench** | Hallucination label on `(document, summary or answer)` | A summary or answer | **No** |
| **RGB negative rejection** | Whether the model emits a refusal string when only noise is supplied | Free text matching a refusal template | **No** |
| **AbstentionBench** | Abstention recall/F1 on unanswerable + correctness on answerable | Free text, LLM-judged | **No** |
| **ALCE / Liu et al. citation P & R** | Whether each cited passage supports its generated statement | Text with **inline citations** | **No** |
| **Conformal filtering of context** (ECIR 2026) | Coverage of relevant snippets retained | Nothing — operates on the retrieved set | **Yes** |
| **Selective prediction** (AURC, risk–coverage, ECE) | Whether a score ranks away its own errors | One scalar per query | **Yes** |

A non-generative engine can therefore be scored on **claim-level faithfulness, evidence coverage, and selective-prediction quality**, and cannot be scored on **answer-level faithfulness, citation precision/recall, or any rejection metric defined over generated strings**.


---

### 3.1 Hallucination detectors with published numbers

#### Vectara HHEM — vendor-reported, every time

HHEM is a fine-tuned `flan-t5-base` (110M parameters; [model card](https://huggingface.co/vectara/hallucination_evaluation_model)) scoring a `(premise, hypothesis)` pair with the template `<pad> Determine if the hypothesis is true given the premise?`. **Every HHEM number in this subsection is vendor-reported by Vectara, which also sells the successor model and runs the leaderboard HHEM powers.**

HHEM-2.1-Open, the last open-weight release:

| Benchmark | Balanced acc. | F1 | Recall | Precision |
|---|---|---|---|---|
| AggreFact-SOTA | 76.55% | 66.77% | 68.48% | 65.13% |
| RAGTruth-Summ | 64.42% | 44.83% | 31.86% | 75.58% |
| RAGTruth-QA | 74.28% | 60.00% | 54.38% | 66.92% |

Two things the card underplays. **HHEM-2.1-Open is worse than HHEM-1.0 on AggreFact-SOTA** (76.55% vs 78.87% balanced accuracy), and RAGTruth-Summ recall of **31.86%** means it misses roughly two summarization hallucinations in three. Its efficiency claim has aged best: under 600 MB RAM at 32-bit, ~1.5 s for a 2k-token input on a modern x86 CPU. HHEM-2.3 is now current and **not open-weight**. The [LLM Hallucination Leaderboard](https://github.com/vectara/hallucination-leaderboard) was **last updated 22 October 2026**, ranks roughly 111 models, and its best hallucination rate is **1.8%** (`antgroup/finix_s1_32b`), then 3.1% (`openai/gpt-5.4-nano-2026-03-17`) and 3.3% (`google/gemini-2.5-flash-lite`). That rate is *defined* as 100% minus HHEM's consistency rate — a vendor instrument reading, not ground truth.

#### The independent re-scoring that sets the ceiling: FaithJudge / FaithBench

Tamber, Bao, Xu, Luo, Kazi, Bae, Li, Mendelevitch, Qu & Lin, *Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards* ([EMNLP 2025 Industry](https://aclanthology.org/2025.emnlp-industry.54/)), Waterloo + Vectara — partly the vendor scoring its own model. Claim-wise mode, accuracy / F1-macro:

| Method | Params | AggreFact-SOTA | RAGTruth-Summ | TofuEval-MB | **FaithBench** | Average |
|---|---|---|---|---|---|---|
| HHEM-1.0-Open | 184M | 76.0 / 71.0 | 66.2 / 52.2 | 54.4 / 49.9 | 59.3 / 58.8 | 64.0 / 58.0 |
| **HHEM-2.1-Open** | 110M | 73.2 / 69.7 | 67.7 / 56.1 | 60.9 / 61.2 | **66.7 / 63.7**\* | 67.1 / 62.7 |
| AlignScore-large | 355M | 73.9 / 69.3 | 67.9 / 54.1 | 56.1 / 52.9 | 62.8 / 59.6 | 65.2 / 59.0 |
| MiniCheck-RoBERTa-L | 355M | 75.7 / 72.5 | 70.5 / 58.6 | 67.6 / 68.5 | 61.6 / 60.0 | 68.8 / 64.9 |
| Bespoke-MiniCheck | 7B | 74.3 / 70.1 | 73.3 / 62.9 | 76.9 / 78.4 | 60.1 / 58.0 | 71.2 / 67.3 |
| GPT-4o, RAGAS prompt | — | 75.9 / 70.3 | 75.7 / 63.7 | 75.2 / 76.7 | 65.3 / 59.0 | 73.0 / 67.4 |

\* asterisked FaithBench cells are models used to *select* FaithBench's adversarial examples, so they read pessimistically there.

Summary-wise, the best average in the table is GPT-4o with the Luo et al. prompt at **77.4 balanced accuracy / 71.9 F1-macro**; o3-mini-high at 76.2 / 71.2. The paper's own verdict: the best claim-wise methods sit "below 78%" balanced accuracy and "below 72%" F1-macro; on FaithBench specifically the best balanced accuracy is **68.8%** (o3-mini-high) and best F1-macro **63.7%** (HHEM). Their FaithJudge approach — an LLM judge primed with human-annotated hallucination examples — reaches **84% balanced accuracy / 82.1 F1-macro**.

**That is the ceiling.** On adversarially-chosen real hallucinations, a purpose-built 110M detector and a frontier LLM judge both land in the mid-60s to low-70s balanced accuracy. High-80s claim-level detection on in-distribution data is a claim about the template, not about hallucination.

MiniCheck ([Tang et al., EMNLP 2024](https://arxiv.org/abs/2404.10774)) claims GPT-4-level accuracy at 400× lower cost with MiniCheck-FT5 (770M); the independent re-scoring above is the better number to quote. **Patronus Lynx** ([Ravi et al., arXiv 2407.08488](https://arxiv.org/abs/2407.08488), 2024; [card](https://huggingface.co/PatronusAI/Llama-3-Patronus-Lynx-70B-Instruct)) reports **87.4% overall accuracy on HaluBench** at 70B and 82.9% at 8B, against GPT-4o 86.5%, GPT-4-Turbo 85.0%, GPT-3.5-Turbo 58.7% — **vendor-reported by Patronus AI**, whose product competes with RAGAS-style pipelines; the same table scores **RAGAS Faithfulness at 66.9%**. I found no 2025–2026 Lynx successor with published numbers (**UNVERIFIED** whether one exists). **IBM Granite Guardian 3.1-8B** ([card](https://huggingface.co/ibm-granite/granite-guardian-3.1-8b)) exposes `groundedness` / `context_relevance` / `answer_relevance` with **AUC 0.86 average** on the TRUE RAG-hallucination suite (0.72 `mnbm` to 0.93 `dialfact`) — **vendor-reported by IBM**.

#### RAGAS Faithfulness

`Faithfulness = (# claims in the response supported by context) / (total # claims in the response)` ([docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)). It is **answer-level by construction** — step 1 is "identify all the claims **in the response**". The evidence on human agreement is genuinely contested: the original paper ([Es et al., 2309.15217](https://arxiv.org/abs/2309.15217)) reports **0.95 pairwise agreement with humans on faithfulness** on its own WikiEval set — but that is preference between two candidate answers, on a small purpose-built dataset, where annotators agreed with each other "in around 90%" of cases. Against that, Patronus measured RAGAS at **66.9% on HaluBench**, and RAGAS-prompted judges average 65–73% balanced accuracy in the FaithJudge table. **Honest summary: a cheap, directionally-correct signal whose absolute human agreement is reported between 66% and 95% depending on who measured it.**


---

### 3.2 Attribution: measured numbers and the ceiling

**ALCE** (Gao, Yen, Yu, Chen; [2305.14627](https://arxiv.org/abs/2305.14627), 2023) defines the metrics everyone now uses. **Citation recall** = fraction of generated statements for which at least one cited passage entails the statement; **citation precision** = fraction of individual citations that support their statement. Both via NLI over decomposed sub-claims.

| Dataset | System | Recall | Precision |
|---|---|---|---|
| ASQA | ChatGPT vanilla, 5 passages | 73.6 | 72.5 |
| ASQA | ChatGPT + rerank | **84.8** | **81.6** |
| ELI5 | ChatGPT vanilla, 5 passages | 51.1 | 50.0 |
| ELI5 | ChatGPT + rerank | **69.3** | **67.8** |
| QAMPARI | ChatGPT vanilla | 57.2 | 12.0 |
| ASQA | LLaMA-13B, 3 passages | 10.6 | 15.4 |

**Liu, Zhang & Liang, *Evaluating Verifiability in Generative Search Engines*** ([2304.09848](https://arxiv.org/abs/2304.09848), 2023) is the human-annotated version over four commercial systems — Bing Chat 58.7 recall / **89.5** precision; NeevaAI 67.6 / 72.0; perplexity.ai **68.7** / 72.7; YouChat 11.1 / 63.6; **average 51.5 / 74.5**, with a near-58-point spread between best and worst recall. The 2026 reading of this pair: **citation precision is a solved-ish problem at 72–90%; citation recall is not, sitting at 51–70% even for frontier systems.** A system claiming 90%+ recall should expect to be asked how its claim decomposition was validated. Notably NeevaAI was the only one of the four that declined to answer some queries — abstention was already a search-engine differentiator in 2023.

The 2025 successor, **CiteEval / CiteBench** ([ACL 2025 long](https://aclanthology.org/2025.acl-long.1574/), Amazon), argues that "binary or ternary supportiveness from cited sources … is a suboptimal proxy for citation evaluation" and grades against the cited source *plus* the full retrieval context, query and generated text. Its contribution is the framework and CiteEval-Auto's human correlation; **UNVERIFIED:** the headline correlation coefficient, which I did not extract, so none is quoted.


---

### 3.3 Abstention as a first-class metric

#### AbstentionBench (FAIR at Meta, 2025)

Kirichenko, Ibrahim, Chaudhuri & Bell, [2506.09038](https://arxiv.org/abs/2506.09038), posted 10 June 2025, revised 24 August 2026. Twenty datasets spanning unknown answers, underspecification, false premises, subjective interpretation and stale information; 20 frontier LLMs.

Headline: **"reasoning fine-tuning degrades abstention (by 24% on average), even for math and science domains on which reasoning models are explicitly trained."** Reasoning models "hallucinat[e] missing context and provid[e] definitive final answers even when their reasoning chains express uncertainty." Post-training is not monotone — abstention rises through SFT and DPO but **RLVR degrades it**.

The released [`abstention_performance.csv`](https://github.com/facebookresearch/AbstentionBench/blob/main/analysis/abstention_performance.csv) permits a per-model view. Mean abstention recall over datasets — **my computation from the released file, not a figure printed in the paper**:

| Model | Mean abstention recall | n |
|---|---|---|
| Qwen2.5 32B | **0.731** | 31 |
| GPT-4o | 0.722 | 31 |
| Gemini 1.5 Pro | 0.704 | 31 |
| o1 | 0.702 | 31 |
| Llama 3.1 8B Instruct | 0.689 | 31 |
| Llama 3.1 70B Instruct | 0.670 | 31 |
| **DeepSeek R1 Distill Llama 70B** | **0.475** | 31 |
| s1.1 32B | 0.441 | 31 |
| Llama 3.1 8B Base | 0.441 | 30 |

The paired comparison carrying the paper's argument shows up cleanly: DeepSeek R1 Distill Llama 70B at **0.475** against its non-reasoning sibling Llama 3.1 70B Instruct at **0.670**. The paper summarises the range as "from near-perfect performance on BIG-Bench Known Unknowns, down to near-zero recall on MediQ."

**What the system must output to be graded:** free text, classified by an LLM judge. The paper validates several and picks **Llama 3.1 8B Instruct** as cheapest at equal accuracy (**acc 0.88, F1 0.85, precision 0.86, recall 0.83**); the judge defines abstention as an "explicit expression of uncertainty" — refusing, questioning the premise, or flagging a mistaken assumption; answering as written without caveats is *not* abstention. Correctness on answerable questions is judged separately, so AbstentionBench **does** grade abstention apart from answer quality — but both halves need generation.

#### RGB (Chen, Lin, Han & Sun; AAAI 2024)

[2309.01431](https://arxiv.org/abs/2309.01431). The negative-rejection testbed supplies **only noise documents** and measures refusal. Two metrics: **Rej**, exact match against the instructed refusal string, and **Rej\***, a ChatGPT judgement of whether a rejection appears. Table 3, negative rejection (%):

| Model | EN Rej | EN Rej\* | ZH Rej | ZH Rej\* |
|---|---|---|---|---|
| ChatGPT | 24.67 | **45.00** | 5.33 | **43.33** |
| ChatGLM2-6B | 10.33 | 41.33 | 6.33 | 36.33 |
| Qwen-7B-Chat | **31.00** | 35.67 | **8.67** | 25.33 |
| Vicuna-7B-v1.3 | 17.00 | 33.33 | 3.37 | 24.67 |
| ChatGLM-6B | 9.00 | 25.00 | 6.33 | 17.00 |
| BELLE-7B-2M | 5.67 | 32.33 | 5.33 | 13.67 |

**The highest rejection rate any of six models reached was 45% (English) / 43.33% (Chinese).** The Rej-vs-Rej\* gap (24.67 vs 45.00 for ChatGPT) is itself a finding: models "often generate unpredictable responses, which make it hard to … use them as state triggers (such as for recognizing rejection)". RGB also reports noise robustness: ChatGPT's accuracy falls **96.33% → 76.00%** from 0 to 0.8 noise, ChatGLM2-6B **91.33% → 57.33%**. **Grading requirement:** a generated string, matched exactly or judged. RGB's own instruction template is the spec — the system must emit the refusal sentence.

#### The Evidence Sufficiency Benchmark (CMC 2026)

Zhang & Wu, *Do LLMs Know When Evidence is Insufficient? An Evidence Sufficiency Benchmark for Answer-Abstention Calibration in RAG*, *Comput. Mater. Contin.* 89(1), 2026, [doi:10.32604/cmc.2026.086343](https://doi.org/10.32604/cmc.2026.086343). This is the paper to point a reviewer at, because it **holds evidence condition fixed and grades the abstention decision as the dependent variable**.

Design: a five-level evidence gradient, **L1 Full Support → L5 Conflicting Evidence** (fully supportive, partially supportive, irrelevant, absent, conflicting); seven LLMs from five families, three prompting strategies. Headline: **under L5 conflicting evidence, all evaluated models show over-answer rates of 65% to 91%.** Models do reduce answer rate as evidence quality falls — the paper's "evidence sufficiency curves" — but abstention "remains unreliable". Human validation on 200 samples supports the automatic scoring; chain-of-thought helps some families and not others. **Grading requirement:** answer-or-abstain, both judged. But *over-answer rate conditional on evidence condition* is the one metric in this literature a non-generative engine could in principle be adapted to, since "answered" can be redefined as "returned a supported verdict" and "abstained" as "returned `no_coverage`". That would be an adapter, and must be declared as one.

#### SQuAD 2.0, and Self-RAG's reflection tokens

SQuAD 2.0 ([Rajpurkar, Jia & Liang, 1806.03822](https://arxiv.org/abs/1806.03822), 2018) adds 50,000+ adversarially-written unanswerable questions and remains the canonical demonstration: **"a strong neural system that gets 86% F1 on SQuAD 1.1 achieves only 66% F1 on SQuAD 2.0."** It is extractive and single-passage, so it measures reading-comprehension abstention, not retrieval-sufficiency abstention. Inside AbstentionBench it appears under underspecified context, where GPT-4o's abstention recall is **0.578** and DeepSeek R1 Distill Llama 70B's **0.483** (from the released CSV).

Self-RAG (Asai et al., [2310.11511](https://arxiv.org/abs/2310.11511), ICLR 2024) defines **ISREL** ∈ {relevant, irrelevant} on a passage, **ISSUP** ∈ {fully supported, partially supported, no support} on a segment given a passage, and **ISUSE** ∈ {1…5}. Its published numbers are citation numbers: ASQA **citation precision 66.9 / recall 67.8** for Self-RAG-7B and **70.3 / 71.3** for 13B, against Ret-ChatGPT's 65.1 / 76.6. **A directly checked negative finding: the words "abstain" and "refuse" appear zero times in the Self-RAG paper.** ISSUP's `no support` label *could* gate a refusal, and later work uses it that way, but Self-RAG reports no abstention rate and should not be cited as an abstention result.

#### Does a benchmark exist that grades correct-evidence **plus** honest-abstention separately from answer quality?

**Partly — three candidates, none cleanly.**

- **AbstentionBench** grades abstention (recall/F1/precision) on unanswerable questions *separately from* response accuracy on answerable ones — but per dataset, not combined into an evidence-quality score, and both halves need generation.
- **The Evidence Sufficiency Benchmark (CMC 2026)** grades answer-versus-abstain *conditional on a controlled evidence condition* — the cleanest separation available — but it is a 2026 journal paper with a small model panel and no public leaderboard as of this writing.
- **RGB** grades negative rejection as its own testbed, but over a refusal-string match on generated output.

I found **no** benchmark that scores "returned the correct evidence" and "abstained honestly when evidence was insufficient" as two *separate* scores over the same query set. The nearest framing in print is a small 2026 study, *Retrieval Quality, Evidence, and Appropriate Abstention in Large Language Models* ([IRE Journals](https://www.irejournals.com/paperhtml/1723870/1723870.html), doi:10.64388/IREV10I4-1723870), whose explicit conclusion is that "factual correctness, evidence faithfulness, and appropriate uncertainty should be evaluated separately." It also reports a striking dissociation: under **low-quality retrieval**, only **3 of 39** responses were fully correct while **36 of 39** showed appropriate abstention. Treat it as an argument, not evidence — the whole study is **13 questions and 117 responses**.


---

### 3.4 Calibration-aware abstention and selective prediction

**Conformal Language Modeling** (Quach et al., [2306.10193](https://arxiv.org/abs/2306.10193), ICLR 2024) calibrates sampling-stop and rejection rules so the returned candidate set "contains at least one acceptable answer with high probability", and marks sentence subsets each independently correct "with statistical guarantees". It is the theoretical anchor for coverage/factuality guarantees — and it is entirely about generator output.

**Conformal filtering for RAG** (Chakraborty, Yang, Khashabi, Lawrie & Duh; [2511.17908](https://arxiv.org/abs/2511.17908), **ECIR 2026**) is the piece a non-generative engine can use. It treats context engineering as a coverage-controlled filter: retain a specified fraction of relevant snippets, discard the rest. On NeuCLIR and RAGTIME with embedding- and LLM-based scorers the filter "consistently meets its target coverage" and **reduces retained context by 2–3×**, with downstream ARGUE F1 improving under strict filtering and stable at moderate coverage. This is the published precedent for promising something about the evidence returned *before any generation happens*.

**CAP: Conformalized Abstention Policies** (Tayebati et al., [PMLR v304](https://proceedings.mlr.press/v304/tayebati26a.html), 2026) learns per-instance abstention over point prediction, set prediction and full abstention. While maintaining **90% target coverage** it reports hallucination-detection **AUROC up to +22.2%**, selective-generation **AUARC +21.2%** and calibration error **down over 70%**, against static conformal baselines — author-reported, and the +22.2% is relative, not an absolute AUROC.

**On what counts as good.** These metrics have no absolute scale. **AURC** is bounded below by a perfect-oracle curve and above by the full no-abstention system's risk; **E-AURC** subtracts the optimal curve so 0 is perfect. AURC's magnitude is a function of base error, so cross-paper comparison is meaningless — **risk at fixed coverage** and **coverage at fixed risk** are the interpretable pair, and a reviewer will ask for both directions. **ECE** plus reliability diagrams are the calibration evidence; anything above ~0.10 reads as poorly calibrated.

For scale, this project's own records give the answerability gate **ROC-AUC 0.8146 [0.7628, 0.8570]**, **balanced accuracy 0.7248**, **AURC 0.2691 [0.2413, 0.3057]**, **AUGRC 0.6681**, **risk at 80% coverage 0.4069 [0.3833, 0.4252]** and **ECE 0.0650 [0.0278, 0.1107]** on a 300-query held-out MultiHop-RAG split (150 null / 150 answerable) — matching `docs/findings-answerability-features.md`. That is a respectable in-distribution selective predictor.

**And the caveat that must travel with it, because a reviewer will find it.** Those same records state that 0.815 is a **within-question-type** number, falling to **0.767 [0.671, 0.827]** with a whole question type held out, and that the gate does **not** transfer out of domain: treating MultiHop-RAG as an out-of-domain split and recalibrating on half of it gives **ROC-AUC 0.509 [0.463, 0.555]** and held-out **balanced accuracy 0.511 [0.472, 0.548]** — chance (`BENCHMARKS.md`, "Measured: negative rejection"). The single-scalar comparator scores **0.520** in the same study, so the defensible claim is a feature-engineering result, not a property of the head.


---

### 3.5 What a 2026 reviewer will demand

Six items, in roughly the order they are asked for:

1. **A fully stated held-out protocol** — split sizes, stratification, within-versus-across question type, confidence intervals. The 0.815-vs-0.767 distinction is exactly what gets a claim downgraded when omitted.
2. **Trivial baselines.** AbstentionBench's best mean recall of 0.73 and RGB's finding that a fixed refusal policy beats every model on Rej\* both mean a raw abstention rate is uninformative without comparison to always-answer, always-abstain and a single-scalar threshold. **0.520** is the number that makes 0.815 meaningful.
3. **A reliability diagram and an ECE number**, not just AUC — AUC is rank-only and says nothing about whether a reported 0.8 means 0.8.
4. **A risk–coverage curve** with both risk-at-fixed-coverage and coverage-at-fixed-risk, plus AURC and preferably E-AURC or AUGRC so the number is oracle-comparable.
5. **Abstention decomposed into precision and recall** — correct abstention *and* over-abstention. AbstentionBench reports abstention F1 and precision specifically because post-training "degrad[es] precision (i.e. over-abstention) for questions about stale data"; high recall with no over-abstention number is half the story.
6. **Out-of-distribution evidence** — the item most often missing and most damaging when found. Given AbstentionBench's result that abstention does not come free with capability, and this project's own record of an in-domain gate scoring at chance out of domain, the defensible form is *"calibrated abstention on domains the gate was calibrated for, with the calibration procedure — not the threshold — as the transferable artefact."*


---

### 3.6 The ceiling, in one place

- Claim-level hallucination detection on adversarially-chosen real data: **best balanced accuracy ~69–77%, best F1-macro ~72%** — purpose-built small models and frontier LLM judges land in the same band (FaithJudge, EMNLP 2025).
- Citation precision **72–90%** (Liu et al. 2023; ALCE 2023); citation recall **51–70%** — the number that has not moved.
- Abstention: best mean recall **~0.73** (AbstentionBench), best negative rejection **45%** (RGB), **65–91% over-answer under conflicting evidence** (CMC 2026), and reasoning fine-tuning **degrades it ~24% on average**.

A system reporting a three-state verdict (`supported` / `contradicted` / `no_coverage`) plus calibrated probabilities operates where the instruments exist. It does not operate where RAGAS faithfulness, citation recall or RGB rejection live — and this section should say so plainly rather than adapt those benchmarks until a number appears.

---

## 4. Evaluation protocol as a differentiator

*Sources accessed 2026-10-29; local copies in `.scratch/research/src/`.*

### 4.1 The reviewer's implicit checklist in 2026

A 2026 reviewer looking at a retrieval/RAG result table asks, in roughly this order:

1. **Is the comparison like-for-like?** Same corpus, same qrels, same cutoffs, same metric code.
   The most common rejection reason in RAG papers is a table whose rows come from different
   corpora or different metric definitions.
2. **Is the difference bigger than the noise?** Paired tests, confidence intervals, and honest
   treatment of many-way comparisons.
3. **Is it contaminated?** Have the queries/answers plausibly been in the generator's or the
   retriever's pretraining data, and what was done about it?
4. **Can I rerun it?** Released code, artefacts, versions, seeds, and a stated compute budget.
5. **Is the claim the claim?** "Best average" vs "best per dataset", "best in class" vs "best in
   class on this envelope".

Points 2–5 are *protocol* advantages, and protocol is the one axis where a small project can beat a
large one on merit rather than on compute.

### 4.2 Contamination: 2026 has made this a first-class obligation

TREC RAG 2026 states the goal outright, as one of four track goals:

> "**Aligning Deep Research Datasets with LLM Pre-Training Corpora** — *Fairness*: Building a testbed
> to fairly evaluate when LLMs are 'memorizing' versus genuinely 'generalizing', since we know what
> the LLM has seen. *Disentanglement*: Building a controlled setting for isolating the contributions
> of retrieval from the contributions of parametric knowledge in agentic search systems."
> — [TREC RAG 2026 track reference](https://github.com/TREC-RAG/trec-rag-skills/blob/main/skills/trec-rag-2026-track-guidelines/references/trec-rag-overview.md), checked 2026-10-29.

That is a benchmark organiser admitting that the field's headline numbers are confounded by
memorisation. The methodological literature is blunter: a 2025 survey of 50 contamination-detection
papers concludes that detection methods rest on "specific assumptions regarding the LMs and
datasets" and that contaminated instances "can have similar performance to random guessing"
([Does Data Contamination Detection Work (Well) for LLMs?](https://aclanthology.org/2025.findings-naacl.291.pdf),
Findings of NAACL 2025). MTEB's submission process has an operational counterpart: results computed
on a **different dataset revision** than the one in the results repository are *skipped with a
warning* ([MTEB: Submit results](https://docs.mteb.org/contributing/submitting_results/)).

**What this means for a system that does not own a generator:** contamination is a *smaller* problem
for the retrieval half and a *larger* problem for the answer half. A bi-encoder or BM25 index cannot
memorise an answer, but a 2026 web-scale dense encoder has almost certainly seen BEIR/NQ/HotpotQA
during training. So the contamination audit that matters for Laya is the *encoder* audit: which
public benchmarks were in the checkpoint's training mix, and what fraction of gold passages can be
recovered **without** the query (a passage-only prior). That is a cheap, honest experiment almost
nobody publishes, and it makes every retrieval number more defensible.

### 4.3 Statistical discipline: the exact tests a small project can afford

The IR community has a settled literature here and it is not flattering to typical RAG tables:

- Significance testing for IR has known pitfalls: test choice matters, and results "at varying
  sample sizes" can disagree between tests (Smucker, Allan & Carterette,
  [A comparison of statistical significance tests for IR evaluation](https://doi.org/10.1145/1321440.1321528), CIKM 2007;
  Carterette, [Agreement among statistical significance tests ... at varying sample sizes](https://dl.acm.org/doi/10.1145/1571941.1572050), SIGIR 2009).
- Running many comparisons inflates false positives unless corrected; the standard reference is
  Carterette, [Multiple testing in statistical analysis of systems-based IR experiments](https://dl.acm.org/doi/10.1145/2094072.2094076), TOIS 2012.
- The practical recipe that survives review: **paired** per-query differences, **bootstrap
  confidence intervals** (or a paired randomization/permutation test) on those differences, an
  explicit statement of how many systems/comparisons were made, and a Holm–Bonferroni or similar
  correction when the table has many rows.

Laya already does the hardest part of this — its README reports paired intervals on deltas
(e.g. MultiHop-RAG `bge-large` hybrid vs `bge-small` hybrid Hits@10 **+0.008 [−0.004, +0.020]**;
depth 10→100 **+0.2118 [+0.2037, +0.2203]**) rather than bare point estimates. That is ahead of
most published RAG tables, including the benchmark papers cited in §1, whose retrieval tables carry
no intervals at all. **Advertising this is nearly free and it is a genuine differentiator.**

One caution: intervals on "best-per-dataset" aggregates are not intervals on a deployable system.
If the headline is a per-dataset maximum, the honest companion number is the best *single*
configuration, with its own interval.

### 4.4 Pre-registration and resumability

The reproducibility literature is unambiguous that the field has a problem and that checklists alone
have not fixed it: "results often fail to confirm original findings" and authors "still don't make
enough resources and information available ... despite reproducibility checklists being introduced
by conferences" (Belz et al.,
[The 2025 ReproNLP Shared Task on Reproducibility of Evaluations in NLP](https://aclanthology.org/2025.gem-1.78.pdf), GEM² 2025).
A 2025 SIGIR paper frames the same problem as a sustainability issue —
[My System Is As Effective As Yours: Reproducibility, Sustainability, and More](https://doi.org/10.1145/3726302.3730345) (SIGIR 2025).

Formal pre-registration for ML/NLP evaluations is emerging (see e.g.
[How to pre-register experimental studies that involve machine learning for text data analysis](https://doi.org/10.1016/j.socec.2025.102414), 2025) but it is not yet a norm a reviewer
will demand. What a reviewer *will* reward is the cheap subset:

- **A pre-committed evaluation plan** written before the numbers: which datasets, which metric
  implementation, which cutoffs, what counts as a win, how many hypotheses.
- **A negative-results register.** This is where Laya is unusually strong: it has recorded, dated
  negatives (coverage-quota selection not transferring off-corpus; listwise LTR not converting;
  hard-negative restriction collapsing the ranker; bridge-feature conditioning hurting the gate).
  In 2026 publishing falsified levers is a credibility signal, not a weakness — and it is exactly
  what the "sustainability" line of work asks for, because it stops the next person repeating them.
- **Resumable, per-query measurement.** The unit of release should be the per-query score file, not
  the aggregate. Per-query files make every later claim re-checkable, make paired tests possible for
  *third parties*, and make a half-finished run resumable instead of lost. MTEB's revision-checked
  result cache is the institutional version of the same idea.

### 4.5 Submission venues a small project can actually reach

| Venue / initiative | What it accepts | Generator needed? | Open now? | Cost of entry |
|---|---|---|---|---|
| **TREC RAG Retrieval (R) task**, 2026 edition | ranked ClimbMix doc IDs per narrative, depth chosen per narrative | **No** | 2026 round **closed** (deadline 8 Aug 2026; TREC 2026 conference Nov 2026) → target 2027 | NIST/Evalbase registration; ≤10 runs/task; run description; priority flag for manual assessment |
| **TREC RAG RAG task** | ≤400-word cited, sentence-split answer | Yes | as above | same |
| **MTRAGEval / MTRAG leaderboard** (SemEval-2026 Task 8) | Subtask A: top-10 passages; B: answers on gold passages; C: full RAG | A no; B/C yes | Evaluation phase closed (papers due 2 Mar 2026); leaderboard opens only during an evaluation phase; **one run per team per task** | Google-form submission; evaluation script released |
| **BrowseComp-Plus** | released fixed corpus + human-verified qrels; retriever table is Cranfield-style; agent table needs answers | retriever **no**, agent yes | Artefacts public; per-retriever results published as datasets (`Tevatron/BrowseComp-Plus-results`) | self-run; no gatekeeper |
| **BEIR** | nDCG@10 over 18 datasets | **No** | Leaderboard exists; submission mechanics not verified in this session | self-run, then a PR/issue |
| **MTEB (v2)** | embedding **models**, via a PR to `embeddings-benchmark/results` with a checklist; mismatched dataset revisions are skipped | **No** | Rolling | `pip install mteb`, run, `mteb submit-results`, fill PR checklist; review "might ask you ... to check for potential data leakage" |
| **ReproNLP** (GEM²) | reproducibility *studies*, i.e. re-running someone else's evaluation | n/a | Annual shared task | a reproduction + error analysis |

Two observations that change strategy:

1. **The MTEB route is not really open to an engine.** MTEB ranks encoders; a hybrid engine is not a
   model. Laya can only appear on MTEB by exposing its dense channel as a `SearchEncoderWrapper`
   (MTEB ≥2.3.0 supports a `FaissSearchIndex` retrieval backend), and even then the score measures
   the encoder, not the engine. The project's own note that it implements MTEB v2's `SearchProtocol`
   but produced no score is therefore *less* damaging than it looks — MTEB was never the right
   placement.
2. **The right placements are the four no-generator tables**: TREC RAG-R, MTRAGEval Subtask A,
   BrowseComp-Plus's retriever table, and BEIR. Of these only TREC RAG-R carries NIST-grade human
   qrels and a fixed, freshly-built corpus (ClimbMix-400b), which is why it is the highest-value
   target and the one worth planning the year around.

### 4.6 What "released artefacts" should mean for this project

The minimum a 2026 reviewer will accept, in increasing order of credibility:

1. A `requirements`-pinned environment and a one-command rerun.
2. Per-query score files + the exact metric implementation (not a description of it).
3. A machine-readable results manifest: dataset revision/hash, corpus snapshot date, model
   checkpoint hash, config hash, seed, wall-clock, peak RSS.
4. Determinism evidence: a byte-identical rerun of at least one cell.
5. **An auditable record of what was run when** — which is where the existing hash-chained audit log
   is genuinely unusual. Very few academic artefacts can say "this number was produced by this
   binary at this time and the record has not been altered". If the log is exposed read-only with a
   published verification command, that is a publishable methodological contribution in its own
   right, and it is trivially cheap to advertise.

### 4.7 The two protocol claims I would *not* make

- **"We are SOTA on BEIR."** No: BEIR-18 best-per-dataset with oracle routing vs published
  single-config systems is not a SOTA comparison, and the BEIR paper's baselines are from 2022.
- **"Our evaluation is pre-registered."** Only if a dated plan predates the numbers. A dated
  changelog and a negatives register are not pre-registration; they are *auditability*, which is a
  different and defensible claim. Say the thing that is true.

---

## 5. Competitive Landscape — open-source / local RAG, as of 29 October 2026

**Method.** Every row was checked on **2026-10-29**: stars, SPDX licence and `pushed_at` from the GitHub REST API; release tags and dates from each repo's `releases.atom`; capability and benchmark claims from the README as served by `raw.githubusercontent.com` that day, plus the benchmark artefacts they link. Nothing is quoted from memory. Gaps are marked **UNVERIFIED** with the attempt made.

**One-line answer to the critical question.** No. As of 2026-10-29 no adopted open-source system is simultaneously (a) fully local, (b) CPU-only, (c) key-free and (d) backed by a held-out retrieval-quality measurement. Those four co-occur only in **retrieval infrastructure** — `bm25s`, Pyserini, Qdrant, Marqo's model evals, Vespa's BEIR write-ups — all of which return ranked documents and none of which returns calibrated probabilities or abstains. Everything that behaves like an *engine* needs a generator; everything that publishes numbers measures the generator's output.

### 5.1 Four groups, not one market

Calling all of these "RAG systems" hides that they are not substitutes.

1. **Generation-required application stacks** (RAGFlow, AnythingLLM, Open WebUI, privateGPT, Kotaemon, Khoj, Onyx, Verba, llmware, mem0/EmbedChain). Product = chat UI + ingestion + LLM. Remove the LLM and there is no user-visible output.
2. **Graph/agentic engines** (GraphRAG, LightRAG, Cognee, Morphik, PageIndex, R2R). They publish numbers, but as LLM-judged answer quality, accuracy-vs-cost or memory scores — measured *through* a generator.
3. **Pure retrieval infrastructure** (Vespa, Qdrant, Chroma, Marqo, txtai, RAGatouille, bm25s, Pyserini). No LLM needed, many CPU-capable, some publish real retrieval numbers — the only genuine functional analogue.
4. **Measurement artefacts** (BEIR, MTEB) — what "rigorous" means here, and why the absence in groups 1–2 is conspicuous.

### 5.2 Comparison table

Stars as of **2026-10-29**. "Needs LLM" = cannot produce its designed output without a generator.

| System | Licence (SPDX) | Stars (2026-10-29) | Last release | Needs LLM? | GPU? | API key? | Published measured numbers | Notes |
|---|---|---|---|---|---|---|---|---|
| Open WebUI | Open WebUI License (`NOASSERTION`, BSD-3 + branding) | 157,131 | v0.12.5, 2026-10-29 | Yes (UI over Ollama/OpenAI) | No | No with local models | None for retrieval; leaderboards are chat-model ELO | Largest footprint here, zero retrieval-quality claim |
| RAGFlow (InfiniFlow) | Apache-2.0 | 93,080 | nightly, 2026-10-29 | Yes | No — DeepDoc parsing is CPU inference in the 1.0 OSS release | Only with a cloud LLM | **None** — "high-fidelity"/"superior context layer", unquantified | ES/Infinity + MySQL + MinIO stack |
| AnythingLLM | MIT | 67,698 | v1.19.0, 2026-10-26 | Yes | Optional — built-in CPU engine | No | None found | Credible key-free desktop option, still generator-bound |
| mem0 / EmbedChain | Apache-2.0 | 68,046 (mem0) | SDK v2.2.1, 2026-09-25 | Yes | No | No | None found | `mem0ai/embedchain` returns **HTTP 404** (2026-10-29): EmbedChain absorbed into a memory layer |
| privateGPT (zylon-ai) | Apache-2.0 | 57,617 | v1.0.1, 2026-06-18 | Yes — "You need a running OpenAI-compatible LLM server" | Practically (example: `qwen3.5:35b`, ~24 GB) | No | None found | The archetypal offline project is now an inference-server front-end |
| LlamaIndex | MIT | 52,729 | v0.14.25, 2026-09-21 | Yes | No | No | None in repo; benchmarking attention redirected to LlamaParse | Orchestration framework; vendor push is hosted parsing |
| LightRAG (HKUDS) | MIT | 40,417 | v1.5.7, 2026-09-02 | Yes (LLM extracts entities) | Practically | No | **README win-rate table**, LLM-judged, 4 domains: comprehensiveness 67.6 / 61.6 / 83.6 / 61.2 % vs NaiveRAG / RQ-RAG / HyDE / GraphRAG; `docs/Reproduce.md`; arXiv:2410.05779 | Vendor-run, generation-based; no nDCG/recall |
| PageIndex (Vectify) | MIT | 39,887 | 0.4.7, 2026-10-24 | Yes — LLM builds the index | No | **Yes**, ≈$0.001/page (`gpt-5.6-luna`) | [Financial QA benchmark](https://github.com/VectifyAI/pageindex-benchmarks/tree/main/financial-qa): 100 questions, 40 10-Ks, 34 configs of 19 LLMs, accuracy vs cost, reproducible | Strongest published protocol in group 2 — measures answers, not retrieval |
| Khoj | AGPL-3.0 | 37,886 | 2.0.0-beta.28, 2026-03-26 | Yes | No | Its one eval used `gemini-1.5-flash-002` | [FRAMES 27.1% → 42.0%](https://blog.khoj.dev/posts/evaluate-khoj-quality/) on a 200-question subset, ±~6%, $5 of API spend (vendor-reported) | Honest about its harness; helpfulness, not retrieval |
| Microsoft GraphRAG | MIT | 36,488 | v3.3.0, 2026-10-08 | Yes (LLM writes the graph) | Practically | No | arXiv:2404.16130: head-to-head **LLM-judged win rates** on comprehensiveness and diversity of *generated answers* vs naive RAG | The graph is unmeasurable without the generator |
| Qdrant | Apache-2.0 | 35,307 | v1.20.0, 2026-10-22 | No | Optional | No | [qdrant.tech/benchmarks](https://qdrant.tech/benchmarks/): RPS, latency, p95, index time, precision on BEIR-derived datasets, re-runnable | Speed at a recall floor — infra, not relevance |
| Onyx (ex-Danswer) | MIT (CE) + EE | 33,071 | v5.0.2, 2026-10-28 | Yes | No | No with local models | None found | Connector-heavy enterprise search |
| Cognee | Apache-2.0 | 32,886 | v1.6.6rc1, 2026-10-23 | Yes | No | No | [BEAM eval](https://github.com/topoteretes/cognee/blob/main/cognee/eval_framework/beam/REPORT.md): 0.79 @100K ctx, 0.67 @10M ctx, LLM judge, 20 held-out questions (vendor-reported) | States its own limitations in the README — best practice |
| Chroma | Apache-2.0 | 29,601 | 2026-10-22 (tag `Latest`) | No | No | No | None in repo | Vector store; research posts only |
| Haystack (deepset) | Apache-2.0 | 26,875 | v3.4.1, 2026-10-21 | Yes — Pipelines/Agents around a Generator | No | No | None in repo (eval components, no numbers) | Cleanest "value is the generator" case |
| Kotaemon (Cinnamon) | Apache-2.0 | 25,836 | v0.12.0, 2026-05-31 | Yes | Optional (`main-ollama` image) | No | None found | Full-stack local chat app |
| llmware | Apache-2.0 | 14,816 | v0.4.7, 2026-10-01 | Yes (small local models) | No — BLING is CPU/"laptop-ready" | No | None in repo; numbers live on HF model cards | Closest philosophical neighbour, still generative |
| txtai (NeuML) | Apache-2.0 | 13,045 | v9.14.0, 2026-10-06 | Optional — embeddings/search need no generator | **No** — documented CPU-only install, `txtai-cpu` image | No | **None** — no table in README, `docs/benchmarks` → 404 (checked 2026-10-29); points at MTEB/HF | Most relevant group-3 framework, publishes nothing of its own |
| R2R (SciPhi) | MIT | 8,049 | v3.6.5, **2025-06-06** (push 2025-11-07) | Yes | No | Optional | None | Dormant ~12 months; README now a pointer |
| Verba (Weaviate) | BSD-3-Clause | 7,691 | v2.1.3, 2025-07-14 | Yes | No | No | None | **ARCHIVED — "Project Discontinued"** banner (2026-10-29) |
| Vespa | Apache-2.0 | 7,153 | Vespa CLI 8.778.45, 2026-10-27 | **No** | No — CPU-capable | No | BEIR zero-shot method + hybrid-search evaluation on [blog.vespa.ai](https://blog.vespa.ai/improving-zero-shot-ranking-with-vespa/) (nDCG@10, P@10, R@100 over 18 datasets); no maintained leaderboard entry | Real engine, real measurement culture; you build the evidence layer |
| Marqo | Apache-2.0 | 5,026 | 2.26.0, 2026-04-07 | **No** (neural search engine) | No | No | [LEADERBOARD.md](https://github.com/marqo-ai/marqo-FashionCLIP/blob/main/LEADERBOARD.md): Recall@1/@10, MRR, P@1/@10, 7 datasets vs 5 baselines, "UPDATED: 2024-08-09" (vendor-reported) | Retrieval-only metrics for embedding models; ~2 years stale |
| RAGatouille | Apache-2.0 | 3,972 | 0.0.9, 2025-02-11 | **No** | No (ColBERT on CPU at small scale) | No | Inherits ColBERT paper numbers; no own table | Late-interaction retrieval, low activity |
| Pyserini | Apache-2.0 | 2,175 | 2.4.0, 2026-08-27 | **No** | No (Java/Lucene) | No | Purpose is reproducing published IR runs (MS MARCO, BEIR) | Research-grade reproducibility, no product surface |
| bm25s | MIT | 1,818 | 0.3.16, 2026-10-28 | **No** | **No** — single-thread Intel Xeon @ 2.70 GHz | No | [README table](https://github.com/xhluca/bm25s): QPS on 15 BEIR datasets vs Elasticsearch, `bm25_pt`, `rank-bm25` (nfcorpus 1196 vs 45.8 q/s; scifact 953 vs 20.8); ships `bm25s.utils.beir.evaluate`; full runs in [bm25-benchmarks](https://github.com/xhluca/bm25-benchmarks) | **Closest existing analogue to Laya's deployment envelope** — published number is throughput, not answer quality |
| Morphik | **BUSL-1.1** (source-available) | 3,719 | No GitHub releases | Yes | Practically (ColPali encoding) | No | None — a vendor blog, no protocol | Licence disqualifies it from "open source"; re-licenses to Apache-2.0 four years after each release |
| Canopy | — | **UNVERIFIED** | — | — | — | — | — | `canopyai/Canopy` **HTTP 404** (2026-10-29); search finds only unrelated forks. Treated as discontinued |
| BEIR | Apache-2.0 | 2,324 | v2.2.0 (push 2025-10-16) | No | No | No | 18 heterogeneous datasets, nDCG@10, official leaderboard (SIGIR 2024 resource paper) | The reference protocol |
| MTEB | Apache-2.0 | 3,481 | 2.26.29, 2026-10-29 | No | No | No | Embedding-model leaderboard | Measures encoders, not engines |

### 5.3 Claims vs publications

Read the two middle columns together and the pattern is stark: of 20 application/engine repositories in groups 1–2, **five** link any published measurement (LightRAG, Cognee, PageIndex, Khoj, and Morphik's blog), and **all five measure generated answers** — LLM-judged win rates, accuracy-vs-cost, or FRAMES helpfulness. Retrieval quality (nDCG@10, Recall@k, MRR) appears only in group 3. Qdrant benchmarks speed at a fixed precision threshold: an infrastructure claim, not a relevance claim.

RAGFlow is the sharpest case: 93,080 stars, Apache-2.0, a release tagged the day of this check, and not one number in the repository. "High-fidelity", "superior context layer", "exceptional efficiency and precision" — all unquantified. The most-adopted dedicated RAG engine publishes nothing you could put in a results table.

What does exist mostly sits on the wrong side of the held-out line. Khoj's post is candid that it is a 200-question random subset at ~±6% against a hosted model. LightRAG's win rates are reproducible in principle but LLM-judged on the vendor's own prompts. Cognee is the most disciplined, warning that its two settings use different conversations, ingestion models and retrieval procedures and must not be compared across systems. PageIndex has the strongest protocol (100 fixed questions, 40 filings, published harness) and still measures answer accuracy, priced per page in LLM calls.

### 5.4 Who is actually comparable to a non-generative evidence engine

Bluntly: almost nobody, and the exceptions are not "RAG systems".

Most of this list is **orchestration whose product is the generator**. Haystack says so in its own README — "explicit control over retrieval, routing, memory, and generation" — and ships Pipelines and Agents around a Generator. LlamaIndex has shifted emphasis to LlamaParse, a hosted parsing service. privateGPT, once the banner carrier for "fully offline", now opens by requiring an OpenAI-compatible server and demonstrates with a ~24 GB model. RAGFlow, AnythingLLM, Open WebUI, Kotaemon and Onyx are chat products; strip the LLM and nothing remains a user would accept as an answer. GraphRAG and LightRAG cannot even be *built* without an LLM, because the LLM writes the graph. Morphik is BUSL-1.1, so not open source in the sense this comparison implies.

Four things share Laya's functional shape — deterministic ranking, no generator in the loop:

- **txtai** — local, CPU-documented, key-free, embeddable, and honest enough to publish no numbers rather than marketing ones. Nearest *deployment* twin, emptiest *evidence* twin.
- **bm25s** — pure NumPy BM25, single-thread CPU, BEIR datasets in its benchmark, reproducible harness in a sibling repo. Nearest *evidence* twin, but it is a ranking function, not an evidence engine: no passage assembly, no calibration, no abstention, no ingestion.
- **Pyserini/Anserini** — the reproducibility standard for first-stage retrieval and the only project whose stated purpose is publishing numbers. Java/Lucene, no LLM, CPU.
- **Vespa** — the only full engine with a real retrieval-measurement culture, and it runs on CPU; but the measurement lives in blog posts, the calibration layer is yours to write, and the operational weight is a cluster.

None of the four returns *calibrated probabilities* or *abstains*. That is not a roadmap oversight — it is outside their product definition. Retrieval middleware returns an ordered list and lets the caller decide. The decision policy — when to answer, when to refuse, with what confidence — is exactly the layer these systems delegate to the generator, and therefore the layer none of them measures.

### 5.5 Is the niche empty?

**Yes on the composite, and structurally so.** The properties separate cleanly:

- Local + CPU + key-free **without** measurement: AnythingLLM, Open WebUI, privateGPT, llmware, txtai. Cheap to build, because nobody has to prove anything.
- Measurement **without** the deployment envelope: LightRAG, GraphRAG, Cognee, PageIndex, Khoj, Onyx — all assume a GPU or an API bill, and all measure through a generator.
- Retrieval-only measurement **without** an engine: bm25s, Pyserini, Marqo's model evals, Vespa's posts.
- All four: **nothing**.

Two caveats keep this honest. First, the niche is not empty *academically* — selective prediction, conformal abstention and "grounded or abstain" QA are well-developed; a web search on 2026-10-29 for local, CPU-only, abstaining RAG engines surfaced only zero-to-low-star personal repositories. The ideas are not novel; the shipped, measured, key-free implementation is missing. Second, "no published numbers" often means "nobody asked", not "the numbers are bad" — RAGFlow's silence is a marketing choice, not evidence of weak retrieval. The claim should therefore be *nobody in this class publishes retrieval-only quality on a held-out protocol*, not *nobody can*.

The strategic reading is narrow: Laya's comparable set is not the 157k-star chat UIs but `bm25s` (CPU retrieval with a reproducible benchmark), Pyserini (published IR protocol) and BEIR/MTEB (the yardstick). Against those, the claim worth making is not "local and private" — a dozen projects already say that — but **retrieval-only quality, published, on a held-out set, measured on a CPU with no key**, an unoccupied position with a well-defined way to occupy it.

### 5.6 Facts I could not verify

- **Canopy** — `canopyai/Canopy` and `canopyai/canopy` both HTTP 404 on 2026-10-29; repo search returns only unrelated forks. Prior star count not recoverable, so no figures are asserted.
- **EmbedChain** — `mem0ai/embedchain` HTTP 404 on 2026-10-29. Historical stars (~8.6k) **UNVERIFIED**; only successor `mem0ai/mem0` (68,046) is measured.
- **Vespa** — `releases.atom` mixes component tags (CLI, language server); the quoted release is the CLI tag, not a canonical platform version, which was not identified.
- **Chroma** — newest tag is literally named `Latest` (2026-10-22); the semantic version behind it is not in the feed.
- **Marqo** — last release 2026-04-07 but pushed 2026-09-03; whether the OSS engine or the hosted product is the live line was not established.
- **RAGFlow** — no measurement found, but the search covered the repository and its README only; a number on `ragflow.io` or in a talk would have been missed.
- **Infinity, Chroma, RAGatouille, Haystack, Onyx, AnythingLLM, Open WebUI, Verba, Kotaemon, llmware, Morphik, R2R** — "no published numbers" means *no table found in the repository on 2026-10-29*; some may publish in blog posts or model cards not crawled (Infinity's model evals and llmware's BLING cards are the likeliest).
- **LightRAG / GraphRAG** — quoted win rates are the vendors' own; no independent replication was located, and neither publishes a retrieval-only nDCG/recall figure.

---

## 6. What a non-generative evidence engine can uniquely claim in 2026

*Sources accessed 2026-10-29; local copies in `.scratch/research/src/`.*

### 6.1 The structural problem, stated plainly

Every benchmark in §1 that awards an "answer quality" crown — TREC RAG's strict-vital and support
layers, MTRAG's `RBalg`/`RLF`, FRAMES, MuSiQue EM/F1, RGB's four abilities, CRAG's truthfulness,
RAGTruth's detection F1, RAGAS's faithfulness — takes **generated natural-language text** as its
input. A system that does not generate cannot be ranked on any of them. This is not a matter of
effort or of a missing script; it is a property of the object being measured.

There are exactly two legitimate responses, and only one of them survives review:

- **Illegitimate:** attach *any* local generator and claim the end-to-end number as the engine's.
  It is the generator's number. The project's own 20-record pilot is the cautionary tale: EM 0.000
  because prose gold vs short entity golds, which measures the harness, not the engine.
- **Legitimate:** compete at the **evidence layer**, on benchmarks that are Cranfield-graded, and
  make the abstention claim at the evidence layer too. That is a smaller claim, and it is winnable.

### 6.2 The evidence-layer leaderboards that exist, and how hard they are

| Benchmark | Metric | Best published | What it proves about an evidence engine |
|---|---|---|---|
| BEIR (18 datasets) | nDCG@10 | SPLADE 0.4740 avg; project's best-per-dataset 0.4799 | general-purpose zero-shot retrieval; saturated-ish, 2022 baselines |
| LoTTE (12 sets) | Success@5 | ColBERTv2 beats SPLADEv2/RocketQAv2 by up to 3.7/8.1 pts | long-tail domains; retrieval-only by construction |
| LegalBench-RAG | P@k / R@k at snippet level | R@64 84.19%, P@1 14.38% (their sweep) | precision at snippet granularity — the closest public analogue to "returns the right passage, not a document containing it" |
| MultiHop-RAG retrieval | MAP@10, MRR@10, Hits@k | MAP@10 0.4795 (voyage-02 + bge-reranker) | multi-document evidence sets; includes 301 `null_query` records |
| TREC RAG-R 2025 | nDCG@30, nDCG@100, R@100 | nDCG@30 0.6934 (4method_merge) | NIST human qrels, 105 deep-research narratives, MS MARCO V2.1 |
| TREC RAG-R 2026 | TBD | — | ClimbMix-400b, 119 narratives, human-judged |
| MTRAGEval Subtask A (2026) | nDCG@5, Recall@5 | nDCG@5 0.578 (AILS-NTUA, sparse) | multi-turn, unanswerable-heavy, **lexical systems at the top** |
| BrowseComp-Plus retriever table | Recall@k, nDCG@k | nDCG@10 20.3 (Qwen3-Embed-8B); BM25 1.6 | web-scale deep-research evidence retrieval; very far from solved |
| BRIGHT (ICLR 2025) | nDCG@10 | **24.3** avg best; BM25 14.5 | reasoning-intensive retrieval is nowhere near solved |

Two facts from this table are strategically important:

1. **On the two hardest, most recent evidence benchmarks, retrieval is at nDCG@10 ≈ 20–24.** BRIGHT
   best 24.3, BrowseComp-Plus best 20.3. The field's *own* numbers say the evidence layer is open,
   which is exactly where a CPU-only engine can be interesting rather than merely cheap.
2. **Lexical retrieval is not dead in 2026.** MTRAGEval Subtask A was won by a sparse system, and
   the organisers' BM25 baseline (0.354) beat BGE-base-1.5 (0.390) only narrowly while the winner
   reached 0.578 — i.e. 0.22 nDCG@5 above BM25, not 0.5. On MultiHop-RAG the project measured BM25
   Hits@10 at 0.782 against the paper's best dense+rerank 0.747, and noted the paper's table has no
   lexical baseline at all. A CPU-only system's honest opportunity is not "beat dense SOTA on
   average"; it is "be at or near the top on evidence-set metrics where fusion and coverage
   dominate, at 1/100th of the cost".

### 6.3 Is there a benchmark that grades "correct evidence + honest abstention" *separately* from "a good answer"?

**Mostly no — and this is the most important finding in this section.** The closest artefacts,
ranked by how directly they support the claim:

1. **Asai & Choi, "Challenges in Information-Seeking QA: Unanswerable Questions and Paragraph
   Retrieval" (ACL 2021)** is the canonical formulation of exactly this decomposition. It isolates
   two subproblems — **paragraph selection** and **answerability prediction**, i.e. "whether the
   paired evidence document contains the answer" — and evaluates them separately, showing that when
   given a gold paragraph *and* told whether the question is answerable, models beat a human
   annotator, while **"predicting answerability itself remains challenging"**
   ([ACL 2021](https://aclanthology.org/2021.acl-long.118.pdf)). It even reports per-category
   answerability prediction over 800 manually annotated unanswerable examples across six languages.
   This is the intellectual ancestor of the answerability gate — and it is a *reading-comprehension*
   setting with an extractive reader, not a modern RAG corpus.
2. **"Sufficient Context" (Joren et al., Google, ICLR 2025)** defines a notion of whether the
   retrieved context contains enough information to answer, classifies instances into
   sufficient/insufficient, and uses that to drive **selective generation** — improving the fraction
   of correct answers among responses by **2–10%** for Gemini, GPT and Gemma. Its key finding is
   precisely a gate finding: strong models "often output incorrect answers instead of abstaining
   when the context is not [sufficient]", while weak models abstain even when it is sufficient
   ([arXiv:2411.06037v3](https://arxiv.org/abs/2411.06037), Apr 2025).
   *Caveat that matters here:* the sufficiency classifier is itself an LLM (`Gemini 1.5 Pro`
   autorater), so the paper demonstrates the *value* of evidence-sufficiency estimation but not a
   generator-free way to compute it.
3. **AbstentionBench (Kirichenko et al., Meta, 2025)** is the best-designed abstention benchmark —
   20 datasets covering unknown answers, underspecification, false premises, subjective
   interpretation, outdated information; 20 frontier LLMs — and its headline result is that
   **reasoning fine-tuning degrades abstention by 24% on average**, and that a system prompt helps
   but does not fix it ([arXiv:2506.09038](https://arxiv.org/abs/2506.09038)). It grades *answers*,
   so it needs a generator, but its taxonomy is the right one to borrow.
4. **MTRAG-UN / MTRAGEval (2026)** carries the answerability types (unanswerable / underspecified /
   non-standalone / unclear) into a RAG leaderboard, and its finding that **abstention is a
   differentiating capability not correlated with overall generation quality** is the strongest
   published support for treating abstention as its own axis. But: the retrieval subtask scores only
   the **answerable** subset, so refusal is never graded on the evidence side.
5. **RGB (AAAI 2024)** measures rejection rate *and* negative rejection — the two-sided view nobody
   else reports — but only over generated answers.
6. **CRAG (NeurIPS 2024 D&B)** assigns 1 / 0.5 / 0 / **−1** to perfect / acceptable / **missing** /
   incorrect, explicitly preferring a refusal over a hallucination — i.e. abstention is a graded
   outcome, not an error. Again over generated answers.

So the honest statement is: **the field has (a) retrieval-only benchmarks with no abstention axis,
and (b) abstention benchmarks that require generation. Nobody publishes a joint
"evidence-correctness × abstention" score for a non-generative system.** There is a genuine hole.

### 6.4 How to fill that hole cheaply, using only public data

The raw materials already exist, and this is the project's strongest available move.

**A public, ready-made evidence-side abstention test set is sitting inside MultiHop-RAG.** Its 2,556
records include **301 `null_query` records whose gold answer is literally "Insufficient
information."** (project README, and arXiv:2401.15391) The benchmark's own retrieval metric excludes
them ("NULL queries are excluded in this experiment because there is no matching evidence"). That is
an invitation: score the *refusal* decision on those 301 records, where the rest of the field scores
only the 2,255 answerable ones. Add the unanswerable subset of MTRAG-UN (97 unanswerable + 78
underspecified out of 507, per the SemEval description) as a second, independent set, and
AbstentionBench-style categories as a third if a text-only subset can be used.

A defensible evaluation artifact then looks like:

- **Axis 1 — evidence quality** (existing): MAP@10 / Hits@10 / nDCG@10 on the answerable subset,
  paired intervals, against published rows.
- **Axis 2 — evidence-side abstention** (new): ROC-AUC / risk–coverage (AURC) of the answerability
  gate on `null_query` + unanswerable subsets, **held out**, reported against two baselines: a
  single-scalar score threshold (the shipped behaviour, AUC ≈0.52) and a BM25-score threshold.
- **Axis 3 — joint score**: e.g. evidence-MAP@10 on answerable × (1 − false-accept rate) on
  unanswerable, so a system cannot buy abstention with blanket refusals. This is the two-sided
  design RGB's negative rejection already argues for, moved to the evidence layer.
- **Axis 4 — cost**: seconds, joules, and peak RSS per query on CPU. BrowseComp-Plus publishes
  dollars per agent run; nobody publishes joules per query. A CPU-only system can own that column.

That is a publishable, novel, cheap, and *honest* contribution — and it is built out of other
people's public data, which is the fastest route to credibility for a small project.

### 6.5 The three-state adjudication is the under-claimed asset

`support / contradict / no_coverage` maps almost one-to-one onto RAGTruth's span-level taxonomy
(evident conflict vs. unsupported/subtle conflict) and onto the three-level support grade TREC 2025
uses (full / partial / no support). What is unusual is that Laya computes it **without generating**:
given a claim and a passage set, it adjudicates. Three reasons this is more valuable than it looks:

1. **RAGTruth's numbers say the problem is unsolved even with a generator in the loop.** Best
   response-level hallucination-detection F1: **GPT-4-turbo 63.4%**, fine-tuned Llama-2-13B 78.7%,
   SelfCheckGPT 58.8%; span-level best 58.2% (ACL 2024, Tables 5–6). A deterministic adjudicator with
   no sampling variance and no API cost has a defensible niche even at comparable accuracy, because
   it is *auditable* — you can say why.
2. **TREC 2025 showed automatic support scoring inflates by +0.11 to +0.15** (manual 0.7792 /
   0.7635 for `bm25-rz7b-2025a` versus 0.9298 / 0.9189 from the GPT-OSS-120B grader; 0.8435 versus
   0.9522 for `no-llm-refined`). A rule-based adjudicator with a published decision procedure is a
   counter-position to LLM-as-judge attribution scoring, and a much easier thing to defend to a
   reviewer than another judge model.
3. **TREC RAG 2026's own research goal is the disentanglement of retrieval from parametric
   knowledge.** A system with no parametric knowledge is a *measuring instrument* for that
   disentanglement: it shows what the corpus alone supports. That is a framing the organisers have
   already published and it costs nothing to adopt.

### 6.6 The strongest counter-argument, and how to answer it

Do **not** claim "better retrieval ⇒ better RAG" without qualification. The 2025 paper
[Redefining Retrieval Evaluation in the Era of LLMs](https://arxiv.org/html/2510.21440) argues that
classical IR metrics mis-model the LLM consumer in two ways — its positional discount is
human-shaped, and it treats all non-relevant passages as equivalent when LLMs are actively harmed by
distractors — and introduces **UDCG**, which correlates with end-to-end answer accuracy up to **36%
better** than traditional metrics (Spearman; significant under bootstrap at p<0.05). Earlier work
(eRAG) makes the same point with a different construction. Correlations in that paper sit around
**0.55–0.70** for UDCG and visibly worse and more volatile for nDCG.

The correct response is to **scope the claim to the instrument, not the downstream answer**:

> The engine is measured as an evidence instrument — recall and ranking of the passages that a
> correct answer requires — under a fixed compute envelope of CPU-only, no-API-key, no-GPU. It does
> not claim that better evidence automatically yields better prose; it claims, and measures, that it
> returns more of the right evidence per query than any other system in its envelope, and that it
> knows when the evidence it holds is insufficient.

That sentence is checkable on five public benchmarks, survives the UDCG critique, and is the only
form of the claim that a 2026 reviewer cannot take apart with one citation.

### 6.7 What cannot be claimed, and should be said out loud

- **"Best RAG system."** Requires a generator; unclaimable.
- **SOTA on MuSiQue/FRAMES/RGB/CRUD-RAG/RAGTruth.** All are answer-string or answer-text metrics.
- **"Our nDCG predicts answer quality."** Contradicted by the UDCG/eRAG line of work as a general
  claim; only defensible per-corpus and measured.
- **"Fully general retrieval SOTA."** BEIR-18 best-per-dataset 0.4799 is above SPLADE's 2022 0.4740,
  not above 2026 encoders, and it is an oracle-routing figure.
- **"Calibrated abstention out of domain."** The project's own measurement says it is not: shipped
  scalar AUC 0.5087, ECE 0.3085 on the only labelled negative class. The in-domain feature gate
  (AUC 0.815) is a measured in-domain gate. Say exactly that.

---

## 7. Confidence and gaps

### 7.1 What I would stake a decision on (high confidence)

These rest on primary sources I read directly in this session — proceedings PDFs, arXiv full texts,
track sites, GitHub API, model cards — not on secondary summaries.

- The **structure of the 2026 grading stack** and its generator dependence (TREC RAG 2025 overview
  §3; RAG task output schema for 2026; MTRAG §6.3.3; SemEval-2026 Table 3).
- **TREC RAG 2025's actual numbers**, because I read Tables 5, 6, 7, 11 and 12 verbatim, including
  the retrieval-run categories and the manual-vs-automatic support-evaluation gap.
- **TREC RAG 2026's shape and closure** (ClimbMix-400b, 119 narratives, AG removed, deadline 8 Aug
  2026, conference Nov 2026, Retrieval task generator-free).
- **BrowseComp-Plus's agent and retriever tables**, including the retriever-swap effect
  (55.90 → 70.12 accuracy; calibration error 13.50 → 9.11) and the retriever-only nDCG@10 of 20.3 /
  1.6.
- **SemEval-2026 Task 8's leaderboards** (Subtask A winner sparse, nDCG@5 0.578; baselines BM25
  0.354 / BGE-base-1.5 0.390 / Elser 0.480) and its abstention finding.
- **RAGTruth's detection ceilings** (GPT-4-turbo 63.4% F1, fine-tuned Llama-2-13B 78.7%, span-level
  58.2%) and **RGB's rejection rates** (45% / 43.3%) and **AbstentionBench's −24% reasoning
  fine-tuning result**.
- **MultiHop-RAG's structure** (2,556 queries, 609 articles, **301 null queries = 11.78%**, Table 4)
  and its retrieval table (voyage-02 + rerank MAP@10 0.4795, no-reranker 0.3143).
- **Search-R1's MuSiQue EM of 0.196** and the surrounding baseline ladder — verified from its
  Table 2.
- **The landscape facts**: stars, licences and release dates from the GitHub API on 2026-10-29;
  RAGFlow's absence of published numbers; Verba archived; Morphik BUSL-1.1; Canopy and EmbedChain
  404.
- **LazyGraphRAG's vendor cost claims** (indexing 0.1% of full GraphRAG, query cost ~700× lower) —
  read from the Microsoft blog, and labelled vendor-reported.

### 7.2 Medium confidence — directionally right, numbers may move

- **FRAMES, CRUD-RAG, LoTTE, LegalBench-RAG** numbers are the papers' own 2022–2024 measurements.
  I found no maintained 2026 leaderboard for any of them; the numbers are the right *shape* but not
  "current bests". Third-party aggregator tables exist but I could not establish their provenance
  and did not use them.
- **"No benchmark grades evidence + abstention jointly."** High confidence that no *mainstream,
  maintained, generator-free* benchmark does. Medium confidence that nothing obscure does: §3 names
  three partial hits (AbstentionBench, the CMC 2026 Evidence Sufficiency Benchmark, RGB) and one
  conceptual ancestor (Asai & Choi, ACL 2021). A literature I did not search thoroughly —
  medical/legal QA sufficiency work — may contain more.
- **The claim that the landscape niche is empty.** True as of 2026-10-29 for *publications found by
  search of repositories*; vendor sites, talks and model cards may carry numbers my search missed.
- **Agentic-retrieval gain sizes.** The +0.35 nDCG@30 at TREC 2025 is real but confounded: the
  "generation-in-the-loop" runs also used better fusion, better query understanding and more compute.
  Treat it as an upper bound on the whole pipeline, not on agentic search alone.

### 7.3 Low confidence / explicitly unverified

- **TREC RAG 2024 overview**: `trec.nist.gov/pubs/trec33/papers/Overview_rag.pdf` and the
  `%5F`-encoded variant both returned 404 on 2026-10-29, and I found no alternative host. The 2024
  description here comes from the 2024 track site, not from the overview paper.
- **Whether TREC RAG 2026 judgements exist yet.** The site says results/judgements "TBD" and the
  conference is November 2026. If judgements are released between writing and reading this, the 2026
  numbers supersede several §1 statements.
- **LiveRAG's 2026 status.** The site content I fetched still describes the SIGIR 2025 edition
  (Falcon3-10B-instruct as the mandated generator, jury reproducibility check). A SIGIR 2026 dataset
  paper exists (arXiv:2511.14531, 895 questions with gold answers and supporting claims, plus IRT
  difficulty/discriminability scores), but I did not verify whether a 2026 *challenge* ran.
- **MTEB submission mechanics for a system.** I read the "Submit results" and retrieval-backend docs;
  I did not verify whether a hybrid engine can be submitted at all other than as an encoder wrapper,
  and the project's own note that a full MTEB run "forked the harness workers and stalled" was not
  reproduced.
- **BEIR leaderboard submission mechanics** were not verified in this session.
- **No 2026 SOTA figures for BEIR-18, FRAMES, MuSiQue or CRUD-RAG** were obtained from a source I
  trust. Do not upgrade "above SPLADE's 0.4740" into "SOTA" on the strength of this report.
- **Two 2026 papers on evidence-calibrated abstention** surfaced in search with exactly the right
  framing ("retrieval coverage, abstention calibration, hallucination-proxy") but appeared in
  journals whose review standards I cannot assess (`jtie.stekom.ac.id`, `journal.jci.co.id`). They
  are cited here as *terminology evidence* only — that this vocabulary is in circulation — and not
  as validation.

### 7.4 What this report does not cover

- **Multimodal / visual RAG** (ColPali, ColQwen, VisRAG), document-layout retrieval, and OCR-heavy
  pipelines. Mentioned only where a competitor depends on them (Morphik's ColPali encoding).
- **Fine-tuning the retriever** in 2026 (late-interaction successors, RL-trained retrievers as
  products). The RL-retriever family is covered as *architectures*, not as a training recipe.
- **Serving/infrastructure** — vector DB performance, ANN recall/latency trade-offs — except where a
  vendor publishes it as a quality claim.
- **Non-English benchmarks** beyond RGB's Chinese half and CRUD-RAG.
- **Pricing of commercial generation** in any detail; only relative cost claims are used.

### 7.5 The single most important thing I could not establish

Whether **evidence-layer quality predicts end-to-end quality well enough to be the headline claim**.
The 2025 evidence says "partially, per corpus, ≈0.55–0.70 Spearman at best" (UDCG/eRAG line). Nobody
has published that relationship *for CPU-only lexical/hybrid systems on modern evidence sets*. That
is both the risk in the positioning of §8 and, if measured, a genuinely novel contribution — and it
is cheap to measure with a local 7B generator on a few hundred queries.

---

## 8. The winnable claim in 2026

### 8.1 The claim that loses, so it can be retired

"Best fully-local CPU-only RAG." It loses on four counts:

1. **"RAG" implies an answer.** The moment a reviewer reads "RAG" they look for a generated answer,
   and the three-layer grading stack of §1 will be applied. Without a generator there is no row to
   enter.
2. **"Best" needs a population.** The population of systems that are local + CPU-only + key-free
   *and publish a measured result* is, per §5, effectively empty — so "best" is vacuous, and a
   reviewer will say so.
3. **CPU-only is currently under-measured.** The project's headline latency ("2–5 s end to end") is
   measured with **MPS enabled on an Intel Mac**; the README itself says CPU-only hosts are slower
   (`README.md`, "Requirements"). Any CPU-only claim must be re-measured with the accelerator
   disabled and reported separately, or the first review comment will be that the envelope was not
   the envelope.
4. **Retrieval quality does not automatically transfer** (§6.6: UDCG/eRAG, ≈0.55–0.70 Spearman at
   best, worse for nDCG). "Best retrieval" is not "best RAG" and in 2025–2026 the field published the
   evidence for that.

### 8.2 The claim that wins

> **Laya is an evidence instrument, not an answer generator.** It is scored, and wins, at the
> evidence layer: on five public benchmarks it returns more of the passages a correct answer
> requires, per query, than any other system that runs on a CPU with no accelerator, no API key and
> no network — and it reports, with a calibrated probability, when the evidence it holds is *not*
> sufficient to answer, which is a quantity the leaderboard-carrying answer models do not report at
> all.

Five separable claims, each checkable, none requiring a generator:

**C1 — Evidence recall at a fixed compute envelope.** Population: systems runnable on CPU, key-free.
Benchmarks: TREC RAG-R (human qrels, ClimbMix), MTRAGEval Subtask A, BrowseComp-Plus retriever table,
MultiHop-RAG retrieval, BEIR-18 as a sanity floor. Metric reported with paired intervals. Target
positioning is *"top of the no-accelerator, no-key class"*, not *"SOTA"*.

**C2 — Evidence-side abstention, as a first-class metric.** This is the genuinely new one, and §6.4
shows it can be built almost entirely from other people's public data: MultiHop-RAG's **301
`null_query` records** (which the benchmark's own retrieval metric *excludes*), MTRAG-UN's 97
unanswerable + 78 underspecified tasks, and AbstentionBench-style categories where a text-only
subset is usable. Report ROC-AUC, risk–coverage (AURC/AUGRC), **precision and recall of abstention**
separately (over-abstention is the failure mode AbstentionBench's precision analysis warns about),
and the OOD split. State the number the way §3.4 does: **0.815 in-domain / 0.767 with a question type
held out / 0.509 out of domain.**

**C3 — Claim adjudication without generation.** `supported / contradicted / no_coverage` is a
*deterministic, auditable* alternative to LLM-judged faithfulness in a literature whose best
claim-level detectors reach only ≈69–77% balanced accuracy on adversarial data (FaithJudge, EMNLP
2025). The defensible framing: not "we are more accurate", but "**we are reproducible, cost zero
tokens, and can explain every verdict** — and here is our accuracy on the same labelled corpora".

**C4 — Measurement discipline as an artefact.** Paired bootstrap intervals on every delta (already
done), per-query score files, dataset/corpus/checkpoint/config hashes, byte-identical rerun proof,
and a **hash-chained audit log** that very nearly no academic artefact can offer. Plus the
**negatives register** (coverage-quota transfer failure, listwise LTR non-conversion, hard-negative
collapse, bridge-feature harm): in a field where ReproNLP documents that "results often fail to
confirm original findings", a dated record of what *did not* work is a credibility asset, not an
embarrassment.

**C5 — Cost as a first-class axis.** Seconds, joules and peak RSS per query on CPU. BrowseComp-Plus
publishes dollars per agent run; nobody publishes joules. Nobody in the landscape section publishes
retrieval quality *per joule*. A CPU-only engine can own that column outright, and it is the honest
version of the "local/cheap" story.

**C6 — A measured retriever router, into a gap the literature names.** §2.4 records that routing
*between* retrieval strategies (dense vs lexical vs graph) is "essentially unmeasured in the
peer-reviewed literature"; what exists is routing between *no retrieval / single-step / multi-step*
(SKR, Adaptive-RAG) and between long context and RAG (SELF-ROUTE). A project whose BEIR-18 result is
already a *best-per-dataset* figure — 0.4799 over eighteen different winning configurations — is
sitting on exactly the per-query data such a router would be trained and tested on. Publishing the
router (features, held-out accuracy, cost per routed query, and where it misroutes) converts a
criticisable aggregate into a contribution, and it is the one place where "we picked a different
configuration per dataset" changes from an admission into a result.

### 8.2b The generator-free lever set, with its measured numbers

Everything below is available to an engine that never generates, and each line has a published
number attached (§2): **HippoRAG's PPR-over-OpenIE** single-step multi-hop retrieval, **+7.3 average
R@5** over ColBERTv2 while matching multi-step IRCoT at **10–30× lower online cost**, with a **local
Llama-3.1-8B OpenIE indexer** competitive with GPT-3.5 on 2 of 3 datasets at $0 API cost;
**Provence's** DeBERTa-fused pruning, **49%** of context removed with negligible-to-no drop in
**25 s / 7.9×10¹⁴ MFLOPS** — 12×–68× cheaper in FLOPs than the LLMLingua family, which additionally
needs a scoring LM and degrades hard on reasoning (BBH 70.07 → 56.85 at 7×); **Adaptive-RAG/SKR**
small-classifier routing, about **three-quarters of the multi-step gain at 64% of the steps**
(44.40 accuracy at 3.55 steps vs 47.00 at 5.53 on HotpotQA); **G-Retriever's** Steiner-tree subgraph
selection, **−99% input tokens** (100,627 → 610); and the coverage/dedup selection this project
already measures. Off-limits by construction, and worth saying so explicitly in any paper: Self-RAG's
reflection tokens, CRAG's trained evaluator, CoRAG's chains, DRAGIN's uncertainty triggers, the whole
Search-R1 / R1-Searcher / ZeroSearch family, and **HyDE** — the largest single retrieval-side gain in
the literature (+16.8 nDCG@10 on DL19, TREC-COVID 27.3 → 59.3), and pure generator output.

### 8.3 The one experiment that would turn positioning into evidence

Everything above is cheap except one thing, and it is the thing that would settle §7.5 and pre-empt
§6.6: **measure the evidence→answer transfer curve on this system.**

Take ~300 queries across two or three corpora, pair them with a single fixed local generator (the
Ollama model already used for the pilot), and plot end-to-end correctness against evidence-layer
score. Then publish the correlation, per corpus. Three outcomes, all publishable:

- **Strong correlation (ρ > 0.7):** the evidence claim is a proxy for end-to-end quality, and the
  project has the one number the UDCG/eRAG literature leaves open for CPU-class systems.
- **Moderate (0.4–0.7):** still valuable — it says when evidence metrics should and should not be
  trusted, and it makes the "instrument" framing rigorous instead of defensive.
- **Weak (<0.4):** a negative result of exactly the kind the project already publishes well, and a
  hard constraint on every future claim.

The pilot needs a metric that can be non-zero: **nugget recall** (TREC/RAGDoll style) or **short-answer
EM/F1 with a constrained output format** (MuSiQue/Search-R1 style), *not* EM on prose gold. The
emitted text should be constrained ("answer in ≤5 words, or say INSUFFICIENT"), which makes grading
mechanical and cheap and sidesteps the gold-format mismatch that produced EM 0.000.

### 8.4 Concrete placements, in priority order

| # | Action | Cost | What it buys |
|---|---|---|---|
| 1 | Evaluate on **MultiHop-RAG's 301 null queries** and publish evidence-MAP × abstention as a joint score | days | a novel, defensible, public artefact; fills the gap §3.5 identifies |
| 2 | **CPU-only re-measurement** of every headline number with the accelerator disabled, latency and RSS included | days | makes "CPU-only" true rather than nominally true |
| 3 | Register for **TREC 2027**, submit to the **Retrieval (R)** task on ClimbMix-400b | a year, cheap in effort | NIST human qrels; the only placement with institutional weight and no generator |
| 4 | **MTRAGEval Subtask A** at the next evaluation phase | weeks | 2026 shared-task comparability; sparse systems lead there |
| 5 | Add a **BrowseComp-Plus retriever row** to the published results set | weeks | hard, modern, fixed corpus with human-verified evidence |
| 6 | Publish the **evidence→answer transfer curve** (§8.3, 300 queries, one local generator) | weeks | the number that makes the instrument claim non-defensive |
| 7 | Package the **audit log + per-query artefacts + negatives register** as a reproducibility release | weeks | satisfies §4's reviewer checklist more thoroughly than most papers do |

### 8.5 The sentence to put on the front page

> Laya returns evidence, not prose — so it is measured where evidence is measured: on public
> retrieval benchmarks (BEIR-18, MultiHop-RAG, MuSiQue, BrowseComp-Plus, and TREC's retrieval task
> when its judgements land), under a CPU-only, no-accelerator, no-API-key, no-network envelope, with
> paired confidence intervals, a contamination audit, per-query artefacts, and a dated register of
> the levers that failed. When the evidence is insufficient it says so, with a calibrated
> probability, and it is scored on that too — an axis the answer-quality leaderboards do not
> measure.

Every clause is checkable. None of it needs a generator. None of it can be taken apart with a single
citation — provided C1 is reported as a *class* claim, C2 is reported with its OOD number, and C3 is
reported as reproducibility rather than accuracy.
