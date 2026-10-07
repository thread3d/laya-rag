# How this system stands against the RAG benchmark landscape

This is the durable record of what the engine has actually been measured against, what it
could be measured against, and what it structurally cannot be ranked on. It is deliberately
separate from [`RANKINGS.md`](RANKINGS.md), which is the retrieval programme; this document is
about *placement outside the project*.

Numbers are nDCG@10 unless stated. Everything marked "measured" was produced by a script in
`eval/benchmarks/` and reproducible on this machine.

## The capability boundary, which decides most of the answers

The engine is a **retrieval and evidence system**, not a generator. It chunks, embeds, does
hybrid recall, re-ranks with Laya's calibrated decision head, gates answerability, and returns
an evidence bundle with a calibrated probability. Text generation is a separate, opt-in step —
`LAYA_RAG_COMPOSE_MODEL` names the model and `LAYA_RAG_COMPOSE_BASE_URL` the endpoint — that
speaks the OpenAI-compatible chat route, by default to a local Ollama.

That single fact determines the answer for most of the list, and it is worth stating plainly
rather than dressing up:

| what a benchmark scores | whose score is it? | can this system be ranked? |
| --- | --- | --- |
| Retrieval quality | **this system's** | yes — run it |
| Answerability / abstention | **this system's** | yes — run it |
| Citation support (does the prose cite real spans) | **this system's** (`verify.py`) | partly — TREC RAG's support assessment is the published analogue, but no run exists |
| Answer correctness / faithfulness | the attached LLM's | only with a generator, and then the number is mostly the LLM's |
| Hallucination detection | nothing this system is | no |

No generator *weights* ship in the repo. What ships is `compose.py`, an opt-in HTTP client to
an external chat model (`LAYA_RAG_COMPOSE_MODEL`, by default a local Ollama), and this host runs
`ornith-1.5:35b`, which produces correctly-cited answers at roughly 10-15 s per query. So
end-to-end runs are *possible* — the Composer supplies the generator half — but every
generation-side number below is a statement about that model, not about Laya.

That is also why the four external targets are each blocked for a *different* reason, and only
two of them turn on prose at all:

| target | what it scores | why the Composer does not settle it |
| --- | --- | --- |
| **MTEB** | a submitted encoder's retrieval quality | scores a model artifact, not text. One *is* packaged in SentenceTransformers format (`eval/distill-nfcorpus-package/`, with a model card carrying its measured numbers) — but it sits under the gitignored `eval/`, was distilled for one corpus, and was never uploaded or submitted; the projection head is a JSON map, not a model |
| **KILT** | entity linking + retrieval + seq2seq generation | the Composer is the seq2seq half, but there is no entity linker or submission writer — and the row would score the attached LLM |
| **TREC RAG** | nugget recall/precision + citation support | the Composer can write cited answers; the blocker is NIST corpus and gold-nugget access plus the official AutoNuggetizer |
| **HHEM** | hallucination-*detection* models | a generator is the wrong kind of module — it is the source of hallucination, not a detector |

## Measured: retrieval-only

### BEIR (already the strongest evidence, from `RANKINGS.md`)

Five datasets, shipped `bge-small` bi-encoder, corrected evaluator. The last two columns are
the multi-model dense ensemble (`bge-small` + `bge-base` + `bge-large` by RRF), a single
configuration rather than a per-dataset choice:

| dataset | shipped lexical | bi-encoder | bi-encoder + fused BM25 | dense ensemble | **ensemble + BM25** |
| --- | --- | --- | --- | --- | --- |
| SciFact | 0.676 | 0.720 | 0.741 | 0.754 | **0.767** |
| NFCorpus | 0.324 | 0.339 | 0.367 | 0.379 | **0.387** |
| ArguAna | 0.417 | 0.596 | 0.568 | 0.656 | **0.651** |
| SciDocs | 0.150 | 0.197 | 0.199 | 0.222 | **0.225** |
| FiQA-2018 | 0.239 | 0.385 | 0.382 | 0.438 | **0.444** |
| **mean** | **0.361** | **0.447** | **0.452** | **0.490** | **0.495** |

**Placement.** On the four datasets where an independent matrix carries every published system,
and counting only systems that can be compared on all four:

| system | params | 4-dataset mean |
| --- | --- | --- |
| **this system — 3-model ensemble + BM25 (single config)** | 478M | **0.507** |
| **this system — best per dataset** | mixed (33M–335M) | **0.495** |
| BGE-large-v1.5 | 335M | 0.491 |
| SPLADE++ | 110M | 0.433 |
| GTR-XL | 1.2B | 0.405 |
| BM25 (Lucene) | — | 0.390 |
| ColBERT | 110M | 0.339 |

The cherry-picked best-per-dataset mean (0.495) is now beaten by a **single** configuration
(0.507) that uses no cross-encoder. The honest caveats are the cost — 478M parameters across
three encoders and three corpus embedding passes — and that the 335M single-encoder
configuration (`bge-small` + fused BM25) is still 0.469, which is the per-parameter claim.

### MultiHop-RAG (measured end-to-end retrieval, official protocol)

MultiHop-RAG is 2,556 records over 609 news articles — 2,255 answerable queries, evidence
spread across 2–4 documents, plus 301 `null_query` records whose gold answer is literally
*"Insufficient information."*

The important part is the **protocol**. The benchmark's own `retrieval_evaluate.py` splits
documents into 256-token chunks, substring-matches each gold evidence *fact* against each
retrieved chunk, and reports Hits@10 / Hits@4 / MAP@10 / MRR@10. Evaluating the same rankings at
whole-article granularity — the obvious first implementation — produces numbers that are not
comparable at all.

Measured here, reimplementing that protocol (`tools/multihop_official.py`).
Our rows use `chunk_overlap=200`; the published rows are the paper's Table 5, which is
**dense-only** — it contains no lexical baseline at all:

| retriever | Hits@10 | Hits@4 | MAP@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| published `bge-large` (no rerank) | 0.672 | 0.522 | 0.342 | 0.430 |
| published `bge-large` + `bge-reranker-large` | 0.718 | 0.636 | 0.476 | 0.563 |
| published best + reranker (`voyage-02`) | 0.747 | 0.663 | 0.480 | 0.586 |
| **this system — `bge-large` hybrid + BM25** | **0.810** | 0.672 | 0.255 | 0.545 |
| **this system — `bge-small` hybrid + BM25** | 0.802 | **0.686** | 0.251 | 0.540 |
| **this system — `bge-base` hybrid + BM25** | 0.799 | 0.668 | 0.249 | 0.530 |
| **this system — hybrid + coverage dedup** | **0.864** | **0.712** | **0.279** | **0.567** |
| **this system — BM25** | 0.782 | 0.638 | 0.253 | 0.550 |
| **this system — `bge-large` dense** | 0.735 | 0.561 | 0.197 | 0.435 |
| **this system — `bge-small` dense** | 0.708 | 0.537 | 0.186 | 0.407 |

Read honestly, that table says four things, and the first one is the least flattering:

- **Most of the win is lexical, not Laya.** Plain BM25 scores 0.782 Hits@10 — above *every*
  published dense system including the reranked ones. Any claim that this engine "beats the
  published baselines" on MultiHop-RAG is mostly a claim about news text having strong term
  overlap. The benchmark's own table simply never included a lexical baseline.
- **Against the published reranked systems**, the hybrid wins on Hits@10 (0.810 vs 0.747) and
  Hits@4 (0.712 with coverage dedup vs 0.663), with no cross-encoder and a 33M dense model.
- **On MAP@10 it loses badly** (0.279 vs 0.480). Hits@10 asks "was at least one piece of evidence
  found"; MAP@10 asks "was *all* of it found, and early". The hybrid is good at the first and
  weak at the second — the multi-hop failure mode you would expect from a lexical channel that
  locks onto one strongly-matching article. The coverage pass (skip a fused candidate whose cosine
  to a kept chunk is ≥ 0.9) is what moves it: MAP@10 0.255 → 0.279, Hits@10 0.812 → 0.864, both
  significant. Which coverage objective wins depends on chunking — at `chunk_overlap=0` the
  better one is query-term coverage, not dedup.
- **The `bge-large` head-to-head is done, and encoder size is not the remaining lever.**
  `bge-large` dense beats `bge-small` dense significantly (+0.027 Hits@10), but once BM25 is
  fused the difference disappears (+0.008 [−0.004, +0.020]) and `bge-base` hybrid is worse on
  Hits@4. Fusion absorbs the encoder gain.

**The protocol sensitivity is large enough to be its own finding.** Chunk overlap is a parameter
the published harness never states (llama-index's `SentenceSplitter(chunk_size=256)` leaves
`chunk_overlap=200` at its default). Changing only that:

| retriever | Hits@10 (overlap 200) | Hits@10 (overlap 0) |
| --- | --- | --- |
| BM25 | 0.782 | 0.844 |
| BM25 (stemmed + title) | 0.795 | 0.862 |
| `bge-small` dense | 0.708 | 0.752 |
| hybrid | 0.802 | 0.859 |

Up to **+0.06 Hits@10 from an undeclared chunking constant**, larger than the gap between most of
the published systems in Table 5. Any RAG leaderboard comparison that does not pin chunking is
comparing harnesses.

## Measured: negative rejection

MultiHop-RAG's 301 `null_query` records have no supporting evidence by construction, so a system
that always answers is wrong on all of them — the same property this engine's answerability gate
exists to handle. Measured through `RagEngine.query` on the indexed corpus (the corpus is a
distractor, deliberately: the articles discuss the entities named in the query, they just do not
answer it), with the shipped thresholds and `bge-small` as the dense channel:

| group | n | abstained | rate |
| --- | --- | --- | --- |
| `null_query` — abstention is **correct** | 301 | 93 | **30.9%** |
| answerable — abstention is an **error** | 300 | 86 | 28.7% |

**The gate does not discriminate on this corpus.** It abstains at essentially the same rate on
questions that cannot be answered and questions that can — a 0.022 difference, well inside the
sampling noise for these counts (SE ≈ 0.037). It behaves like a fixed ~30% abstention policy
rather than a decision about the evidence.

This is the most important negative result in this document, and it contradicts the intuitive
story that a calibrated answerability head is the natural answer to negative rejection. Caveats,
stated because they cut against the conclusion as much as for it:

- The gate is calibrated on this project's own corpora and question shape. MultiHop-RAG is
  2023 news with multi-entity, metadata-dependent questions — a domain shift, and thresholds
  were left at their shipped defaults rather than tuned here.
- These are the **hard** negatives: every `null_query` names entities that genuinely appear in
  the corpus; only the combination is unanswerable. A gate that keys on entity overlap will be
  fooled, and that is exactly what ~30% flat suggests.
- The dense channel was `bge-small`, not Laya's own encoder, so the passages reaching the gate
  are not the ones the gate was calibrated against.

What it implies is concrete rather than fatal: **abstention is not a portable, zero-shot
capability as things stand.** Claiming it on any external benchmark requires recalibrating the
answerability head on that domain first, and the calibration procedure is the thing that would
need to transfer — not the threshold.

**The recalibration test has now been run** (`tools/multihop_recalibrate.py`): treat
MultiHop-RAG as the out-of-domain split, fit a threshold — and a monotone Platt rescaling — on
half of the null/answerable queries, and score the other half. It does not transfer. ROC-AUC over
all 601 queries is 0.509 [0.463, 0.555]; over the 506 that reached the head it is 0.529 [0.479,
0.579]. The Platt slope fitted on the calibration half is −0.18. Across 300 stratified half-splits
the held-out balanced accuracy is 0.511 [0.472, 0.548] — chance. So this is **not a miscalibrated
threshold but the absence of a transferable signal**: recalibrating this head on a held-out
MultiHop split produces no discrimination, and the gate must not be cited as a strength out of
domain.

## The rest of the landscape

Verdicts, with the reason rather than a bare label. "Adapter" means the work is in this repo,
not in the benchmark.

### End-to-end RAG

| benchmark | what it actually scores | verdict |
| --- | --- | --- |
| **TREC RAG Track** | nugget recall/precision via AutoNuggetizer + citation support | **blocked** — corpus and gold nuggets need NIST access, and scoring needs the official assessor. The system's `verify.py` is the closest structural match to the support-assessment half of this track, but there is nothing to submit it to |
| **CRAG (KDD Cup 2024)** | answer accuracy over *provided* noisy web pages + mock KG/API tool calls | **blocked** — the retrieval is handed to you (5 HTML pages per question), so the engine's search is bypassed; it needs an HTML extractor and a tool-calling loop it does not have. Task 3 alone is ~50 full pages × 4,409 questions |
| **RGB** | noise robustness, negative rejection, information integration, counterfactual | **not adapted — investigated and declined.** The English files (`data/en*.json`, JSONL) ship per-query `positive`/`negative` passage lists, but every record has positives and `answer` is an alias list, so there is **no clean no-answer label** to score negative rejection against; the official rejection metric is judge-based, which this project does not run. A retrieval-side adapter would report noise robustness only and would not exercise the gate, so it was not built. `tools/bright.py` covers retrieval-only breadth instead |
| **FRAMES** | multi-hop factual QA over Wikipedia, exact-match accuracy | **runnable** — 824 questions, 0.5 MB, ships the Wikipedia links needed to build the corpus. Adapter: hours for retrieval; scoring needs a judge model |
| **MultiHop-RAG** | retrieval (above) + answer accuracy + negative rejection | **measured** (retrieval + rejection); generation not run |
| **2WikiMultiHopQA** | distractor-setting multi-hop retrieval: Hits@k, MAP@10, MRR@10 | **measured** — `tools/twowiki.py` over the 56,651-paragraph context pool: BM25 Hits@10 **0.9927** / MAP@10 **0.5037**, biencoder **0.9986** / **0.5223**, hybrid-biencoder **0.9997** / **0.5337** (all paired deltas p<0.0001). **The coverage objective does not travel**: coverage-0.7 and covquota-3-0.7 are **significantly worse** than the fused order handed to them (MAP@10 **−0.0046** / **−0.0056**, both p<0.0001) while preserving Hits@4 — the MuSiQue failure is not idiosyncratic. Retrieval-only, no generator; the relevance term is the fused RRF score, not a cross-encoder (`docs/findings-2wiki.md`) |
| **MTRAG** | multi-turn conversational RAG | **adapter needed** — the engine is single-turn; conversation needs query rewriting over history. Not a benchmark-config change |
| **RAGBench** | generation faithfulness/RAGAS-style metrics over *given* contexts | **not applicable to the engine** — it supplies the retrieved contexts, so it scores the attached LLM |
| **KILT** | entity linking + retrieval + generation, single leaderboard | **blocked** — needs an entity linker (BLINK) and a seq2seq generator; the leaderboard expects a full stack |

### Classic QA

| benchmark | verdict |
| --- | --- |
| **Natural Questions, TriviaQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, ASQA, ELI5** | **open-corpus variants blocked at full scale; the distractor retrieval setting is runnable and measured.** The open-corpus variants need a Wikipedia index (the DPR corpus behind NQ/HotpotQA is ~21M passages — a multi-day build even where disk allows it, and this host has 471 GB free). HotpotQA's distractor setting, 2WikiMultiHopQA and MuSiQue ship the gold paragraphs, so retrieval can be measured **inside the provided pool** — which is what `tools/twowiki.py` does (2Wiki above) and `tools/musique.py` does for MuSiQue; retrieval is not bypassed, only scoped to the distractor corpus. **Answer scoring** still needs a generator and a judge, so any EM figure would be a statement about them |

### Retrieval only

| benchmark | verdict |
| --- | --- |
| **BEIR** | **measured — 13 of 18 datasets**, counted from the reports present in `eval/beir/` (`make consistency-check`): SciFact, NFCorpus, ArguAna, SCIDOCS, FiQA-2018, Quora, DBPedia-entity, NQ, HotpotQA (**full 7,405 queries**, nDCG@10 0.6352), webis-touche2020, **Climate-FEVER** (lexical, nDCG@10 0.1610), **FEVER** (lexical, **nDCG@10 0.6323** over all 6,666 queries) and **TREC-COVID** (lexical, **0.6188** over the full 171,332-document collection, against a published BM25 of **0.595**; the earlier 40,000-document subsample read 0.3062 and is superseded). Membership is **stated, not implied**: the count is taken over the 18 datasets `Resources for Brewing BEIR` tabulates, and the gate reports `missing: bioasq, cqadupstack, robust04, signal1m, trec-news` beside the measured set. Robust04 / TREC-NEWS / `specter` / `signal1m` are **404 on the mirror with gated HuggingFace copies** and BioASQ (14.9M documents) is gated — this is **not** disk-bound (534 GiB free, so the "28 GB free disk" this row previously cited is stale). **MS MARCO (8.8M, lexical nDCG@10 0.4321) is measured but is *not* one of the 18** — it is the training collection, not a BEIR test dataset (`docs/findings-msmarco.md`) — and **CQADupStack**, the canonical member the earlier list omitted, is public on the mirror but needs a per-forum adapter |
| **MTEB (retrieval subset)** | **now first-class representable** — its retrieval tasks are largely BEIR's, and **MTEB v2 defines a `SearchProtocol`**, which `tools/mteb_search.py` implements over the engine itself (`.[mteb]`, `make mteb-eval`): chunking, fusion, the second stage and the gate are what get scored, not one encoder. Structural conformance is verified against mteb 2.22.5 and the corpus-id plumbing is tested; a full evaluation run and the upload remain external |
| **BRIGHT** | **partially measured** — `tools/bright.py` runs six CPU-cheap subsets (603 queries total). Macro mean nDCG@10: hybrid **0.1436** > BM25 0.1324 > dense 0.0991, and dense-only is *significantly* worse than BM25 on pony (−0.0823) and biology (−0.1013), which is the benchmark's premise. The hybrid is significantly better than BM25 on psychology (+0.0504); other per-subset deltas are ties at 76–112 queries. The large subsets (leetcode 414k, aops/theoremqa_questions 188k, stackoverflow 107k) are not run. Findings: `docs/findings-bright.md` |
| **MIRACL** | **blocked as-is** — 18 languages and per-language Wikipedia indexes (disk), and the dense channel would need a multilingual encoder. Laya's own checkpoint is multilingual, which is the one favourable fact |
| **LoTTE** | **runnable** — search/forum splits, long-tail; several collections are large |
| **ViDoRe** | **not applicable** — visual documents (screenshots/charts); this engine takes text only and has no OCR or vision path |

### Domain-specific

| benchmark | verdict |
| --- | --- |
| **FinanceBench** | **adapter + generator** — 10k questions over SEC PDFs; no PDF loader in this repo, and the score is answer accuracy |
| **LegalBench-RAG** | **runnable** — it is retrieval precision/recall over legal text, which is exactly this engine's shape. Adapter: corpus ingestion |
| **MIRAGE** | **adapter + generator** — medical RAG, scored end-to-end |

### Faithfulness and hallucination

| benchmark | verdict |
| --- | --- |
| **RAGTruth** | **not applicable** — it annotates hallucination spans in an LLM's output. Nothing here is a hallucination detector; `verify.py` checks citation *mechanics* (do cited spans exist), which is a different and narrower question |
| **Vectara HHEM leaderboard** | **no ranking** — it ranks hallucination-detection models, which this system is not |

### Tooling for fair comparison

| tool | verdict |
| --- | --- |
| **FlashRAG / BERGEN** | **integration, not a ranking** — either could host this engine as a retriever, and doing so is the right way to get a like-for-like number against their reproduced pipelines. Neither is a score |
| **RAGAS / ARES** | **runnable on generated output** — faithfulness, answer relevance and context relevance could be computed over composed answers; again these measure the generator against the retrieved evidence, and the context-relevance half is the only part that is this system's |
| **DeepEval / TruLens** | same class as above — metrics, not cross-system rankings |

## What can be claimed today, and what cannot

**Can be claimed.** Retrieval quality on BEIR is above published-BGE-large level: a single
three-model dense ensemble reaches **0.495 on five datasets and 0.507 on the four** the published
matrix carries (BGE-large 0.491), with no cross-encoder, and the 33M `bge-small` + fused BM25
configuration is 0.469 as the per-parameter claim. On MultiHop-RAG, Hits@10 and Hits@4 beat every
published system in the benchmark's own Table 5, reranking included — with the honest qualifier
that the published table has no lexical baseline and plain BM25 already beats it. A model-free
coverage pass lifts MAP@10 from 0.255 to 0.279.

**Cannot be claimed.** There is no end-to-end **ranking**: the first `engine + <generator>` row
exists only as a **pilot** — 12 answerable plus 8 `null_query` records through the engine at its
shipped defaults with the local `ornith-1.5:35b`, where **EM 0.000 sits beside a 0.667
gold-containment rate on the same nine answers** (prose against short entity golds) and ~36 s/query
puts the full 2,255-record row at 22–25 h on this host. See
[`docs/findings-endtoend-pilot.md`](docs/findings-endtoend-pilot.md). There is no KILT, MTEB, HHEM or
TREC RAG placement, because those need a submission, a full stack, or access this machine does not
have. **Answerability does not transfer**: on MultiHop-RAG the gate abstains at ~30% whether or not
the question is answerable, an AUC of 0.51, and a threshold recalibrated on a held-out MultiHop
split scores at chance (0.511 balanced accuracy) — so calibrated abstention is a property of a
calibrated corpus, not a portable capability. **Stronger scorers did not pay off here**: a 335M
pointwise reranker over the fused top-100 lost nDCG@10 on all five BEIR datasets, full-split
fine-tuning with mined hard negatives scored below zero-shot, and all three Phase 2 ordering heads
failed — a topical late-interaction model, a relational NLI stance score, and a cross-encoder
adapted on MuSiQue's own train pairs all scored *below* the baselines they were meant to improve.
And any faithfulness or answer-accuracy number produced with a local 35B model is a statement about
that model, not about this engine — it is labelled that way wherever it appears.

## What was expected to move this, and what actually did

The forecast list has been run; [`README.md`](README.md) and [`RANKINGS.md`](RANKINGS.md) carry
the measurements. In brief:

1. **`bge-large` under the MultiHop-RAG protocol — done.** It helps dense retrieval and does
   nothing for the fused hybrid; encoder size is not the gap. The head-to-head is now real.
2. **Recalibrate the answerability head out of domain — done, negative.** The split does not
   transfer (AUC 0.51), so there is no signal to recalibrate.
3. **Fix MAP@10 with a coverage objective — done.** A near-duplicate pass on the fused top-100 is
   worth +0.024 MAP@10 and +0.052 Hits@10; a model-free win where a bigger ranker did not help.
4. **A multi-model dense ensemble — done, and it beat its estimate** (+0.043 macro nDCG@10).
5. **The second stage (reranker) and full fine-tuning — both done, both negative.**

What remains open is a listwise or late-interaction ranker rather than another pointwise one, a
distillation of the three-model ensemble back into one encoder, and the scale work needed to run
the large BEIR corpora (the on-disk postings and streaming index are in place; BioASQ-scale still
needs a bounded external merge and an on-disk ANN).

## Reproducing

The harnesses are tracked in `tools/`; the downloaded datasets, indexes and reports stay in the
gitignored `eval/`:

```
.venv/bin/python tools/multihop_doclevel.py      # whole-article framing — NOT comparable
.venv/bin/python tools/multihop_official.py      # the benchmark's own 256-token protocol
.venv/bin/python tools/multihop_gate.py          # negative rejection through the engine
.venv/bin/python tools/multihop_recalibrate.py   # does a threshold fitted on one split transfer?
.venv/bin/python tools/multihop_coverage.py      # coverage objectives for MAP@10
.venv/bin/python tools/beir_routing.py           # per-corpus channel selection
.venv/bin/python tools/ondisk.py                 # on-disk postings + streaming dense index
```

`multihop_doclevel.py` is kept deliberately: it is the naive framing that produced numbers up to
0.08 above the honest ones, and it is the reason the protocol section above exists.
