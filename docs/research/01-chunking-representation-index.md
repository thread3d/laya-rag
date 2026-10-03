# Brief: chunking, representation, and index architecture (2024–2026)

Scope: additions for `laya-rag`, grounded in its own numbers — fact loss when a fact exceeds the
400-char overlap and spans >1 sentence; multi-hop MAP@10 **0.2835** against an oracle reorder
ceiling of **0.7307** on the fused top-100 (corrected 2026-10-03 to **0.609**; see the note below);
BEIR fused mean 0.452 / ensemble 0.495; passage-level indexing and pointwise CE reranking already
measured and **rejected** on BEIR. The ceiling is the key fact: the binding constraint is
**selection inside a strong first stage**, so recall-oriented additions must justify themselves
against a 0.73 ceiling the pool already contains.

Legend: **[Q]** query-time, **[I]** ingest-time. "Parity" = preserves the existing chunk ranking.

## Ranked additions

### 1. Atomic retrieval units + parent-window return (model-free small-to-big) — [I]+[Q], parity: no
Index sentences (or the existing sentence atoms) as retrievable units, then return the enclosing
window. **Attacks the stated fact-loss class**: a two-sentence fact longer than the overlap is
split today, but if *either* sentence is retrieved the window returns both. Also raises multi-hop
MAP — MAP@10 substring-matches gold facts, so atom-sized slots pack more distinct facts per
top-10. Maturity: production pattern in
[LlamaIndex SentenceWindowNodeParser](https://developers.llamaindex.ai/python/framework/module_guides/loading/node_parsers/modules/)
and LangChain [ParentDocumentRetriever](https://reference.langchain.com/python/langchain-classic/retrievers/parent_document_retriever/ParentDocumentRetriever),
but **no peer-reviewed benchmark exists for the parent-window pattern itself**. Closest formal
evidence:
[Raina & Gales, 2024](https://arxiv.org/abs/2405.12363) find retrieving atomic statements beats
retrieving chunks on recall. Their setup generates **15 ChatGPT-3.5 questions per atom** and the
corpus grows from 2,067 chunk vectors to **~205k–252k atom vectors (~100–120×)** — and they report
the gains are "less clear" under the stronger `e5-base-v2` embedder. The paper is an arXiv preprint
with no venue; its code is promised "if accepted". Cost: days, no LLM —
keep chunk-level BM25, add an atom index + `atom → parent` map + parent dedup on the fused list.
Hypothesis: MAP@10 **+0.01–0.04**, fact-loss class removed, near-zero BEIR change. Falsify: atom
index → top-100 → parents vs the fused baseline on MAP@10/Hits@4. Skeptical: short units destroy
BM25 term statistics, and 600-char passage indexing was already rejected here (SciFact
0.682→0.659, TREC-COVID 0.306→0.245); mitigate by using atoms on the dense side only.

### 2. Deterministic structural context prefix (contextual retrieval without the LLM) — [I], parity: no
Prepend a heading breadcrumb + document title + path + first line to each chunk **before both**
the dense encode and the BM25 index. Attacks the context-starvation half of the fact gap (a chunk
saying "revenue grew 3%" with no company/period). Maturity: [Anthropic Contextual
Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) (Sep 2024) reports 35%
fewer top-20 failures from contextual embeddings, 49% with contextual BM25, 67% with reranking —
vendor-run, non-peer-reviewed, Claude-generated. Their exact cost assumptions: **800-token chunks,
8k-token documents, 50-token instructions, 100 tokens of generated context → $1.02 per million
document tokens** with Claude 3 Haiku, reranking top-150 → 20, and prompt caching cutting cost up
to 90%. The deterministic prefix is the cheap 80%; the
repo's own `bm25-title` was a small kept win. Cost: hours, no LLM, requires re-index. Hypothesis:
**+0.005–0.02** on title-dependent corpora (SciDocs), small on multi-hop. Falsify: A/B the prefix
on five BEIR corpora + MultiHop. Skeptical: repeated context tokens inflate `df` and dilute
distinctive terms; title repetition was significant only on ArguAna here. Do **not** run the LLM
version at scale: even Anthropic's $1.02/M-document-tokens assumes prompt caching; a local 35B
Ollama call at ~10 s/chunk is thousands of hours for 100k chunks.

### 3. Late chunking (embedding-side context, no LLM) — [I], parity: no
Embed the whole document with a long-context encoder, then mean-pool token vectors *per chunk*
after the transformer ([Günther et al., 2024/25](https://arxiv.org/abs/2409.04701)). Reported
**+3.63% relative nDCG@10 (≈1.9 abs)** over sentence chunking, 3 models × 4 BeIR tasks —
author-reported; gains shrink with chunk size and naive chunking won on some
reading-comprehension sets. Notably, their Appendix A.2 finds **chunk overlap gave no clear
advantage** (256-token chunks ±16), which is direct evidence against the value of this repo's
400-char overlap. Makes a boundary-split fact present in both chunk embeddings; does
**nothing for BM25**. Cost: **high** — long-context encoder (jina-v3 / bge-m3, ~570M) vs the 33M
bge-small, plus token-offset pooling. Hypothesis: **+0.01–0.03** on long documents; ~0 on BEIR
abstracts and 256-token MultiHop chunks. Falsify: re-embed the long MultiHop articles vs
bge-small chunks. Skeptical: `bge-large` gave no gain over `bge-base` here, RRF absorbs encoder
changes, and the ceiling says representation is not binding.

### 4. Proposition / atom indexing (Dense X Retrieval) — [I]+[Q], parity: no
Index atomic propositions so each unit is one fact
([Chen et al., EMNLP 2024](https://arxiv.org/abs/2312.06648)): reported **Recall@20 +10.1**
(unsupervised) / **+2.7** (supervised) over passage indexing, author-reported. Fits a
fact-substring MAP metric conceptually. Cost: **LLM at ingest, prohibitive locally** — the
reference pipeline expands 41.39M passages to **256,885,003 propositions** (6.3 per passage,
~6.2× the index) using Flan-T5-Large fine-tuned on 42,857 GPT-4 pairs (extraction F1 0.822) and
consumes **~500 P100 GPU-hours**. EMNLP 2024 Main. The model-free proxy is #1. Hypothesis (LLM):
MAP@10 **+0.02–0.05**. Falsify: same-pool atom vs chunk index at equal depth. Skeptical: the existing
atom-level *selection* already harvests +0.0288 from the same top-100, so retrieval-side atoms may
be redundant; LLM propositions hallucinate and inflate the index.

### 5. Late-interaction (ColBERT/PLAID) second stage — [Q]+[I storage], parity: no
Score the fused top-100 by MaxSim over token embeddings
([Khattab & Zaharia, 2020](https://arxiv.org/abs/2004.12832);
[ColBERTv2, 2021](https://arxiv.org/abs/2112.01488);
[PLAID, 2022](https://arxiv.org/abs/2205.09707)). Attacks the distributed single-vector
matching gap in multi-hop retrieval. **It is not a relational fix**: ColBERT MaxSim is still
topical token matching, so it does not repair ArguAna (the 33M `answerai-colbert-small-v1` scores
50.09 there and 76.11 on HotpotQA — see brief 02/03). Cost: **high** — multi-vector
store, residual compression, NumPy MaxSim, feasible only at rerank depth. Hypothesis: MAP@10
**+0.03–0.08** on multi-hop. Falsify: MaxSim over bge-small token vectors vs `bge-reranker-base`
(already +0.046 MAP there). Skeptical: without ColBERT-style training the token embeddings may
underperform and the reranker is cheaper.

### 6. Structure/code-aware chunking (Markdown headings, tree-sitter AST) — [I], parity: no
Split Markdown on headings and code on function/class boundaries; keep config files whole. The
engine ingests code but chunks it as prose. Maturity: active 2024–25 area, but be careful —
[CodeRAG-Bench](https://arxiv.org/abs/2406.14497) (Findings of NAACL 2025) tests only **fixed
token-window chunking** (200–800 tokens optimal) and finds repo-level gains "rather limited"; it is
**not** evidence for AST/function-boundary chunking. Treat the structure-aware idea as a local
hypothesis, not a cited result. Cost: low — a pure-Python heading splitter
needs no dependencies; tree-sitter is optional. Hypothesis: **large for code** (function-level
facts stop being cut), ~0 for prose BEIR/MultiHop. Falsify: function-level code retrieval set over
the repo's own source tree, gold containment vs fixed windows. Skeptical: no code-RAG benchmark
exists here, so the gain is unmeasured against the stated gaps.

### 7. Matryoshka truncation + binary/scalar quantization with fp32 rescore — [I]+[Q], parity: yes (with rescore)
Short-vector first pass for memory/speed, then fp32 rescore of top-k. Attacks the scale gap, not
quality ([Kusupati et al., 2022](https://arxiv.org/abs/2205.13147);
[HF embedding quantization, 2024](https://huggingface.co/blog/embedding-quantization);
[RaBitQ, SIGMOD 2024](https://arxiv.org/abs/2405.12497): **one bit per dimension** (D bits per
D-dim vector, 4-bit query), an unbiased estimator with an O(1/√D) error bound). Cost: moderate —
NumPy popcount plus the existing mmap fp32 store. Hypothesis: 8–32× smaller dense index, 2–5× scan
speed, recall parity after rescore. Note the HF "~96% quality retained" figure is
**model-specific**: 96.45% for mxbai-embed-large-v1 but 93.79% (all-MiniLM-L6-v2), 87.7%
(nomic-v1.5) and 74.77% (e5-base-v2) — write "up to ~96%, setup-dependent". Falsify: extend
`tools/ann_bench.py` with a quantized first pass + rescore.
Skeptical: bge-small is not MRL-trained, so truncation is out of distribution; the repo's bench
shows exhaustive scan is only 11.6 ms at 1M vectors, so no measured quality problem exists yet.

### 8. RAPTOR-style recursive summary tree / document units — [I]+[Q], parity: no
Cluster and recursively summarise into a tree queried alongside leaves
([Sarthi et al., ICLR 2024](https://arxiv.org/abs/2401.18059)). Cost: **LLM at ingest** at every
level plus UMAP/GMM clustering; high. The headline QuALITY number (82.6%, vs prior best 62.3%) is
against the prior published best; the **controlled UnifiedQA-3B ablation is only +1.6–2.2 points**
(+1.7 SBERT, +2.2 BM25, +1.6 DPR). Falsify: a
two-level summary index on MultiHop, MAP@10. Skeptical: the ceiling shows recall inside top-100
is not binding, and Anthropic reports summary-based indexing as low performance and generic
document summaries as "very limited gains" (2024). Lowest priority.

### 9. doc2query / summary-augmented chunks — [I], parity: no
Append generated queries (or a parent-doc summary) to each chunk before BM25 indexing so lexical
matching sees the query's vocabulary. doc2query is a **production standard for lexical retrieval**
([Nogueira et al., 2019](https://arxiv.org/abs/1904.08375): MS MARCO dev MRR@10 18.4→21.5 at
~1.8× query latency, 50→90 ms/query, ~7× faster than a neural reranker), but that predates modern
dense retrievers; note the model is a **from-scratch 6-layer encoder/decoder Transformer** trained
in OpenNMT on ~500k MS MARCO pairs — not a T5 fine-tune;
[summary-augmented chunking, 2025](https://arxiv.org/abs/2510.06999) reports less document-level
retrieval mismatch in legal RAG. Cost: **LLM at ingest** (small seq2seq or local summariser);
index grows. Hypothesis: **+0.01–0.04** lexical recall, mostly on short low-overlap queries.
Falsify: doc2query expansion on five BEIR corpora with BM25 fixed. Skeptical: BM25
pseudo-relevance feedback was already rejected hard here (SciFact 0.682→0.424); doc2query is a
learned document-side version of the same expansion and may drift the same way.

## Deprioritized
- **Semantic (embedding-similarity) chunking**: compute cost "not justified by consistent
  performance gains" ([Qu et al., 2024](https://arxiv.org/abs/2410.13070)) — an arXiv preprint
  whose scope is 3 tasks / 10+5 datasets / 3 chunkers / 3 embedders; Jina measured
  semantic boundaries as the *smallest* of three chunkers' gains.
- **Metadata extraction + self-query filtering**: product framing; noisy extraction silently
  removes true positives, and no precision gap is measured here.
- **DiskANN / ScaNN**: research-grade, Linux/x86-oriented; irrelevant below the regime where the
  repo's IVF already returns recall 1.00. **HNSW**: IVF dominates it here (recall 1.00 at 16× vs
  0.744 at 724×); revisit with real embeddings.

## Verdict
#1–#2 are cheap, LLM-free, and attack the two *measured* gaps (fact loss, missing document
context). #5 is the largest quality lever but the most expensive and most in tension with the
measured CE result. #3–#4 are representation bets the oracle ceiling argues against. #7–#9 are
scale/lexical plays with no measured quality gap behind them yet. Everything except #7 changes the
candidate set or the index, so none preserves ranking parity by default — budget a re-index and a
held-out paired bootstrap each.

> **Correction, 2026-10-03.** The ceiling quoted here (0.7307 at depth 100) divided the oracle AP by the
> gold facts *the pool matched* rather than by the query's true gold count, which inflated it on exactly
> the queries whose recall was failing. Corrected it is **0.609**, and it **rises** with depth
> (0.397 / 0.609 / 0.659 at depth 10 / 100 / 500). 36% of depth-100 pools are missing a gold fact, so
> recall is a live lever, not a solved one. See the README diagnostics subsection.
