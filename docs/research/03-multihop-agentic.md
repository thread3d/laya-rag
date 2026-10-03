# Multi-hop / agentic retrieval: ranked additions for laya-rag

Theme: query decomposition, IRCoT, multi-hop graphs, late interaction, doc-vs-passage
aggregation, set-coverage, no-generator hop-2. Target: raise candidate-pool recall and beat
MAP@10 **0.2835** (best fused+coverage) / **0.2986** (CE rerank, 400-query subset).

**Cross-cutting finding (read first).** Two mechanisms dominate your gap. (a) The **second-hop
tax**: [redhop, 2025](https://github.com/vysakh0/redhop/blob/main/docs/findings/SECOND_HOP_TAX.md)
measured directly, with bootstrap CIs, that *any* relevance-based selection — cross-encoder
rerank, threshold filtering, density pruning — discards the low-query-relevance hop that is
linked to the answer only through a bridge entity. **Verified against the primary source
(2026-10-02):** on the *distractor-filtered* strategy, second-hop retention falls from
**0.968 [0.958, 0.977]** at τ=0.05 to **0.439 [0.411, 0.468]** at τ=0.30 (HotpotQA, n=1327,
mean relevance gap 0.234), and from **0.751** to **0.043 [0.034, 0.053]** (MuSiQue, n=1484, gap
0.293). A `reasoning_preserving` strategy that readmits low-relevance chunks lexically linked to
a kept seed recovers **+2.1 / +8.4 / +23.2 pts at τ=0.10/0.20/0.30** on HotpotQA and **+28.8 pts
at τ=0.20** on MuSiQue (CIs non-overlapping from τ=0.10). The source's own honest limits:
the rescue signal is **lexical** (a pure-paraphrase bridge with no shared surface tokens is not
rescued); **retention is reachability, not answer quality**; and the MuSiQue density-pruning
sub-result **did not survive** a sharper grounding signal (the filtering tax is the robust
two-dataset core). Laya's relevance gate and your CE both pay this tax. (b) **Single-vector
topical matching**: one query embedding matches the bridge entity, not the answer facts. Your
own numbers said recall was decent (Hits@10 0.864, pool ceiling 0.7307 at depth 100 — **corrected to
0.609**, which changes this premise: see the note at the end of this file) — but
that ceiling is far below 1.0, so recall still bounds everything. Caveat throughout: published
0.479 is reranked; your 0.2986 already exceeds 0.2835, so rank the list below against **0.2986**,
not 0.2835.

---

**1. Bridge-linked second-hop rescue (reasoning-preserving rerank).** Keep the query-relevant
seeds, then readmit low-relevance candidates that are *lexically/semantically linked to a kept
seed* via a bridge entity; drop only unlinked junk. · Attacks (a) directly; it is the only item
that targets the measured cause of the MAP gap, and it composes with your CE. · Maturity:
measured in 2025 on two benchmarks, with CIs; small project, not peer-reviewed. ·
[redhop](https://github.com/vysakh0/redhop/blob/main/docs/findings/SECOND_HOP_TAX.md). · Cost:
**days**, pure Python, reuses existing BM25/entity terms and CE scores; no model. · Hypothesis:
recovers ~10–20 pts of second-hop retention → MAP@10 **+0.02 to +0.04**, and larger at depth 500.
· Falsify: log per-query second-hop retention (lowest-relevance gold chunk) in
`tools/multihop_official.py`; if the rescue does not raise it, or raises it without raising MAP,
the tax is not your bottleneck. Skeptical: authors' own caveat — retention ≠ answer quality;
lexical linkage misses paraphrase bridges.

**2. Late-interaction multi-vector retrieval (ColBERT-style MaxSim).** Replace/augment the
single bge vector with per-token embeddings so different query tokens match different evidence
facts. · Attacks (b); the highest-recall, no-generator lever for multi-hop. · Maturity: mature.
[ColBERTv2, NAACL 2022](https://arxiv.org/abs/2112.01488),
[PLAID, 2022](https://arxiv.org/abs/2205.09707); small model
[answerai-colbert-small-v1, Aug 2024](https://huggingface.co/answerdotai/answerai-colbert-small-v1)
(33M params, BEIR-competitive with far larger encoders). Read
[the PLAID reproducibility study](https://arxiv.org/abs/2404.14989) before trusting PLAID's
latency claims. · Cost: **1–2 weeks**; no GPU. Make it CPU-viable with
[MUVERA fixed-dimensional encodings, 2024](https://arxiv.org/abs/2405.19504) or
[SLIM sparsified late interaction, 2023](https://arxiv.org/abs/2302.06587) instead of full PLAID.
· Hypothesis: MAP@10 **+0.03 to +0.07**, biggest on the low-relevance second hop. · Falsify:
same-protocol A/B against bge-small on the 400-query subset; if it does not beat hybrid on
pool recall at depth 100, it is not worth the storage. Skeptical: ColBERTv2 ≈ bge-large on
single-hop BEIR, so the win must appear on multi-hop or not at all.

**3. Generator-free hop-2 expansion from bridge entities (IRCoT-lite).** After pass 1, mine
salient entity phrases from the top chunks that are absent from the query, issue them as extra
dense+BM25 queries, and RRF-fuse into the pool. · Attacks recall (raises the ceiling itself). ·
Maturity: the LLM version is well-established —
[IRCoT, ACL 2023](https://aclanthology.org/2023.acl-long.557/),
[self-ask, 2022](https://arxiv.org/abs/2210.03350) — but the *generator-free* variant is a
build, not a published system. · Cost: **days**, pure Python + existing index. · Hypothesis:
pool ceiling 0.609 → **0.70–0.75** (re-based), MAP@10 **+0.02 to +0.05**. · Falsify: measure ceiling at
depths 100 and 500 before/after. If the ceiling moves but MAP does not, the problem is fully
selection-side and item 1 subsumes this. Skeptical: this is adjacent to the BM25 PRF you already
rejected — use *entity* terms (capitalised n-grams / noun chunks filtered by corpus DF), not raw
top-document terms, and always fuse rather than replace, or it will reproduce the PRF failure.

**4. Non-LLM entity graph + personalized PageRank (HippoRAG-lite).** Build entity nodes and
passage nodes offline from spaCy NER/noun-chunk co-occurrence (no OpenIE LLM), run PPR seeded on
query entities, retrieve passages by PPR mass. · Attacks (b) structurally: bridge entities become
graph paths instead of embedding collisions. · Maturity: strong with LLMs —
[HippoRAG, NeurIPS 2024](https://arxiv.org/abs/2405.14831),
[HippoRAG 2, ICML 2025](https://arxiv.org/abs/2502.14802) — but published gains depend on
GPT-3.5/4 OpenIE. [GraphRAG, 2024](https://arxiv.org/abs/2404.16130) needs an LLM for community
summaries and is expensive; treat vendor claims skeptically. · Cost: **1–2 weeks** (spaCy CPU,
scipy sparse PPR). · Hypothesis: ceiling **+0.03 to +0.06** on associative hops. · Falsify: if
PPR ranking of the depth-100 pool does not beat RRF on gold-fact coverage, the graph is noise.
Skeptical: entity linking/alias resolution, not PPR, is the failure point; a noisy non-LLM graph
may underperform plain BM25.

**5. Document-level aggregation + per-document chunk emission.** Retrieve at article level
(max/mean-pool chunk scores, or index concatenated articles), then emit the best chunks per
retrieved document. · Attacks the protocol: MAP@10 credits *facts* against 256-token chunks, so
splitting one article across ranks leaks credit; a doc-level first stage then intra-doc chunk
ranking is structurally better suited to the set-aware metric. · Maturity: standard practice, but
the literature evidence is thin and largely single-hop; this is an ablation, not a proven win.
Anchor to your own harness and [MultiHop-RAG's `simple_retrieval.py`](https://github.com/yixuantt/MultiHop-RAG). ·
Cost: **days**. · Hypothesis: MAP@10 **+0.01 to +0.03**, mostly from de-duplicating repeated
chunks of the same article. · Falsify: run with `overlap=0` (your `coverage-ov0` already wins
there) and compare. Skeptical: doc-level retrieval can *hurt* recall when the query needs one
sentence from a long article; do not ship without the A/B.

**6. CE × set-coverage joint selection (extend what you have).** Feed cross-encoder scores into
`facility_location_select` as the relevance term, add an explicit bridge-rescue term (item 1) and
a per-document quota. · Attacks ordering inside the pool; lowest-risk path because both
components already measured positive independently (coverage 0.2835; CE 0.2986). · Maturity:
your own measured components plus classic submodular/MMR selection. · Cost: **days**, edit
`tools/multihop_coverage.py` / `engine._coverage_select`. · Hypothesis: **0.2986 → 0.31–0.33**.
· Falsify: if CE+coverage does not beat CE alone on the held-out split, coverage is subsumed on
this corpus. Skeptical: the CE is the dominant term; the coverage gain may vanish under it.

**7. RAPTOR-lite: extractive cluster routing (no summaries).** K-means the chunk embeddings,
build cluster nodes from *extracted* central sentences and retrieve at the node level as a
routing layer. · Attacks long-range aggregation. · Maturity:
[RAPTOR, ICLR 2024](https://arxiv.org/abs/2401.18059) is real but needs LLM summarisation and its
gains are on long-document QA (QASPER/NarrativeQA), not short news. · Cost: **days**. ·
Hypothesis: Hits@10 **+0.01 to +0.02**, MAP flat. · Falsify: cheap — build, measure, delete.
Skeptical: MultiHop-RAG articles are short; hierarchy may have nothing to add. Do this last.

**8. Local small-generator decomposition — ⚠ NEEDS A GENERATOR (local, no API).** Few-shot
sub-question generation with a 0.5–1B model, run through your existing Ollama path, then RRF the
sub-question results. · Attacks (b) at the source. · Maturity: high evidence
([self-ask 2022](https://arxiv.org/abs/2210.03350),
[Query2Doc, EMNLP 2023](https://arxiv.org/abs/2303.07678),
[IRCoT ACL 2023](https://aclanthology.org/2023.acl-long.557/)); non-LLM precursor
[unsupervised QA decomposition, EMNLP 2020](https://aclanthology.org/2020.emnlp-main.713/). ·
Cost: **1–2 weeks** + model download; MPS OK, CPU slow. · Hypothesis: **+0.03 to +0.06**, highly
generator-quality-dependent. · Falsify: if sub-questions do not retrieve gold facts absent from
pass 1, drop it. Skeptical: small local models decompose poorly; decomposition hurts atomic
queries; it also violates "no API keys" only in spirit (local model is fine, but it is an LLM).

**9. Evaluation expansion (diagnostic, not a retrieval method).** Add
[MuSiQue](https://arxiv.org/abs/2108.00573),
[2WikiMultiHopQA](https://arxiv.org/abs/2011.01084),
[FRAMES, NAACL 2025](https://arxiv.org/abs/2409.12941) and
[BRIGHT, ICLR 2025](https://arxiv.org/abs/2407.12883) as retrieval-only sets (no answer
generation needed). · Attacks *falsifiability*: your whole multi-hop conclusion rests on one
benchmark. · Cost: **hours–days** (downloads/disk). · Hypothesis: the second-hop tax replicates;
if MultiHop-RAG is idiosyncratic, item rankings change. Skeptical: BRIGHT and FRAMES official
scores need a generator; compute recall/nDCG yourself and label them non-comparable. Also see
[beam retrieval, NAACL 2024](https://aclanthology.org/2024.naacl-long.96/) and
[intermediate-representation hop retrieval, 2025](https://arxiv.org/abs/2503.04796) if you later
want trainable beam rerankers.

**Not recommended / already dead.** Re-running BM25 PRF (you rejected it — item 3 must use
entities and fusion), full re-sort cross-encoders (BEIR-negative, MultiHop-positive; keep it as a
*filter*/blend), GraphRAG-style LLM community summaries on one CPU/MPS host (cost without
multi-hop news evidence).

> **Correction, 2026-10-03.** The ceiling quoted here (0.7307 at depth 100) divided the oracle AP by the
> gold facts *the pool matched* rather than by the query's true gold count, which inflated it on exactly
> the queries whose recall was failing. Corrected it is **0.609**, and it **rises** with depth
> (0.397 / 0.609 / 0.659 at depth 10 / 100 / 500). 36% of depth-100 pools are missing a gold fact, so
> recall is a live lever, not a solved one. See the README diagnostics subsection.
