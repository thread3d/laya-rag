# The best RAG system in the world — what that would take, in October 2026

A frontier assessment written after the 1.2.2 correctness pass. It answers one question: **given what
the field grades in late 2026, what would make this the best RAG system in the world — and what would
not?**

It is a delta on three documents, not a replacement for them: [`ROADMAP.md`](../ROADMAP.md) §0 (the
narrowed target, stated so it can be falsified), [`RESEARCH.md`](../RESEARCH.md) (the seven-theme
survey and the do-not-retry ledger) and [`docs/research/08-frontier-2026.md`](research/08-frontier-2026.md)
(the 2026-10-03 delta pass). Most of *that* programme has since been executed and reported in
[`docs/FINDINGS.md`](FINDINGS.md); §6 is the updated ledger, so nothing here re-proposes a measured
negative.

Evidence is labelled throughout, because the labels are the argument:

- `[repo]` — measured in this repository; the findings document is named.
- `[measure]` — measured during this pass on this host; the artefact path is given.
- `[web]` — read from a cited source on 2026-10-29; the URL is given.
- `[unverified]` — looked for and not found. Never used to support a claim.

---

## 0. The one-paragraph answer

The headline number the field reports — best retrieval nDCG, best multi-hop, best answer quality — is
not winnable from a single CPU host in 2026. The retrieval leaders are 0.6B–8B encoders or closed APIs,
the multi-hop leaders are RL-trained agentic systems built around a generator, and every end-to-end
leaderboard scores *generated text*. But three things changed in the last eighteen months, and
together they define a winnable claim.

**First, the field started grading what this architecture already computes**: retrieval under
unanswerable and underspecified queries, abstention as a first-class score, and support/attribution of
a returned evidence set. SemEval-2026 Task 8 says out loud what this programme has argued since v1.0 —
*"abstention is a differentiating capability not correlated with overall generation quality"*
`[web]` ([SemEval-2026 Task 8](https://aclanthology.org/2026.semeval-1.447.pdf)).

**Second, "small won" in retrieval.** The current leader of MTEB's English retrieval board is a
**0.596B** model beating every 7–14B open model and Google's embedding API by 6.3 points `[web]`
([model card](https://huggingface.co/IEITYuan/Yuan-embedding-2.0-en), board snapshot 2026-10-29), a
**25M** MIT-licensed model is first on the new domain-heavy RTEB board at *zero* task overlap `[web]`,
and a zero-parameter BM25 baseline is within **0.7 points of the leader on long-document retrieval**
`[web]`. The component market moved underneath this project's most load-bearing negatives, which were
measured with 2023–2025 components.

**Third, the niche is empty.** As of 2026-10-29 no adopted open-source RAG system is simultaneously
fully local, CPU-only, key-free and backed by a held-out retrieval measurement; everything that
behaves like an *engine* needs a generator, and everything that publishes numbers measures the
generator's output `[web]`.

So the claim to aim at is not "highest nDCG". It is **the reference evidence engine** — the system that
returns the passages that support an answer, with a calibrated probability that the corpus can answer
at all, refuses when it cannot, cites mechanically and verifiably, and is the only RAG system whose
every published number is re-derivable from a report a resumable job produced, including the
unfavourable ones. The work between here and that claim is a component refresh under this project's own
protocol, three measured gaps, and two leaderboard entries that do not need a generator.

---

## 1. What the field grades in 2026, and what a non-generative engine can enter

### 1.1 The grading stack gained two layers, and one of them is ours

Between TREC RAG 2024 and 2025 the evaluation moved from "did the answer match the gold string" to a
four-layer stack `[web]` ([TREC RAG](https://trec-rag.github.io/)):

1. **Retrieval** under the Cranfield paradigm — human qrels, nDCG@k, Recall@k. **No generator.**
2. **Response quality** as *nugget recall*: key information units extracted from judged-relevant
   documents, scored by how many *vital* nuggets a response fully supports.
3. **Support / attribution** — every answer sentence graded against its cited document, reported as
   weighted precision and weighted recall.
4. **Answerability conditioning** — MTRAG's IDK judge decides *first* whether a response is a full
   answer, a partial answer or an explicit refusal, and only then scores quality
   `[web]` ([MTRAG, TACL 2025](https://aclanthology.org/2025.tacl-1.36.pdf)); SemEval-2026 Task 8 turns
   that into a score table where an explicit refusal on an unanswerable question scores **1** and a
   confident answer scores **0**.

Layer 1 is scorable on evidence alone. Layers 2–4 take a string of natural-language answer text. The
corollary the programme should state plainly: **the engine cannot be *ranked* on layers 2–4, but it can
be *scored* on the inputs those layers are built from** — nugget coverage of a returned evidence set,
support of a claim against its cited passage, and a calibrated probability that a question is
answerable at all. `tools/nugget_recall.py`, `verify.py` and `tools/calibration_report.py` already
compute those three; none of them is published as a protocol.

### 1.2 Venues whose submission path needs no generator

| Venue | What is submitted | Metric | 2026 state |
| --- | --- | --- | --- |
| **TREC RAG — Retrieval task** | ranked document ids; "a custom retrieval system" is explicitly allowed, submission depth `k` per narrative `[web]` | nDCG@30/@100, Recall@100 | 2026 round closed 8 Aug 2026 with **no leaderboard yet** (conference Nov 2026); next is TREC 2027. 2025's best was nDCG@30 **0.6934** by an LLM-in-the-loop pipeline against BM25's **0.3468** |
| **SemEval-2026 Task 8 — Subtask A** | ranked passages over MTRAG-UN (unanswerable, underspecified, non-standalone, unclear) `[web]` | nDCG@5, Recall@5 | winner **0.578**, a *sparse* system; organisers' baselines BM25 0.354 < BGE-base-1.5 0.390 < Elser 0.480 |
| **LongEmbed** (MTEB) | rankings over long documents | nDCG@10 (6 tasks) | **bm25s 0.9087 against a leader at 0.9153** — a hybrid stack is a genuine candidate here `[web]` |
| **RTEB(eng, beta)** (MTEB) | rankings, 21 domain-heavy tasks: finance, health, code, legal, SWE-bench | mean nDCG@10 | new in 2026; leader is a **25M** MIT model at 0.5216 with zero task overlap, BM25 at 0.3411 `[web]` |
| **BrowseComp-Plus — retriever split** | ranked docs over a released corpus with human-verified evidence | Recall@k, nDCG@k | published; **BM25 nDCG@10 1.6**, Qwen3-Embed-8B **20.3** `[web]` ([arXiv 2508.06600](https://arxiv.org/abs/2508.06600)) |
| **MultiHop-RAG — retrieval table** | rankings; the paper's primary table is retrieval-only `[web]` ([arXiv 2401.15391](https://arxiv.org/abs/2401.15391)) | MAP@10, MRR@10, Hits@4/@10 | peak `voyage-02` + `bge-reranker-large` MAP@10 **0.4795**; no-reranker **0.3143**; **no lexical baseline in the table at all** |
| **MTEB via `SearchProtocol`** | a *system*, not an encoder (`tools/mteb_search.py` implements the protocol) | nDCG@10 per task | submission rules require the exact dataset revision `[web]` ([MTEB submission](https://docs.mteb.org/contributing/submitting_results/)) |

Two of these deserve emphasis because they invert the usual intuition:

- **LongEmbed is a free-win zone.** A zero-parameter BM25 baseline sits 0.7 points below the leader and
  *above* a 0.149B late-interaction model `[web]`. Long-document retrieval is still dominated by exact
  matching — which is this engine's shipped default channel.
- **On the two hardest 2025–2026 evidence benchmarks the absolute numbers are 20–24 nDCG@10**
  (BrowseComp-Plus 20.3, BRIGHT 24.3) `[web]`. The evidence layer is unsolved, which is precisely where
  a CPU-only engine is interesting rather than merely cheap.

### 1.3 The trap in comparing this repository's numbers to the board

There are three denominators in the wild and a 2026 reviewer will notice if they are mixed: MTEB's BEIR
board is **15** tasks, the BEIR paper is **13**, and this repository's macro is **18**. On the
official board (2026-10-29), BM25 scores **0.3984**, `bge-large-en-v1.5` **0.5434**, Qwen3-Embedding-0.6B
**0.5552**, and NV-Embed-v2 **0.6321** over BEIR-15 `[web]`. The repository's **0.4799** is a
*best-per-dataset macro over 18* datasets, not a BEIR-15 or BEIR-13 mean, and its own comparison row
(SPLADE 0.4740) is a 2022 published system, not 2026 SOTA. Any external claim must say all of three
things: the denominator, "best per dataset", and the year of the comparison row.

---

## 2. Where this system stands, measured

All `[repo]`; the findings document is named. These are the numbers an external claim rests on, with the
qualifier a reviewer will demand travelling beside each one.

| axis | number | the qualifier that must travel with it |
| --- | --- | --- |
| BEIR best-per-dataset macro | nDCG@10 **0.4799** over 18 datasets | a per-dataset maximum over routing configurations, **not** one deployable configuration; the single-config five-dataset mean is **0.452** ([`findings-encoder-adaptation.md`](findings-encoder-adaptation.md)) |
| BEIR-5 vs published | **0.495** best-per-dataset against BGE-large-v1.5's 0.491 and SPLADE++'s 0.433 `[repo]` | margin is inside the noise band: *level with a 335M BGE-large*, not ahead ([`RANKINGS.md`](../RANKINGS.md)) |
| MultiHop-RAG, engine, full protocol | MAP@10 **0.3128**; Hits@10 **0.896** with coverage + quota, reranked | beats the paper's no-reranker rows and its Hits@10 column; **loses** to its reranked rows (0.4795) ([`findings-engine-multihop.md`](findings-engine-multihop.md)) |
| MuSiQue, engine vs harness | MAP@10 **0.3761** vs **0.3852** (−0.0091 [−0.0120, −0.0062]) | retrieval MAP over supporting paragraphs, **not** answer EM — MuSiQue's own SOTA is EM ≈0.196 `[web]` ([`findings-engine-musique.md`](findings-engine-musique.md)) |
| pool-reorder ceiling | **0.609** at depth 100, rising to 0.659 at 500 | 35.4% of depth-100 pools lack a gold fact, so the loss is recall **and** construction ([`findings-multihop-diagnostics.md`](findings-multihop-diagnostics.md)) |
| extractive answer row | citation clean **1.0000** (0 of 9,806 sentences uncited) | mechanical citation *validity*, not semantic support ([`findings-extractive-row.md`](findings-extractive-row.md)) |
| extractive completeness | answer nugget recall **0.1906** vs pool claim-recall@10 **0.7438** | an extract carries only what the selection picks; this is the measured cost of extractive-by-construction |
| abstention | feature gate held-out AUC **0.8146** (9 features) / **0.8290** (12), risk@80 **0.4069**; shipped scalar AUC **0.5087**, risk@80 **0.4927** | in-domain only; out of domain the scalar is at chance ([`findings-calibration-row.md`](findings-calibration-row.md), [`findings-gate-shape-features.md`](findings-gate-shape-features.md)) |
| abstention on answerable queries | **28.7%** abstain at shipped gates | a deployment abstains for two reasons, reported separately |
| trust / adversarial | 12 deterministic checks; poison leak 0.20 → **0.00** at zero false refusals; injection recall **1.0000** with 0 false positives on two controls | ([`findings-trust-report.md`](findings-trust-report.md), [`findings-adversarial-variants.md`](findings-adversarial-variants.md)) |

Two facts from this pass that belong in the same breath because they are not favourable:

- **There is no end-to-end ranking.** The only `engine + <generator>` row is a 20-record pilot, EM 0.000
  against a 0.667 gold-containment rate, ~36 s/query `[repo]`. Any "best" claim on answer quality needs
  a generator this project does not own — and the `engine + <model>` label already exists for when it
  does.
- **The decision pass is 99.8% of query latency** `[measure]`. Retrieval is 0.03 s; the decision model
  costs 12–24 s for 10–40 candidates on this CPU host (`.scratch/cost-probe.json`). That is the right
  shape for a batch measurement harness and the wrong shape for an interactive one.

---

## 3. Seven gaps, each with a mechanism

### G1 — The component market moved, and the flagship negatives were measured before it

The repository's central retrieval finding is that **the dense channel never earned its weight**: Laya's
own encoder scores nDCG@10 ≈ 0.014 on SciFact, `bge-small` fusion is the shipped compromise, and a
larger encoder was measured twice to buy nothing `[repo]`. Every one of those measurements used
2023–2025 components.

On 2026-10-29 `[web]`:

- **Small dense.** A **25M** MIT-licensed model leads MTEB's 21-task domain-heavy RTEB board at
  **0.5216** against BM25's 0.3411, with MTEB recording **zero** task overlap, at 134 docs/s on a desktop
  CPU `[web]`. A **179M** model reaches **0.6248** on MTEB(eng,v2) retrieval — rank 14 of 221, within 0.4
  points of a 7.85B model — using 128 dimensions `[web]`. Qwen3-Embedding-0.6B reads **0.5552** on BEIR-15
  against `bge-large`'s 0.5434 at 1.8× the parameters `[web]`.
- **Learned sparse.** The best sparse row on the BEIR-15 board is **0.5782** (0.593B, Apache-2.0,
  released 2026-10-12) against BM25's 0.3984 `[web]`. The repository's sparse channel measured **+0.0338**
  on SciFact using an OpenSearch distil from 2025 — an older encoder on one dataset.
- **Listwise reranking.** On one controlled grid over a single strong first stage, a **0.6B listwise**
  reranker adds **+6.04 nDCG@10** where small pointwise cross-encoders add **+0.47 to +0.70** `[web]`
  ([arXiv 2509.25085](https://arxiv.org/abs/2509.25085)). The repository's cross-encoder result
  (+0.0464 MAP@10 on MultiHop, negative on BEIR) is the *small pointwise* row of that grid, reproduced
  exactly. The listwise row was never tested here: `jina-reranker-v3` was set aside on licence, and
  licence-compatible 2026 alternatives were not evaluated. Note the honest size of the opportunity: at
  **equal parameter count** listwise beats a cross-encoder by about **one point**, not by the historical
  2–4× — the multiplier that survives is on reasoning-heavy retrieval (BRIGHT), which is not
  CPU-feasible `[web]`.

None of this contradicts a single recorded negative — each was, and remains, correct *for the component
it measured*. What it contradicts is reading them as statements about 2026. Two of the repository's
findings have in fact been *independently published* since: SPLADE-v3's own table reads **SciFact 71.0 →
71.0** between SPLADE++ and v3, which is precisely the "+0.0004" this repository measured and once
suspected of being a conversion bug `[web]`; and a 2026 IEEE Access paper, *The Fusion Tax*, reports that
the best fusion ranker sits below its better component on 13–48% of queries and loses to dense-only on
8–21%, worst for RRF — the published form of this repository's −0.091 fusion loss on Touché, reaching the
same conclusion that per-query gating is the answer and that it does not transfer across corpora `[web]`.

**One economic constraint travels with the component refresh**: a 0.6B encoder is query-time affordable
on CPU (~1.1 s per item) but costs 10²–10³ CPU-hours per million documents to index, while a 25M encoder
indexes a million documents in about two hours `[web]`. The refresh therefore has to be *routed by corpus
size* — a small model where the corpus is large, a 0.6B model where it is not — which is the same routing
argument the repository already makes about the dense channel and the second stage.

### G2 — The extractive answer has no objective (0.1906 of 0.7438 available)

`compose_extractive` selects sentences by their **parent document's retrieval score** and dedups by text
(`src/laya_rag/extractive.py`). It has no query-relevance term and no coverage term, so the sentence it
keeps is the first one in the first-ranked passage rather than the one carrying the fact.

`[measure]` On the real multilingual checkpoint over a 345-chunk corpus, five queries: the current
composer covers **0–5** distinct query terms; a greedy max-coverage selection over the **same** candidate
sentences covers **2–7**, and the two selections share **no sentence at all** in three of the five
queries (`.scratch/extractive-probe.json`). Query-term coverage is a crude proxy for nugget recall —
not a substitute — but it isolates the mechanism: the selection is blind to the question.

This is the change class the repository's own ledger says *wins*: the failures it records are almost all
"trust a stronger pointwise model more"; the wins are almost all "change the objective or the routing"
`[repo]`. It is model-free and scored by a row that already exists.

### G3 — Abstention does not travel (AUC 0.51 out of domain; 28.7% false abstention in domain)

The gate that decides "can this corpus answer this question" is calibrated in domain and at chance out
of it. The mechanism is not mysterious: it was fitted on one corpus's negatives (601 labelled queries),
and negatives from another corpus have a different feature distribution.

The fix that needs no human labels is **cross-corpus negative mining** — a query from corpus A run
against corpus B is unanswerable by construction and yields a real feature vector. Two of the twelve
features (evidence-shape: max passage coverage, coverage spread, pairwise redundancy) need **no model
call at all**, and the 12-feature arm held AUC **0.8323** across a chunking change that dropped the
9-feature arm to 0.7862 `[repo]` — so a model-free gate is trainable at corpus scale.

The literature says this is the right axis to invest in: AbstentionBench found reasoning fine-tuning
*degrades* abstention by 24% on average `[web]` ([arXiv 2506.09038](https://arxiv.org/abs/2506.09038)),
"Sufficient Context" showed strong models answer instead of abstaining when context is insufficient
`[web]` ([arXiv 2411.06037](https://arxiv.org/abs/2411.06037)), and no benchmark grades a
*retrieval-only* answerability predictor with a held-out AUC. That absence is both the gap and the
opportunity.

### G4 — Cost is unmeasured and concentrated, and the headline latency is an accelerated number

`[measure]` `docs/findings-cost-envelope.md`, taken after R0 landed the instrument: with the device
refused (`LAYA_RAG_DEVICE=cpu`), 16 torch threads and `RETRIEVE_K=20`, a query costs **6.71 s median wall
and 96.3 s median CPU** — the decision pass is **5.29 s of CPU per candidate** against 0.03 s for
retrieval, so the second stage is more than 99% of the work. CPU time is a thread-sum and comparable only
at a fixed thread count, which is recorded beside it. The pre-registered falsifier (±5% across repeats)
fired at **8.1%**, and the same configuration read **15.83 s** while two other processes were busy and
**6.71 s** once they stopped: a 2.4× swing from co-tenancy alone, which is the fourth time this
repository has watched `seconds` invent a speed difference.

There is a second, larger problem with the same root. The README's headline — *"the engine auto-selects
MPS and answers a query in 2–5 s end to end"* — is an **accelerated** number from an Intel Mac. The same
engine with the accelerator refused is the 6.71 s median above. Any claim phrased as "CPU-only" must
report the accelerator-disabled envelope as the primary number and the accelerated one as a footnote, or
the first review comment will be that the envelope measured was not the envelope claimed. R0 has now done
that measurement and landed the fields that make it repeatable.

The field has started asking for the rest of the axis: BrowseComp-Plus reports calibration error and
search calls beside accuracy `[web]`; nobody publishes joules per query. A CPU-only system can own that
column — after the instrument exists.

### G5 — There is no ranked artefact

Every quality number above comes from stock components wired well. The field's unit of comparison is a
*model*: BEIR ranks encoders, MTEB ranks encoders. `tools/mteb_search.py` implements MTEB v2's
`SearchProtocol`, so the engine is *representable* as a submission and is not submitted; the
distillation that would make an encoder submission worth making is measured on one dataset
(0.3385 → 0.3505 on NFCorpus, ~29% of the ensemble's gain `[repo]`).

### G6 — The evidence-layer abstention axis exists as a hole nobody fills

The field has (a) retrieval-only benchmarks with no abstention axis and (b) abstention benchmarks that
require generation. **Nobody publishes a joint "evidence-correctness × abstention" score for a
non-generative system** `[web]`. The intellectual ancestor is Asai & Choi's decomposition of
information-seeking QA into paragraph selection and answerability prediction, which concluded that
"predicting answerability itself remains challenging" `[web]`
([ACL 2021](https://aclanthology.org/2021.acl-long.118.pdf)).

The raw materials for filling it are public and already in this repository's toolchain:

- **MultiHop-RAG ships 301 `null_query` records** whose gold answer is literally "Insufficient
  information", and its own retrieval metric **excludes** them `[web]` `[repo]`. Scoring refusal on the
  records everyone else drops is a cheap, novel, defensible move.
- **MTRAG-UN** carries the answerability taxonomy into a 2026 leaderboard, but scores only the answerable
  subset on the retrieval side `[web]`.
- **AbstentionBench**'s taxonomy (unknown answers, underspecification, false premises, subjective
  interpretation, outdated information) is the right one to borrow `[web]`.

### G7 — Scope is one language and one modality

The default checkpoint is multilingual and no multilingual benchmark has been run. MIRACL, MMTEB and the
non-English retrieval task types are unentered. A text-only CPU engine will never do ViDoRe or MIEB, but
"the best CPU-only retrieval in eleven languages" is a claim with a published table behind it and nobody
competing in that envelope `[web]`.

---

## 4. The claim to make

Stated so it can be falsified, in the form the field would accept:

> **The reference evidence engine.** Fully local, CPU-only, no API key: returns the passages that support
> an answer with a calibrated probability that the corpus can answer at all, refuses when it cannot,
> cites mechanically and verifiably, and is the only RAG system whose every published number — including
> the unfavourable ones — is re-derivable from a report a resumable job produced.

Four numbers defend it, and each is partly in hand:

1. **Abstention that transfers** — and it must be stated as three numbers, not one, because a reviewer
   will find the gap otherwise: **0.8146 in domain / 0.767 [0.671, 0.827] with a whole question type held
   out / 0.509 [0.463, 0.555] out of domain** `[repo]`. Target: raise the out-of-domain arm on a corpus
   the gate was not fitted on, and report abstention precision and recall *separately*, since
   over-abstention is the failure mode this architecture is most exposed to.
2. **Attribution that is a measurement** — citation clean **1.0000** today, deterministically checked. The
   claim is not "we cite better than an LLM"; it is "our citation number is not an LLM judgement", which
   TREC's own 0.44-manual-versus-0.95-auto-grader split makes valuable `[web]`.
3. **Completeness of a non-generated answer** — answer nugget recall **0.1906** with the pool's **0.7438**
   as the stated ceiling. The claim is that the ratio is *known*, not that it is high.
4. **Cost per point of quality** — to be measured properly; today rerank dominates at ~0.6–0.8
   s/candidate on CPU. The compression literature is unusually clear about which half is cheap: at RAG
   scale, **int8 is lossless and *not faster*** (`0.3653 → 0.3661` nDCG@10 on 8.8M passages), **binary
   alone keeps 93% but binary plus a float rerank is lossless at 32×**, and at equal bytes **bits beat
   Matryoshka truncation** `[web]` ([BitBudget](https://sjmoran.github.io/bitbudget/)). So the index side
   of the cost axis is nearly free and the *compute* side — the decision forward pass — is where every
   remaining second is.

One claim the programme should **retire** rather than defend: *"our retrieval quality predicts answer
quality."* It does not, in general. The UDCG and eRAG lines of work show retrieval metrics correlating
with end-to-end answer accuracy at only **0.55–0.70** Spearman at best — visibly worse for nDCG `[web]` —
so "best retrieval" is not "best RAG", and the honest framing is the one above: an evidence instrument
scored at the evidence layer, not a predictor of someone else's generator.

---

## 5. The programme, ranked

Ranked by expected value per unit cost, each with the falsifier that closes it. Tags: `[offline]` runs on
this host today · `[data]` needs a download · `[model]` needs a model · `[web]` needs a submission.

### R0 — Instrument the CPU-only envelope (hours, `[offline]`)

Two things, both small. Add `cpu_seconds` and `candidates_scored` to `QueryResult` timings and record them
in every engine report beside wall clock; and re-measure the headline latency with the accelerator
**disabled** (`DEVICE=cpu`), so the "2–5 s on MPS" claim and the CPU-only claim are separate sentences.
Without this, G4 cannot be measured on a shared host and every future cost claim inherits the 253.9 s
artefact — and the claim the project most wants to make ("no accelerator") is the one currently quoted
from an accelerated run.

**Falsifier:** if CPU-seconds per candidate still varies by more than ±5% across repeats at fixed depth,
the host is not a valid instrument; say so and defer the cost axis rather than publish a jitter number.

**Outcome (done).** The instrument landed and the falsifier **fired at 8.1%**, so the cost axis is
published as medians with their observed spread rather than as point estimates — enough resolution to
compare a `RETRIEVE_K=10` cascade against `RETRIEVE_K=40`, nothing like enough for a 3% variant. The
measurement, its two clock definitions and the co-tenancy warning are in
[`findings-cost-envelope.md`](findings-cost-envelope.md).

### R1 — Refresh the components, under this repository's own protocol (days, `[model]` `[offline]`)

The highest-value item in this document, because it attacks the flagship negative with a measurement the
harness already performs. Run `tools/beir.py` — and then `tools/engine_beir.py`, because *a replica is
not the system* `[repo]` — over the five measured datasets with:

- a **2026 small dense encoder** (25M–0.6B, licence-clean: bekko-25M is MIT, Qwen3-Embedding-0.6B and
  `uibk-embedding-v1` are Apache-2.0) replacing `bge-small` as `EMBED_MODEL`;
- a **2026 learned-sparse encoder** in the existing `SPARSE_MODEL` channel (the two 2026 Apache-2.0 rows
  are 0.188B and 0.593B);
- the shipped hybrid + RRF fusion on top, unchanged.

**Prediction:** fused nDCG@10 ≥ **+0.03** over the current 0.452 single-config mean, and the dense channel
re-earning a non-zero `DENSE_WEIGHT` on at least three of five datasets. The arithmetic behind the
prediction: on MTEB's official board Qwen3-Embedding-0.6B reads **0.5552** against `bge-large`'s 0.5434
and BM25's 0.3984 over BEIR-15 `[web]`, and discounting this repository's best-per-dataset inflation
leaves **+0.03 to +0.05** from the encoder alone. Route by corpus size: the 0.6B model where indexing is
affordable, the 25M model on the 5M-document members. **Falsifier:** if the fused row does not beat the
lexical-only row by ≥0.02 on ≥3 of 5 at matched chunking, the "dense does not pay" finding is a statement
about encoders *in general* on these corpora and should be re-stated as such. Either outcome is worth
publishing, and the cost is CPU-hours, not weeks.

### R2 — Give the extractive composer an objective (days, `[offline]`)

Replace rank-order selection with greedy max-coverage over candidate sentences: relevance to the query
(lexical, and optionally the passage's calibrated probability) minus redundancy against already-selected
sentences, under the existing verbatim + citation constraints. Keep today's behaviour behind a flag so
[`findings-extractive-row.md`](findings-extractive-row.md) stays reproducible.

**Prediction:** answer nugget recall **0.1906 → 0.35–0.50**, citation clean stays 1.0000.
**Falsifier:** no rise of ≥0.10 at unchanged citation validity on the full row ⇒ selection is not the
binding constraint and G2 closes.

### R3 — Make abstention portable, and grade it on the records everyone else drops (1–2 weeks, `[offline]` `[data]`)

Two arms, both cheap: (a) a **shape-only gate** — the three model-free evidence-shape features — trained
and evaluated with no decision-model forward at all, so negatives can be mined at corpus scale; (b)
**cross-corpus negative mining** — corpus A's queries against corpus B, labelled unanswerable, with a
shuffled-corpus control — evaluated on the corpus held out.

Then evaluate on **MultiHop-RAG's 301 `null_query` records** (which the benchmark's own retrieval metric
excludes) and the **MTRAG-UN** unanswerable/underspecified subsets.

**Prediction:** out-of-domain AUC **0.51 → 0.70+**, risk@80 below the scalar's 0.49, reported as a three-way
split (in domain / question type held out / corpus held out) with abstention precision and recall stated
separately.
**Falsifier:** mined negatives failing to beat in-corpus-only training on the held-out corpus ⇒ the
failure is a feature problem, not a data problem, and the honest output is a stated negative.

### R4 — Publish the evidence-layer protocol, and take two leaderboard placements (weeks, `[web]`)

Package what already exists as **one named protocol with one command and one report**, on four axes:

1. **Evidence quality** — MAP@10 / Hits@10 / nDCG@10 on the answerable subset, paired intervals, against
   published rows.
2. **Evidence-side abstention** — ROC-AUC, AURC, risk–coverage on `null_query` plus an independent
   unanswerable set, against two baselines (the shipped scalar and a BM25-score threshold).
3. **Joint score** — evidence-MAP@10 × (1 − false-accept rate), so abstention cannot be bought with
   blanket refusals. This is RGB's two-sided negative-rejection idea moved to the evidence layer.
4. **Cost** — seconds, peak RSS, index bytes, and joules per query on CPU. Nobody publishes the last one.

In parallel, produce the artefacts for **LongEmbed** (a hybrid against a 0.9153 leader that BM25 reaches
0.9087 of) and **MultiHop-RAG's retrieval table** (already strong; needs the paper's protocol and a stated
single-configuration row), and prepare a **TREC RAG 2027 Retrieval** run.

**Falsifier:** if an external reviewer cannot reproduce the four-axis report from a fresh clone in under
an hour, it is a tool and not a protocol, and the claim reduces to the leaderboard placements.

### R5 — Test the listwise reranker the licence question previously blocked (weeks, `[model]`)

The controlled grid puts **+6.04 nDCG@10** on a 0.6B listwise reranker against **+0.47–0.70** for small
pointwise cross-encoders `[web]`. This repository measured the small pointwise row and rejected
`jina-reranker-v3` on licence (CC BY-NC). Licence-clean 2026 candidates exist. Route it as the ledger
already demands — per corpus, with a rank-1 guardrail and a calibrated blend — never globally.

**Falsifier:** if listwise does not clear the pointwise cross-encoder by ≥+0.02 MAP@10 on the multi-hop
protocol at equal pool, the gain is a BEIR-only artefact and does not transfer to sets.

### R6 — Cut the decision cost (weeks, `[offline]` `[model]`)

Three stages, cheapest first: measure whether candidates are already batched into one forward pass; then
a **cascade** (cheap lexical/embedding score keeps a wide pool, the decision model scores only the top
*m* — a *routing* change, the class the ledger finds wins); then distil into a small cross-encoder trained
on cached scores and measure the calibration loss honestly.

**Prediction:** 3–10× fewer decision seconds at ≤0.005 MAP@10.
**Falsifier:** if the student's calibration degrades more than its ranking, the asset is the
*calibration* — ship the cascade and stop.

### R7 — Measure one multilingual collection (days, `[data]`)

A MIRACL subset through the engine beside lexical and dense controls, with the result stated even if it
is poor. A CPU-only multilingual retrieval row appears in no table this pass found `[web]`.

### R8 — Own a submittable artefact (weeks–months, `[model]`)

Finish the ensemble distillation and submit the *engine* through `SearchProtocol`, or the encoder to
MTEB. This is the only route from *compared* to *ranked*.
**Falsifier:** if the distilled student misses the ensemble's fused score by more than 0.005 on the five
datasets, do not submit — a ranked row below the stock `bge-small` row is worse than no row.

### What not to do

The 2026 evidence strengthens the existing ledger and adds three closures of its own:

- **Do not enter an answer-quality leaderboard by adapting its metric.** All of them grade generated text
  `[web]`; the `engine + <model>` rule is the honest path and it puts the number in the generator's
  column.
- **Do not quote an LLM-judged citation or faithfulness score as comparable** to the deterministic one:
  TREC's own 0.44-versus-0.95 gap is the evidence `[web]`.
- **Do not chase the agentic-retrieval premium without owning a generator.** It is the largest
  documented gain in the field (≈+0.35 nDCG@30 over BM25 on TREC 2025's retrieval task) and it is paid in
  generator calls `[web]`.
- **Do not build a vision path or a knowledge graph.** ViDoRe/MIEB are out of reach for a text-only
  system, and the graph's measured gain is a conditioning signal this engine already has without it.
- Everything in [`RESEARCH.md`](../RESEARCH.md) §2 still stands.

---

## 6. The ledger, updated

| item from the 2026-10-03 programme | outcome | where |
| --- | --- | --- |
| Bridge-linked second-hop rescue | **rejected** — worse than the cross-encoder at every weight (0.2786/0.2714/0.2682 vs 0.2986) | [`findings-rerank-selection.md`](findings-rerank-selection.md) |
| Tripartite / bridge-conditioned judge | **rejected as a gate column** — AUC −0.0023, risk@80 +0.0012, shuffled control flat | [`findings-bridge-gate-measurement.md`](findings-bridge-gate-measurement.md) |
| Learned selector / listwise LTR (the pool bet) | **closed** — pointwise 0.2939, listwise 0.2955, both below the hand rule's 0.3147; coverage-quota does not transfer off-corpus | [`FINDINGS.md`](FINDINGS.md) |
| Coverage objective as a general win | **falsified outside MultiHop-RAG** — −0.0109 on MuSiQue, −0.0046/−0.0056 on 2Wiki; costs 0.02–0.08 nDCG@10 on BEIR | [`findings-2wiki.md`](findings-2wiki.md), [`findings-coverage-scope.md`](findings-coverage-scope.md) |
| Atomic units, context prefix, three-state adjudication, numeric verification, ACL/tenancy, injection defence, staleness/dup detection | **all landed and measured**; the capability matrix in the 2026-10-03 pass is out of date in those rows | [`FINDINGS.md`](FINDINGS.md) |
| Distil the ensemble, submit to MTEB | distillation at one dataset (+0.0120 NFCorpus); **submission still not made** | [`findings-distillation.md`](findings-distillation.md) |
| MaxSim / late interaction; DISCo coverage index | still open; the deferral stands — and the 2026 grid independently confirms late interaction as a *reranker* loses (−1.32 nDCG@10) `[web]` | [`RESEARCH.md`](../RESEARCH.md) §4 |
| Sparse channel, doc2query, quantisation | **sparse pays +0.0338** on SciFact through the encoder already in the tree; doc2query **+0.0095** at 21 days corpus scale; fp16 free, int8/PQ not — note the 2026 sparse rows are a different generation of model (§3/G1) | [`FINDINGS.md`](FINDINGS.md) |

---

## 7. A 90-day sequence to a claim someone else can check

1. **Week 1** — R0 (instrument the CPU-only envelope). Nothing else in this list can be reported honestly
   until the accelerator-disabled numbers exist.
2. **Weeks 1–2** — R1 (component refresh) and R2 (composer objective). Both offline, both measured against
   rows that already exist, and R2 is the only item here that can move a *published* number this month.
3. **Weeks 3–5** — R3 (portable abstention). The flagship scientific claim: it converts "AUC 0.815 in
   domain" into a three-way transferable statement or an honest negative, on the axis the field started
   grading.
4. **Weeks 6–8** — R4 (protocol + two placements) and R5 (listwise, if a licence-clean candidate
   reproduces). The first external artefacts.
5. **Weeks 9–12** — R6 (cascade/distillation) and R7 (one multilingual collection). Cost and breadth,
   which turn "good numbers on one host" into "a system with an envelope".
6. **R8 runs in parallel** if and only if the distillation reproduces; it has the longest tail.

Each step publishes its own `findings-*.md`, positive or negative. That is the actual competitive
advantage: the [`measurement-failures`](findings-measurement-failures.md) record and the do-not-retry
ledger are stronger evidence of trustworthiness than any single score, and 2026's graders have started
asking for exactly that.

---

## 8. Verification status and what this document does not claim

- `[repo]` numbers are quoted from the named findings documents as of 1.2.2 and were not re-derived here.
  They were produced under Laya 0.3.23 and are compatibility-checked, not measurement-re-verified, under
  0.4.14.
- `[measure]` numbers come from two probes run during this pass on this host, with artefacts at
  `.scratch/cost-probe.json` and `.scratch/extractive-probe.json`. They are single-host, single-run
  measurements; the cost probe is visibly contaminated (§3/G4). They support *mechanism*, not *effect
  size*.
- `[web]` numbers were read on 2026-10-29 from the cited URLs. Vendor-reported figures are called that
  where they appear. Three things this pass could not verify: a published 2026 SOTA for the BEIR-18
  denominator specifically (so "above SPLADE" means above the 2022 systems the BEIR paper contains);
  the TREC RAG 2024 overview PDF (404 on 2026-10-29); and any independent replication of the
  leaderboard-leading small models — those are leaderboard submissions, not third-party re-runs.
- **This document proposes no answer-quality claim.** A system that does not generate text cannot be
  ranked on the leaderboards that score generated text; the `engine + <model>` labelling rule is the
  correct response to that, not an obstacle to be worked around.
- Every prediction in §5 is a hypothesis with a stated falsifier, in the form
  [`RESEARCH.md`](../RESEARCH.md) established. None has been measured.
