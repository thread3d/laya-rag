# Programme status

A single page for whoever picks this up next. `ROADMAP.md` carries each item's live status and
commands; this states where the programme actually landed, what it refuted, and what is left with the
cost of getting it.

**Release:** `1.7.2` is the current tagged release; its notes are [`CHANGELOG.md`](../CHANGELOG.md). The
version is what `laya_rag.__version__` reports and what the HTTP API publishes in its OpenAPI
schema, so a client can tell which engine it is talking to.

## Where it landed

| phase | state | headline |
| --- | --- | --- |
| 0 — claim ledger | **complete** | `make claim-check`: **58** prose numbers pinned to the reports that produced them, plus `make consistency-check`, which pins documents to *each other* — the repository's own and the `best-in-the-world/` audit set above it — and computes the BEIR coverage count from disk |
| 1 — engine full-split runs | **complete** | MultiHop-RAG 2,255 both configurations; MuSiQue 2,417 hybrid plus a 200-query default |
| 2 — ordering heads | **complete, entirely negative** | late interaction, relational NLI and an adapted cross-encoder all lose; 2.4 not adopted on licence |
| 3 — end-to-end and cost | **complete** | FRAMES, a generation pilot, citation mechanics, and the cost axis |
| 4 — scale | **complete** | 4.1, 4.3, 4.5 done; **4.2 at 18 of 18 datasets** (Climate-FEVER, FEVER and the full TREC-COVID collection landed 2026-10-06 and after; Robust04, TREC-NEWS, Signal-1M, CQADupStack and BioASQ afterwards); 4.4 is an external submission |
| 5 — trust | **complete** | trust report, three-state adjudication, verifier precision, poison-dose probe |

The engine **beats the harness it replaced on MultiHop-RAG** (MAP@10 0.3128 against 0.2510) and
**loses to it on MuSiQue** (0.3761 against 0.3852). Neither generalises, which is the finding.

## The October 2026 frontier round (R0–R7)

The programme above measured the system as it stood in early October. The
[frontier assessment](best-in-the-world-2026-10.md) then re-scoped what "best" could mean against dated
2026 sources and laid out R0–R8; this is how far that ran, and every item below has its own
`findings-*.md` with a prediction recorded before the measurement.

| item | state | headline |
| --- | --- | --- |
| **R0** cost envelope | **done** | `cpu_seconds` and the *resolved* `device` travel with every result. With the accelerator refused: **6.71 s median wall / 96.3 s median CPU** per query at 16 threads, and the decision pass is **99.8%** of it. R0's own falsifier fired — repeat spread **8.1%** — so the cost axis publishes medians with their spread, not point estimates |
| **R1** component refresh | **2 of 5 datasets** | A 2026 encoder is measurable again (the harness could not pool `last` or apply a per-family instruction). On SciFact Qwen3-0.6B is **−0.018** on dense, on NFCorpus **+0.024**; fused is **+0.007 / +0.006** for **15–21× the indexing**. Verdict: **routing, not a swap** |
| **R2** composer objective | **done, prediction falsified** | Answer nugget recall **0.2060 → 0.2610** (**+0.0549 [+0.0240, +0.0841], p=0.0005**), citation validity unchanged at 1.0000, both arms at 5.00 sentences per answer. Half the pre-registered 0.10 bar, so selection is *a* binding constraint, not *the* one |
| **R3** portable abstention | **gate shipped; transfer open** | On the benchmark's own 301 `null_query` records, the gate fitted over 40 stratified splits reads **AUC 0.8377** (risk@80 **0.4043**); the protocol's independent half-split run reads **0.8159** against the shipped scalar's **0.5208**. At the deployed threshold that run answers **69.3%** of unanswerable queries where the gate answers **30.7%**, and **72.0%** of answerable ones where the gate answers **76.7%**. Cross-corpus transfer still needs MTRAG-UN |
| **R4** evidence protocol + placement | **done and reproducible; the full row is owed** | `make evidence-protocol` prints five axes — the fifth says *which configuration* produced the numbers — naming an absent axis rather than zeroing it, and the chain runs from a fresh clone in **~68 min**. Its first report is the number the programme is built on: **production answers 69–72% of the questions the corpus cannot answer**. The placement half is `make multihop-row`, which formats the engine's row for the benchmark's retrieval table and **refuses to place a subset** — on the 150-query subset it shows Hits@10 **0.940**, Hits@4 **0.847**, MRR@10 **0.688** against the best published row's 0.747 / 0.663 / 0.586, with MAP@10 **0.371 against 0.480** |
| **R5** reranker generation | **pointwise answered** | `bge-reranker-v2-m3` (2026, 568M) beats `bge-reranker-base` (2024, 278M) by **+0.0313 MAP@10 [+0.0052, +0.0578], p=0.0089** on the same pool, and both beat no reranker (**0.3128 → 0.3398 → 0.3710**). The gain is ordering, not completeness. **The ArguAna question was then measured against a registered prediction**: at depth 50 on a seeded 400-query subset the first stage reads **0.5664** against **0.4722** for the 568M model — first stage **+0.0942 [+0.0530, +0.1335]**, still losing on 290 of 400 queries — where the 278M model reads 0.2736 (first stage **+0.2928**). The prediction (0.35–0.50, below 0.568) held and its falsifier did not fire: capacity recovers **two thirds** of the opposition damage without reversing the sign, so a pointwise ranker pays for opposition in proportion to how little it knows. The listwise arm is still untested |
| **R6** decision cost | **done** | `SECOND_STAGE_TOP_M=10` gives **43% less CPU at identical quality** (99.2 s → 56.1 s per query; MAP@10, Hits@10, nugget recall and citation all unchanged). **M=5** costs 0.023 MAP@10. Corpus-specific by construction, so off by default **Its interactions with the other levers were then measured**, since each was measured alone: at the gate the aggregate is nearly indifferent (**−0.009** mean probability, answerable rate unchanged at M=10) but **4 of 20 verdicts flip**; at the composer nugget recall falls **0.2005 → 0.1862** and abstention rises 0.20 → 0.25, because an abstention emits the fixed no-answer sentence the checker counts as an uncited claim. So the cascade is free for the ranker and slightly expensive for the answer |
| **R7** multilingual | **priced and scoped** | The smallest MIRACL collection is ~**28 hours** for one language here, so it was declined rather than truncated. What was measured instead: with the shipped lexical channel non-English recall is **1/3 at best and acronym leakage**, and forcing the dense channel on drops every non-English language to **0/3**. The README's "handles non-English text" claim was corrected |
| **R8** submittable artefact | **gate shipped; artefact owed** | `make submission-check` audits any exported encoder against five properties — it loads, its vectors are unit, it is deterministic, its pooling and revision are pinned, the SearchProtocol adapter is present — and returns SUBMITTABLE or not with every failing check named. It deliberately does not score, so R8's own falsifier stays a scoring criterion. The training run that would produce a winning artefact is the one item this host cannot finish quickly |

### The audit layer, which turned out to be the round's second product

Eight contracts were added while the measurements ran, and **six of them found something**. Every failure
was in a surface that *describes* the work rather than in the work:

| contract | what it found |
| --- | --- |
| protocol axis 5 — *which configuration produced these* | a row recording a `--out`-derived index path the run never used |
| measurement-failures class 14 | five provenance defects, three caught by *identical arms* |
| environment variables documented | **nothing** — 91 of 91, a clean sweep worth recording |
| findings-index reachability | one orphaned implementation record |
| `/v1/stats` behaviour block | the levers that change behaviour were unreportable |
| version consistency (five files) | one value, now enforced |
| tool paths written in prose | **nine dangling, two of them the round's own** |
| `make help` completeness | **fifteen targets undiscoverable** |
| the front page's claim about its own gates | **"every quoted figure" was 54 of 8,370** |

The rule the ledger now states: **when a configuration variable changes what a number means, it has to
appear in the artefact — or the artefact is a filename.**

Two properties of this round are worth stating because they are the point of it:

- **Three of the eight items ended in a correction rather than a win** — the composer's level prediction
  was twice too optimistic, the multilingual claim was false as documented, and the encoder refresh
  turned out to be routing rather than a swap. Those are recorded as such.
- **Four independent levers now have numbers**, and none of them was the one the programme expected in
  October: the **gate** (halves false accepts, raises true accepts), the **composer objective**
  (+0.055 nugget recall), the **reranker generation** (+0.031 MAP@10), and the **cascade** (43% CPU for
  free). The cost axis finally has both a lever that spends and one that saves.

## What the programme refuted

Negatives are results here, and these are the ones worth keeping:

- **Every Phase 2 ordering head lost.** Late interaction by −0.1548 MAP@10, at ~337× the storage.
- **The dense channel is a per-corpus instrument, not an improvement.** Off is worth **+0.0119** on
  MultiHop-RAG and costs **−0.0540** on MuSiQue.
- **The channel reversal holds in ten corpora** (`docs/charts/channel_reversal.png`): dense wins on
  eight, lexical wins on Touché by two to one, and Touché and FRAMES are both long Wikipedia text that
  disagree — so the axis is the query against the document, not length or domain.
- **Fusion is not a free win.** It loses to BM25 alone by 0.091 on Touché and loses 0.041 recall@1
  when fused *into* the FRAMES ensemble.
- **The plan's target for Phase 4.1 was wrong twice**: build memory is not the term dictionary (flat in
  vocabulary to 400k terms) and not the merge (401 runs cost nothing); it is the block and allocator
  retention.
- **Two of my own hypotheses were falsified by the runs that tested them**: that the lexical shortfall
  grows with corpus size (DBPedia at 4.6M documents is −0.008), and that our lexical settings cause it
  (the published-settings control row is *further* from the anchor).

## What is left, and what it costs

1. **Phase 4.2, the remaining datasets.** Not a code problem, and the reason has changed twice since
   this page was first written. The UKP mirror returns **404** for `robust04`, `trec-news`,
   `specter`, `signal1m` and `bioasq` (not a transient stub — the files are gone), and the Wayback
   Machine has **no snapshot** of any of them. The `BeIR/*` HuggingFace organisation does **not**
   carry those four corpora at all, and its **401** to an anonymous request is the same answer it
   gives for a name that does not exist, so the earlier reading that they are "gated" there was
   wrong. Three of the four are in fact available from a **public, non-gated third-party mirror in
   BEIR layout** and are now measured as lexical rows: **Robust04** (**0.4077**, against a published
   BM25 of **0.407** — the control reproduces its anchor to 0.001), **TREC-NEWS** (**0.3351** against
   0.398) and **Signal-1M** (**0.3131** on the engine row, with the published-settings flat control
   at **0.3640** against a published 0.330) — see `docs/findings-beir-recovered.md`. **Both of the
   "blocked" 5.4M-document corpora are done** as lexical rows: **Climate-FEVER** (nDCG@10 **0.1610**)
   and **FEVER** (**0.6323** over all 6,666 queries, against a published 0.753 — the largest
   lexical-control gap measured, and now **~80% explained**: a per-field BM25F control on the same
   queries and corpus scores **0.7288**, so the gap narrows to **−0.0242** and what looked like a
   quality deficit was mostly a flat-vs-multi-field settings difference). NQ (2.68M) completed in ~25 min and HotpotQA's full 7,405-query set
   reads **0.6352**; the full **TREC-COVID** collection (171,332 documents, 50 queries) is measured at
   **0.6188** — against a published BM25 of **0.595**. **CQADupStack was never 404 at all** — it is
   twelve independent sub-collections behind one name, which the one-corpus harness could not express,
   so it now has its own runner (`tools/cqadupstack.py`, `make cqadupstack`): twelve per-sub-forum
   retrieval runs, macro-averaged, where the flat control reads **0.2988** against a published BM25 of
   **0.299** (a −0.0002 replication) and the engine row **0.2943**. **BioASQ** — the one blocker that
   survived an honest search, needing a **free BioASQ account** for both the task 9a corpus and the
   8B golden sets — is now measured too: `bm25-engine` **0.5193** over **14,914,604 documents** and
   500 queries, against a published BM25 of **0.465** and the best published system's **0.504**
   (`tools/bioasq.py`, `docs/findings-bioasq.md`). Coverage is therefore **18 of 18** — the benchmark
   is covered, and for the first time the engine's macro is over the complete BEIR-18: **0.4799**
   after the title-weighted and adapted-dense pass (**0.4665** when coverage completed), above
   SPLADE's **0.4740**, the strongest published system — as a best-per-dataset row, not one
   configuration. **MS MARCO
   (8.8M) is measured as an *extra*, not as coverage**: it is the collection the BEIR encoders are
   trained on, not a test dataset, and the gate counts over an explicit membership set. The
   public BioASQ corpus mirror (`BeIR/bioasq-generated-queries`) was **not** used: it holds
   14,100,000 records against the 14,913,939 the qrels score against, a 5.46% truncation that would
   score the TREC-COVID subsample defect rather than a smaller measurement. The three
   recovered corpora
   are third-party redistributions of NIST/TREC-agreement collections, so their licensing block is a
   policy question the measurement does not settle.
   `--limit-queries` gives a labelled seeded subset (~1 h per corpus).
2. **Phase 4.4, the MTEB submission.** Requires an upload; not offline work.
3. **HotpotQA's full query set — done.** The full **7,405** queries read **nDCG@10 0.6352** (MRR@10
   0.8065) against a published 0.603 — a **+0.032** gap. The earlier 600-query subset estimate (0.6365,
   +0.034) is superseded: `tools/compare_more.py` now prefers a full report over a seeded estimate, so
   the lexical control uses the full row.

4. **The two research bets are now answered, and neither paid as hoped.**
   **P11, the pool bet**: stage 1 is validated (gold-fact coverage **0.8326 → 0.8937**, pool ceiling
   **0.5955 → 0.6203**), and stage 2 *converts on MultiHop-RAG* — a coverage-quota blend reads **0.3147
   MAP@10**, significantly above `ce` (**+0.0266 [+0.0230, +0.0303]**) and the learned `ltr`
   (**+0.0208 [+0.0130, +0.0287]**). But **it does not transfer**: on MuSiQue held-out it is **below
   `ce`** (0.3984 vs 0.4032), while a plain **`blend-0.75` does transfer** (**+0.0120 [+0.0088,
   +0.0150]**, n = 2,417) — the arm that generalises does not need the pool. And **hard-negative
   restriction destroys the ranker**: training on golds plus the five nearest non-golds drops it from
   **0.2939 to 0.0257** MAP@10, a training/test distribution shift. Reports:
   `eval/benchmarks/musique/ce-scores-{train,val-full}.json`,
   `eval/benchmarks/multihop-rag/ltr-hardneg-5.json`; detail in `SCORE-LIFT.md` §P11.
   **Both `.bin` directions are blocked by the platform, not by choice**: SPLADE-v3 and every
   doc2query checkpoint on the Hub ship `pytorch_model.bin`, and torch 2.2.2 — the newest macOS
   x86_64 wheel — is refused by `transformers` 4.57 under CVE-2025-32434. Neither is a tuning
   question; both would need that guard bypassed, which was declined deliberately. The 864 MB
   doc2query model is left in `eval/hf-doc2query` in case the platform moves.

**The runs are queued one at a time, and the queue is empty.** Three corpus builds plus a harness
launched together **crashed this host**; no work was lost (every job resumes), and the rule is now
enforced by an exclusive lock in 26 long-running tools (`make heavy-status`). All four planned runs
have now finished, one at a time: **Climate-FEVER → FEVER → HotpotQA → 2WikiMultiHopQA**.

## Newest results (2026-10-06 pass)

**Settings-matched controls, seven cells (2026-10-21).** A per-field BM25F control against each cell's registered
anchor moves the mean gap from **−0.0379 to −0.0080** — four fifths of the average deviation was configuration, and
three of the seven cells now sit at or above their published anchor. Two (TREC-NEWS, Climate-FEVER) still lose ground, and
**Robust04 cannot be tested at all**: its corpus carries no titles, so there is no second field to weight. That is the
measurement behind "the 0.6154 deficit is an upper bound", and it is a cell-by-cell fact rather than a generalisation.

- **The owed bridge-conditioned gate column is measured, and it is a negative.** Adding
  `bridge_conditioned` (the query's own top passage as the bridge, max conditioned score over the
  rest) to the twelve-feature answerability gate moves AUC **−0.0023 [−0.0032, −0.0014]** and
  risk@80 **+0.0012 [+0.0003, +0.0022]**, while a shuffled-bridge control is flat: the true first hop
  makes the gate worse than an arbitrary one, so `GATE_BRIDGE_FEATURE` stays off. The ranker's bridge
  signal (AUC 0.559 against 0.530) does not transfer to the gate. An earlier run of the two arms was
  identical to four decimals — the harness read JSON passage dicts with `getattr`, always empty —
  and printed a spurious gain; a control that matches its treatment exactly is a bug signal.
- **TREC-COVID on the full 171,332-document collection**: `bm25-engine` nDCG@10 **0.6188** against a
  published BM25 of 0.595; the 40,000-document subsample (0.3062) was a corpus truncation, not a
  subset, and is superseded. Coverage **12 → 13 of 18**. **MS MARCO** (8,841,823 documents) is
  measured at **0.4321** as an *extra* — it is the training collection, not one of the 18.

- **Generator-free iteration, Stage 1 — the mechanism holds.** One round of entity-probe pool
  expansion on 2,255 MultiHop queries raises gold-fact coverage **0.8326 → 0.8937**
  (**+0.0611** [+0.0555, +0.0670], p<0.0001), complete-evidence queries **61.2% → 72.8%**, and the
  pool's **ceiling MAP@10 0.5955 → 0.6203** (**+0.0248**); **+0.0510 / +0.0210** survive a
  shuffled-probe control. By type, **inference queries gain +0.1067 coverage** (five times
  comparison's +0.0229). One round saturates; the pool triples. The ceiling is a bound, not an
  achieved MAP@10 — realizing it is Bet 2.
- **Answerability calibration.** The shipped scalar is at chance and miscalibrated on MultiHop-RAG
  (AUC **0.5087**, ECE **0.3085**, risk@80 **0.4927**); a held-out 12-feature gate reads AUC
  **0.8244** / ECE **0.0892** on the same queries. Reliability diagram committed.
- **Fusion ablation (SciFact).** Min-max convex is the best arm (**0.7412**) but only **+0.0092
  over RRF k=60, not significant**; it is significantly better than max-normalised CombSUM
  (+0.0263). The rule matters; RRF is not clearly beaten.
- **Nugget and whole-fact recall.** Claim-recall@10 **0.7438 fused vs 0.7803 CE**; whole-fact
  recall@10 **0.4644 vs 0.5452**; both **0.8480** at depth 100 — the CE's gain is ordering, and 15.2%
  of gold facts are missing from the pool.
- **2WikiMultiHopQA, first run.** 12,576 dev questions over the 56,651-paragraph distractor pool:
  BM25 **MAP@10 0.5037 with Hits@10 0.9927** — the gold is found and the order is the score; a
  retrieval-trained bi-encoder beats it (**+0.0186**) and RRF fusion adds again (**+0.0300**, all
  p<0.0001). **And the coverage rule loses here too**: coverage **−0.0046** and covquota **−0.0056**
  MAP@10 against the fused order (both p<0.0001) while preserving Hits@4, so the MuSiQue failure is
  not idiosyncratic.
- **The learned selector (Bet 2).** On the full 2,255-query protocol (1,127 held out) the logistic
  scorer reaches MAP@10 **0.2939** — not significantly above the cross-encoder (**+0.0058**, p=0.10)
  and significantly below the hand coverage-quota rule (**−0.0208**, p=0.0001). Its bridge feature is
  weighted **5× the shuffled control's** while the held-out ranking is unchanged, so the bridge signal
  is real but redundant. The null now has scale and a control.

## Method notes worth carrying

Each of these cost a round to learn, and each is recorded in `docs/FINDINGS.md`:

- **Measure on the corpus you intend to run.** Two errors came from extending a ratio measured on a
  convenient corpus to another (`merge is the next lever`, `scoring dominates`).
- **Re-measure after the change it justified.** The 400k-document sweep said a small block cap was
  free; the merge that made it safe then cost 65% of build time, reversing the conclusion.
- **A result off by an order of magnitude is a bug signal, not a finding.** HotpotQA's 12× shortfall
  was qrels dilution in a flag written an hour earlier — the same trap already recorded twice in this
  repository, and since found in two more tools.
- **A uniform partial result is an API semantic.** One article per 20-title batch was TextExtracts
  returning a single extract per request, not throttling.
- **Checkpoint the call you are waiting on, not the one you report.** The engine row was resumable;
  the slower control row was not, and cost three hours twice.
- **Pin the prose, not just the table.** Tables are generated and cannot drift; the sentence
  summarising one was wrong twice until `tools/lexical_control.py` generated it too.
- **Verify the guard against the change it guards.** Six claims passing is only meaningful because an
  anchor matching no line *fails* rather than skips.

## Starting again

```sh
make setup && make test          # ~1,760 tests, offline and deterministic
make claim-check                 # prose claims (65 today) against their reports
make consistency-check           # documents agree with each other, the audit set included
make charts                     # regenerate every figure in the README
make job-status                  # any long job and how far it got
make heavy-status                # is a heavyweight job running? run long jobs one at a time
```

Everything long is resumable: `--lexical-checkpoint` per query, `--embed-chunk` per embedding batch,
jobkit per work unit. A job stopped mid-flight costs only the batch in flight.
