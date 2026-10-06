# Programme status

A single page for whoever picks this up next. `ROADMAP.md` carries each item's live status and
commands; this states where the programme actually landed, what it refuted, and what is left with the
cost of getting it.

## Where it landed

| phase | state | headline |
| --- | --- | --- |
| 0 — claim ledger | **complete** | `make claim-check`: **58** prose numbers pinned to the reports that produced them, plus `make consistency-check`, which pins documents to *each other* and computes the BEIR coverage count from disk |
| 1 — engine full-split runs | **complete** | MultiHop-RAG 2,255 both configurations; MuSiQue 2,417 hybrid plus a 200-query default |
| 2 — ordering heads | **complete, entirely negative** | late interaction, relational NLI and an adapted cross-encoder all lose; 2.4 not adopted on licence |
| 3 — end-to-end and cost | **complete** | FRAMES, a generation pilot, citation mechanics, and the cost axis |
| 4 — scale | **partial** | 4.1, 4.3, 4.5 done; **4.2 at 11 of 18 datasets** (Climate-FEVER landed 2026-10-06); 4.4 is an external submission |
| 5 — trust | **complete** | trust report, three-state adjudication, verifier precision, poison-dose probe |

The engine **beats the harness it replaced on MultiHop-RAG** (MAP@10 0.3128 against 0.2510) and
**loses to it on MuSiQue** (0.3761 against 0.3852). Neither generalises, which is the finding.

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

1. **Phase 4.2, the remaining datasets.** Not a code problem, and the reason has changed since
   this page was first written. The UKP mirror now returns **404** for `robust04`, `trec-news`,
   `specter`, `signal1m` and `bioasq` (not a transient stub — the files are gone), and the
   HuggingFace copies under `BeIR/*` are **gated** (401 anonymously): they need a signed license
   or a token, so the block is **access/licensing**, not disk. **Climate-FEVER is done** as a
   lexical row (nDCG@10 **0.1610**, 1,535 queries over 5.4M documents); `fever` is next in the
   queue, resumable from `--lexical-checkpoint`; NQ (2.68M) completed in ~25 min; HotpotQA degrades
   to **~0.04 q/s** on its later queries, which is 10–40 hours for one lexical-only row.
   `--limit-queries` gives a labelled seeded subset (~1 h per corpus).
2. **Phase 4.4, the MTEB submission.** Requires an upload; not offline work.
3. **HotpotQA's full query set.** Its 600-query subset reads **0.6365** against a published 0.603; the
   full 7,405-query run is owed and resumes from a checkpoint (**3,062/7,405** at the last restart).

**The runs are queued one at a time.** Three corpus builds plus a harness launched together
**crashed this host**; no work was lost (every job resumes), and the rule is now enforced by an
exclusive lock in 26 long-running tools (`make heavy-status`). Current order: **Climate-FEVER →
FEVER → HotpotQA → 2WikiMultiHopQA**.

## Newest results (2026-10-06 pass)

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
make setup && make test          # 1251 tests, offline and deterministic
make claim-check                 # 58 prose claims against their reports
make consistency-check           # documents agree with each other on the shared quantities
make charts && make charts-more  # regenerate every figure in the README
make job-status                  # any long job and how far it got
make heavy-status                # is a heavyweight job running? run long jobs one at a time
```

Everything long is resumable: `--lexical-checkpoint` per query, `--embed-chunk` per embedding batch,
jobkit per work unit. A job stopped mid-flight costs only the batch in flight.
