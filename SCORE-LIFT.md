# Raising the scores — a ranked programme

**Written 2026-10-10.** This is a *forward* document: where the measured scores actually sit
relative to the published systems this project compares itself to, and what would move them.
It is not a findings document — every number quoted as "measured" is reproduced from a report
already in the tree, and everything else is labelled **expected** and carries a falsifier.

Companion documents, and what each is for:

| document | question it answers |
| --- | --- |
| [`RANKINGS.md`](RANKINGS.md) | the retrieval programme: what was measured, on which harness, and why |
| [`BENCHMARKS.md`](BENCHMARKS.md) | how the system is *placed* outside the project, and what cannot be claimed |
| [`docs/PROGRAMME-STATUS.md`](docs/PROGRAMME-STATUS.md) | the live status of every phase and open item |
| [`../best-in-the-world/`](../best-in-the-world/) | the long-horizon "what would make this the best in the world" corpus |
| **this file** | **given all of that, what to run next, in what order, for how much** |

**How to read it.** §1–§6 are the score programme: what to run next, ranked, with expected deltas
and falsifiers. §7 is *where to be ranked* — which leaderboards are reachable and what a survey of
the landscape did and did not add. **§8 is a separate, lower-ranked portfolio programme**: the
specification for crossing the capability boundaries (multilingual, tables, vision, multi-turn) that
currently make several benchmarks "not applicable". **§9 turns the RAG-evaluation survey's
bibliography into 14 numbered tasks** — datasets to run, and two methods (eRAG, RAGEval) for building
the own-domain benchmark §7.2 says is missing. **§10 is the execution layer**: one order across all
three sequences, every pre-registered bar in one table, how to reproduce §1, and what could invalidate
the document. Nothing in §8 or §9 raises a score on the corpora measured today; §9's T8 is nonetheless
the highest-value item outside §1–§6.

> **Revision note.** The `best-in-the-world/README.md` "open bets" framing is **stale** relative to
> the measurements taken after it was written: Bet 2's set-aware selector is a measured null
> (`docs/PROGRAMME-STATUS.md`), Bet 3's encoder adaptation **shipped** as the default dense channel,
> and the 2WikiMultiHopQA replication is done and negative for coverage. Its own §"open" list is
> therefore not the right work queue, and this document does not use it as one.
>
> **Second revision note (added after §7).** `BENCHMARKS.md`'s TREC RAG verdict — "blocked: corpus
> and gold nuggets need NIST access, and scoring needs the official assessor" (lines 43, 200) — is
> **stale for the retrieval half**. TREC RAG 2025 added a *Retrieval-only* task scored with
> `trec_eval` on nDCG@30/@100 and recall@100 over 43 published runs, with no generation step. See
> §P10a. The `BENCHMARKS.md` text describes the RAG and AG tasks and does not mention it.
>
> **Third revision note.** An earlier draft of this document described the system as
> "generator-free". That was **wrong**: it ships two answer paths, an external-LLM `Composer` and a
> model-free `compose_extractive`, and only the second is generator-free. The consequence is not
> cosmetic — the extractive path makes an end-to-end row whose score is entirely the engine's, and
> it is ~3 s/query against the LLM path's ~36 s/query, which removes the 22–25 hour cost estimate
> that `BENCHMARKS.md` uses to explain why no end-to-end ranking exists. See §P12 and §7.1.
>
> **Fourth revision note.** §8 was added on request: a specification and staged plan for the
> capability boundaries that make several benchmarks "not applicable". One finding there is a live
> **bug**, not just a boundary — the lexical channel tokenises Russian, Arabic, Greek, Hindi and
> Thai to **zero tokens** and splits German at the umlaut, with no ingest guard, and the lexical
> channel is the shipped default. See §8.3.
>
> **Fifth revision note.** §8.9's original B2 verdict — "caption-only is cheap, visual embedding is
> a different product" — was **wrong**, and the correction is recorded there. Academic papers are
> visually rich *text* documents, not document-images, so they are the dominant case rather than a
> separate product; [MMDocIR](https://arxiv.org/html/2501.08828v3) is the benchmark for them and
> supplies OCR-text and VLM-text for every image layout, so a text-only engine can enter it without
> a vision model. B2 is now staged **B2a** (PDF + layout-level chunks + bbox locators, weeks) →
> **B2b** (visual embedding channel) → **B2c** (local VLM-text generation, deferred).
>
> **Sixth revision note.** §9 added on request: the RAG-evaluation survey
> ([arXiv:2504.20119](https://arxiv.org/abs/2504.20119)) turned into 14 tasks. Its prose names few
> datasets, so the PDF was fetched and extracted locally and its 71-entry bibliography triaged;
> **all 24 arXiv identifiers used here were fetched and returned 200**, and titles were read back
> rather than trusted. That check caught the survey's `RAG-ConfusionQA` now resolving as **ELOQ**,
> and caught three substring false positives (`UDA`, `eRAG`, `VERA` inside "aver**age**") that would
> otherwise have been recorded as already-present.
>
> **Seventh revision note.** §1 was rewritten to be **machine-generated**, and §10 was added as the
> execution layer. `tools/channel_coverage.py` (`make channel-coverage`) now regenerates §1.1 and
> §1.2 from `eval/beir/*.json`; running it falsified three hand-entered cells in the first draft —
> FEVER's published column is 0.812 (uniCOIL) not 0.796 (SPLADE), and Signal-1M and Touché were being
> compared against **BM25** columns where every other row used a model column. The headline arithmetic
> is unchanged (**+0.022** half-closed, **+0.040** nine tenths) but the count is **nine** cells behind,
> not ten, and the summed deficit is **0.8054**, not 0.809.

> **Eighth revision note.** Wiring `make channel-coverage` into `make consistency-check` (the
> seventh note's follow-up, done at last) turned the whole of §1.2 over and **falsified its `ours`
> column on nine of eighteen cells**. That column had been transcribed from `README.md`'s prose,
> and the title-weighted and adapted-dense passes landed *after* the transcription: FEVER 0.6323 →
> **0.6749**, HotpotQA 0.6352 → **0.6563**, CQADupStack 0.2943 → **0.3046**, plus SciFact, NFCorpus,
> SCIDOCS, ArguAna, FiQA and Signal-1M moving by a digit as their reports were refined. The column
> is now computed through `tools/charts_compare.our_rows` — the repository's one implementation of
> "best row" — so a hand-entered value cannot come back. **The consequences are not cosmetic:** the
> macro is **0.4799**, not the 0.4760 this section stated, and the deficit arithmetic moves with it
> — summed deficit **0.7085**, macro cost **0.0394**, half-closed **+0.020**, nine tenths **+0.035**
> (the seventh note's 0.8054 / +0.022 / +0.040 are the same computation over the stale column and
> are superseded). Signal-1M also becomes the one member whose *own best measured row* is the flat
> control, so it reads **+0.068**, not +0.017. The §2 ranking is unaffected: the nine losing cells
> are the same nine, and P1 is still the largest single gain available.


---

## The answer in one paragraph

The largest single score gain available to this system is not a modelling idea. **Eight of the
eighteen BEIR cells this project counts as "covered" have never had a dense or learned-sparse row
at all — only a lexical one** — and three more have only Laya's adapted encoder fused. The nine of
those cells that sit behind the published matrix carry a summed deficit of **0.7085 nDCG@10**;
closing half of it is **+0.020 on the BEIR-18 macro**, closing nine tenths is **+0.035** (0.4799 →
~0.50–0.52). That is larger than every modelling result the programme has produced so far (fusion
tuning +0.004, SPLADE +0.004–0.023, per-corpus routing +0.006, the three-model ensemble +0.043 on
five datasets), and it is engineering the repo has already built the machinery for. Only after
that does the second lever — a 2025-era encoder, which the current code cannot load *correctly*
without four small changes — have room to pay.

---

## 1. Where the numbers actually are

### 1.1 What "18 of 18" hides

`BENCHMARKS.md`, `PROGRAMME-STATUS.md` and the consistency gate all report **BEIR coverage 18 of
18**. That count is a statement about *dataset membership*, and it is correct — every member has a
report. It is not a statement about *channel coverage*, and the difference is large.

**This section is machine-generated, not hand-built.**
`make channel-coverage` → `tools/channel_coverage.py` reads the retriever rows actually present in
`eval/beir/*.json` and prints the two tables below. Every number in §1.1 and §1.2 comes from that
run. (The tool was written *after* the first draft of this section and immediately falsified three
of its hand-entered cells — see the note under §1.2.)

| dataset | documents | rows measured | richest channel measured |
| --- | ---: | --- | --- |
| arguana | 8,674 | 13 rows | full |
| bioasq | 14,914,604 | bm25, bm25-engine | **lexical only** |
| climate-fever | 5,416,593 | bm25-engine, bm25-title | **lexical only** |
| cqadupstack | 457,199 | biencoder, bm25, bm25-engine, bm25-title, hybrid-biencoder | dense |
| dbpedia-entity | 4,635,922 | bm25-engine, bm25-title | **lexical only** |
| fever | 5,416,568 | bm25-engine, bm25-title | **lexical only** |
| fiqa | 57,638 | 9 rows | full |
| hotpotqa | 5,233,329 | bm25-engine, bm25-title | **lexical only** |
| msmarco | 8,841,823 | bm25, bm25-engine | **lexical only** *(not a member)* |
| nfcorpus | 3,633 | 16 rows | full |
| nq | 2,681,468 | biencoder, bm25-engine, bm25-title, hybrid-biencoder | dense |
| quora | 522,931 | biencoder, bm25-engine, bm25-title, hybrid-biencoder | dense |
| robust04 | 528,155 | biencoder, bm25, bm25-engine, bm25-passages@600, bm25-title, hybrid-biencoder | adapted dense |
| scidocs | 25,657 | 9 rows | full |
| scifact | 5,183 | 17 rows | full |
| signal1m | 2,866,316 | bm25, bm25-engine, bm25-title | **lexical only** |
| trec-covid | 171,332 | biencoder, bm25, bm25-engine, bm25-passages@600, bm25-title, hybrid-biencoder | adapted dense |
| trec-news | 594,977 | biencoder, bm25, bm25-engine, bm25-passages@600, bm25-title, hybrid-biencoder | adapted dense |
| webis-touche2020 | 382,545 | biencoder, bm25-engine, bm25-title, hybrid-biencoder | dense |

**lexical-only members (6 of 18):** bioasq, climate-fever, dbpedia-entity, fever, hotpotqa, signal1m
**members with no report:** none
**non-members present (not coverage):** msmarco

### 1.2 The macro, cell by cell

`0.4799` is the project's best-per-dataset BEIR-18 macro (`tools/charts_compare.our_rows` reads it
off the reports; `README.md` §"What cannot be claimed" states it). Against the **model columns** of
`tools/published_beir18.py` (the parsed *Resources for Brewing BEIR* table — BM25 excluded by
convention, because the paper's BM25 is a multi-field index and this repository's control is flat):

| cell | ours | best published model column | gap | channel we have there |
| --- | ---: | ---: | ---: | --- |
| Quora | 0.8876 | 0.865 | +0.023 | biencoder |
| SciFact | 0.7667 | 0.699 | **+0.068** | hybrid-ensemble |
| TREC-COVID | 0.6871 | 0.711 | −0.024 | hybrid-biencoder |
| FEVER | 0.6749 | 0.812 | **−0.137** | bm25-title |
| ArguAna | 0.6563 | 0.521 | **+0.135** | dense-ensemble |
| HotpotQA | 0.6563 | 0.686 | −0.030 | bm25-title |
| BioASQ | 0.5193 | 0.504 | +0.015 | bm25-engine |
| FiQA-2018 | 0.4441 | 0.351 | **+0.093** | hybrid-ensemble |
| webis-touche2020 | 0.4419 | 0.298 | **+0.144** | bm25-engine |
| Robust04 | 0.4405 | 0.473 | −0.033 | hybrid-biencoder |
| NFCorpus | 0.3867 | 0.345 | **+0.042** | hybrid-ensemble |
| TREC-NEWS | 0.3745 | 0.428 | −0.054 | hybrid-biencoder |
| Signal-1M | 0.3640 | 0.296 | **+0.068** | bm25 |
| NQ | 0.3264 | 0.544 | **−0.218** | bm25-title |
| DBPedia-entity | 0.3166 | 0.442 | **−0.125** | bm25-title |
| CQADupStack | 0.3275 | 0.345 | −0.018 | hybrid-biencoder |
| SCIDOCS | 0.2248 | 0.165 | **+0.060** | hybrid-ensemble |
| Climate-FEVER | 0.1657 | 0.237 | **−0.071** | bm25-title |

The columns are the tool's: `ours` is the best measured row per member, `channel we have there` is
that row's name, and the published column is the cell's best **model** system. A dash is not needed
anywhere — every member has both a measured row and a published one.

**What the tool falsified, in two rounds.** The first draft of this table was hand-built, and the
tool killed three of its cells at once: FEVER's published column is **0.812 (uniCOIL)**, not 0.796
(SPLADE) — I had used SPLADE as the ceiling everywhere, which *understated the largest gap in the
table*; and Signal-1M and Touché were being compared against their **BM25** columns (0.330, 0.367)
while every other row used a model column, so the best model system there is 0.296 and 0.298.

The **second** round is the one worth dwelling on. The whole `ours` column used to be transcribed
from `README.md`'s prose, and by the time `make channel-coverage` was wired into `make
consistency-check` it disagreed with the reports on **nine of eighteen** cells: the title-weighted
pass landed FEVER 0.6323 → **0.6749**, HotpotQA 0.6352 → **0.6563**, CQADupStack 0.2943 → **0.3046**
*after* the column was copied, and four ensemble cells (SciFact, NFCorpus, SCIDOCS, ArguAna) had
moved by a digit as their reports were refined. The column is now computed through
`tools/charts_compare.our_rows` — the repository's single implementation of "best row" — so a
hand-entered value cannot come back. **The macro is 0.4799, not the 0.4760 this section used to
state**, and every number below moved with it.

**On the two cells where BM25 beats every model column.** Touché (BM25 0.367) and Signal-1M
(BM25 0.330) are the only members where the strongest published entry is a *lexical* system — which
is the published field independently agreeing with §1.3's per-corpus finding rather than a defect in
the table. Signal-1M is also the one member whose *own best measured row* is the flat control
(0.3640), so the computed column credits the control there and the cell reads **+0.068** rather than
the −0.017 a naive reading of "ours 0.3131" gives.

**The deficit arithmetic.** Summing the **nine** negative gaps — FEVER 0.1371, NQ 0.2176, DBPedia
0.1254, Climate-FEVER 0.0713, TREC-NEWS 0.0535, TREC-COVID 0.0239, CQADupStack 0.0177, HotpotQA 0.0297,
Robust04 0.0325 — comes to **0.7085 nDCG@10**. Divided by 18, the gap costs the macro **0.0394**.
Half of it is **+0.020**; nine tenths is **+0.035**. Everything in §2 is either an attack on that
arithmetic or a separate contribution. (The tool prints these figures; the seventh revision note's
0.8054 / +0.022 / +0.040 are the same arithmetic over the *stale* column and are superseded here.)

**Ten cells need channel work** — the nine above with a negative gap, plus BioASQ, where the
published dense models *fail* (Contriever 0.383 against BM25 0.465) and the right answer may
therefore be "no dense channel here". Quora is the eleventh mixed cell: it has a dense row but has
never had an ensemble or sparse row.

**Every cell where the system loses to the published matrix is a cell where its best row is
lexical-only or adapted-dense-only.** All five full-channel cells win, four of them by more than
0.04. That is not a coincidence and it is the whole story of this document.

### 1.3 The second-order finding: the shipped lexical config is not even the best lexical config

Two of the eleven cells are led by the *control*, not the engine:

| dataset | flat `bm25` control | shipped `bm25-engine` | delta |
| --- | ---: | ---: | ---: |
| Signal-1M | **0.3640** | 0.3131 | **−0.0509** |
| CQADupStack (12-forum macro) | **0.2988** | 0.2943 | −0.0045 |
| BioASQ | 0.5168 | **0.5193** | +0.0025 |
| TREC-COVID | 0.5573 | **0.6188** | +0.0615 |

On Signal-1M the shipped settings (stemming on, query stopwords dropped, title weighted) cost
**five points** against the flat control. The repo already knows the instruments reverse per
corpus — dense beats fusion on Quora and ArguAna, fusion loses to BM25 by 0.09 on Touché, the
cross-encoder wins on MultiHop-RAG and loses on all five BEIR sets — but it has never acted on that
knowledge for the *lexical* channel, which is the only channel the eight large cells have.

---

## 2. The programme

Ordered by `expected score gain ÷ effort`, with the falsifier stated for each so a negative result
is as publishable as a positive one. `[data]` marks items that need no modelling work at all.

### P1 — Run the channels that already exist on the eleven empty cells `[data]`

**The first cell needs a code change, and the run that discovered it took one minute.** My first
specification was wrong twice, in the same way P4's was, and it is worth recording both because they
are the same mistake:

- **`tools/beir.py` refuses CQADupStack by design**: *"cqadupstack is twelve sub-collections behind
  one BEIR name, not one corpus; this harness loads one corpus per dataset. Run
  `tools/cqadupstack.py` instead."* The 457k documents are the aggregate of twelve forums, each
  20–40k — so the per-forum embedding cost is **minutes, not hours**, and §10.1's "days" is the
  aggregate across eight members.
- **`tools/cqadupstack.py` is lexical-only.** It takes `--retrievers` but rejects anything outside
  `LEXICAL_RETRIEVERS`, so it cannot produce the dense or hybrid row P1 needs. **That is the
  blocker, and it is a code change**: the dense pass (the `--biencoder-model` / `--dense-weight`
  path in `tools/beir.py`) has to be reachable from this tool, which already solves the
  twelve-forum bookkeeping P1 depends on.

**What the dense row must reproduce** is still recorded from the five full-channel cells
(`eval/beir/five-adapted.json`): `biencoder_model` **`models/laya-retrieval`**, `dense_weight`
**0.3**, `lexical_weight` 1.0, `top_k` 100, `bm25_k1` 0.9, `bm25_b` 0.4, `rrf_k` 60, `title_boost` 2,
`include_titles` true. And at aggregate scale the two flags that are not optional are
`--lexical-backend ondisk` (the memory index keeps a `Counter` per document) and a checkpointed
dense pass — lessons this programme has already paid for twice.

**So P1's next concrete step is the extension, not a run** — and the good news is that the per-forum
size makes it testable in minutes once the path exists.

**The extension, to the point where it is a transcription rather than a design.** `beir.py` already
has every mechanism; `cqadupstack.py` uses four of them and the dense row needs three more, so the
change is: add `--biencoder-model` (default `models/laya-retrieval`), `--dense-weight` (default 0.3)
and `--embed-chunk` to the parser; build the encoder **once** in `main` rather than per forum; and
give `rank_subforum` an `encoder`/`vectors` parameter plus two branches, mirroring
`beir.py`'s own dense block (lines ~1559–1600):

```python
vectors = beir.cached_embed(encoder, corpus, cache_dir / f"{forum}.{encoder.name}",
                            use_cache=True, chunk=embed_chunk)
bi = beir.dense_rankings(corpus, queries, encoder, vectors, top_k)      # 'biencoder'
...
rankings["hybrid-biencoder"] = beir.hybrid_rankings(bi, lexical or {}, top_k, ...)
```

Two signatures have to be read before writing it — **`beir.hybrid_rankings`** (its full parameter
list, since the row records a dense weight and a lexical weight) and **`beir.active_biencoder`**
(how a dataset name resolves to a model, which is what decides whether the forum id or an explicit
model has to be passed) — and `LEXICAL_RETRIEVERS` in `cqadupstack.py` is what currently rejects the
two new names. **`--embed-chunk` matters here for the same reason it does everywhere else**: the
twelve forums embed separately and a kill should cost one forum, not the member.

**P1's first cell is measured, on one sub-forum, and the dense channel is worth having.** The
extension landed and the prescribed cheap launch ran — `--subforums android`, the smallest forum at
**22,998 documents**, embedding in **568.9 s**:

| cqadupstack/android, 1 of 12 sub-forums | nDCG@10 | nDCG@100 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| `bm25-engine` (the row this member already had) | 0.3681 | 0.4178 | 0.4595 | 0.3696 |
| `biencoder` (dense alone) | 0.3580 | 0.4118 | 0.4673 | 0.3510 |
| **`hybrid-biencoder`** (dense w=0.3 + `bm25-engine`) | **0.4067** | **0.4456** | **0.5079** | **0.4079** |

**+0.0386 nDCG@10 over the lexical row** — and the shape matters: dense *alone* is slightly worse
than BM25 (0.3580 against 0.3681), while the fusion is clearly better than either, which is the
§1.2 prediction for the lexical-only members stated as a measurement rather than an expectation.
Recall@10 moves +0.0484, the largest of the four.

Two caveats travel with it, and the tool prints the first itself: it is **1 of 12 sub-forums**, so it
is a subset run and not the member's row, and the member is then macro-averaged with seven others.
The **cost** is now measured per forum: 568.9 s to embed 23k documents at ~40 docs/s, so the twelve
forums are ~2–3 hours of embedding and the eight members are where §10.1's "days" comes from — the
`--embed-chunk` checkpointing and the per-forum embedding cache are what make that resumable.

**A subset report has to live outside `eval/beir/`, and that cost two gate failures to learn.** The
run above was written to `eval/beir/p1-cqadupstack-android.json`, and two authorities scan that
directory's top level (`glob("*.json")`): the channel-coverage gate read the android dense row as
evidence that **CQADupStack itself** now has a dense channel — changing §1.1's lexical-only count
from 8 to 7 — and the lexical selector saw `cqadupstack`/`bm25-engine` disagree with itself, because
one report held the 12-forum macro and the other a single forum's 0.3681. Both are the correct
behaviour for a member row and the wrong reading for a **subset**. The report now lives in
`eval/beir/subsets/`, both gates pass, and the convention is: **anything run with `--subforums`,
`--limit`, `--limit-docs` or a seed goes to `eval/beir/subsets/`**, because a subset is evidence
about a subset and the authorities cannot tell from the numbers alone.

**The member run is in flight, and the caching is doing what it was built for.** The twelve-forum
pass is running now (`--out eval/beir/cqadupstack-dense`, `--retrievers biencoder,hybrid-biencoder`,
`--embed-chunk 5000`). Two things worth recording from its first minute:

- **android re-ran from cache**: it printed `22998 documents, 699 scored queries` immediately and
  moved to `english` without embedding anything, so the per-sub-forum vector cache means a re-run
  costs queries rather than embeddings — and a kill costs at most the sub-forum in progress;
- the run asks for **only the dense rows**, so `cqadupstack-dense.json` carries no second
  `bm25-engine` for the lexical selector to disagree with, while the coverage gate will see the
  member's dense channel when it lands. That is also why the gates stay green while it runs: the
  report is written at the end, and §1.1 will need regenerating **then**, not now.

Expect roughly three hours of embedding for the remaining ~434,000 documents at the measured ~40
docs/s. The comparison will be against the recorded member rows in `eval/beir/cqadupstack.json`, and
the android datum above says what to expect in sign: the fusion ahead of the lexical row, the dense
channel alone behind it.

**The resumability claim is verified, and it has one wart worth recording.** The per-sub-forum
caches are on disk — `eval/beir/cache/android.laya-retrieval.mean.npy.{npy,ids.json,meta.json}`,
written at 16:31 by this run — so a kill really does cost only the sub-forum in progress, which is
what the android re-run demonstrated. The wart: because `cqadupstack.py` passes
`cache_root / f"{name}.{encoder.name}.npy"` and `cached_embed` appends its own suffix, the file ends
up **`.npy.npy`**, where the older BEIR datasets cache as a single `.npy`. It is cosmetic and it is
*not* being fixed while this run is live: changing the path would orphan the android and english
vectors and cost ~20 minutes of re-embedding to correct a filename. Recorded so the next cold cache
aligns the two, since a cache directory with two naming schemes is the kind of thing that wastes an
afternoon later.

**The member row is in, and it moves the deficit arithmetic — the first time P1 has.** All twelve
sub-forums, macro-averaged:

| cqadupstack (12-forum macro) | nDCG@10 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: |
| `bm25-engine` (the row this member had) | 0.2943 | 0.3748 | 0.2891 |
| `bm25-title` (the row §1.2 was scoring it on) | 0.3046 | 0.3886 | 0.2984 |
| `biencoder` (dense alone) | 0.2882 | 0.3811 | 0.2792 |
| **`hybrid-biencoder`** (dense w=0.3 + `bm25-engine`) | **0.3275** | **0.4159** | **0.3221** |

**+0.0229 over the row §1.2 was using**, and the android datum's shape held exactly: the fusion
ahead of both lexical rows, the dense channel **alone** behind both (0.2882 against 0.2943 and
0.3046). Its gap to the published column closes from **−0.040 to −0.018**.

**And that is the first movement of the BEIR-18 arithmetic this programme has produced:**

| | before | after |
| --- | ---: | ---: |
| summed deficit | 0.7314 | **0.7085** |
| macro cost | 0.0406 | **0.0394** |
| nine tenths closed | +0.037 | **+0.035** |
| lexical-only members | 8 of 18 | **7 of 18** |

Seven members remain, and if their dense rows behave like this one's the arithmetic keeps moving the
same way. §1.1's table is regenerated from the authority in this commit — *generated, not typed*, as
the tool says — and the registry's pinned literals moved with it.

**The second member is running, and it is a different size of job.** `nq` — **2,681,468 documents
and 3,452 scored queries** — went up with `--retrievers biencoder,hybrid-biencoder`,
`--lexical-backend ondisk` and `--embed-chunk 20000`, out to `eval/beir/nq-dense`. Scaling from
CQADupStack (457k documents in ~3 hours) puts this at **15–18 hours of embedding**, plus an on-disk
BM25 build over the same corpus before the queries can be scored. It will span many rounds, and its
report lands only at the end.

**One precondition was checked before committing fifteen hours to it, and it was worth checking.**
`hybrid-biencoder` fuses against the engine's lexical channel, and for a 2.7M-document corpus an
in-memory BM25 would keep a `Counter` per document; `bm25_rankings` takes `backend`/`index_dir`
precisely so that does not happen, and the engine's lexical row is built with `**on_disk("engine")`.
So `--lexical-backend ondisk` **does** feed the fused rows rather than only the row that names it —
which is the difference between a 15-hour run and a 15-hour run that dies at hour fourteen.

**Order from here is smallest-first among what remains**: nq 2.68M, signal1m 2.87M, dbpedia-entity
4.64M, hotpotqa 5.23M, climate-fever and fever 5.42M each, bioasq 14.9M. And the number they are
chasing is now concrete: the summed deficit stands at **0.7085**, and CQADupStack's cell shows what a
dense row does to one.

**The on-disk build's cost curve now reaches 2.68M documents, and it is close to linear.** `nq`'s
lexical index finished at 20:22, having started at 19:53 — **~29 minutes for 2,681,468 documents**,
against the 240–396 s the class docstring records at 400,000. That is 6.7× the documents for 4.4–7.3×
the time, so the spill-and-merge design is not degrading at this size, and the earlier extrapolation
(15–18 h of *embedding*) is unaffected — the build is not the long pole, the embedding is.

It is also the last phase in the harness that reported nothing while it worked, and the fix for that
landed one round before the *next* member needs it: the remaining six are all larger than nq.

**A note on how this round went, because the pattern has now happened twice.** Both "one flag"
tasks (P4 and P1) opened with a specification of mine that was wrong, and both were corrected within
a minute of actually running something. The lesson is not that the specifications were wrong — it is
that **a cheap launch is a better specification tool than more reading**, and both times the
expensive version would have been a mis-specified hours-long run. For P1 specifically, the cheapest
possible launch is one forum with `--retrievers bm25-engine` to prove the twelve-forum plumbing
before the dense path is written.

### P2 — Fix the lexical channel per corpus, starting with the two cells the control leads `[data]`

**What.** (a) Make the lexical settings — stemming on/off, query-stopword retention, title boost,
`k1`/`b`, and field weighting — a **per-corpus** choice rather than a global one, selected on the
corpus itself; (b) implement **BM25F** (per-field `b` and weight) rather than the current
"repeat the title twice" proxy.

**Why it is not already falsified.** The Signal-1M 0.3131-vs-0.3640 control reversal is measured
and recorded but never acted on. `tools/beir.py` already has `--no-stem`, `--title-boost N` and
`--k1/--b`; putting them behind a per-corpus selector is a harness change, not a research bet. The
BM25F half is genuinely untried: the repo's own `published_beir18.py` docstring records that the
published BM25 column is a **multi-field** index while its control is a **flat** one — so the
−0.121 FEVER gap, the largest measured, sits exactly on the axis BM25F addresses, and
`findings-bm25-parameters.md` (k1/b indistinguishable) does not cover field weighting.

**Expected.** Signal-1M up to +0.05 (it is currently below its own control), CQADupStack +0.005,
and a partial FEVER recovery if field weighting is the mechanism — **+0.002 to +0.010 macro**,
plus a real shot at the FEVER anomaly.

**Measured for (a), and the first falsifier is refuted.** `tools/lexical_selector.py`
(`make lexical-selector`) profiles the four configurations the harness already exposes against the
rows already on disk, so the answer cost no new runs (`docs/findings-lexical-selector.md`):

- **The picks differ per corpus** — `bm25-title` ×11, `bm25-engine` ×6, and the flat published
  `bm25` control ×1. The selector has something to choose, so §P2(b) and the routing question are
  live rather than closed.
- **The oracle is +0.0109** on the lexical-only macro (0.4195 → 0.4303), at the top of the
  +0.002–+0.010 band. Signal-1M alone contributes **+0.0509**; FEVER **+0.0426**, HotpotQA +0.0211,
  TREC-COVID +0.0201, NQ +0.0146, DBPedia +0.0113.
- It is an **oracle**: it selects on the same qrels it scores. A label-free rule reaching even half
  of it is §P5(b)'s transductive selector, which is untried.
- Signal-1M is the sharpest reading: not a missing channel but a **misconfigured** one — the member
  has no dense row at all, and its only channel was tuned where its settings won.

**Measured for (b), and the proxy was not the problem.** `laya_rag.lexical.BM25FIndex` and the
harness's `--bm25f`/`--title-weight`/`--bm25f-b-title`/`--bm25f-b-body` flags are in
(`docs/findings-bm25f.md`). A weight sweep over the five small full-channel corpora:

| corpus | `bm25-engine` | title ×2 proxy | best BM25F | weight |
| --- | ---: | ---: | ---: | ---: |
| SciFact | 0.6762 | 0.6824 | 0.6829 | 3 |
| NFCorpus | 0.3244 | 0.3248 | 0.3255 | 3 |
| ArguAna | 0.4169 | 0.4275 | **0.4407** | 12 |
| SciDocs | 0.1497 | 0.1523 | 0.1557 | 12 |
| FiQA-2018 | 0.2386 | 0.2386 | 0.2386 | any |
| **macro** | **0.3612** | **0.3651** | **0.3687** | — |

The optimal weight is **corpus-dependent and not in one direction** (3 on two, still rising at 12 on
two, flat on FiQA), so there is no single deployed title weight — the same shape §P2(a) found for the
other knobs. The proxy is *close* (0.3651 against 0.3662 at a shared weight) but not equivalent, and
it fails exactly where theory says it should: **ArguAna +0.0132**, a corpus whose "titles" are shared
topic keyword lists, so repeating them inflates the field that is already the whole signal. At `w=1`
— fields separated but unweighted — ArguAna *loses* 0.011 against plain BM25, which locates the gain
in the weight rather than the split.

**The falsifier's second branch is still open, for a structural reason.** No FEVER row exists: the
member is 5.4M documents, `BM25FIndex` is in-memory, and FEVER's existing lexical rows went through
the on-disk postings backend, which has no field-aware mode. Extending that backend is the
unblocking work: the postings format, the build loop and the memory accounting all assume one length
per document.

**Cost.** Hours for the selector profile and the BM25F sweep (done — neither needs embeddings); the
harness switch that would act on the selector is a small change to `tools/beir.py` and the engine's
lexical config. The on-disk BM25F backend is the remaining large piece.

**Falsifier.** The per-corpus selector picks the same settings on every corpus, or BM25F does not
move FEVER. The first is **refuted above** — the picks are 11/6/1 across the four configurations.
The second is **untested**: `bm25-title` is a proxy for field weighting, not field weighting, and
it recovers +0.0426 of FEVER's −0.137.

---

### P3 — Encoder refresh: teach `BiEncoderModel` the 2025 encoder interfaces

**What.** The shipped dense model is `BAAI/bge-small-en-v1.5` (33M, 2023). The 2025 CPU-feasible
tier is a different axis from the one already measured. Replace or add one modern encoder.

**Why it is not already falsified — and why the existing negative does not cover it.** `RANKINGS.md`
concludes "encoder size saturates at `bge-base`" from `bge-small` (33M) / `bge-base` (110M) /
`bge-large` (335M) — **three checkpoints of one 2023 family, differing only in width**. It is a
statement about scaling a family, not about better-trained models. A 2025 encoder differs in
initialisation (LLM-init backbones), objective (geometric distillation from a proprietary teacher),
hard-negative weighting, instruction-awareness and MRL — the axes that actually moved MTEB
between 2024 and 2025. The repo's own `best-in-the-world/sota-landscape.md` §1 already says so and
recommends a 300M-class replacement; nothing in `eval/` tests it.

**What this host can load — verified today, not assumed.** Probed on this machine with the pinned
`transformers` 4.57.6 / `torch` 2.2.2 (config + tokenizer fetch, writable `HF_HOME`):

| model | loads? | architecture | note |
| --- | --- | --- | --- |
| `Qwen/Qwen3-Embedding-0.6B` | **yes** | `Qwen3ForCausalLM` → `Qwen3Model` | needs **EOS/last-token** pooling + an instruct template |
| `BAAI/bge-m3` | **yes** | `XLMRobertaModel` | 568M; CLS pooling — **nearly drop-in**; also supplies learned-sparse |
| `intfloat/multilingual-e5-large-instruct` | **yes** | `XLMRobertaModel` | needs `query: `/`passage: ` prefixes |
| `Snowflake/snowflake-arctic-embed-l-v2.0` | **yes** | `XLMRobertaModel` | 568M multilingual |
| `mixedbread-ai/mxbai-embed-large-v1` | **yes** | `BertModel` | 335M, CLS pooling (the `auto` heuristic would wrongly pick mean) |
| `google/embeddinggemma-300m` | **no — gated** | — | needs the Gemma licence accepted + an HF token; friction, not a blocker |
| `nomic-ai/nomic-embed-text-v2-moe`, `jinaai/jina-embeddings-v3` | not reached | — | need `trust_remote_code`, which `model.py` never passes |

**The failure mode is silent, and it is verified.** `AutoModel.from_config` resolves
`Qwen/Qwen3-Embedding-0.6B` to `Qwen3Model` with hidden size 1024 — so `BiEncoderModel` would
*load it and run it today*, with no error, and embed it with **mean pooling**, because
`"bge" not in "qwen/qwen3-embedding-0.6b"` and the heuristic therefore returns `mean`. Qwen3
Embedding is trained with last-token pooling. That is precisely the class of mismatch
`_resolve_pooling`'s own docstring warns is silent — "Getting this wrong is silent, it just embeds
worse" — and it is why a naive `EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B` run would produce a
misleading negative.

**The four code changes** (all in `src/laya_rag/model.py::BiEncoderModel`, ~150–250 lines):

1. **Pooling.** `_resolve_pooling` accepts only `cls` and `mean`. Add `last` (last non-padding
   token) for Qwen3-Embedding and the decoder-only rerankers, and extend the `auto` heuristic by
   family (`mxbai`/`arctic`/`bge-m3` → cls, `e5` → mean, `qwen3-embedding` → last).
2. **Prompt templates.** There is one hard-coded bge prefix and one boolean
   (`EMBED_QUERY_INSTRUCTION`). Modern encoders need a *pair* of templates (query and document) —
   Qwen3's `Instruct: {task}\nQuery: {q}`, EmbeddingGemma's `task: search result | query: `,
   e5's `query: `/`passage: `. Add `EMBED_QUERY_TEMPLATE` / `EMBED_DOC_TEMPLATE`.
3. **MRL truncation.** Add `EMBED_DIM` so 1024- or 768-dim models can be cut to 256 — the single
   highest-leverage memory decision for the P1 programme, and it makes the 39M-document embedding
   pass 3–4× cheaper.
4. **`trust_remote_code`.** Pass it through only when explicitly enabled.

**Expected.** On the five full-channel BEIR datasets, a modern 300M-class encoder should land at
or above the current three-model 478M ensemble's 0.495, i.e. **+0.00 to +0.03 on the five-set mean**
while cutting parameters ~1.6× and corpus embedding passes from three to one. `bge-m3` is the
lowest-risk probe because it is nearly drop-in and additionally supplies a learned-sparse channel
from the same forward pass; `Qwen3-Embedding-0.6B` is the higher-ceiling one.

**Cost.** Days for the code plus one embedding pass per candidate on the five small corpora, then
on the eleven large ones only if it wins. `make beir` already caches embeddings keyed on model
contents (`fix(beir): key embedding caches on model contents`), so re-runs are cheap.

**Falsifier.** A pre-registered bar: the candidate must beat `bge-small`'s `hybrid-biencoder` on
**at least four of five** datasets with paired intervals excluding zero, and must not lose to
`bge-small` on MultiHop-RAG MAP@10. Miss it and the 2023 family stands, recorded as a third
independent confirmation that size and pedigree do not transfer here.

---

### P4 — A modern, and where possible listwise, second stage on MultiHop-RAG

**What.** MultiHop-RAG MAP@10 is the system's worst published comparison: **0.319 best measured
(0.3128 engine default) against a published 0.480**, and the published 0.480 is *reranked*. The
reranker the repo measured is `bge-reranker-base` (2023, BertForSequenceClassification) at depth
100 over a fused pool. Three things are untried and cheap.

**Why it is not already falsified.** The rejection is specific — "a *pointwise* MS MARCO
cross-encoder over the fused top-100" — and the repo says so itself: what remains open is "a
listwise or late-interaction ranker rather than another pointwise one". `jina-reranker-v3` was
declined on **licence**, not on measurement. Two Apache-2.0 alternatives load on this host today:

| reranker | loads here | drop-in for `CrossEncoderModel`? | note |
| --- | --- | --- | --- |
| `BAAI/bge-reranker-base` (measured, loses) | yes | yes | 2023 baseline |
| `BAAI/bge-reranker-v2-m3` | **yes**, `XLMRobertaForSequenceClassification` | **yes — zero code change** | 568M, 2024, multilingual |
| `Qwen/Qwen3-Reranker-0.6B` | **yes**, `Qwen3ForCausalLM` | no — needs a yes/no-logit path | Apache-2.0, listwise-capable |

**The next run, specified so it is not mis-specified.** `tools/engine_multihop.py` already takes
`--rerank-model`, so arm 1 needs no code:

```
.venv/bin/python tools/engine_multihop.py --rerank-model BAAI/bge-reranker-v2-m3 \
    --limit 200 --sample-seed <s>
```

Three things to get right, each of which otherwise wastes hours:

- **The reranker is query-time only**, so the index must be *reused* rather than rebuilt — keep the
  tag the recorded row used, or the run pays a full MultiHop ingest for a change that cannot affect
  the index.
- **A limited run is a smoke, not a result.** `--limit` with `--sample-seed` matches
  `tools/multihop_official.py`'s sampling, so a 200-query pair (the base reranker against `v2-m3` on
  the *same* seed) is comparable and bracketable; it must be labelled a subset wherever it appears.
- **jobkit refusing a fingerprint change is correct** — a new `--rerank-model` is a different
  measurement, so it needs its own output stem rather than `--fresh` over the recorded row.

Arm 2 (`Qwen3-Reranker-0.6B`, listwise) needs the yes/no-logit path and is real work; arm 1 is a
flag and an afternoon.

**Arm 1 has been run, and it produced a defect signal rather than a result.** MultiHop-RAG, the same
200-query sample (`--sample-seed 0`), two rerankers:

| `--rerank-model` | Hits@10 | Hits@4 | MAP@10 | MRR@10 | seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| `BAAI/bge-reranker-base` (184M, 2023) | 0.9200 | 0.7300 | **0.3189** | 0.6142 | 170.6 |
| **`BAAI/bge-reranker-v2-m3`** (568M, 2024) | 0.9200 | 0.7300 | **0.3189** | 0.6142 | 318.4 |

**Every metric is identical to four decimal places, and the runs are not the same run**: the recorded
`_settings` differ in `rerank_model` (`bge-reranker-base` against `bge-reranker-v2-m3`) and the wall
clock differs by 1.9× (170.6 s against 318.4 s), so the larger model *was* loaded and *did* compute.
Identical hits, MAP and MRR across 200 queries while a different ranker scores every candidate means
**the reranked order is not what the report is ordering by** — the cross-encoder's scores are being
discarded somewhere between `rerank()` and the passages the metrics read.

That is not a measurement of either reranker, and the honest state of P4 is that **its premise needs
checking before its arms are worth running**: if the second stage cannot change the order, then the
"pointwise cross-encoder over the fused top-100" rejection this section rests on was never tested
either. The check is bounded and code-level — whether `RagEngine.query` returns the reranker's order
(or a `blend < 1` nudge of the fused order), and what `tools/engine_multihop.py`'s summary reads.
**It is answered, and the answer is worse than "not ordering by the reranker".** All **200 of 200
per-query rankings are byte-identical** between the two runs (`per_query[i]["ranking"]` and the
`chunks` order match exactly), so the cross-encoder's scores demonstrably do not reach the returned
order at all. The mechanism is now narrow, and it is in the coverage pass rather than the reranker:

- `RagEngine.query` builds `cross_scores` from `self.cross_encoder.score_pairs([...])` and passes them
  into `_coverage_select_from` as the relevance signal (engine.py:1286–1315);
- `resolved_rerank_coverage_lambda` returns **0.7 automatically whenever a `rerank_model` is set**
  (config.py:427), so configuring a reranker *always* takes the coverage path;
- the coverage pass then sets `order = "recall"` and `blend = 1.0`, so the final ordering is the
  **coverage selection's**, and if that selection degenerates to the input order the reranker's scores
  are consumed and discarded;
- the 1.9× wall clock (170.6 s → 318.4 s) shows the 568M model really did run, so the cost is paid
  and the effect is zero.

**What this means for the section above.** The rejection this section rests on — "a *pointwise* MS
MARCO cross-encoder over the fused top-100" loses — **cannot have been testing a cross-encoder** if
the cross-encoder cannot change the order. The recorded `engine-hybrid` rows, and every reranker row
taken with `--rerank-model`, are measuring the coverage/recall order. That is a defect report on the
measurement, not a result about rerankers, and it is the largest one this programme has produced.

**That probe has been run, and it refutes the mechanism above.** `_coverage_select_from`, called
directly with a four-chunk pool and `use_redundancy=False`, returns

```
relevance all zero   ->  ['a', 'c', 'b', 'd']
relevance flipped    ->  ['b', 'd', 'a', 'c']
```

so the function **does** depend on its relevance argument, and the coverage pass is not degenerate.
The "consumed and discarded" paragraph above was wrong, and the surviving explanation for identical
rankings is the opposite one: **the two runs' cross-encoders were the same object** — a 568M model
that fails to load (or is silently replaced by a fallback) leaves the recorded `rerank_model`
different, the ranking unchanged, and only the wall clock moved by the failed attempt (170.6 s →
318.4 s).

**That probe is answered too, and it moves the question from the code to the environment.** The
cross-encoder's *identity* is right — `CrossEncoderModel` is constructed with the requested model and
records it (`{'reranker': 'BAAI/bge-reranker-base', 'kind': 'cross-encoder', 'loaded': False}`) — but
it is **lazily loaded, and on this host it cannot load at all**:

```
laya_rag.model.ModelUnavailableError: failed to load rerank model 'BAAI/bge-reranker-base'
on device 'mps': PermissionError at ~/.cache/huggingface/hub/models--BAAI--bge-reranker-base
when downloading BAAI/bge-reranker-base. Check cache directory permissions.
```

**That reading was too strong, and the same round refuted it.** A one-query run of the same tool
with `--rerank-model BAAI/bge-reranker-base` and **`HF_HOME` pointed at a writable directory**
completed without error (`indexed 609 docs / 4592 chunks in 151s`, then the query), so the model
loads here — the failure above is an **intermittent shared-cache condition**, and the error text says
so itself: *"another user is downloading the same model (please wait); a previous download was
canceled and the lock file needs manual removal"*. This is a shared host, and the rerank cache is a
shared resource.

So the defect claim above is withdrawn in favour of the weaker, supported one: **the two P4 arms
produced identical rankings, and the load failure I observed is one candidate explanation among
several rather than the established cause** — the same failure mode this section has now hit three
times, which is why the corrected pair is being re-taken with a private `HF_HOME` before anything
else is said about rerankers.

**The corrected pair is in, and the observation survives the fix.** With `HF_HOME` pointed at a
private writable cache — so both models genuinely load, which the wall clocks corroborate — the two
arms are **still identical**, and the per-query rankings are identical for **200 of 200**:

| `--rerank-model` | Hits@10 | Hits@4 | MAP@10 | MRR@10 | seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| `BAAI/bge-reranker-base` | 0.9200 | 0.7300 | **0.3189** | 0.6142 | 170.8 |
| `BAAI/bge-reranker-v2-m3` | 0.9200 | 0.7300 | **0.3189** | 0.6142 | 317.4 |

So the load failure was **not** the cause, and the defect claim stands on three independent checks
now: the two models load, `_coverage_select_from` demonstrably depends on its relevance argument, and
the returned order does not move. **The configured cross-encoder does not affect the MultiHop-RAG
result.**

**The locus is one call.** After the coverage pass, the engine sets `order = "recall"` and
`blend = 1.0` and calls `rerank(..., scores=cross_scores, order="recall")` — deliberately, per the
comment, so that the reranker "preserves that order rather than re-sorts by the score again". But the
coverage call is made with `limit = len(pairs)`, i.e. **every candidate**, so the selection is a
*permutation of the whole pool*; if `rerank(order="recall")` re-sorts by the first-stage score rather
than preserving the order it was handed, every such permutation is undone and the output is the fused
order for any reranker. **That question is answered, and the answer is a one-line defect that is now fixed.** `rerank`'s
`order="recall"` branch read

```python
above.sort(key=lambda passage: (-passage.recall_score, -passage.relevance))
```

— i.e. it **sorted by the first-stage score**, which re-derives the order the caller *started* from
and silently discards any reordering applied before the hand-off. The coverage pass permutes the whole
pool (`limit = len(pairs)`) and then asks for `recall` precisely so its order survives, so this sort
undid every coverage decision and made the cross-encoder's scores — which drove that permutation —
irrelevant to the result. The branch's own documentation ("keeps the retrieval order") and its
caller's comment ("the reranker must preserve that order rather than re-sort by the score again") both
described the intended behaviour, and the implementation did the opposite.

`order="recall"` now **preserves the order it is handed**, with three tests pinning it: a permuted
pool survives, the result is *not* the `recall_score` order when the caller reordered, and
`order="relevance"` still ranks by the decision. All 1,665 tests pass, which also says no existing
test had encoded the buggy behaviour.

**The invalidation, scoped by an audit rather than by guesswork.** Scanning the report trees for
settings that name a reranker finds **18 reports**, and they are exactly the affected set — each one's
ordering was its first stage's, so each row is that first stage's number wearing a reranker's name:

```
eval/benchmarks/multihop-rag/   15 reports: rerank-experiment{,-full}.json, rerank-depth20{,-thr}.json,
                                rerank-qcov.json, rerank-estimator-full.json, rerank-bridge.json,
                                official-rerank-subset.json, and this session's p4*/p4b*/p4c* probes
eval/engine-beir/                3 reports: scifact-chauto-ce.json (nDCG@10 0.7417),
                                scifact-ceauto.json (0.7274), scifact-ceplain.json (0.7091)
```

**A control arm shows the audit was over-inclusive, and the rule is sharper than "names a
reranker".** `scifact-ceplain` sets `rerank_coverage_lambda: 0.0`, which means the coverage branch is
skipped entirely and the cross-encoder orders through the ordinary `order="relevance"` path — never
touching the defect. Re-taken with the fix it reads **nDCG@10 0.7091, identical to its recorded
0.7091**, so the unaffected path is unperturbed and the fix is not a blanket change. The affected set
is therefore reports with a cross-encoder **and a coverage λ above zero** (the auto 0.7 counts), not
every report that merely names a model; `ceplain` and anything like it stays valid.

**The whole SciFact CE family is now re-taken**, which makes the conclusion stronger than a single
row could:

| scifact, `ms-marco-MiniLM-L-6-v2` | λ | recorded | re-taken | Δ |
| --- | ---: | ---: | ---: | ---: |
| `ceplain` | 0.0 (CE orders via `relevance`) | 0.7091 | **0.7091** | **0.0000** — the control |
| `ceauto` | auto 0.7 | 0.7274 | **0.6762** | **−0.0512** |
| `chauto-ce` | auto 0.7 | 0.7417 | **0.6422** | **−0.0995** |

**On SciFact this cross-encoder is worse than the fused order in every configuration.** Where it
genuinely ordered (`ceplain`, λ=0) it read **0.7091**; where the defect discarded its order the rows
read 0.7274 and 0.7417 — so those two were the *fused* order wearing a cross-encoder's name, and the
engine's SciFact number is the fused **0.7417**, not any CE ordering. The repository's own older note
that a pointwise cross-encoder "lowered nDCG@10 on both BEIR collections" is now measured on this one
at three configurations, with a control that does not move.

**The SciFact rows matter most**, because the repository quotes them: `docs/findings-reranker-fairness.md`,
`docs/findings-rerank-selection.md` and the README all reference this family, and
`scifact-chauto-ce`'s **0.7417 is competitive with the quoted `hybrid-biencoder` 0.7413**, which is why
it was the first re-take worth doing.

**The re-take is in, and it refutes the direction I guessed.** Same flags, same model, the fix in
place, new tag:

| scifact, `cross-encoder/ms-marco-MiniLM-L-6-v2`, `--chunk-auto` | nDCG@10 |
| --- | ---: |
| recorded (pre-fix — the fused order, CE discarded) | **0.7417** |
| **re-taken (post-fix — the CE's coverage order actually applied)** | **0.6422** |

So the cross-encoder is **not** being denied an improvement — given the chance, it makes SciFact
**0.0995 worse**. Three things follow:

- the recorded 0.7417 was the *fused* order, and it **stands as the engine's SciFact number**; the
  invalidation does not understate the engine here, which is what I speculated last round and is now
  withdrawn;
- it corroborates the repository's own older note in `rerank`'s docstring — that reordering BM25's
  candidates by a pointwise cross-encoder *lowered* nDCG@10 on BEIR — by measuring it directly and
  finding the same sign and a larger size;
- and it is the second case in this programme where fixing a defect that made a reranker inert shows
  the reranker was **harmful**, not merely unused (MultiHop is the first, where v2-m3 helps). Both
  directions exist, and neither was knowable while the order could not change. Everything else recorded in this document — the fusion rows, the
extractive rows, the MTEB scores — configured no reranker and is untouched.

**What this means for every recorded reranker row**: they were taken with a configured cross-encoder,
a coverage λ of 0.7 auto, and this sort — so their order was the fused order and the reranker's
contribution was zero. They need re-taking, and §P4's arms can now be run for real.

**And the re-take confirms the fix immediately.** The same 200-query sample, same
`--rerank-model BAAI/bge-reranker-base`, before and after the one line:

| | Hits@10 | Hits@4 | MAP@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| before the fix (fused order) | 0.9200 | 0.7300 | 0.3189 | 0.6142 |
| **after the fix** | 0.8950 | **0.7700** | **0.3384** | **0.6362** |

**MAP@10 +0.0195 and Hits@4 +0.0400**, at the cost of Hits@10 (−0.0250): a cross-encoder is doing
what a cross-encoder does — better at the top of the ranking, slightly worse at breadth. That is the
first time a reranker has moved a number in this repository, and it is a 0.02–0.04 move from a line
that sorted when it should have preserved.

**And the comparison is P4's answer.** Same 200-query sample, one protocol, the fix in place:

| `--rerank-model` | Hits@10 | Hits@4 | MAP@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| `BAAI/bge-reranker-base` (184M, 2023 — the recorded baseline) | 0.8950 | 0.7700 | 0.3384 | 0.6362 |
| **`BAAI/bge-reranker-v2-m3`** (568M, 2024) | **0.9350** | **0.8200** | **0.3721** | **0.7023** |
| *recorded "best measured" before the fix* | 0.9200 | 0.7300 | 0.3189 | 0.6142 |

**`bge-reranker-v2-m3` wins on every metric** — MAP@10 **+0.0337** over the same-sample base and
**+0.0532** over the recorded best — and it is a drop-in with zero code change, which is what this
section predicted when it listed it. Hits@4 **+0.0500**, MRR@10 **+0.0661**.

**It does not close the gap.** The published bar is **0.480** MAP@10 and this reads **0.3721**, so the
honest summary is: *a modern 568M reranker recovers ~0.05 of the 0.16 that separates this engine from
the published number, on a 200-query subset.* That also makes the remaining ~0.11 the interesting
quantity — it is not "the reranker is old", which is now measured rather than assumed.

**The full protocol confirms it, and the subset was accurate.** Same configuration, all 2,255
queries:

| | Hits@10 | Hits@4 | MAP@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| recorded best before the fix | 0.9200* | 0.7632 | **0.3190** | 0.5983 |
| subset, 200 queries | 0.9350 | 0.8200 | 0.3721 | 0.7023 |
| **full protocol, 2,255 queries** | **0.9384** | **0.8417** | **0.3702** | **0.7026** |

**MAP@10 0.3190 → 0.3702 (+0.0512), Hits@4 0.7632 → 0.8417 (+0.0785), MRR@10 0.5983 → 0.7026
(+0.1043)** — on the system's worst published comparison, from a drop-in reranker and a one-line
ordering fix. The subset's 0.3721 was within **0.002** of the full protocol, which is the sampling
discipline doing its job rather than luck.

**It still does not reach the published 0.480**, so the honest summary is now a full-protocol one: a
modern 568M pointwise reranker plus the ordering fix recovers **~0.05 of the ~0.16** that separates
this engine from the published number, while improving Hits@4 and MRR far more than MAP. The
remaining **~0.11 is the interesting quantity**, and it is not "the reranker is old".

(*The Hits@10 entry for the old row is the fixed-path number, since the pre-fix rows have no valid
Hits@10 to quote; the MAP and MRR columns are the recorded rows' own.)

**The base arm has landed, so the pair is full-against-full and P4's first question is answered.**

| MultiHop-RAG, full 2,255 queries, post-fix | Hits@10 | Hits@4 | MAP@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| recorded "best measured", pre-fix (the fused order) | — | 0.7632 | 0.3190 | 0.5983 |
| `BAAI/bge-reranker-base` (184M, 2023) | 0.9055 | 0.7761 | **0.3294** | 0.6353 |
| **`BAAI/bge-reranker-v2-m3`** (568M, 2024) | **0.9384** | **0.8417** | **0.3702** | **0.7026** |
| **v2-m3 − base** | **+0.0329** | **+0.0656** | **+0.0408** | **+0.0673** |

So on the full protocol the newer drop-in reranker is worth **+0.0408 MAP@10, +0.0656 Hits@4 and
+0.0673 MRR** over the older one — and even the *older* one beats the recorded 0.3190 by +0.0104,
because until the ordering defect was fixed neither could do anything. Both remain short of the
published **0.480**.

**The subset's verdict on itself**: the 200-query base arm read 0.3384 against the full 0.3294
(overstated by 0.009) and the subset v2-m3 read 0.3721 against 0.3702 (understated by 0.002). It got
the direction and the ordering right and the magnitudes to within ~0.01 — which is what a seeded
subset with `--sample-seed 0` is for, and it is worth recording that it was not exact.

**Arm 2 has run, and it falsifies the section's own hypothesis: a listwise ranker is much worse.** Same
200-query sample, same configuration, the ordering fix in place — only the reranker differs:

| reranker, 200-query subset | Hits@10 | Hits@4 | MAP@10 | MRR@10 | seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| none (the fused order) | 0.9200 | 0.7300 | 0.3189 | 0.6142 | — |
| `BAAI/bge-reranker-base` (pointwise, 184M) | 0.8950 | 0.7700 | 0.3384 | 0.6362 | 170.6 |
| **`BAAI/bge-reranker-v2-m3`** (pointwise, 568M) | **0.9350** | **0.8200** | **0.3721** | **0.7023** | 317.4 |
| `Qwen/Qwen3-Reranker-0.6B` (**listwise**, 0.6B) | 0.8650 | 0.5950 | **0.2047** | 0.4220 | ~1,100 |

**MAP@10 0.2047 against 0.3721 for the pointwise model — a 45% relative drop — and below the fused order
it was meant to improve.** The section said the rejection it rested on was specific to *pointwise* MS
MARCO cross-encoders and that "what remains open is a listwise or late-interaction ranker". On this
corpus, at this size, through the repository's scoring path, **the listwise ranker is not the answer
either** — the third §P4 hypothesis to be measured and rejected, after "the reranker is old" (recovering
only 0.05 of 0.16) and "the ordering is fine" (it was not).

**The instruction moves the result, so arm 2's negative is bounded by the line it was scored with.**
Scoring the same pair under three instructions on the same model:

| instruction | relevant | irrelevant | separation |
| --- | ---: | ---: | ---: |
| *(empty)* | 0.0101 | 0.0001 | 0.0100 |
| the model's own web-search default (**what arm 2 used**) | 0.0505 | 0.0000 | 0.0505 |
| "Given a question, retrieve passages that answer the question" | **0.0698** | 0.0001 | **0.0697** |

**A 7× spread in the separation across three instructions**, and arm 2 ran the middle one — the model's
web-search default — on a question-answering corpus where the third separates 38% better. So the honest
statement of the arm-2 result is narrower than "a listwise ranker is worse": **a listwise ranker scored
with a web-search instruction is much worse, and the instruction is a variable large enough that the
comparison has to be re-run before the hypothesis is finally rejected.**

`RERANK_INSTRUCTION` now exists as a knob (empty = the model's default), documented in the README because a
knob with no row is half-built, with three tests — one of which checks that the *prompt* reads the field,
which is the failure mode this session opened with. The re-run waits on the heavy lock, which `signal1m`
holds; it is a 200-query subset, about twenty minutes when the lock frees.

**And the scoring loop is batched now, which the re-run needed more than it needed a knob.** Scoring one
pair at a time meant 20,000 forwards for a 200-query subset; `score_pairs` batches with the tokenizer's
own padding. Two measurements came out of doing it, and the second is the useful one:

**And writing that loop's test found a latent bug the measurements could not have.** My first version
located each row's answer with *the count of real tokens minus one*, which is correct under **right**
padding and **wrong under left** — and Qwen tokenizers commonly left-pad for generation, where the answer
sits in the final column. Under left padding it would have read the second-to-last pad token and returned
a plausible, wrong score for every row of every batch. **The batching parity check passed anyway**, because
this tokenizer happens to right-pad: a green measurement over a latent bug, which is the shape this
document's §10.3 catalogue is about, caught here by a unit test written for a helper rather than by the
experiment it was serving. The helper now takes the last real token's index, which is correct either way.

**And the question the fix raises has been checked, because a fixed bug raises one: was the published
number affected?** No, on two counts. `Qwen3-Reranker-0.6B`'s tokenizer reports `padding_side: right` — so
under the old code the answer was at the last column and the scores were right, which is why the parity
check passed rather than failing; and **the arm-2 result was produced by the single-pair path with no
padding at all** (`logits[0, -1]` on one sequence is the last real token under either convention). So the
0.2047 stands as measured, and the fix is insurance against a different tokenizer rather than a correction
to a published figure.

**The engine-integrated path works, and running it corrected an over-generalisation of mine.** With
`RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B` on a three-document index, `RagEngine.query` returns the right
passage first and reports the right reranker — `{'kind': 'generative-reranker', 'loaded': False}` before
the call, ordered passages after, with the answerability gate satisfied. That is the path the arm-2 re-run
will use, exercised end to end for the first time since batching changed it.

**And it scored 0.4455** for the relevant passage, where my earlier probes scored the same model at
0.0505. The difference is the passage: the earlier one was `"Refunds are processed within five business
days."` and the engine's chunk was `"Refunds are processed within five business days. This is the only
sentence about refunds."` — a passage that *states* its relevance. **So the scale is not uniformly
compressed**; 0.0505 was a property of that pair, and the sentence I wrote two rounds ago saying the model
"answers in a range two orders of magnitude below a cross-encoder's" was generalised from two hand-picked
passages and should not have been. The *instruction sensitivity* measurement stands as measured — the same
three instructions on the same pair, a 7× spread — but the absolute level is pair-dependent, and the
threshold guidance in the README is about *the default* rather than about this model's ceiling.

* **parity holds to float32 batching noise**: the same eight pairs score identically to four decimals
  through batch 1 and batch 8, with a maximum difference of **6.5e-6** — so the path validated two rounds
  ago is still the path, and batching is not a different retriever;
* **the speedup is 1.2×, not the 3× I first measured.** My first timing included the model *loading*
  inside the first call and read as a 3× win; warming both instances first gives **1.58 s against 1.32 s**
  for eight short pairs. Honest conclusion: batching is correct and slightly faster, and the twenty-minute
  re-run stays about twenty minutes. The bigger win would come from longer prompts, which this corpus does
  not have.

Two caveats still travel: **one model, one corpus, 200 queries**, so the falsification holds for this
configuration rather than for listwise ranking in general.

**The path is validated with the real weights, and the validation turned up the trap the run would
have hit.** `Qwen/Qwen3-Reranker-0.6B` is now fetched into the writable `eval/hf-rerank` (the shared
`~/.cache/huggingface` is not writable here — the lesson the SciFact re-take already paid for), and
scoring three pairs gives the right *ordering* and a scale that would quietly break the engine:

| pair | `P("yes")` |
| --- | ---: |
| "Refunds are processed within five business days." (relevant) | **0.0505** |
| "Returns require a receipt within thirty days." (partly) | 0.0023 |
| "Shipping is free on orders over fifty dollars." (irrelevant) | 0.0000 |

The ordering is correct and the level is **two orders of magnitude below a cross-encoder's**. The
engine's `RELEVANCE_THRESHOLD` defaults to **0.5** (`_env_float("RELEVANCE_THRESHOLD", 0.5)`), so a run at the default would find *nothing* above it and return an empty result set for every query. **Two scopings matter, and I had the first one wrong for a commit.** `tools/engine_multihop.py` and `tools/engine_beir.py` both set `LAYA_RAG_RELEVANCE_THRESHOLD=0.0` for their own runs — they want the ranking, not the gate's kept set — so **the arm-2 benchmark row is unaffected** and needs no threshold tuning. The trap is for the engine used *directly*: the API, the library, a service, where the default stands and a generative reranker would look like it had broken retrieval.— a reranker that looks like it
destroyed retrieval. It has not: the model answers a different question in a different scale, and the
threshold has to be set for it (or the score calibrated). That belongs in the run's configuration, not
in a post-mortem, which is why it is written here before the run exists.

**The run itself waits for the heavy lock**, and its question is whether a listwise ranker beats the
pointwise ceiling now measured at **0.3702** full-protocol. The full-set re-take is the obvious
next run, and it is the one that would replace the recorded 0.319 — **it is running now**
(`--out eval/benchmarks/multihop-rag/engine-p4-full`, `--rerank-model BAAI/bge-reranker-v2-m3`, the
full 2,255 queries, under the heavy lock). That is incompatible with the P4 runs having
produced numbers at all: a `query` that reached `score_pairs` would have raised. So the two arms'
identical rankings most likely mean **the cross-encoder never scored anything in either arm** —
the failure was absorbed somewhere between `score_pairs` and the report, and the wall-clock
difference (170.6 s → 318.4 s) is the cost of the failed *attempt*, not of inference.

**Every sentence in this section that treats a reranker as measured is therefore suspended**, not
just arm 1's. Two things must happen before any reranker row is quoted: make a failed load **loud**
(no silent path from `ModelUnavailableError` to a reported number), and take the measurements where
the model can actually load — `HF_HOME` pointed at a writable directory, as this repository already
does for MTEB (`HF_HOME=$PWD/eval/mteb-hf`). Arm 2 stays on hold.
| `mixedbread-ai/mxbai-rerank-large-v2` | **yes**, `Qwen2ForCausalLM` | no — same custom path | 1.5B |

So the first experiment is literally
`--rerank-model BAAI/bge-reranker-v2-m3 --rerank-depth 100` on the existing harness.

**Plus two levers the diagnostics already justify:**
- **Deeper pool.** `findings-multihop-diagnostics.md` measures the reordering ceiling rising
  **0.609 → 0.659** as depth goes 100 → 500, and `findings-reranker-fairness.md` shows R@100 is
  0.848 with the reranker excluded from the pool. Selecting at depth 500 is therefore an untapped
  +0.05 of *ceiling*, and the cost is linear in K.
- **Union-then-fuse across the generator-free iteration.** Stage 1 raised gold coverage
  0.8326 → 0.8937 and the pool ceiling 0.5955 → 0.6203, but "no ranker here has converted that into
  MAP@10" (`docs/findings-generator-free-iteration.md`). The iteration was lexical-only with no
  controller; the union of per-hop lists was never handed to a *stronger* second stage than the
  one that failed.

**Expected.** **+0.02 to +0.05 MAP@10** (0.319 → 0.34–0.37) from the reranker swap plus depth
alone; MAP@10 is the one metric where this system is far behind a published reranked run, so it has
the highest claim value per point.

**Falsifier.** `bge-reranker-v2-m3` at depth 100 lowers MAP@10 on the 400-query deterministic
subset the way `bge-reranker-base` did on BEIR — that would say the first stage is too strong for
any pointwise stage here and re-close the question. Depth 500 with no gain would confirm the
ceiling is unreachable rather than merely unmeasured.

---

### P5 — Convex and weighted fusion, and a *transductive* per-corpus selector

**What.** (a) Replace `FUSION_K=5` RRF with a **tuned convex combination** of normalised channel
scores, per-channel weights, and an ablated `k` — the challenger the repo has already stubbed
(`FUSION_MODE=convex`) but never tuned. (b) Select the per-corpus configuration **without labels**,
from the corpus itself.

**Why it is not already falsified.** (a) The fusion analysis literature the repo cites in
`sota-landscape.md` §2 is explicit that a convex combination of normalised lexical and semantic
scores beats RRF in and out of domain, and is sample-efficient. The one local measurement
(`--convex`, SciFact, untuned weights) gave +0.0092 with p = 0.139 — not significant, and not a
tuned comparison. `FUSION_K=5` is also an unusually sharp rank prior chosen on two corpora.
(b) `tools/beir_routing.py` routes **supervised**, on a labelled half, and recovers +0.0061 on five
datasets — exactly the oracle. But it has only ever been exercised on those five, and a supervised
router needs qrels *on the target corpus*, which a benchmark's train split supplies and a
deployment does not. A transductive selector — pick the configuration from a label-free property
of the corpus itself, such as channel rank-agreement, query-length distribution, or top-k
self-consistency under each candidate — is untried, needs no labels, and is the only version of
routing that can be applied to the eleven mixed cells (and to a real corpus) rather than to the
five the router was fitted on.

**Measured for (a), and it is a win where the earlier attempt was a tie.** `tools/fusion_tuning.py`
(`make fusion-tuning`) tunes the dense weight on a seeded half of each corpus's queries and reads
it on the other half, against **RRF at `k=5`** — the fusion the engine actually ships, where the
earlier comparison used `k=60` and untuned weights. Six corpora (`docs/findings-fusion-tuning.md`):
**three significant wins, no significant losses, three ties** — ArguAna **+0.0266**
[+0.0133, +0.0403], FiQA **+0.0296** [+0.0177, +0.0413], Quora **+0.0215** [+0.0186, +0.0245] over
10,000 queries; SciFact +0.0004, SciDocs +0.0056 and NFCorpus **−0.0038**, all intervals including
zero. And the deployable version of the same runs: one **global** dense weight of 2.0, chosen on the
pooled train halves, beats the shipped RRF by **+0.0127** on the held-out halves (0.5244 vs 0.5117),
with every corpus's own optimum in 1.5–3.0 — the direction is not a per-corpus lottery. Touché-2020,
where lexical wins by two to one and the convex rule is most at risk, is **not** measured: its run
was stopped in the corpus build after ~50 minutes. **The engine-level row is now measured**:
`SECOND_STAGE=0` (a new opt-in mode, `docs/findings-second-stage.md`) returns the first stage's own
order with no decision model in the path and cut the run from ~10 minutes to **37 seconds** per
configuration. SciFact, 300 queries, depth 10:

| arm | nDCG@10 |
| --- | ---: |
| RRF `k=5` (shipped) | 0.7269 |
| convex, dense weight 1.0 | 0.7280 |
| **convex, dense weight 2.0** | **0.7331** |
| full engine with the second stage, RRF (recorded) | 0.7264 |

**+0.0062 for the convex rule over the shipped RRF at the engine level**, where the harness's own
SciFact comparison had convex *worse* (0.7362 against 0.7408) at the same weight. The sign differs
because the replica scores whole documents while the engine collapses its top-10 passages to distinct
documents — a reason to measure rather than infer, and the reason this row was the one asked for. The
second stage itself is worth **+0.0005 here** (0.7269 → 0.7264), consistent with the recorded finding
that Laya's decision is a better filter than a ranker.

**Expected.** (a) **+0.005 to +0.015** on the full-channel cells. (b) Larger in expectation on the
eleven mixed cells: Quora wants dense only (−0.025 for fusion), Touché wants lexical only (−0.091
for fusion), ArguAna wants dense, MultiHop-RAG wants the cross-encoder, BEIR wants it off. The
oracle for routing across the five-dataset set is 0.4576, and per-corpus lexical settings alone
would recover Signal-1M's 0.05.

**Cost.** Hours for (a) — no embedding. Days for (b).

**Falsifier.** The transductive selector must beat global fusion on held-out halves by a paired
interval that excludes zero; if it cannot, that is the rigorous "fixed fusion dominates learned
routing at this scale" negative that `sota-landscape.md` §6 notes is publishable because the
routing literature is under-powered.

---

### P6 — Sparse: a stronger sparse model, and document expansion

**What.** (a) SPLADE-v3 rather than the OpenSearch neural-sparse encoder, which was chosen because
it ships `safetensors` — a packaging constraint, not a quality one. `naver/splade-v3` is a
safetensors conversion away. (b) **doc2query** document expansion for the lexical channel.

**Why it is not already falsified.** (a) The current sparse channel is the first channel addition
that has ever paid (+0.004–0.023 fused), so it is under-explored rather than refuted. (b) doc2query
is listed untried in the ledger, and it is the known cheap lever for exactly the corpora where the
lexical row underperforms: FEVER, NQ, DBPedia, Climate-FEVER. The local generator needed to build
the expansion is already on this host, and a distilled doc2query-T5 (~220M) is CPU-runnable — the
cost is at ingest, not at query time, and the expanded index is still pure BM25.

**Expected.** (a) +0.005–0.015 on the datasets where sparse runs. (b) Historically **+0.02–0.05
nDCG@10** on BEIR for sparse retrievers; realistically **+0.02–0.04** on the four lexical-only
cells it can be afforded on, i.e. up to **+0.005 macro**.

**Cost.** (a) hours (checkpoint conversion + one run). (b) days per corpus at 5M+ documents.

**Falsifier.** doc2query must beat plain `bm25-engine` on the same corpus with a paired interval
excluding zero, and must not repeat PRF's failure mode (expansion drift on short queries) — the
repo's PRF result is a warning that expansion is not automatically safe here.

---

### P7 — Passage-level retrieval on the two genuinely long corpora

**What.** Robust04 (466 words/doc) and TREC-NEWS (635 words/doc) were reached from the recovered
mirror only *after* the passage work was written off. `RANKINGS.md` states the position exactly:
passage retrieval is "**untested, not disproven**" for long documents, and the two corpora it was
written for are now in hand. SPLADE-v3's authors independently report long-document passage
decomposition as a real, fixable failure mode.

**Why it is not already falsified.** The measured rejections were at ~230-word abstracts
(SciFact, NFCorpus, TREC-COVID — the last of which is *shorter* than NFCorpus, so it never tested
the hypothesis either).

**Measured, and the falsifier is met — passages lose on both, so the technique is closed at every
document length.** `tools/beir.py --datasets robust04,trec-news --retrievers bm25-engine --passages
600` (one index build each, 528k and 595k documents), nDCG@10:

| corpus | `bm25-engine` (whole documents) | `bm25-passages@600` | delta |
| --- | ---: | ---: | ---: |
| Robust04 | **0.4077** | 0.3522 | **−0.0555** |
| TREC-NEWS | **0.3351** | 0.2456 | **−0.0895** |

Every supporting metric agrees in sign and roughly in size (Robust04 MRR 0.6886 → 0.5908, MAP@100
0.1986 → 0.1741, Recall@100 0.3821 → 0.3614; TREC-NEWS 0.7599 → 0.6323, 0.2214 → 0.1594, 0.4461 →
0.3737). The expected direction was wrong, and the *size* of the loss grows with document length —
which is the opposite of the hypothesis that long documents are where decomposition helps. The
standing "untested, not disproven" caveat is now a result: **the engine's BM25 is better
max-pooled over whole documents than over 600-character pieces at 466 and 635 words per document**,
which closes the technique for this engine at every length it has been tested.

**Expected (withdrawn).** +0.01 to +0.04 on two cells → up to +0.004 macro.

**Cost.** ~35 minutes of lexical indexing, measured — both corpora are under 600k documents and
neither needs embeddings.

**Falsifier — met.** Passage retrieval loses on Robust04 **and** TREC-NEWS, so the technique is
closed for this engine at every document length, which is a cleaner statement than the caveat it
replaces. The second clause is **implemented, and half of it is measured**: `tools/beir.py` takes
`--passages p90` (each document's budget is the p90 of **its own** sentence lengths, clamped, rather
than the same 600 characters for every document) and `--passages-aggregate max|sum` (max-pool, the
historical behaviour and the one every refutation used, or CombSUM over a document's pieces). Nine
tests pin the mechanisms, including that `max` still orders by each document's best piece. **The
aggregation half is measured and negative** — CombSUM reads 0.2428 on Robust04 (−0.1094 against
max-pool) and 0.2719 on TREC-NEWS (+0.0263), and in neither case reaches the whole-document row — and
the **adaptive** half is not measurable at this scale (three attempts, no report; the cost is
stated below).

**Aggregation beyond max-pool is now measured, and it does not rescue the technique.** CombSUM over
a document's 600-character pieces against max-pool on the same corpora: **Robust04 0.3522 → 0.2428
(−0.1094)**, **TREC-NEWS 0.2456 → 0.2719 (+0.0263)** — and in neither case does it reach the
whole-document row (0.4077 / 0.3351). The bias the knob exposes is real and corpus-dependent, and
the verdict is negative. The **adaptive** budget remains unmeasured.

**The adaptive rows are pending, and the shape of the wait is itself a weakness.** The
`--passages p90` run **finished Robust04** and is still working through TREC-NEWS; when it completes
it writes `eval/beir/passages-p90.json` with both corpora. The engine row was briefly reported as
*lost* — the shell wrapper was killed and the Python process survived — which is the more useful
finding: `tools/beir.py` writes the report **once, after every dataset in the run**, so a genuine
kill loses every finished dataset with it. The passage rows have no per-dataset checkpoint, where
the engine row (`--lexical-checkpoint`) and the dense pass (`--embed-chunk`) both do. Until the
number lands the honest statement is **"fixed-size passage decomposition is refuted; adaptive
decomposition is implemented, tested and pending"**. The gap that nearly cost the work is closed:
`tools/beir.py` writes its report after **every** dataset now, printing which one landed, so a kill
costs at most the dataset in flight — two tests pin it, one by failing a later dataset on purpose.

---

### P8 — Fix the FEVER anomaly, and re-check BioASQ

**What.** FEVER is the largest lexical-control gap measured anywhere here: **ours 0.6323 against a
published BM25 of 0.753 (−0.121)**. BioASQ is the largest unexplained *positive*: ours 0.5193
against a published BM25 of 0.465 (+0.052), five times the control tolerance. Both are integrity
questions before they are score questions.

**Why it matters more than its size.** A −0.121 deviation on one cell is 0.007 of the macro, but it
is also the number a reviewer would use to dismiss the whole table — and the repo's own
`findings-beir-breadth.md` explicitly leaves the cause "**unidentified**". The mechanisms already
ruled out are worth restating so they are not re-tried: corpus size (DBPedia kills it), the
engine's stemming/stopword settings (the flat control is *further* from the anchor on the small
sets). The untested candidates are **field weighting** (P2 — the published column is multi-field,
the control is flat), **corpus release/duplicate handling** on FEVER (near-duplicate claims), and
**query-set version**. The experiment that separates them is a published-settings control row on
FEVER itself, which the ledger records as **abandoned at 502/3,452 queries**.

**Expected.** +0.00 to +0.12 on FEVER → **+0.000 to +0.007 macro**, and removal of the largest
credibility hole in the benchmark table.

**Cost.** Hours to a day of compute (5.4M-document lexical build).

**Falsifier.** The published-settings control also reads ~0.63, in which case the discrepancy is in
the corpus release or the evaluator convention, and the honest fix is a footnote — but a *measured*
footnote, not an unexplained one.

---

### P9 — Make the answerability gate portable, or say plainly that it is an in-domain instrument

**What.** The single highest-value *novel* contribution available, per
`best-in-the-world/sota-landscape.md` §9: nobody benchmarks the **calibration** of a retriever's
relevance or answerability probabilities, and the one leaderboard that reports calibration error
shows a spread of 9% to 69%. This system emits calibrated per-passage and per-set probabilities;
it is unusually well placed.

**Why it is not already falsified.** The measured position is subtle and correctly stated in the
repo: the shipped single scalar is at chance out of domain (AUC 0.509), a threshold fitted on a
MultiHop half transfers at chance (0.511), **but** a twelve-feature gate over retrieval signals the
engine already has scores **held-out ROC-AUC 0.815** in-domain and 0.767 with a whole question type
held out. So the signal exists; the *portability* is what has no measurement. The calibration row
exists on **one corpus only**.

**Expected.** Not a leaderboard point. A reliability diagram plus ECE/Brier/risk-coverage on
BEIR-10 + MultiHop + MuSiQue + FRAMES, plus a phantom-hit flag (high reranker score, low dense and
low lexical) — a documented failure mode in *Drowning in Documents* for which no fix is published.
This is the "best in the world" claim rather than the "best score" claim, and it is the one place
the system's architecture is an advantage instead of a deficit.

**Cost.** Days, no new retrieval code — the features and the first artifact already exist.

**Falsifier.** The twelve-feature gate's AUC collapses on a fresh out-of-domain corpus the way the
scalar's did. Then the honest claim narrows to "a calibrated *and recalibratable* gate, given a
labelled sample of the target domain", which is still more than anyone else has measured.

---

### P10 — Submit to a public leaderboard: TREC RAG's retrieval task first, then MTEB

**What.** Two external placements are reachable, and the first one is better than the project's own
standing verdict says.

**(a) TREC RAG 2025 introduced a *Retrieval-only* task (R).** Participants submit a ranked list of
the top 100 segment IDs from the MS MARCO v2.1 *segment* collection for each narrative topic; it is
scored with `trec_eval` on `ndcg_cut_30`, `ndcg_cut_100` and `recall_100`, with **no generation
step at all**. Forty-three runs were published in the TREC 34 proceedings. The span is wide and the
baselines are named:

| run | org | nDCG@30 | nDCG@100 | recall@100 |
| --- | --- | ---: | ---: | ---: |
| `4method_merge` | UTokyo | **0.6934** | 0.6134 | 0.2571 |
| `lucerank` | digsci | 0.6805 | 0.5927 | 0.2435 |
| `splade-v3-arctic-l` | coordinators | 0.6009 | 0.5683 | 0.2435 |
| `rag25_test_splade-v3` | coordinators | 0.5957 | 0.5387 | 0.2321 |
| `RRF_colbert_minlm` | cfdalab | 0.5262 | 0.2907 | 0.0951 |
| `Anserini_bm25_only` / `anserini_bm25_top100` | cfdalab / coordinators | 0.3468 | 0.2900 | 0.1106 |
| `bm25_NITA_JH` | NITATREC | 0.3966 | 0.3001 | 0.1120 |
| `sentence-transformers-all-MiniLM-L6-v2` | clip2025 | 0.4029 | 0.3163 | 0.1225 |

**Why this matters more than any other external target on the list.** It is a *retrieval and
chunking* benchmark with a public leaderboard and a named baseline grid — exactly this system's
shape — and its relevance scale is graded **0–4 by how many parts of the narrative the segment
answers**. That makes it, in effect, a published **coverage** benchmark. The project's most
contested internal result is precisely whether a coverage objective helps: it won on MultiHop-RAG
(+0.0244 MAP@10), lost on MuSiQue (−0.0109) and again on 2Wiki (−0.0046), and the honest position
today is "corpus-shape instrument". TREC R's graded scale is where that question can be settled
against 43 real runs instead of against this project's own harness. Its topics are also 2–3
sentence narratives — the long-query regime where the repo's PRF failed and where query-side BM25
and document expansion are supposed to pay.

**The blocker, stated honestly.** The R task needs the MS MARCO v2.1 *segment* collection, which is
an order of magnitude past the largest corpus measured here (BioASQ, 14.9M documents; MS MARCO v1
at 8.8M was measured, and its dev queries number only 43). This is a **scale** blocker — registration
plus a bounded external merge for the term dictionary and an on-disk ANN, both already named as the
remaining scale work — not the *gold-nugget/AutoNuggetizer* blocker this repo's `BENCHMARKS.md`
records. That verdict (`BENCHMARKS.md` lines 43 and 200) describes the RAG and AG tasks and does not
mention the 2025 R task; it is **stale for the retrieval half** and should be corrected.

**And the RAG and AG tasks are no longer generator-attributed either.** The AG task supplies the
top-100 candidate segments and asks for a ≤400-word cited answer; the RAG task asks you to build
your own retrieval over the same collection. Both can be run through `compose_extractive`, which
has no model in the loop, so their attribution and nugget scores would be the engine's rather than
an attached LLM's — see P12. The R task remains the cleanest entry point because it needs no
answer at all, but it is no longer the only one.

**(b) MTEB v2 `SearchProtocol` — the "stall" was throughput and silence, and the silence is fixed.**
The code blocker is gone: `tools/mteb_search.py` implements the protocol with structural conformance
verified against mteb 2.22.5 (18 tests). The recorded failure — "forked ~33 workers and stalled, no
progress output for 30+ minutes" — **does not reproduce**. Re-run with `num_proc=1` under
`faulthandler`, the process shows **20 consecutive stack dumps over 30 minutes, every one inside the
engine's own scoring** (`mteb_search.py:search` → `engine.query` → `rerank` → `score_relevance` →
Laya's ModernBERT forward), and `mteb`'s `_create_dataloaders` passes `num_workers=0` whenever
`num_proc <= 1`, so no pool is involved.

The cost is structural rather than a defect: MTEB asks for a `top_k=100` ranking, the adapter widens
`retrieve_k` to match, and the engine then runs its **322M-parameter decision model on 100 candidates
per query** — a multi-hour task for SciFact's 300 queries, which is the same shape
`findings-engine-parity.md` measured on BEIR. What *was* broken is observability: a working run was
indistinguishable from a hung one. `tools/mteb_search.py` now prints a rate and a remaining-time
estimate every 25 queries, and `--limit-queries` is an explicit **smoke** cap that says in the log
that a subset is not a score. The instrumented run was stopped after **60 stack dumps over ~90
minutes with no result** — the measured wall clock the "accept it as a long job" option is weighed
against, and why the silence mattered: an operator watching for an hour and a half could not tell it
from a hang.

**The silence had a cause and it was ours.** The five-query stack probe shows
`engine.query → rerank → score_relevance → predict_many` at 30-second intervals: `main` printed the
`--first-stage-only` banner and **set nothing on the adapter**, so every "first-stage-only" MTEB run
still ran the 322M decision on every candidate. It is fixed (`main` sets the attribute, with a
regression test naming the bug), and it also explains the 282,984-token warning — that is the
**rerank prompt**, query plus candidate passages. The engine-level rows were never affected:
`engine_beir.py` sets `LAYA_RAG_SECOND_STAGE=0` through the environment, which works.

**The first score under an external protocol exists — and with the retrieval-trained encoder it is
above published BM25.** MTEB's SciFact, `--first-stage-only`, two encoders:

| encoder (as filed) | `main_score` | nDCG@10 | MAP@10 | MRR@10 | search rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `laya-rag/engine-multilingual` (decision checkpoint) | 0.37 | 0.36987 | 0.31667 | — | 1.05 s/query |
| **`laya-rag/engine-retrieval`** (retrieval-trained) | **0.69** | **0.69482** | 0.64817 | 0.65887 | **0.21 s/query** |

**And the level is now a like-for-like reading, because the baseline was run through MTEB**:
`BAAI/bge-small-en-v1.5` scores **0.71271** nDCG@10 (`main_score` 0.71) on the same task under the
same harness, against this engine's **0.69482** — so the engine sits **0.018 below a stock
`bge-small` bi-encoder**, measured rather than inferred, and **~5× slower** (263.9 s against 54.8 s).
**A second task agrees**: NFCorpus `bge-small` 0.34339 against the engine's 0.33318 (−0.010), again
~4× slower — with the engine's **MAP@10 ahead** (0.12463 against 0.12280), which is what a system
that orders the top slightly worse but covers more relevant documents looks like.
Reading 0.69482 against the BEIR-convention numbers (BM25 0.665, `bge-small` 0.72) would have been a
cross-protocol claim; this is not. `--embed-model` now *requires* `--model-name`, so a
run cannot be filed under a name that does not describe its encoder. **Nothing is uploaded.**

**The earlier, weaker version of this:** `mteb.evaluate` on SciFact,
`--first-stage-only`, completed locally: MTEB's `TaskResult` reports **`main_score=0.37`**, with
**nDCG@10 0.36987** (nDCG@100 0.39378, MAP@10 0.31667) and MTEB's own `evaluation_time` **193.6 s**.
This is the first number in this repository that nobody here computed, and read against SciFact's
published BM25 (0.665) or a `bge-small` bi-encoder (0.72) **it is a starting point, not a
placement** — the configuration removes the decision model, and the registered encoder is the
multilingual *decision* model's, which was never retrieval-trained. **Nothing is uploaded.** The
honest next step is the same run with `models/laya-retrieval`, which is one flag, not a defence of
0.37.

**The corrected cost model: an MTEB `SciFact` task is ~8.5 minutes, not hours.** With the flag
applied, the same five-query probe reads index **183 s** and search **1.05 s/query** — against ≥31
s/query before, with no `rerank.py` frame and no 282,984-token warning in either dump. The rate
agrees with the engine-level 1.3 s/query, which is the cross-check that the two configurations are
now the same one. So the harness was never the problem: it forks nothing under `num_proc=1`, does
nothing between `index` and `search`, and was faithfully driving an adapter whose flag did nothing.

**Two hypotheses were tested before that and both failed, which was part of the cost.**
(a) *The query embedding*: with the decision model removed it is the only per-query model work left,
and the engine level refutes it — **20 queries in 26 s with the same 322M encoder**, 1.3 s/query.
(b) *MTEB drives `search` once per query* (its evaluator carries a `qid`): reading
`abstasks/retrieval.py` refutes it — one evaluator per subset, whole query collection, **one**
`search` call. So `search` is called once over ~300 queries, fewer than 25 of which scored in 13+
minutes, while the engine answers at 1.3 s each; the remaining differences (`retrieve_k = 100`, the
adapter's own loop) do not look like a 20× factor. **The next step needs no guess**:
`--limit-queries 5 --stack-dump-every 30` against the 178-second index names the frame in 30-second
intervals.

**Where it stands after the second-stage mode landed, with the phase named.** `--first-stage-only`
takes the decision model out of the *search* phase — the engine-level measurement of that is 37 s for
300 queries against ~10 minutes, in `docs/findings-second-stage.md`. Re-run end to end on SciFact
with the index instrument, the phases separate cleanly: **index 5,981 chunks in 178 s**, and then
**fewer than 25 of ~300 queries in 13+ minutes** — ≥31 s/query, a multi-hour task. So indexing is not
the wall clock; the search phase is, and the stack dumps put it in model inference. **The obvious explanation is
measured false**: with the decision model removed the only per-query model work left is the query
embedding, and the engine level answers 20 queries in **26 s with the same 322M adapted encoder** —
1.3 s/query, not 31. So the next step is what `mteb.evaluate` does *between* the index call and the
first `search` call, not the engine. The adapter now reports
the **index** phase too (stage, documents, chunks, embedded, and a done line) and takes
`--stack-dump-every N` for `faulthandler`, which is the instrument that corrected this section in
the first place. One observation is recorded and not resolved: a **282,984-token** sequence, once, at
the start of indexing — not a document (the longest `title + text` is 10,126 characters), not a
query (204), and present in first-stage-only mode so not the reranker. **Two local MTEB scores now exist** — 0.37 and 0.69, below — and **nothing is uploaded**.

**Cost.** (a) is a project, not a submission: corpus acquisition plus the scale work. (b) needs one
of three things, in order of principle: an **engine first-stage-only mode** (the engine has none —
`order="recall"` still computes the relevance probabilities, so it changes the ordering and not the
work), accepting the **wall clock** (~2–3 h per task at the measured rate) as a long job, or
lowering `retrieve_k`, which changes what the benchmark measures and is therefore last. Then
`ResultCache.submit_results`. Both should be scoped as *placements* with a named owner rather than as
benchmark runs.

**Falsifier.** For (a), the run itself is the result — even a mid-table row with a measured
nDCG@30 is worth more than every internal claim in this repository, because it is the first number
here that nobody in this project computed. For (b), the earlier form of this bar — "if the harness
cannot be made to run without forking" — is **withdrawn**: the harness runs. The bar is now whether
one task completes end to end within the wall clock an operator will actually wait for, and if it
cannot, the honest fallback is the public `Evaluation Request` issue template **naming the measured
per-query cost**, not silence.

---

### P11 — Realise the pool bet, or close it

**What.** The generator-free iteration is the programme's best-validated mechanism and its worst
conversion. Stage 1: gold-fact coverage **0.8326 → 0.8937 (+0.0611)**, pool ceiling **0.5955 →
0.6203**, both surviving a shuffled-probe control. Stage 2 — converting that into MAP@10 — has
been attempted once (a logistic/pointwise scorer at 0.2939 MAP@10, **−0.0208 against the hand
rule**) and never with cross-corpus training, pool-mined hard negatives, or a listwise objective.

**Why it is not already falsified.** Those are the three named untested axes. Equally important:
Stage 1's probe was **lexical-only with no controller**, one round (it saturates), and the union
was never handed to a stronger ranker.

**Expected.** The measured ordering headroom is **+0.1254** at the ceiling; a realistic capture is
**+0.02 to +0.05 MAP@10**. This is the only route that attacks the *measured* cause of the MAP gap
— 35.4% of depth-100 pools are missing at least one gold fact, and hop-2 retention is 0.719 against
0.950 for hop-1 — rather than the selection symptom.

**Cost.** Weeks. CPU-trainable; features and labels already exist.

**Falsifier.** If the listwise/cross-corpus selector cannot beat `covquota-3-0.7` (0.3280 subset /
0.3192 full), the pool bet is closed honestly, and that is a contribution in itself: it is the
third independent instrument to fail on the same headroom.

---

### P12 — Score the extractive composer end to end: the row that is entirely the engine's

*Numbered last only because it was found last. It is hours of work and belongs in week 1.*

**What.** `compose_extractive` produces an answer in which every emitted sentence is a verbatim
substring of a retrieved passage, carrying its source path and chunk id. There is no model in the
loop, so an end-to-end score over its output is **attributable to the engine with no generator
confound at all** — the strongest form of the claim this project could make, and it does not exist
yet. `docs/findings-extractive-composer.md` states the gap explicitly.

**Why this is bigger than it looks — it removes the cost blocker.** Every end-to-end target in
`BENCHMARKS.md` is marked blocked or "pilot only" on the strength of one number: **~36 s/query**,
"nearly all of it generation", putting the full MultiHop-RAG set at **22–25 hours**. That is the
*LLM* path's cost. Retrieval in the same pilot is **~3 s/query**, so the extractive path puts the
full 2,556-record end-to-end row at roughly **2 hours** on this host. The reason the project has no
end-to-end ranking is a cost that only applies to the path it chose to measure.

**Three concrete runs, cheapest first:**

1. **MultiHop-RAG end to end, extractive** — the same 2,556 records the retrieval work already
   covers, with `tools/nugget_recall.py` (the LLM-free nugget/claim-recall protocol, already
   written: claim-recall@10 0.7438 fused vs 0.7803 CE) plus the deterministic citation checks. This
   yields the first end-to-end row whose every component is the engine's. Expect **near-perfect
   attribution by construction** and weaker fluency/completeness — which is the finding, not a
   defect.
2. **FRAMES retrieval + extractive answers** — the adapter exists (`tools/frames_eval.py`,
   gold coverage 0.9977 after redirect resolution). **The corpus is not 0.5 MB**: that is the
   *questions* file (`eval/frames/test.tsv`, 476 KB); the corpus is `eval/frames/corpus.jsonl` at
   **74 MB, 2,509 full Wikipedia articles**, and every question carries a gold `Answer`, so the
   completeness half is scorable without a judge. Longer articles mean more chunks per document than
   MultiHop-RAG's news corpus, so the index build dominates the run rather than the queries.
3. **A held-out own-domain set** (§7.2) scored the same way, where extractive prose is a plausible
   product rather than a compromise.

**The surfaces are no longer the blocker.** `query_extractive` was reachable from the engine and
the CLI (`laya-rag extract`) and **nowhere else** — `grep -c extract` was **0** in both `api.py` and
`mcp_server.py`. It is now **`POST /v1/extract`** (the one route whose output is entirely the
engine's, with the passages returned alongside the answer so `/v1/verify` runs over the result) and
the MCP **`extract_answer`** tool. The engine method also takes `include_evidence`, and the API
response is the `/v1/query` shape plus `answer`/`abstained`/`sentences`/`sentences_considered`.

**Building that surface found a bug in the citation format, and it was load-bearing.** The composer
emitted the marker *trailing* its sentence (`"Refunds take five days. [1]"`). `verify_citations`
splits an answer into claims **after** the terminal punctuation, so that `[1]` was orphaned onto the
*next* claim: the sentence it supported read as uncited and the final marker dangled as a citation
to nothing — **every extractive answer failed the project's own deterministic check**, which is the
exact failure the composer was built to remove. The unit tests missed it because they only asserted
that each sentence's number appeared somewhere in the answer, which the trailing form satisfies. The
marker now leads (`"[1] Refunds take five days."`), matching how `build_messages` numbers the
evidence, and the `/v1/extract` → `/v1/verify` round trip is a test.

**Run 1 of the three, measured on the full query set — the first end-to-end row whose every
component is the engine's.** `tools/engine_extractive.py` (`make engine-extractive`) over **all
2,255 answerable MultiHop-RAG queries** (the `engine-hybrid` index, thresholds zeroed so the
retrieval half is comparable), in `docs/findings-extractive-row.md`:

| metric | full set | 400-query subset |
| --- | ---: | ---: |
| Hits@10 / Hits@4 / MAP@10 / MRR@10 | 0.9082 / 0.7632 / 0.3009 / 0.5983 | 0.9075 / 0.7525 / 0.3057 / 0.5901 |
| **citation clean** | **1.0000** (0 uncited of 9,806) | **1.0000** (0 of 1,728) |
| abstention | 0.0000 | 0.0000 |
| answer nugget recall | 0.1906 | 0.1921 |

**The retrieval half is identical to four decimals on all four metrics** to the retrieval-only row
(`engine-hybrid.json`), and to `engine-hybrid-400.json` on the subset — a cross-check that composing
an answer from a passage does not disturb the retrieval it composed from. **Attribution is 1.0 by construction and now by measurement**, which is
the claim the composer's own finding had only asserted. **Completeness is the finding**: the answer
carries **0.1906** of each gold fact's nuggets against the **0.7438** the retrieved pool carries — a
five-sentence extract is not a written answer, and that gap is what attribution-by-construction
costs, with `--max-sentences` the obvious lever and unmeasured. Getting to 1.0 took three defects, one of them a wrong fix recorded in the finding: the
trailing citation marker, a quotation cut by a sentence boundary (16 of 400 answers, none of them a
bad citation), and — the wrong fix — suppressing the boundary *inside* a quotation, which made the
checker split there and left 24 of 1,637 sentences uncited. Both checks need one boundary rule.

**Run 2 is measured too — FRAMES, 814 questions over 2,482 Wikipedia articles, gates zeroed.**
Retrieval saturates (recall@10 **0.9853**, MRR@10 0.9032, nDCG@10 0.7420), so the run's interest is
the answer half: **answer nugget recall 0.1922**, against MultiHop-RAG's 0.1906. Two corpora in
opposite regimes — multi-hop news sentences against full Wikipedia articles whose gold answers are
one or two words — and the **same number**, which says the limit is the five-sentence budget rather
than the corpus, and makes `--max-sentences` the lever. Attribution is 0.9779 rather than 1.0 and the
18 failures decompose exactly: **227 of 814 answers quote raw Wikipedia table markup and 18 of those
fail; the other 587 are clean**, with 0 uncited sentences of 3,751 and no citation out of range. The
deterministic quote checker is being handed table syntax; the fix is to strip wiki markup in corpus
preparation, not to change the composer. A first attempt at this row ran with the shipped gates and
read an abstention of 0.163 that was the *gate's* refusal, not the composer's — `--zero-thresholds`
is now a flag with a test.

**Still open here.** Run 3 (an own-domain set) is unrun; no interval is quoted (per-query values are
persisted for both runs). And `findings-extractive-composer.md` names two ends that bear on the score:
sentence scoring is the *document's* relevance rather than a per-sentence scorer, and
**contradiction gating is absent** — `adjudicate_claims` can mark a claim `contradicted` and the
composer will still emit the span. Gating emission on adjudication is the difference between
"verbatim" and "supported".

**Expected.** Not a delta on an existing number: a **new number that did not exist**, on the axis
(citation support, abstention, nugget recall without a judge) where the field has no leaderboard
and this system is structurally strong. It is the cheapest route to "we measure something nobody
else measures".

**Falsifier.** If extractive answers score so poorly on completeness that no reader would accept
them, say so with the number — that is a real, publishable statement about the limits of
attribution-by-construction, and it is more useful than the current untested caveat.

---

## 3. Sequencing

**Week 1 — hygiene, then the cheap wins.** Fix the four `--limit-queries` dilution sites named in
`measurement-audit.md` §6 and the `jobkit` fingerprint gap before any new headline number is
produced; a stale or diluted number silently invalidates everything below it. Then run P2 (lexical
selector + BM25F), P5(a) (convex fusion) and **P12** — all three are hours, all three are pure
harness and surface work, and P12 is what makes an end-to-end row that is entirely the engine's.
The MTEB half of P10 also belongs here, because a public row changes what every other number in
this repository is allowed to claim.

**Weeks 2–3 — the empty cells.** P1. Largest-last, behind `tools/heavylock.py`, with the small
corpora (Signal-1M, CQADupStack, TREC-COVID, Robust04, TREC-NEWS) landing first so the macro moves
early. This is the item that produces the headline.

**Weeks 3–4 — the second stage.** P4, in parallel with the P1 embedding jobs (it is a different
machine resource: P1 is embedding-bound, P4 is scoring-bound). Also P3's code changes, so the
encoder probe can start the moment P1's harness changes are proven.

**Weeks 5–8 — the two research bets.** P3's encoder measurement, and P11. P6(b), P7 and P8 are
fill-in work for idle compute. P9 is a write-up and runs whenever a human is available.

**Standing, not scheduled — P10.** The TREC RAG retrieval-only run is a *scale* project and should
be scoped as such (corpus acquisition, bounded external merge, on-disk ANN), not slotted as a
benchmark run.

## 4. Prerequisites — measurement hygiene

These are not score items, they are the conditions under which a score means anything. Each is
already named in `measurement-audit.md`; **items 1–3 are now closed — and closing each of them
found a defect that the convention was hiding — while items 4–5 are standing practice.**

1. **`--limit-queries` dilution — fixed, and the guard that proves it was itself broken.** Found
   in four tools (`arguana_stance`, `order_eval`, `beir.py`'s HotpotQA path, `ondisk.py`, then
   `engine_beir.py`); all five now pair a capped ranking set with `beir.restrict_qrels`, pinned by
   `tests/test_qrels_dilution.py`. A diluted sample is not a subset; any row produced through it
   must be marked, not quoted. **The audit that checks this had stopped working**: it raised an
   uncaught `UnicodeDecodeError` on `eval/bioasq-src/allMeSH_2020.json` (a 24 GB latin-1 MeSH dump —
   `UnicodeDecodeError` is not an `OSError`), so `make leakage-audit --check` died on the first
   undecodable file rather than reporting. Fixed, and its coverage is now stated rather than
   implied: **590 reports scanned, 142 exclusions named** (oversized corpora above a 32 MiB cap,
   empty HuggingFace `.no_exist/` sentinels, list-shaped id caches), zero findings.
2. **`jobkit` fingerprints did not cover the index they were measured against — fixed.** An
   engine runner reuses whatever index is on disk, so changing a chunking knob against an existing
   index directory produced a *new* settings fingerprint over an *old* index: a wrong number with a
   fresh timestamp. Each engine index now carries an `index-identity.json` recording the corpus
   file and the ingest knobs that built it; that identity is folded into the run fingerprint, and
   a run whose identity differs **rebuilds the index and refuses the stale per-query results**.
   An index built before the sidecar existed is adopted once and stamped, and its checkpoint
   manifest is rebound rather than discarded, so no measured work was thrown away. `--fresh` is
   still the tool for a deliberate reset, but it is no longer the *only* guard.
3. **The harness and the shipped engine disagree on BM25 parameters — a convention, now
   checked.** `tools/beir.py` runs `k1=0.9, b=0.4` (Anserini) for its `bm25` control, the shipped
   engine `1.2/0.75`; at the engine level the two are indistinguishable on SciFact (p=0.45), so
   `BM25_K1`/`BM25_B` exist to *reproduce a baseline exactly*, not because one is better. A
   `bm25-engine` row is the engine's configuration reproduced in the harness and a `bm25` row is
   the published baseline reproduced there; **neither is the engine itself**, and a fused row must
   not be quoted against the control.

   Making that convention checkable showed how much the *other* knobs matter to the number.
   Trying to pin the replica's overstatement of the engine on SciFact (engine **0.7264**, replica
   **0.7408**, **+0.0144**) I first concluded the replica value reproduced from no report — two
   reports read **0.7413** for the same retriever and `rrf_k`. It does reproduce: the two differ by
   **`--title-boost`**, which those reports carry and the parity run did not (`title_boost=1`
   → 0.7408, `title_boost=2` → 0.7413, and likewise 0.7320 / 0.7310 at `rrf_k=60`). The
   engine-matched comparison is the un-boosted one, because the engine indexes each document's
   title inline once. Both endpoints and the delta are pinned in `make consistency-check` from
   `eval/engine-beir/scifact-base.json` and **`eval/beir/engine-parity.json`** — a report produced
   at the engine-matched configuration so the document quotes a row that can be regenerated.
4. **Minimum detectable effect.** MAP@10 MDE is 0.0076–0.0112 on 2,255 queries; most BEIR ablations
   are below theirs. Publish the MDE next to the delta, and pre-register the family before looking.
5. **Significance, always.** Every row in this programme should carry a paired-bootstrap interval
   over queries, and the `bm25` control should be checked against its pinned anchor — the ArguAna
   incident (`ignore_identical_ids`) is the standing proof that a drifting control is the
   harness's fault and that nobody notices.

---

## 5. Do not retry

The consolidated ledger is `../best-in-the-world/proposal-ledger.md` §1 and §3, and `RANKINGS.md`
§"Experiments tried, measured, and rejected". Repeated here only because this document's
recommendations are adjacent to them and a reader will wonder:

- BM25 PRF (SciFact 0.682 → 0.424); fixed 600-char passages at abstract scale; CombSUM.
- Larger encoders *within the BGE family*; `bge-large` over `bge-base`.
- **Pointwise** cross-encoders as a full re-sort on BEIR (−0.027 … −0.318).
- Laya's decision head as a ranker; naive full-split fine-tuning with mined hard negatives.
- Late-interaction MaxSim over **Laya's own token states** and with `answerai-colbert-small-v1`
  (MAP@10 −0.0430); NLI stance reranking on ArguAna; bridge-linked lexical rescue.
- Coverage/dedup as **BEIR** defaults; the single answerability scalar recalibrated out of domain;
  engineered gate features; atom-level lexical as a standalone BEIR channel.
- `jina-reranker-v3` (licence, not result).

Note what is **not** on that list, and is in this one: a **listwise** ranker, a **convex** fusion,
**BM25F**, **document expansion**, **passages on long documents**, a **2025-era encoder**, and
**running the existing stack on eight corpora that have never had one**.

---

## 6. What this document does not claim

- It does not claim the +0.035 figure. It is "closing nine tenths of the 0.7085 summed deficit"
  across the ten cells that sit behind the published matrix, which assumes the local pipeline
  reproduces published behaviour on corpora where it has never been run. The +0.020 half-gap figure
  is the one to hold the programme to, and the per-cell rows are the result — not the macro.
- It does not claim any external model's numbers will reproduce here. The loadability probe above
  is a *config-and-tokenizer* probe on this host's pinned stack; it says a model can be
  instantiated, not that it scores what its card says.
- It does not claim the eleven cells that need channel work (the ten behind the published matrix,
  plus BioASQ) are the only explanation for the losing cells. FEVER is a −0.121 lexical gap *before*
  any dense channel is added, and the largest single-cell recovery on this list is a
  control-integrity question (P8), not a modelling one.
- It does not re-open the "open bets" list in `../best-in-the-world/README.md`. Three of those
  bets have since been measured, two of them negatively; the work queue there is stale, and this
  document is derived from the measurements rather than from that framing.

---

## 7. Where to be ranked — and why the standard landscape advice partly inverts here

A common framing of the RAG benchmark landscape divides it into end-to-end RAG, classic QA, and
retrieval-only, with the advice: *hold the generator fixed or report retrieval and generation
separately; TREC RAG, CRAG and BEIR/MTEB are your best bets; run baselines yourself with
FlashRAG/BERGEN; and add a benchmark on your own domain.*

That framing is accurate, and it is essentially the same landscape this repository already
carries — `BENCHMARKS.md` §"The rest of the landscape" covers every one of those names and, unlike
a list, assigns each a **verdict** ("measured", "runnable", "blocked", "not applicable", "declined")
with the reason. What follows is where the generic advice **inverts** when the system being placed
is generator-free and retrieval-and-evidence-shaped, and the two items it adds that this document
had under-weighted.

### 7.1 Two of the six tips need restating, and one inverts

**"Hold the generator fixed."** This needs restating rather than inverting — the earlier draft of
this document said "there is no generator to hold", and that was **wrong**. The system ships *two*
answer-production paths:

| path | surface | model? | attribution |
| --- | --- | --- | --- |
| `Composer` (`compose.py`) | `POST /v1/compose`, MCP `compose` — the API docstring calls it "the only route that returns generated text" | external LLM (Ollama, default `ornith-1.5:35b`); no weights ship | probabilistic |
| `compose_extractive` (`extractive.py`) | `POST /v1/extract`, MCP `extract_answer`, `RagEngine.query_extractive` and `laya-rag extract` | none | **structural**: every emitted sentence is a verbatim substring of a retrieved passage, with a path + chunk id |

So the honest statement is not "generator-free" but **generator-optional**, and the two paths have
different attribution properties:

- Through the **LLM** path the absolute number is `engine + <model>`. But the project's own rule —
  "any end-to-end answer accuracy is `engine + <model>`, never the engine's" — is **too strong in one
  direction**. Under a *fixed generator and prompt*, the **delta** between two retrieval systems is
  the retrieval's; that is exactly the controlled comparison the advice recommends, and this system
  is fully eligible for it. What it cannot do is claim an absolute end-to-end score as its own.
- Through the **extractive** path there is no generator at all, so the score *is* the engine's — an
  end-to-end row with the confound **removed rather than managed**. And it has never been measured:
  `docs/findings-extractive-composer.md` says so outright — "No answer-quality measurement… the
  end-to-end row (accuracy, sentence-level attribution precision, the abstention curve) that Bet 5
  asks for is not run here. What is guaranteed is attribution, not usefulness."

**"TREC RAG, CRAG and BEIR/MTEB are your best bets."** Two of the three are right, for different
reasons than the advice gives. **MTEB** is right and is now genuinely reachable (§P10b).
**TREC RAG** is right — and *more* right than the advice knows, because the extractive composer
makes its RAG and AG tasks enterable with a **fully owned** row: no LLM, so nugget recall and
attribution are the engine's numbers. **CRAG is actively wrong for this system**: it hands the
participant five web pages per question plus mock APIs and a mock KG, so the engine's retrieval —
the only part that is this system's — is bypassed entirely. `BENCHMARKS.md` already reaches this
verdict; the generic advice does not.

**"KILT has a leaderboard."** It does, and it is unreachable here for a reason the advice elides:
KILT expects a full stack (entity linking, retrieval, seq2seq generation) and a leaderboard row
would score the attached LLM. Likewise **RGB** looks ideal on paper — noise robustness, negative
rejection, counterfactual — but this project investigated the English files and found **no clean
no-answer label**, with the official rejection metric judge-based; **HHEM** ranks hallucination
*detectors* and **RAGTruth** annotates spans in an LLM's output, so both are structurally the wrong
module class. Each of those is a "no", and the reason matters more than the label.

### 7.2 Two items the advice adds that this document under-weighted

**Hosting the engine as a retriever in FlashRAG or BERGEN.** `BENCHMARKS.md` calls this
"integration, not a ranking" and moves on. That is correct about what it *is* and too dismissive
about what it *buys*: a like-for-like number against pipelines those toolkits have already
reproduced, under identical conditions, with the harness mismatch (`k1`/`b`, chunking, top-k)
neutralised by construction. Given that this project's own audit finds its replica **overstates**
the engine by 0.0144 on SciFact and **understates** it on MultiHop-RAG, an external harness is not
a nicety — it is the only control on the harness itself. This deserves to be a scoped item, not a
footnote.

**A benchmark on the project's own domain.** The advice is right and the gap is real: the only
in-house corpus is `eval/corpus-planted` — **13 files, 455 words**, a regression fixture for
chunking and trust tiers, not a domain benchmark. Everything the project knows about domain
transfer is therefore *negative* (answerability AUC 0.509 out of domain, encoder adaptation
regressing out of domain, coverage reversing on MuSiQue and 2Wiki). A small, hand-built,
own-domain set with real qrels would let the two things this engine uniquely emits —
**answerability calibration** and **citation mechanics** — be scored where no public benchmark
grades them at all (see P9). That is the cheapest route from "we measure ourselves well" to "we
measure something nobody else measures".

### 7.3 What does not change

The critique is about *where to be ranked*; §2 is about *what score moves*. The three-group
taxonomy says nothing about the fact that eight BEIR cells have never had a dense row, that the
shipped lexical settings cost Signal-1M five points against their own control, that FEVER sits
−0.121 below its published anchor, or that MultiHop-RAG MAP@10 is 0.319 against a reranked 0.480.
**P1 remains the largest single gain available and its ranking is unaffected.** What the critique
and the generation question change is everything downstream of it: **P10** is upgraded from one
external placement to two and re-scoped, **P12 is new** — and it is the item that turns the
project's best structural property (attribution by construction) into a measured number for the
first time — and two items (external-harness hosting, an own-domain set) move from footnote to
programme.

---

---

### 7.4 Did the advice name anything new?

**Systems and benchmarks: no.** Every one of the 31 names in that framing is already in
`BENCHMARKS.md` §"The rest of the landscape", each with a verdict rather than a category. The
mapping, verbatim from that file:

| group | verdicts already recorded |
| --- | --- |
| end-to-end | TREC RAG **blocked** †, CRAG **blocked**, RGB **declined**, FRAMES **runnable/measured**, MultiHop-RAG **measured**, MTRAG **adapter needed**, RAGBench **not applicable**, KILT **blocked** |
| classic QA | NQ / TriviaQA / HotpotQA / 2Wiki / MuSiQue / ASQA / ELI5 — **open-corpus variants blocked, distractor setting measured** |
| retrieval | BEIR **measured 18/18**, MTEB **representable**, BRIGHT **partially measured**, MIRACL **blocked as-is**, LoTTE **runnable**, ViDoRe **not applicable** |
| domain | FinanceBench **adapter + generator**, LegalBench-RAG **runnable**, MIRAGE **adapter + generator** |
| faithfulness | RAGTruth **not applicable**, HHEM **no ranking** |
| tooling | FlashRAG / BERGEN **integration, not a ranking**, RAGAS / ARES / DeepEval / TruLens **metrics, not rankings** |

† stale for the retrieval-only task; see §P10a.

"Not benchmarked against" and "not previously identified" are different questions. Twenty-odd of
those are indeed unrun — but the advice did not add one to the catalogue, and in two places it is
*less* complete than what is already here: it treats **BEIR as a single row** when the 18 members
diverge so far that dense loses to lexical on Touché and beats fusion on Quora, and it omits the
**TREC RAG 2025 retrieval-only task** entirely.

**Methodologies: four, of which two are real gaps.**

| methodology the advice names | status here |
| --- | --- |
| **Fixed-generator controlled comparison** | **Not practised.** Zero hits for "fixed generator" anywhere in `docs/`. The project separates retrieval from generation, but has never run a two-retriever A/B under one generator — the standard way the field isolates a retriever inside an end-to-end pipeline. Cheap, and now unblocked (P12). |
| **Counterfactual robustness as a named axis** | **Not formalised.** It appears only inside the *description of RGB* in the briefs (`docs/generation-placement.md`, `docs/research/verification/05`, `07`), and RGB was declined. The project measures negative rejection (MultiHop `null_query`) and poisoned *evidence* (the adversarial audit: fabricated claims, injected instructions, false numbers), but never a **false-premise query** — a question whose presupposition the corpus contradicts. That is a distinct failure class and the planted corpus could host it. |
| **External harness as a control (FlashRAG / BERGEN)** | **Assessed, not executed** — see §7.2. |
| **Own-domain benchmark** | **Assessed, not executed** — 13 files, 455 words, see §7.2. |

**Methodologies the advice implies that this project already does better.** Nugget-based scoring:
the advice points at TREC RAG's AutoNuggetizer, but there is already an **LLM-free** nugget/claim
recall protocol here (claim-recall@10 0.7438 fused against the CE's 0.7803), which is strictly more
ownable. Citation checking as a separate graded axis: `verify.py` does it deterministically, with
no judge. Cost reporting: the advice lists latency and tokens as things a credible row should
carry; `docs/findings-cost-axis.md` already reports 1.66 s mean / 2.14 s p95 / peak RSS / index
bytes per chunk — ahead of most published RAG rows.

**What the advice was actually worth.** Not targets — the target list was already here. It was
worth two things: it prompted the TREC RAG re-check that found the 2025 retrieval-only task and
exposed a stale verdict in `BENCHMARKS.md` (§P10a), and it named the **fixed-generator comparison**,
which is the one method this programme has genuinely never run and which the extractive composer
(P12) makes both cheap and fully ownable.

### 7.5 A second, larger survey — and this one *is* mostly new

A later survey of the landscape named 26 further systems. Checked the same way, across all 659
Markdown files in this repository and `best-in-the-world/`:

| status | items |
| --- | --- |
| **already known** | `BrowseComp` (4 hits), `BrowseComp-Plus` (4), `MS MARCO` (30, measured at 8.8M), `PopQA` (1), `DelucionQA` (1) |
| **known only as an aside** | `ALCE` — exactly **one** occurrence, in a corrections table in `RESEARCH.md`, never assessed as a target |
| **absent** | RAGQA-Leaderboard, EnterpriseRAG-Bench, DRAGOn, MIRAGE-Bench, CRUD-RAG, RAGEval, PruningRAG, ChatRAGBench, DomainRAG, GraphRAG-Bench, WideSearch, FaithJudge, HaluEval, FinQA, ConvFinQA, TAT-DQA, RAGCare-QA, Dyn-VQA |

(A first pass reported `RAGEval` as present; that was a **false positive** — the match was inside
`MTRAGEval` in `ROADMAP.md:25`. RAGEval is genuinely absent.)

**The two worth acting on, and why the survey under-rates one of them.**

**`ALCE` (EMNLP 2023) is mis-described by the survey as "scored with response-level metrics like
exact match or ROUGE".** Its headline metrics are **citation precision and citation recall**
(NLI-checked against the cited passages), alongside QAMPARI **list recall**. That is precisely the
axis `BENCHMARKS.md` records as *"partly this system's (`verify.py`) — TREC RAG's support assessment
is the published analogue, but **no run exists**"*. ALCE is a **better analogue than TREC RAG's
support assessment**: public, small, no NIST corpus, no official assessor, with a released eval
script. Its QAMPARI axis is **set coverage** — the same quantity as the MAP@10 coverage fight that
won on MultiHop-RAG and lost on MuSiQue and 2Wiki.

*The caveat to state up front:* ALCE's main setting supplies the retrieved passages, so it scores
evidence *selection and citation*, not retrieval. It is a citation-mechanics and set-coverage
benchmark. And through the extractive composer (P12) citation precision/recall is ~1.0 **by
construction** — so the informative numbers become correctness and abstention, and a citation score
*below* 1.0 would be a bug rather than a result.

**`BrowseComp-Plus` is the strongest addition on either survey, and the reason is not "it supports
retrieval-only".** The repository already cites it four times — but always as **evidence**
(calibration error 9.11%–68.96%; swapping only the retriever moves agent accuracy 3.86% → 10.36%),
**never as a target**. Its corpus is ~100K documents, which is *smaller than four SciDocs* and
trivial for this engine; it is a fixed-corpus Cranfield setup in TREC format with retrieval-only
components supported, and it grades **calibration error** as a headline metric. It is the only
benchmark on either survey that is simultaneously (a) fixed-corpus, (b) calibration-grading and
(c) backed by a published results dataset — the exact shape §P9 describes as the unclaimed row.

**Two whose *shape* matches an unmeasured strength, which is a different reason to care.**

- **`CRUD-RAG`** (create/read/update/delete) would score the mutation story: `audit.py`'s
  hash-chained log, delete-by-id, prune-on-reindex, drift reporting. All shipped, all measured
  in-house, none ever exercised by an external benchmark.
- **`EnterpriseRAG-Bench`** is generation-scored, but it is the one setting where **trust tiers,
  per-tenant ACL, provenance and the audit log** are load-bearing rather than invisible. The in-house
  numbers (poison leak 0.20 → 0.00 at zero false refusals, tenant isolation, anonymous denial) have
  no external comparison at all.

**Not applicable, and each names a real capability boundary.** `FinQA` / `ConvFinQA` / `TAT-DQA` are
**text-and-table** QA and this engine has no table representation — `chunking.py` is a text loader.
`Dyn-VQA` (and ViDoRe) need a vision path. `MIRAGE-Bench` (18 languages) and `DRAGOn` (Russian) need a
multilingual lexical channel; the stopword list is English and the stemmer is hard-wired
(`docs/research/06`), so Laya's multilingual checkpoint is the one favourable fact, not enough.
`GraphRAG-Bench` needs an LLM at graph-construction time, and the repo's own position is that the
conditioning signal, not the graph, carries the gain. `HaluEval` / `DelucionQA` / `FaithJudge` are
hallucination *detectors*; for the extractive path faithfulness is ~1.0 by construction, so they are
useful only as a negative control — a score below 1.0 is a bug.

**`RAGEval` is a methodology, not a target, and it answers §7.2.** It automates dataset generation
for testing, which is the cheapest known route to the own-domain set this project lacks.

**Two caveats on the survey itself.** Maintenance status is unverified here too, and two of its
citations look unconfirmed — `EnterpriseRAG-Bench`, whose arXiv identifier is 2026-dated, and
`MIRAGE-Bench`, whose link resolves to an author page marked unverified. Neither should be acted on
before the papers are read. And like the first survey, it omits the two things that matter most
*here*: the 18-way divergence of BEIR's members, and TREC RAG 2025's retrieval-only task.

**Net change to the programme: one item.** Add **BrowseComp-Plus** (calibration, fixed corpus) and
**ALCE** (citation support, set coverage) to P10 as the two external targets small enough to run
now, and note `RAGEval` as a build route for §7.2. Everything else is either not applicable for a
stated structural reason — which is itself useful, because the reasons are capability boundaries
(tables, vision, multilingual) rather than scope choices — or already catalogued.

---

## 8. Expanding the capability boundaries

§7.5's "not applicable" verdicts are not scope choices. They are **capability boundaries** — a
document loader that reads text, a tokeniser that reads Latin script, an evidence contract that
carries character offsets and nothing else. This section is the specification and the plan for
moving them, so the benchmarks behind them become runnable later.

### 8.0 The invariant, and what it forbids

Everything here must preserve the four properties that make a number *this system's*:

1. **Verbatim evidence** — returned spans are the corpus's own words; nothing is invented.
2. **Structured citations** — every emitted span carries `path` + `chunk_id`, checkable without a
   judge.
3. **Calibrated probability** — the answerability decision is a probability, not a token sample.
4. **No generation surface** in the default path.

This is why the plan below is a **representation** programme, not a modelling one. Boundaries that
can only be crossed by putting an LLM on the critical path (agentic tool-calling; graph
construction) are either deferred or explicitly declined — not because they are hard, but because
crossing them would make the resulting number someone else's.

**The recurring consequence: machine-derived text may be indexed, never returned.** Every loader
below produces text that a model wrote — OCR of a scanned paragraph, a VLM's description of a
chart, a generated table summary. None of it is the corpus's own words, so it cannot be evidence.
The mechanism already exists and is already proven in this codebase: `Chunk.context` is auxiliary
text that the retrieval channels see and the reader and citation checker never do. Every such
projection goes there, and the *evidence of record* for a non-text region is its **locator** — a
page and a bounding box, which a human opens and verifies. §8.9 makes this concrete for figures;
C3/C4 inherit it for tables.

**This section adds addressable surface, not score.** None of it raises a number on the current 18
BEIR cells or on MultiHop-RAG. It is ranked below P1–P12 and should be scheduled behind them —
with one exception worth flagging, because it is a **live bug** rather than a boundary: the lexical
channel cannot tokenise most of the world's scripts (§8.3).

### 8.1 The capability registry

| # | boundary | what blocks it today (file:line) | unlocks | effort | verdict |
| --- | --- | --- | --- | --- | --- |
| **B3** | **Multilingual** | `_TOKEN_RE` `lexical.py:35` matches Latin/CJK only; `STOPWORDS:37` is English; `_stem_word:75` is hard-wired `english` | MIRACL, MIRAGE-Bench, DRAGOn, MMTEB | **days** (lexical) / weeks (dense) | **do first** |
| **B1** | **Tables** | `Chunk.text` is a `str` slice (`chunking.py:51`); suffix allowlist is text-only (`chunking.py:19`) | FinQA, ConvFinQA, TAT-DQA, FinanceBench | weeks | do second |
| **B9** | **Structured/PDF ingestion** | suffix allowlist; `_looks_binary` rejects (`chunking.py:122`, `:257`) | FinanceBench, MIRAGE, LegalBench-RAG | weeks | do with B1 |
| **B6** | **Multi-turn** | `RagEngine.query` is single-turn; no history type | MTRAG, MTRAG-UN | weeks | do, generator-free |
| **B4** | **Entity linking / graph** | no linker; no PPR | KILT, GraphRAG-Bench, HippoRAG-lite | weeks–months | partial only |
| **B7** | **Judge harness** | no judge; deterministic checks only | RGB, FRAMES EM, any generation-scored row | days | **enabling**, opt-in |
| **B5** | **Agentic / tool-calling** | single-pass, no planner, no HTML extractor | CRAG, WideSearch | months | **declined** — see §8.8 |
| **B2** | **Visually-rich documents** | no image path; `_looks_binary` rejects | **MMDocIR**, ViDoRe v3, Dyn-VQA, MIEB | weeks (B2a) → months (B2b) | **do, staged — see §8.9** |
| **B8** | **Hallucination detection** | wrong module class | HHEM, RAGTruth, HaluEval, DelucionQA, FaithJudge | weeks | **recommend against** — §8.10 |

### 8.2 The shared refactor: a chunk becomes a record, not a string

Three of the nine boundaries (B1, B2, B9) fail at the same seam, and one change serves all three.
Today a chunk is a text slice (`chunking.py:50`):

```python
@dataclass(frozen=True)
class Chunk:
    chunk_id: str; doc_id: str; path: str
    text: str; index: int; start_char: int; end_char: int
    context: str = ""
    @property
    def index_text(self) -> str:  # "the text a channel should index"
        return f"{self.context}\n{self.text}" if self.context else self.text
```

**The key observation is that `context` already is the pattern.** It is auxiliary text that the
retrieval channels see and the reader and citation checker never do — documented in the docstring,
pinned by tests. A table's linearised form, a figure's caption, a PDF page's header block and a
scanned paragraph's OCR are the *same kind of thing*. So the specification is an extension of a
mechanism that already exists rather than a new pipeline:

```python
class Chunk:
    ...
    kind: str = "text"                 # text | table | figure | formula
    locator: Locator | None = None     # page/bbox/sheet-cell, in addition to char offsets
    derived: tuple[Derived, ...] = ()  # index-time-only searchable projections
    @property
    def index_text(self) -> str:
        parts = [self.context, self.text, *(d.text for d in self.derived)]
        return "\n".join(p for p in parts if p)
```

and the evidence contract (`schemas.py:87`) gains the same two fields:

```python
class Passage:
    ...
    kind: str = "text"                       # what the span *is*
    locator: dict[str, Any] | None = None   # page/bbox/cell, when char offsets do not apply
```

Three consequences worth stating explicitly, because they are what make this safe:

- **The verbatim guarantee survives.** A table chunk's `text` is the table's own cells rendered
  canonically; a figure chunk's `text` is its caption and surrounding prose, both the document's
  words. Nothing on the returned side is machine-written. `extractive.py`'s "a sentence that is not
  a substring of a retrieved passage is never emitted" still holds.
- **The locator generalises `start_char`/`end_char`** rather than replacing them, so existing
  reports, tests and `verify.py` are unaffected on text corpora.
- **`derived` is where an LLM may write, and it is index-time only.** A generated table summary
  can help recall and can never be *returned* as evidence. That is the same containment the
  system prompt already applies to retrieved content, applied to the ingest side.

### 8.3 B3 — Multilingual: the cheapest boundary, and it is already half-open

**This is not "needs a multilingual model". It is a one-line regex and two registries**, and there
is a measured repro:

```
$ .venv/bin/python -c "from laya_rag.lexical import tokenize; print(tokenize('привет мир'))"
[]          # Russian   -> ZERO tokens
[]          # Arabic, Greek, Hindi, Thai -> ZERO tokens
['你','好','世','界']   # CJK       -> works (the regex already carries the ranges)
['ckerstattung','der','rechnung']  # German 'Rückerstattung' -> the umlaut splits the word
```

The lexical channel is the **shipped default** (`DENSE_WEIGHT` resolves to 0 for Laya's encoder), so
a Cyrillic or Arabic corpus is silently indexed as empty documents today, and there is **no ingest
guard** anywhere in `src/laya_rag/` — I checked. Even Latin-script German is mangled. B3 is
therefore simultaneously a capability boundary *and* a latent correctness bug on any corpus that is
not English or CJK.

**Specification.**

1. **Script-aware tokeniser.** Replace `_TOKEN_RE` with a Unicode-property expression covering
   Latin+diacritics, Cyrillic, Greek, Armenian, Hebrew, Arabic, Devanagari, Thai/Lao/Khmer/Myanmar,
   plus the existing CJK ranges. Acceptance: the eight samples above all yield >0 tokens and German
   yields `rückerstattung` as one token.
2. **A language registry.** `STOPWORDS` becomes `STOPWORDS: dict[str, frozenset[str]]`. Add a
   deterministic, model-free detector (script histogram, then a stopword-frequency vote over the
   top-2 scripts) with `LANGUAGE=auto|xx`, defaulting to the corpus majority and recorded in the
   index metadata so a query is processed the way its corpus was — the same invariant the stemmer
   already documents.
3. **Per-language stemming.** `snowballstemmer` **already ships 36 languages and is already
   installed**; `_stem_word` hard-wires `stemmer("english")`. Make the cache key `(lang, token)`
   and select the stemmer from the registry. No new dependency.
4. **An ingest guard.** A chunk whose `index_text` yields zero tokens while the lexical channel is
   the only enabled channel is a *reported* anomaly (`LoadReport.skip`-style, surfaced in
   `stats()`), never a silent empty document. This is the "ingest anomaly quarantine" the ledger
   lists as untried, and B3 is what makes it matter.
5. **Dense half.** Once the lexical half works, MIRACL's blocker becomes the 18 per-language
   Wikipedia indexes (disk) and a multilingual encoder. Laya's checkpoint *is* multilingual, and
   the adapted-encoder work (`models/laya-retrieval`) already exists, so this is a measurement, not
   a new model — but it is a separate, larger item and is not on B3's critical path.

**Acceptance test.** A multilingual planted corpus (extend `eval/corpus-planted/i18n/` from 2 files
to ≥6 languages) retrieves gold in every language with the lexical channel alone. **Falsifier:** if
per-language stemming does not beat no-stemming on a per-language basis, the stemmer registry is
dead weight and only the tokeniser change is kept.

### 8.4 B1 + B9 — Tables and structured ingestion

**Specification.**

1. **New loaders behind the existing suffix allowlist** (`chunking.py:19`): `.csv`, `.tsv`, `.html`
   (table extraction only), `.pdf` (text + table layer). Each returns a `Document` whose `text` is
   a canonical rendering, so every downstream consumer keeps working unchanged.
2. **A table becomes a chunk with two views.** `text` is the canonical rendering (the table's own
   cells, Markdown-pipe style, so it stays verbatim); `derived` carries a **row-wise linearisation
   with headers expanded** — `"revenue | FY2024 | 3.2M"` — which is what makes a natural-language
   question match a cell. This is the whole trick, and it is the same index-time-only containment
   as `context`.
3. **`locator` on tables and PDFs**: `{"sheet": ..., "rows": [i, j]}` or `{"page": n, "bbox": [...]}`,
   so a citation points at a region a reader can open, not an offset they cannot.
4. **HTML needs a real extractor** and must be treated as untrusted input — `sanitize_text` and
   `invisible_payload_flags` already exist for text and should run over extracted cell content
   too, since a poisoned table is a new injection surface.

**Deliberately out of scope: numeric aggregation.** FinQA/TAT-DQA ask "what was the 3-year CAGR",
which is an *answer* capability, not a retrieval one. Adding a numeric index to serve it would make
the system a QA engine and put the returned number outside the verbatim guarantee. The defensible
target is **retrieval of the gold table and its rows** (recall@k over gold tables), reported as
such.

**Acceptance test.** FinQA gold-table recall@10 with the lexical channel; a planted numeric
document set retrieves the right row for a natural-language cell query. **Falsifier:** if the
row-wise projection does not beat the raw canonical rendering, keep the rendering and drop the
projection.

### 8.5 B6 — Multi-turn, without a generator

**Specification.** Conversation state is `list[Turn]`, and the follow-up query is rewritten by a
**deterministic, model-free** rule: carry over noun phrases and proper nouns from the previous
turn's kept passages that are absent from the follow-up, then **union-then-fuse** the per-turn
ranked lists by RRF — the same construction already specified for the multi-hop iteration in P11,
and for the same reason (union preserves recall that sequential replacement destroys).

**Why not an LLM rewriter.** It would be the cheapest route, and it is the one the field uses. It
also puts a model on the path that produces the *query*, which makes every downstream number
partly the model's. The generator-free version is worse and ownable, which is the project's
consistent trade.

**Acceptance test.** MTRAG retrieval metrics on the multi-turn split, measured as *deltas* against
the same engine with history ignored. **Falsifier:** if union-then-fuse does not beat
last-turn-only, the state is not worth carrying and the finding is that this engine should be
documented as single-turn.

### 8.6 B4 — Entity linking and graph retrieval: partial only

**Specification.** Productise what is already validated rather than build a linker. The
generator-free iteration (`tools/generator_free_iteration.py`) already mines entity probes and
raises gold coverage **0.8326 → 0.8937 (+0.0611)** with a shuffled-probe control. Expose it as
`bridge_entities(query, passages) -> list[tuple[str, float]]` with a calibrated score.

**Where this stops, stated plainly.** KILT needs a real entity linker (BLINK-class) and
GraphRAG-Bench needs an LLM at graph-construction time. A linker is a *model*, and it can be
treated like `EMBED_MODEL` — an optional, named, external component whose identity appears in the
report — but graph construction as the repo's own analysis notes has the conditioning signal, not
the graph, carrying the gain. **Recommendation: build `bridge_entities`, decline the graph.**

### 8.7 B7 — A judge harness, as an enabling tool and never as a claimed number

**Specification.** `tools/judge.py`: provider-agnostic, reads `LAYA_RAG_JUDGE_MODEL` and
`LAYA_RAG_JUDGE_BASE_URL` from the same OpenAI-compatible surface `compose.py` already speaks;
records model id, prompt hash and token cost in every report it writes; and refuses to run unless
an env flag is set. It is never invoked by `make test` or by any row the engine labels as its own.

**What it unblocks.** RGB's negative-rejection metric, FRAMES' answer accuracy, and the "judge
validated against a human sample with a reported κ" discipline the field expects. **What it does
not do:** make a judged number the engine's. Every report it writes must carry the `engine + <model>`
label that `findings-endtoend-pilot.md` already uses.

### 8.8 B5 — Agentic tool-calling: declined, with the reason

CRAG hands the participant five web pages per question plus mock APIs and a mock KG. Crossing that
boundary means building an HTML extractor and a tool-calling loop — and both are only useful if an
LLM plans the calls. That puts a model on the critical path to the *retrieval*, which is the one
thing this system currently owns outright. It is not that it is hard; it is that the resulting row
would be a different system's.

**Recommendation: decline CRAG permanently, and record the reason in `BENCHMARKS.md` as a
capability-boundary verdict rather than a scope choice.** The generator-free controller (P11, the
pool bet) is the version of "agentic" that stays inside the invariant, and it is already priced.

### 8.9 B2 — Visually-rich documents: the academic-paper case, and a corrected verdict

*Revised. An earlier draft split this into "cheap caption-only" versus "visual embedding, which is a
different product". That split was wrong, and the correction matters.*

**The axis I got wrong.** ViDoRe and MMDocIR are not the same task shape. In ViDoRe the *document*
is an image; in MMDocIR the visual content sits **inside** a text document, in the same corpus and
the same retrieval task. Academic papers are the second case — the dominant one, and a stated use
of this system. Filing them under "different product" was a mistake.

**The evidence, and it is stronger than I assumed.** [MMDocIR](https://arxiv.org/html/2501.08828v3)
(EMNLP 2025) is the right benchmark: 313 documents averaging **65.1 pages**, 1,658 expert-annotated
questions, with **1,658 page-level and 2,638 layout-level labels**, annotation consistency F1 95.2
(page) and 87.1 (layout). Its measurements:

- **Text is only 52.7% of content area** in the source corpus; images are 29.2% and tables 12.8%.
  Half the page is invisible to a text-only pipeline.
- **"A layout" is a paragraph, heading, equation, table, figure or chart with a bounding box** — and
  a document page holds 5–15 of them.
- **Visual-driven retrievers significantly outperform text-driven ones.**
- **VLM-text beats OCR-text** for text-driven retrievers — by 1.5× (tables) to 3.8× (figures) in length,
  because a figure's information is visual elaboration that OCR cannot recover.

**The part that makes this affordable.** MMDocIR ships **three representations of every image
layout — the original image, OCR-text, and VLM-text** — precisely so that text-only systems can
compete. So the first row does not need a vision model, a VLM, or an OCR pipeline: it needs
ingestion of shipped (or locally produced) text. And the engine's fit is unusually good, because
**its chunk already is what MMDocIR calls a layout** — a bounded, retrievable region with a
citation — so `locator` (§8.2) needs only to carry `{page, bbox}`.

**Feasibility, checked on this host rather than assumed.** `tesseract` and `pdftotext` are **already
installed** (`/Users/threaded/miniconda3/bin/tesseract`, `/usr/local/bin/pdftotext`); `PyMuPDF`
1.28.2 ships a **`macosx_10_15_x86_64` wheel**, so it installs here; `pytesseract` and `pypdf` are
pure Python. Layout detection is the one genuinely hard piece — MMDocIR uses MinerU, a heavyweight —
but PyMuPDF natively yields text blocks and image blocks with bounding boxes, which is a crude and
sufficient first cut.

**The design decision this forces, and it should be made explicitly.** OCR and VLM text are
**machine-written**, so returning them as evidence would break invariant #1. MMDocIR's own three-way
distinction is the answer, and it maps exactly onto the mechanism §8.2 already specifies:

| representation | where it goes | returned as evidence? |
| --- | --- | --- |
| the document's own extracted text | `Chunk.text` | **yes** — verbatim, as today |
| the **image region** for a figure/chart | `Chunk.locator` = `{page, bbox}` | **yes** — the evidence of record is the region, which a reader opens and verifies |
| OCR-text / VLM-text of that region | `Chunk.derived` | **no** — index-time only, exactly like `context` |

So a figure's citation resolves to a **region a human can look at**, not to a machine-written string
that cannot be checked against the page. That is the same containment the system prompt already
applies to retrieved content, applied at ingest.

**Staged plan.**

- **B2a — do it. PDF ingestion, layout-level chunks, bbox locators, provided/local OCR-text and
  VLM-text in `derived`.** Target: **MMDocIR layout-level retrieval, text-driven arm**, reported
  honestly as mid-pack, because the paper's headline is that visual retrievers beat text ones.
  Effort: weeks. This alone converts "not applicable" into a measured row on the benchmark that
  matches the stated use case.
- **B2b — price it. A visual embedding channel** (CLIP/SigLIP-class, CPU-runnable at the ~10⁴-layout
  scale of an MMDocIR-sized corpus) so the visual-driven arm becomes enterable. This is what the
  paper says is needed to be *competitive*, so it is not optional for a top row — but B2a does not
  depend on it, and MMDocIR's own §5.7 cascade (text first, visual second) is the engine's own
  recall→rerank shape.
- **B2c — defer. Document-VLM inference at ingest** to *generate* VLM-text locally. MMDocIR supplies
  it; producing it here for 313×65 pages needs a VLM this host cannot run at that scale. Source it
  from the benchmark, or from a one-off external pass, and record the provenance.

**Also fold in the original cheap case, which was right as far as it went.** A standalone image file
with a caption and alt text is still a text document for `B2a`'s purposes; there is no separate
"caption-only" stage to build.

### 8.10 B8 — Hallucination detection: recommend against

The engine is a *source* of hallucination, not a detector, and the extractive composer makes
faithfulness **1.0 by construction**. Building a detector to enter HHEM, RAGTruth, HaluEval,
DelucionQA or FaithJudge would be building a different product to compete on a leaderboard that
does not describe this system. **The correct use of those datasets here is as a negative control:**
feed the extractive composer's output to them and treat any score below 1.0 as a **bug** — either a
verbatim-guarantee violation or a citation-numbering error. That is a genuine test, it costs
nothing, and it turns five unusable leaderboards into one regression fixture.

### 8.11 Sequencing, dependencies and gates

| stage | work | depends on | gate |
| --- | --- | --- | --- |
| **C1** | B3 lexical (tokeniser, registries, guard) + multilingual planted corpus | — | ≥6 languages retrieve with the lexical channel alone; German tokenises intact |
| **C2** | §8.2 chunk-as-record + `Passage.kind`/`locator` | C1's schema churn lands first | all existing suites green; text-corpus reports byte-identical |
| **C3** | B9 structured ingestion (CSV/TSV/HTML/PDF text+table) | C2 | FinQA gold-table recall@10 reported |
| **C4** | B1 table projections | C3 | row-wise projection beats canonical rendering, or is dropped |
| **C5** | B6 multi-turn (deterministic rewrite + union-then-fuse) | — | MTRAG delta vs history-ignored |
| **C6** | B7 judge harness | — | a judged row carries `engine + <model>`; no `make test` path invokes it |
| **C7** | **B2a** PDF ingestion + layout-level chunks + bbox locators + OCR/VLM-text in `derived` | C2, C3 (shared PDF loader) | MMDocIR layout-level retrieval reported, text-driven arm, labelled mid-pack |
| **C8** | B4 `bridge_entities` | P11's mechanism | coverage delta reproduces +0.0611 outside the probe |
| **C9** | **B2b** visual embedding channel (CLIP/SigLIP-class) | C7 | MMDocIR visual-driven arm enterable; per-layout storage budget stated |
| **—** | B5 agentic, B8 detectors, B2c local VLM-text generation | — | **declined / deferred** — reasons recorded, not silently dropped |

**Dependency worth noting:** §8.2 is the single highest-leverage change in this section — three
boundaries (B1, B2, B9) all fail at the same seam, and its cost is one dataclass plus a schema
field, because the index-time-only auxiliary-text mechanism already exists and is already pinned
by tests. **Ordering rule: do §8.2 once, correctly, before any loader work**, or each loader will
invent its own incompatible shape. C3 and C7 share the PDF loader, so they should land together.

**The honest ranking.** B3 is days and fixes a live bug. §8.2 is days and unblocks three
boundaries. B7 is days and enabling. **B2a is weeks and is the highest-*product*-value item in §8**:
it serves the academic-paper use case directly, and MMDocIR is the one benchmark here that measures
a capability this system's users actually need — with its OCR/VLM-text supplied, so the first row
needs no vision model. C3/C4 and B2a are weeks and share a loader. C9 (the visual channel) is the
price of *competing* on MMDocIR rather than merely appearing on it. B6 and B4 are weeks and
speculative. B5 and B8 should not be done at all. **Nothing in §8 outranks P1**, and a reader should
treat the whole section as a portfolio decision rather than a score one — it decides which
leaderboards exist for this system in two years, not what it scores this month.

---

## 9. Datasets and methods from the RAG-evaluation survey — tasks

**Source.** *Can LLMs Be Trusted for Evaluating RAG Systems? A Survey of Methods and Datasets*
(Brehme, Ströhle, Breu; IEEE SDS25; [arXiv:2504.20119](https://arxiv.org/abs/2504.20119)). A
systematic review of 63 papers, claiming **87 existing QA datasets** used to benchmark RAG systems.

**What I actually did, and one honest limit.** The survey names few datasets in its prose — the
catalogue *is* its bibliography. I pulled the PDF, extracted it with the locally installed
`pdftotext`, and read the 71-entry reference list. The "87 datasets" figure is a count, not a
published table, so what follows is the set of **named, linked artifacts I could extract and
verify**, not the full 87. Every arXiv identifier below was fetched and returned HTTP 200, and the
titles were read back rather than trusted from the survey's prose.

**Two corrections the verification produced.** The survey cites `[23]` as *"RAG-ConfusionQA"* at
`2410.14567`; that identifier now resolves to **ELOQ: Resources for Enhancing LLM Detection of
Out-of-Scope…** — same IBM authors, renamed. And an automated scan of this repository first reported
`UDA`, `eRAG` and `VERA` as already present; those were **substring false positives** (inside
"aver**age**" and "a**vera**ge"). A word-boundary check returns zero hits. All three are genuinely new
here.

### 9.1 Triage

Tiers are by *fit to this system*, not by the survey's emphasis. Most of its references are
generator-evaluation frameworks, which is the axis §7.1 established is largely not this system's.

| tier | meaning |
| --- | --- |
| **T-A** | retrieval-side, public, runnable here — becomes a row |
| **T-B** | a **method** for building benchmarks, not a benchmark — adopt it |
| **T-C** | relevant to a §8 boundary — scheduled *with* that boundary |
| **T-D** | read-only; framework or survey, no row |
| **T-E** | negative control only — see §8.10 |

### 9.2 T-A — retrieval-side benchmarks to run

These are the ones that produce a comparable row without a generator. **All are retrieval-scored**,
which is why they survive §7.1's test.

| id | dataset | link | why it fits | effort |
| --- | --- | --- | --- | --- |
| **T1** | **ClimRetrieve** | [2406.09818](https://arxiv.org/abs/2406.09818) | a genuine **retrieval** benchmark over corporate climate disclosures, with graded relevance. Small, public, single-domain, and no generation. The most directly runnable new row on this list. | days |
| **T2** | **CORAL** | [2410.23090](https://arxiv.org/abs/2410.23090) | multi-turn conversational RAG scored on **Recall / MRR / MAP / NDCG** — it is the benchmark for the B6 boundary (§8.5) *and* it grades retrieval rather than prose. | days after C5 |
| **T3** | **IRSC** | [2409.15763](https://arxiv.org/abs/2409.15763) | zero-shot IR evaluation benchmark framed for RAG scenarios; retrieval-only, so it is a clean addition. | days |
| **T4** | **UDA** | [2406.15187](https://arxiv.org/abs/2406.15187) | benchmark suite for RAG in **real-world document analysis** — document-shaped, which is this engine's native object. | days–weeks |
| **T5** | **RAG-QA Arena** | [2407.13998](https://arxiv.org/abs/2407.13998) | domain-robustness evaluation for long-form RAG; useful as a *domain-transfer* probe, which is where every in-house result here has been negative. | weeks |
| **T6** | **RAD-Bench** | [2409.12558](https://arxiv.org/abs/2409.12558) | retrieval-augmented **dialogues**; the second multi-turn row, and a cross-check on T2. | weeks |
| **T7** | **FeB4RAG** | [2402.11891](https://arxiv.org/abs/2402.11891) | **federated search** — retrieval across multiple collections. The engine ships one index; this is the benchmark that would force a decision about multiple corpora. | weeks |

**Task shape for T1–T7.** Each is: (a) a loader in `tools/` following the existing
`tools/twincorpus.py`/`musique.py` pattern; (b) a run through **the shipped engine defaults**, not a
harness replica, since `findings-engine-parity.md` records the replica overstating the engine by
0.0144 on SciFact; (c) per-query scores persisted so the paired bootstrap applies; (d) a report with
an interval, not a point estimate. **Do T1 first** — it is the smallest and the only one whose
scoring is unambiguous.

### 9.3 T-B — methods for *building* benchmarks, which is what §7.2 actually needs

The survey's most useful contribution here is not a dataset. It is that three of its references
describe how to construct an evaluation set when you have no qrels — which is exactly the
own-domain gap §7.2 names as the project's largest missing measurement.

| id | method | link | what it gives this project |
| --- | --- | --- | --- |
| **T8** | **eRAG** — *Evaluating Retrieval Quality in RAG* | [2404.13781](https://arxiv.org/abs/2404.13781) | **Reference-free chunk relevance.** Generate a question from each retrieved chunk and check whether the system's answer to it is correct; the chunk is relevant if so. Adopting this lets the project label an own-domain corpus **without human qrels or an LLM judge over the whole set** — the single highest-value item in §9. |
| **T9** | **Sub-question coverage** — *Do RAG Systems Cover What Matters?* | [2410.15531](https://arxiv.org/abs/2410.15531) | A published framing of the **set-coverage** axis. This project's most contested result is exactly that axis (coverage won on MultiHop-RAG, lost on MuSiQue and 2Wiki); this gives it a published metric to be reported against instead of an in-house one. |
| **T10** | **Long²RAG — key point recall** | [2410.23000](https://arxiv.org/abs/2410.23000) | A generator-free *key-point-recall* protocol for long-context, long-form evaluation, complementing `tools/nugget_recall.py` (claim-recall@10 0.7438 fused vs 0.7803 CE) and giving the P12 row a second, independently-defined measure. |
| **T11** | **RAGEval** | [2408.01262](https://arxiv.org/abs/2408.01262) | Scenario-specific RAG **dataset generation** framework. Already cited in `ROADMAP.md`; it is the build route for §7.2 and pairs with T8 — generate, then filter with eRAG. |

**Concretely:** T8 + T11 + a small hand-reviewed corpus is a **complete, generator-free own-domain
benchmark pipeline**, and it is the cheapest way to close the gap §7.2 calls the project's largest.

### 9.4 T-C — scheduled with a §8 boundary

| id | artifact | link | lands with |
| --- | --- | --- | --- |
| **T12** | **Face4RAG** — factual consistency, Chinese | [2407.01080](https://arxiv.org/abs/2407.01080) | **§8.3 (B3, multilingual)** — it is the test that proves B3's tokeniser and stemmer work on a non-Latin script, and it is already labelled. |
| **T13** | **MIRAGE-Bench** — multilingual arena, 18 languages | [2410.13716](https://arxiv.org/abs/2410.13716) | **§8.3** — the wide-language version of T12, after the per-language registry exists. |
| **T14** | **CRUD-RAG** | [2401.17043](https://arxiv.org/abs/2401.17043) | **§7.5** — scores the mutation story (`audit.py`, delete-by-id, prune-on-reindex) that has never been exercised externally. Chinese-language, so it also depends on C1. |

### 9.5 T-E — negative controls, per §8.10

Adopt these **only** as regression fixtures for the extractive composer, where any score below 1.0
is a bug: **FaithEval** ([2410.03727](https://arxiv.org/abs/2410.03727)) and **ELOQ** / formerly
RAG-ConfusionQA ([2410.14567](https://arxiv.org/abs/2410.14567) — out-of-scope and confusing
questions, the closest published analogue of the **false-premise** axis §7.4 names as a genuine gap).
Do not build detectors to enter their leaderboards.

### 9.6 T-D — read, do not run

Frameworks and methodology papers, worth a reading pass before any of the above is designed:
**VERA** ([2409.03759](https://arxiv.org/abs/2409.03759), statistical validation of RAG evaluation),
**RAGElo** ([2406.14783](https://arxiv.org/abs/2406.14783), Elo-based ranking),
**FaaF** ([2403.03888](https://arxiv.org/abs/2403.03888), facts-as-a-function),
**RAGProbe** ([2409.19019](https://arxiv.org/abs/2409.19019), automated probing),
**CoFE-RAG** ([2410.12248](https://arxiv.org/abs/2410.12248), full-chain evaluation),
*A Methodology for Evaluating RAG Systems* ([2410.08801](https://arxiv.org/abs/2410.08801)),
*Evaluating the Retrieval Component in LLM-Based QA* ([2406.06458](https://arxiv.org/abs/2406.06458)),
and *Beyond Benchmarks: Evaluating Embedding Model Similarity* ([2407.08275](https://arxiv.org/abs/2407.08275)
— CKA, Jaccard and a `RankSimilarity` score for **comparing retrievers**, which is the closest thing
in the survey to a method for the channel-reversal problem this project keeps hitting).

### 9.7 Sequencing and the honest ranking

| stage | tasks | depends on | note |
| --- | --- | --- | --- |
| **S1** | **T1** ClimRetrieve | — | days; the smallest clean new retrieval row |
| **S2** | **T8 + T11** eRAG + RAGEval → own-domain set | — | the highest-value item in §9: it closes §7.2 |
| **S3** | **T9, T10** coverage and key-point-recall metrics | S2's corpus | gives the contested coverage axis a published metric |
| **S4** | **T2, T6** CORAL, RAD-Bench | §8.5 (C5, multi-turn) | two independent multi-turn rows |
| **S5** | **T3, T4, T5** IRSC, UDA, RAG-QA Arena | — | further retrieval and domain-transfer rows |
| **S6** | **T12, T13, T14** Face4RAG, MIRAGE-Bench, CRUD-RAG | §8.3 (C1) | gated on the multilingual work |
| **S7** | **T7** FeB4RAG | — | forces a multi-corpus decision; do last |
| **—** | **T-D** reading list | — | no rows |

**The ranking, stated plainly.** T8 (eRAG) is the one that changes what this project *can* measure —
a generator-free route to an own-domain benchmark, which §7.2 calls the largest missing measurement
here. T1 (ClimRetrieve) is the cheapest new row. T9 and T10 give the coverage fight a published
metric instead of an internal argument. Everything else is either gated on a §8 boundary or is a
second confirmation of something already known. **Nothing in §9 outranks P1 or P12**, but **T8
outranks everything else in §8 and §9.**

**One caveat carried forward from the verification.** The survey's "87 QA datasets" is a count
without a published table, so this section is deliberately the *extractable and verified* subset,
not the whole claim. A reader wanting the other ~60 should go to the survey's own references and
apply the same HTTP-and-read-back check this section used — three of the names here would otherwise
have been recorded as already-present on the strength of a substring match.

---

## 10. Executing and checking this document

Three separately-numbered sequences now live here — **P1–P12** (score), **C1–C9** (§8 boundaries),
**S1–S7** (§9 datasets) — and read on their own each is coherent but the reader cannot answer "what
do I do on Monday". This section is the missing execution layer: one order, one register of the
bars that decide whether each item worked, an honest account of how to check this document, **§10.5**'s
live record of what has actually been run and what it returned, and **§10.6**'s queued runs as
executable commands, because a planning
document that does not say which of its own items are done makes the reader reconstruct it from
thirty findings pages.

### 10.1 One order, and the resource that serialises it

**Assumption, stated because it changes everything: one operator, one host.** Every "days" and
"weeks" below is one person's calendar, and `tools/heavylock.py` permits **one heavy job at a time**,
so the embedding-bound work serialises rather than parallelises.

| phase | items | resource | why here |
| --- | --- | --- | --- |
| **W0 — hygiene** | §4 items 1–3; `make channel-coverage` wired into the gate | none | a stale or diluted number invalidates everything after it |
| **W1 — free wins, no heavy compute** | **P2** selector + BM25F · **P12** extractive end-to-end · **P10b** MTEB stall · **P5a** convex fusion · **T8+T11** own-domain pipeline | CPU-light | all hours; P12 and T8 change what is *measurable*, not just what is measured |
| **W2–W4 — the headline** | **P1** the eight lexical-only cells, largest-last; **C3/C7** share its PDF loader | embedding-bound, serialised by the lock | the only item that moves the BEIR-18 macro |
| **W2–W4, in the gaps** | **P4** modern reranker at depth 100 → 500 · **P11** the pool bet | scoring-bound | contends with P1 for cores; run when a P1 job is between corpora |
| **W5–W8 — the two research bets** | **P3** encoder refresh · **P11** if not already closed | embedding + training | P3 needs W1's `BiEncoderModel` changes |
| **W6+ — portfolio** | **C1 → C2 → C5/C6**, then **T1, T2/T6, T3–T5**, gated **T12–T14** | mixed | §8/§9 |
| **standing** | **P6b, P7, P8** fill idle compute; **P9** and **T-D** are reading and writing | — | never on the critical path |

**The critical path is W2–W4 and it is embedding-bound.** Everything in W1 is short and should be
finished before the first 5M-document job starts, because a harness change discovered mid-run
invalidates the run. **Land C3/C7's PDF loader inside W2–W4**, since it shares P1's ingest path and
duplicating it later costs the same work twice.

**Operating notes, learned the expensive way.** This section is about the lock, so these belong
here rather than in a findings page nobody running P1 would open. Every one of them cost this
programme at least a round.

- **A background job's shell wrapper is not the job.** Killing the wrapper leaves the Python process
  running — a `beir.py` run survived its wrapper for thirty minutes, and a five-task sweep left
  **101 `mteb_search.py` processes** resident. Kill the *process* (`pkill -9 -f <script>`), and check
  afterwards; `make heavy-status` shows what holds the lock.
- **Never let a wrapper filter a long run's output.** A sweep launched as `... | grep main_score`
  discarded the progress lines, so a working run looked exactly like a hang — the failure mode this
  programme spent four rounds on, self-inflicted.
- **A flag that is parsed and never read is a knob that silently does nothing, and `tools/` now has
  none.** `tests/test_tools_flags_are_used.py` fails on any *new* one, and the five it found are all
  resolved, each by the route that fitted it: `multihop_hop2.py --rrf-k` and
  `laya_finetune_probe.py --seed` named mechanisms their tools do not have (no fusion; no randomness)
  and `generator_free_iteration.py --samples` named a limiting step that already exists as
  `--limit-queries`, so all three were **removed**; `multihop_diagnostics.py --bootstrap-samples`
  **could act** and is now wired into `_mean_ci`.
  *When a flag cannot act, deleting it is the fix; when it can, wiring it is.* And a third case the
  check has to know about: **a flag that acts by being the default** — `generation_eval.py --list` picks
  the branch that already runs, so it is a documented spelling rather than a lie, and the test keeps
  those in a separate `DEFAULT_MODE_ALIASES` set rather than treating them as defects. This is the class that cost four rounds here:
  `tools/mteb_search.py --first-stage-only` printed a banner and set nothing, and by the time it was
  found 18 recorded reports were their first stage's order wearing a reranker's name. **If you add a
  flag, read it; the check is one commit away from telling you that you did not.**
- **A documented command that fails when run is a claim nobody checked, and the checks are now two.**
  The status page told a reader to run make charts and then make charts-more; the second did not
  exist because `make charts` already calls `tools/compare_more.py`. Fixed, and
  `tests/test_docs_make_targets_exist.py` now fails on any **new** `make X` written inside code where `X`
  is not a Makefile target — scoped to code because prose is full of "make up" and "make for", which
  produced twenty-one false hits in a first pass that scanned everything. (This note is written without
  backticks around the two dead examples for exactly that reason — my first version of it failed the new
  check, which is the check working.) Four references in
  `docs/generation-placement.md` stay, listed as **proposed** rather than dead: that document calls them
  "planned hook" and its own Makefile sketch says "not yet in Makefile", so a reader is told the truth.
- **The commands in this document are checked too, and the checks that do it taught me two scopings.**
  `tests/test_docs_tool_flags_exist.py` fails when a documented invocation names a flag its tool does not
  declare — **including every command in §10.6**, which is the point: a runbook nobody executes is prose.
  Two scopings, both found by being wrong first: a **commented-out** command is not a reference (the
  generation document writes its unimplemented hooks as `# ...` and says in the next sentence that they
  are unimplemented), and a flag scan must accept **single quotes** — `tools/multihop_gate.py` declares
  its flags that way, and a double-quote-only regex produced four false hits against it.
- **A subset report belongs in `eval/beir/subsets/`, and two authorities will tell you if it does
  not.** `tools/channel_coverage.py` and `tools/lexical_selector.py` both scan `eval/beir/*.json`
  and both read a report as *the member's*. A one-sub-forum CQADupStack row therefore claimed the
  member had a dense channel (§1.1's lexical-only count moved 8 → 7 on a twelfth of the corpus) and
  made `bm25-engine` disagree with itself across two reports. Neither is a bug in the authorities —
  they cannot tell a subset from the member, which is why the path has to say it. The existing
  labelled exceptions (`smoke.json`, `trec-covid-bounded.json`, `fusion-tuning.json`) are at the top
  level deliberately and are named so.
- **Prefer one invocation to a shell loop.** `tools/beir.py` takes `--datasets`, `tools/mteb_search.py`
  takes a comma-separated `--task`; both take the lock once and print per-item results. The loop was
  the source of the process problem above.
- **A long run's report must survive a kill.** `tools/beir.py` writes its JSON **after every
  dataset** for this reason. Anything that only writes at the end should be distrusted at this scale.
- **Where a tool offers sharding, use it for anything over an hour.** `tools/engine_multihop.py` has
  `--shard i/n` and `--merge N` precisely because a 2,255-query cross-encoder pass reports *"2255 of
  2255 queries to score (0 checkpointed)"* — a kill costs the whole pass, and on this host runs are
  killed routinely (five times this session, for lock contention and for being wrong about what they
  measured).
- **Contention masquerades as pathology.** Slow is not stuck. Before diagnosing a run, check what
  else is resident — several rounds here were spent on a "stall" that was a contended host.
- **State what a number is not.** Claims of *absence* ("no score exists") and of *level* (an MTEB
  score read against BEIR baselines) are the two easiest to leave behind, because the number can be
  real while the surrounding sentence is wrong. Five separate cleanups were needed for the first
  kind alone.

### 10.2 The pre-registered bars

Every item carries a falsifier in place, which is the right form but the wrong shape for the moment
of decision — nobody re-reads thirty sections to find the bar they are about to clear. Collected
here, because a bar chosen after seeing the number is not a bar.

| item | the bar | the falsifier |
| --- | --- | --- |
| **P1** | ≥6 of the 11 members beat their published BM25 anchor; macro **≥ +0.015** | the dense row is not significantly better than `bm25-engine` on a majority, or the hybrid loses to its best single channel |
| **P2** | the selector must *not* pick identical settings on every corpus | it does — which closes lexical routing — or BM25F does not move FEVER |
| **P3** | beats `bge-small` `hybrid-biencoder` on **≥4 of 5**, intervals excluding zero; no loss on MultiHop MAP@10 | miss it, and the 2023 family stands |
| **P4** | `bge-reranker-v2-m3` raises MAP@10 on the 400-query subset | it lowers it, as `bge-reranker-base` did on BEIR |
| **P5a/b** | transductive selector beats global fusion on held-out halves, interval excludes zero | it cannot — the publishable "fixed fusion dominates routing" negative |
| **P6** | doc2query beats `bm25-engine`, interval excludes zero | it repeats PRF's expansion drift |
| **P7** | passages win on Robust04 and TREC-NEWS | they lose — the technique is closed at every length |
| **P8** | the published-settings FEVER control reads ≠ 0.63 | it reads ≈0.63 → footnote, not bug |
| **P9** | the 12-feature gate holds its AUC on a **fresh out-of-domain** corpus | it collapses like the scalar's |
| **P12** | the extractive row exists, with attribution and abstention scored | extractive prose is unusable — a result, not a failure |
| **C1** | ≥6 languages retrieve with the lexical channel alone; German tokenises intact | per-language stemming does not beat none |
| **C4** | the row-wise projection beats the canonical rendering | it does not — keep the rendering |
| **C5** | union-then-fuse beats last-turn-only on MTRAG | it does not — document as single-turn |
| **T8** | eRAG's labels agree with a hand-labelled sample | they do not — reference-free labelling is not adopted |
| **document** | the nine-cell deficit is closable | **if P1 finds no dense channel beats `bm25-engine` on a majority of the eight, §1's framing is wrong** — the ranking in §2 survives, the headline does not |

### 10.3 How to check this document

**§1 is machine-generated and reproducible.** `make channel-coverage` reads the retriever rows in
`eval/beir/*.json` and prints both tables. This was added *after* the first draft, and running it
falsified three hand-entered cells (§1.2) — two of them by using SPLADE as the ceiling on every
dataset instead of the actual per-cell maximum, and one by comparing against a BM25 column when
every other row used a model column. That is the argument for the tool, made by the tool.

**Everything else here quotes other documents.** The numbers are transcribed, and transcription
drifts. Two things followed, and the first is now closed:

1. **§1.1, the §1.2 arithmetic and the harness parity are anchored to the gate — done.**
   `consistency_check.py` imports `tools/channel_coverage.py` and pins, as one `Quantity` per
   figure: the summed deficit **0.7085**, the macro cost **0.0394**, the half-closed **+0.020** and
   the nine-tenths **+0.035**; plus a `check_channel_coverage` pass that requires every §1.1 table
   row *and* the three summary lines — including the lexical-only count — to appear verbatim, and
   every §1.2 cell to state the computed "ours", published and gap values. The SciFact
   harness-parity pair is pinned the same way (§4.3): the engine's 0.7264 and the replica's 0.7408
   from its two reports, and the **+0.0144** overstatement derived from them across eleven sites
   in six documents. `consistency-check` went from **20** checked sites to **43**, and to **47** once the summary line at
the top of §1 was anchored too (it had been the most visible unguarded figure in the document); a two-line edit
   to the deficit, to a member's channel or to the parity figure now fails the gate rather than
   passing silently. Both halves were negative-tested (perturbing 0.7085 in §1.2 and `nq`'s channel
   in §1.1 each produces exactly one failure; so does moving either parity endpoint). **The
   parity pin immediately found one**: the figure had been hand-typed as 0.7408 / +0.0144 and
   reproduces from no report on disk.

   **The gates were then adversarially tested rather than assumed, and one of them had a hole.**
   Perturbing `claim-check`'s anchored figures fails it with the anchor named, so that gate is live; so is
   `make leakage-audit --check`, which caught a planted **dilution signature** (a report whose settings
   record `limit_queries: 10` while carrying fifty per-query entries) and exited 2 — the rule the audit
   was *not* enforcing while it crashed on a 24 GB latin-1 dump, which is why it is worth re-planting
   rather than trusting that a fixed audit still detects.
   Perturbing the **summary blockquote at the top of §1** did *not* fail `consistency-check` — the
   gate was fine, that line simply was not one of its sites, and it is the first number a reader
   meets. It is anchored now (**43 → 47 sites**) and the perturbation fails as it should. The lesson
   generalises past this document: **a gate that catches a wrong number and misses the number looks
   identical to a working gate until someone perturbs the right line**, which is why each of these
   was perturbed rather than trusted.

   **A fourth class of stale claim turned up in the same sweep, and it is the one aimed at newcomers.**
   `docs/PROGRAMME-STATUS.md`'s "Starting again" block — the first thing a new contributor runs — said
   `make test   # 1251 tests` and `make claim-check   # 58 prose claims`. The suite runs **1,718** and the
   checker reports **65**. Both are now approximate or attributed ("~1,700 tests", "65 today") rather
   than exact, because a count in a comment is a claim nothing checks and every commit invalidates it.
   The general rule this session keeps arriving at: **do not write a number where a checked number would
   do, and do not write an exact number where the thing it counts changes weekly.**

   **A third reference class was scanned — paths — and it is faithful too, and deliberately not
   automated.** Every backticked `eval/…`, `tools/…`, `docs/…` path in the tracked documents was
   resolved: **zero genuine stale pointers**. The seventeen that failed to resolve are all one of three
   legitimate kinds — a report a *running* job will write (`eval/beir/nq-dense.json`), a tool a *plan*
   proposes (`docs/generation-placement.md`'s `tools/kilt.py`, `tools/lotte.py`, … in a section headed
   "proposed"), or a **historical** mention (`eval/beir/p1-cqadupstack-android.json` is named in the
   sentence that explains why it was moved to `eval/beir/subsets/`). The last one is the instructive
   case: a checker cannot tell a dead pointer from a sentence about a dead pointer, and automating this
   class would need a proposed/historical marking convention that the documents do not have. **A
   checker that guesses intent is worse than no checker**, which is why the flag and `make` checks exist
   and this one does not.
2. **The rest of the document is still transcription, and registering it is a separate job.**
   `grep -c SCORE-LIFT tools/consistency_check.py tools/claim_check.py` is no longer **0 and 0** —
   `consistency_check.py` matches — but the ~30 quantities *outside* §1 and §4.3 are still
   unanchored. The order to do them in is unchanged: the TREC RAG 2025 retrieval-task figures
   (§P10a) next, since those are the ones a reader is most likely to act on, then the P1/P4 bars
   quoted from `RANKINGS.md` and `docs/findings-*`. A `claim-check` `Claim` still needs its pinned
   count updated in three documents when one is added, which is why these anchors went into the
   consistency registry (additive, no pinned count) rather than into `claim-check`.

**A gate can be green because it passed or because it had nothing to do, and the difference has to be
said.** `make audit-verify` exits 0 today with *"no log at eval/audit.jsonl (nothing to verify)"* — the
log is written by running the engine with `LAYA_RAG_AUDIT_LOG=eval/audit.jsonl`
(`docs/findings-audit-log.md`), no target produces it, and none exists on disk. So that gate is
**vacuously green**, not satisfied. Everything else in the ledger below was checked against an artefact
that exists: the diagnostics report for `claim-check`'s 65, 590 scanned reports for the leakage audit,
the documents themselves for the rest. A reader told "all gates are green" deserves to know which of
them merely had no work.

**And "vacuous" is not "suspect" — that verifier is tested, including the case that matters.**
`tests/test_audit.py` builds a log through `AuditLog`, verifies it (exit 0), then **overwrites it with a
tampered record and asserts exit 1**; separately it tests that editing, removing and reordering records
each break the chain, and that a missing log is skipped rather than failed. So `audit-verify` is *idle on
this checkout*, not broken — which is a different thing from the leakage audit, whose rule was not
running at all when it was finally looked at. The distinction is worth the sentence: the first kind
means "run the engine with `LAYA_RAG_AUDIT_LOG` and this gate has work to do", the second meant "this
safety net has had a hole for months".

**The verification ledger, so the next session does not re-do it.** Eight classes of claim in this
repository have been checked, each by the method named. Six are automated; the two that are not have
their reasons written down, because *a checker that guesses intent is worse than no checker*:

| class | result | how |
| --- | --- | --- |
| numeric claims against their reports | live; **one hole found and fixed** (the summary blockquote had no site) | `make consistency-check` (47 sites), `make claim-check` (65 checks) |
| query dilution in reports | a planted signature is caught | `make leakage-audit --check` |
| a patch that doubled its own text | one found in the tree, repaired | `tests/test_docs_no_doubled_text.py` |
| a flag parsed and never read | **five found, all resolved** (three removed, one wired, one reclassified as a default-mode alias) | `tests/test_tools_flags_are_used.py` |
| a documented `make` target that does not exist | one real (`charts-more`), fixed; four documented as *proposed*, listed | `tests/test_docs_make_targets_exist.py` |
| a documented tool flag that does not exist | **zero** | `tests/test_docs_tool_flags_exist.py` |
| documented paths that do not resolve | **zero genuine** — the seventeen are pending reports, proposed tools, or a historical mention | **not automated**: a checker cannot tell a dead pointer from a sentence about one |
| the documented configuration surface | **five knobs were missing** (`SECOND_STAGE` among them); now zero | **not automated**: a completeness check would have to decide which internal knobs deserve a row |
| every test file is actually collected | **zero silently uncollected** — 140 test modules collected, and the three that are not are `__init__.py`, `support.py` and the `real_weights`-marked file | one command: the collected modules against `tests/*.py` |

Two of the six automated checks produced false positives on their own first run — nineteen English
"make up" hits, and four single-quoted flags read as undeclared. **Checking a checker is a different job

The twelve instances behind this table, each with what it looked like, how it was caught and
what catches it now — plus what the classes cost in commits, counted rather than remembered — are
collected in **[findings-measurement-failures.md](docs/findings-measurement-failures.md)**. This
table is the summary a maintainer needs; that page is the one to hand a new contributor.
from running it**, and it is now the job this section exists to make routine.

### 10.4 What could invalidate this document, and how it would look

**A document-level falsifier is in §10.2, and it is the only one that retires the framing rather
than an item.** Separately, several external dependencies could rot — this project has already been
burned once, when the UKP mirror 404'd four BEIR members and the "blocked" verdicts turned out to be
about a dead link rather than a licence.

| dependency | how it rot | mitigation |
| --- | --- | --- |
| **TREC RAG retrieval task** (§P10a) | the track could change or drop the R-only task, as it added it in 2025 | the results are **published proceedings** — the 43 runs stay citable even if the track changes |
| **MTEB v2 `SearchProtocol`** (§P10b) | v2 was already a large refactor; a v3 would stale `tools/mteb_search.py` | pin the mteb version in the eval extra and record it in the report |
| **Benchmark mirrors** | already happened: Robust04 / TREC-NEWS / Signal-1M went 404 and were recovered from a third party | the recovered corpora are on disk; a re-fetch plan should be recorded with them |
| **Gated checkpoints** | `google/embeddinggemma-300m` returns *"You are trying to access a gated repo"* — verified on this host | prefer ungated Apache-2.0 alternatives (Qwen3-Embedding, bge-m3, arctic-embed) |
| **This host's HF cache** | the **default** cache directory is **not writable** here — a plain `AutoConfig.from_pretrained` fails with `PermissionError` until `HF_HOME` is redirected; verified during §P3's probe | every tool that loads a model must set `HF_HOME` explicitly, or the failure looks like a missing model |
| **The torch pin** | `torch` 2.2.2 on macOS Intel means `transformers` cannot `torch.load` `.bin` checkpoints, and can never upgrade past the CVE-2025-32434 fix | this is why the sparse channel uses a `safetensors` encoder, and why new models must be checked for `safetensors` **before** they are planned around |

**The last three are environmental, and they are the ones most likely to waste a week**, because
each presents as a modelling failure rather than an infrastructure one.

### 10.5 Where the programme stands — the record of what has been run

This section is **live status**, not planning. Every row below is a measurement that exists, with the
finding page that carries it; a reader picking this document up should be able to tell what is done
from what is aspirational without reading thirty sections. The bars are in §10.2.

**W0 — hygiene: closed, four for four, and each one found something.**

| item | state | what running it returned |
| --- | --- | --- |
| `make channel-coverage` in the gate | done | §1.1 verbatim + the §1.2 arithmetic pinned; checked sites **20 → 43 → 47**. It immediately falsified §1.2's `ours` column on **nine of eighteen** cells — the macro is **0.4799**, not 0.4760, and the deficit **0.7085 / +0.020 / +0.035** |
| `--limit-queries` dilution | closed | the five tools were already fixed; **the audit proving it had stopped running** (an uncaught `UnicodeDecodeError` on a 24 GB latin-1 dump). Fixed; **590 reports scanned, 142 exclusions named** |
| `jobkit` fingerprints vs the index | done | each engine index records an `index-identity.json`; a changed chunk size rebuilds and refuses stale rows, with legacy manifests rebound rather than discarded |
| harness/engine BM25 divergence | done | the quoted parity was **0.7408 / +0.0144**, not the 0.7413 / +0.0149 an over-hasty "correction" produced — the two differ by `--title-boost`. Pinned to `eval/beir/engine-parity.json` |

**W1 — free wins: closed. Every item measured, two of them negatively.**

| item | state | what running it returned |
| --- | --- | --- |
| **P2a** lexical selector | measured | picks are `bm25-title` ×11, `bm25-engine` ×6, `bm25` ×1 — so the first falsifier is **refuted**. Oracle **+0.0109** on the lexical-only macro (`findings-lexical-selector.md`) |
| **P2b** BM25F | mechanism done; **FEVER's row now waits only on the lock** | the proxy was not the problem: optimum weight is corpus-dependent (3, 3, 12, 12, flat) and ArguAna takes **+0.0132** over repetition; macro 0.3612 → 0.3651 → **0.3687**. The blocker was never the idea but the index: `ondisk.OnDiskBM25F` now composes one per-field index with the union DF accumulated at build time, and `--bm25f --lexical-backend ondisk` reaches it, so FEVER needs a flag and the lock rather than a capability (`findings-bm25f.md`) |
| **P5a** tuned convex | measured | held-out, against the shipped `k=5`: **3 significant wins, no losses, 3 ties**; a global weight of 2.0 gives **+0.0127**. Engine-level first-stage-only: convex 0.7331 vs RRF 0.7269, **+0.0062** |
| **P10b** MTEB | **scored locally; nothing uploaded** | the block was our own flag — `--first-stage-only` printed a banner and set nothing, so every run still paid the 322M decision per candidate. Fixed and confirmed: index 183 s, search **0.21–1.05 s/query**, no rerank frames, no 282,984-token warning. MTEB SciFact `main_score` **0.37** (multilingual decision encoder) → **0.69** (`models/laya-retrieval`), i.e. nDCG@10 **0.69482** against published BM25 0.665 and `bge-small` 0.72 (`findings-mteb-search-protocol.md`)  **Like-for-like under MTEB** (baseline flag added, since reading an MTEB row against BEIR numbers is a cross-protocol claim): SciFact `bge-small` **0.71271** vs engine **0.69482**; NFCorpus **0.34339** vs **0.33318**, ArguAna **0.60413** vs **0.51630**, SCIDOCS **0.20518** vs **0.16251**, FiQA2018 **0.40347** vs **0.26067**. The quality gap ranges from **2.5% (SciFact) to 35% (FiQA2018)** and **does not track document length** — my three-task reading of that was wrong, and SCIDOCS is the counterexample (short documents, a gap four times SciFact's). The two where the lexical channel carries the engine sit at 2.5–3.0%; the three where relevance is citation-, argument- or opinion-based sit at 14.5–35%. That is a hypothesis, stated as one, and FiQA2018 *joined* the large-gap group rather than breaking it — one surviving prediction, not evidence. **What all five tasks agree on is the cost**: 3.3–7.4× the baseline every time, so the engine is neither competitive on quality nor comparable on cost. **The harness's ~33-worker fork was real** (re-counted: 101 helpers from one launch, which is why indexing read 411 s against 183 s on a quiet host); only the inference that it deadlocked was wrong. **All five tasks are complete** (SciFact, NFCorpus, ArguAna, SCIDOCS, FiQA2018): quality gap 2.5%–35%, cost 3.3–7.4×, nothing uploaded |
| **P12** extractive end to end | runs 1–2 measured | 2,255 MultiHop queries: **citation clean 1.0000, 0 uncited of 9,806**, nugget recall **0.1906**. FRAMES: recall@10 0.9853, nugget recall **0.1922** — the same completeness in the opposite regime (`findings-extractive-row.md`) |
| cross-cutting | done | `SECOND_STAGE=0` returns the first stage's own order with no decision model: ~10 min → **37 s** per configuration (`findings-second-stage.md`) |

**W2 — started, one clause closed and one measured.** **P7 is closed, negatively**: passage BM25
loses on Robust04 (−0.0555) and TREC-NEWS (−0.0895), and the loss grows with document length, so a
fixed cut is refuted at every length tested. Of its second clause, **aggregation is measured and also
negative** — CombSUM over 600-character pieces reads 0.2428 for Robust04 (−0.1094 against max-pool)
and 0.2719 for TREC-NEWS (+0.0263), and in neither case reaches the whole-document row (0.4077 /
0.3351). The **adaptive budget** is implemented, unit-tested and **not measurable here at this scale**:
three attempts, no report, because a `max(200, ceiling/4)` per-document budget nearly doubles the
passage count on a 528,155-document corpus on top of the whole-document index every run builds. It
is not refuted — it does not fit this host's budget, which is a stated reason rather than a result
(`findings-passages-long.md`).

**P4 is measured, and it produced a defect as well as a result.** The task was one flag — a modern
drop-in reranker — and it opened with two different rerankers producing **byte-identical** rankings.
The cause was one line: `rerank(order="recall")` **sorted by `recall_score`**, which re-derived the
pre-permutation order and discarded every coverage decision, so a configured cross-encoder could not
affect the result and **18 recorded reports were their first stage's order wearing a reranker's
name**. Fixed to preserve the order it is handed (3 tests; a λ=0 control reproduces its recorded
number exactly, 0.7091). What the re-takes then measured, in both directions:

| | recorded | re-taken | Δ |
| --- | ---: | ---: | ---: |
| MultiHop-RAG MAP@10, `bge-reranker-v2-m3`, **full 2,255 queries** | 0.3190 | **0.3702** | **+0.0512** |
| MultiHop-RAG MAP@10, `bge-reranker-base`, **full 2,255 queries** | 0.3190 | **0.3294** | **+0.0104** |
| SciFact nDCG@10, `ms-marco-MiniLM`, λ auto (×2 configs) | 0.7274 / 0.7417 | 0.6762 / 0.6422 | −0.0512 / −0.0995 |

So the same defect was hiding a **helpful** reranker on MultiHop (+0.0512 MAP@10, Hits@4 +0.0785,
MRR@10 +0.1043, against a published 0.480 that remains ~0.11 away) and a **harmful** one on SciFact
(where the engine's real number is the fused 0.7417, not any CE ordering). Neither direction was
knowable while the order could not change. The base arm at full protocol and the listwise arm are
the two runs still outstanding.

**Not started, and why — the honest remainder, so a reader does not have to diff this page against
its own findings.**

| item | state | cost |
| --- | --- | --- |
| **P4 arm 2** | **measured, and negative** | the listwise `Qwen3-Reranker-0.6B` reads **MAP@10 0.2047** against **0.3721** for the pointwise `bge-reranker-v2-m3` on the same 200-query sample — a 45% relative drop and below the fused order it was meant to improve. §P4's hypothesis is falsified for this configuration: one model, one corpus, 200 queries, with the model's compressed score scale recorded as a suspicion rather than a finding |
| **P1** — the eight lexical-only cells | **2 of 8 measured, rest queued** | `cqadupstack` is in (`hybrid-biencoder` **0.3275** against the 0.3046 §1.2 was scoring — the summed deficit fell **0.7314 → 0.7085**) and **`nq` is in too, without moving it**: hybrid **0.3212** against the engine's 0.3118, but NQ's recorded best is already 0.3264 (`bm25-title`), so the member's `ours` value does not change. Coverage: lexical-only members **8 → 6 of 18**. Next: signal1m, dbpedia-entity, hotpotqa, climate-fever, fever, bioasq — largest last, one command each in §10.6 |
| **P3 / P11** | unstarted | week-scale research bets |
| **P8 / P9** | unstarted | a published-settings control on 5.4M documents; a fresh out-of-domain corpus |
| **§P12 run 3 / T8+T11** | blocked on a build | the own-domain set is a separate tool, and run 3 depends on it |
| **on-disk `BM25F`** | **complete, wired, parity-checked; FEVER's row waits on the lock** | `ondisk.OnDiskBM25F` composes one `OnDiskBM25` per field (fields are query-time weights, so a sweep needs no rebuild) and accumulates the union DF at build time — the count is **not** derivable from the per-field DFs (`max` is a lower bound, the sum an upper). **27 tests**, two of them parities: one field with `b=0` reproduces `OnDiskBM25`'s scores exactly, and the on-disk row reproduces `bm25f_rankings` exactly through both the adapter and the CLI |
| **P6b**, Touché-2020 in the fusion sweep | unstarted | bounded; Touché's corpus build was stopped once already |
| **the `reranker` rows' re-takes beyond SciFact** | partial | SciFact's three are done with a control; the MultiHop pair is done full-protocol; nothing else names a reranker |

**P4 is measured on the full protocol**: with the ordering defect fixed, `bge-reranker-v2-m3` reads
**MAP@10 0.3702, Hits@4 0.8417, MRR@10 0.7026** over all 2,255 queries, against the recorded 0.3190 /
0.7632 / 0.5983 — **+0.0512 / +0.0785 / +0.1043** on the system's worst published comparison, and
still short of the published 0.480.

### 10.6 The queued runs, as commands

Every measurement the programme is waiting on needs the heavy lock, so they queue. This is the queue as
executable lines rather than as a list of intentions — each was specified or built in an earlier round,
and each carries the trap that round found. **One at a time**; check `make heavy-status` first.

**P1's remaining members** (largest last; the next one is `signal1m`). `nq`'s command is the pattern:

```
# CQADupStack — done, member row in eval/beir/cqadupstack-dense.json
.venv/bin/python tools/cqadupstack.py --retrievers biencoder,hybrid-biencoder \
    --dense-weight 0.3 --embed-chunk 5000 --out eval/beir/cqadupstack-dense

# nq — RUNNING. Report lands at eval/beir/nq-dense.json when the 2.68M-document
# embedding finishes, ~13 h from a 20:22 start.
.venv/bin/python tools/beir.py --datasets nq \
    --biencoder-model models/laya-retrieval --dense-weight 0.3 \
    --retrievers biencoder,hybrid-biencoder \
    --lexical-backend ondisk --embed-chunk 20000 --out eval/beir/nq-dense

# then signal1m, dbpedia-entity, hotpotqa, climate-fever, fever, bioasq — same flags,
# one --datasets name each, its own --out. ondisk and embed-chunk are not optional.
```

Two traps: **the on-disk lexical backend is required** above ~1M documents (the memory index keeps a
`Counter` per document), and **dense rows only** keeps the lexical selector from seeing a second
`bm25-engine` for the member. A run with `--limit*`, `--subforums` or a seed writes to
`eval/beir/subsets/` instead — see §10.1.

**What "healthy" looks like, checked mid-flight on `nq`** so a later reader can tell normal from
broken without re-deriving it: the log's last line is a `N/2681468 (Xs)` counter — measured at **65.7 documents per
second** (340,000 documents in 5,176 s), so a million documents is about 4.2 hours and the member's
2.68M is **~11 hours**, which is what the counters show: a quarter of the way in, ten hours to go — the process appears in `make heavy-status`, the
embedding cache grows to **34 GB** for the whole member, and the volume has **357 GB free**. Zero
matches for `error`, `Traceback` or `no space` in the log. Disk is the failure that would waste a
night silently, which is why it is the first thing to look at on a multi-hour job, not the last.

**And a pile of processes is not automatically a runaway — check the parent.** `pgrep -f beir.py`
against this run returns **33 pids**, which looks alarming and is the embedding's own worker pool:
`pgrep -P 30375` shows **32 direct children**, all alive under the run that owns the lock. The
sixty-seven-process incident earlier in this session was a different animal — orphans whose *wrappers*
had been killed, each holding memory for a job nobody was waiting for. The discriminating question is
therefore **"whose children are they?"**, not "how many are there?", and it is worth asking in that order
because the second question is the one that panics an operator into killing a healthy run.

**And the checkpoints are real, which is what makes the resumability claim worth anything.** The run
writes `eval/beir/cache/nq.laya-retrieval.mean.parts/`, and the directory carries **part files dated
within the last few minutes** — so `--embed-chunk 20000` is doing what it says and a kill costs the block
in flight rather than six hours of embedding. Verified by looking at the filesystem rather than by
trusting the flag, for the same reason every other claim in this record was: **a flag being read is not
the same as a flag doing its job.** (The check that a flag is *read* exists;
`tests/test_tools_flags_are_used.py`. That it *works* is only ever visible in the artefacts.)

**And one link out of 195 pointed a directory above the repository root.** The sweep is now eleven
classes; this one is as crisp as any and had exactly one instance: `docs/findings-multihop-diagnostics.md`
cited `../../eval/benchmarks/multihop-rag/ceiling-decomp.md`, so the report was two levels away from
where a reader would look and the link resolved nowhere. Fixed, and
`tests/test_docs_links_exist.py` checks all 195 relative links, skipping `eval/` targets when `eval/` is
absent — the same absent-artefact convention the leakage audit and `claim-check` use.

**`nq`'s report has landed, and the honest result is that it does not move the deficit.** All
2,681,468 documents embedded in 37,342.9 s (10.4 h), twelve hours after the first one:

| nq, 2.68M documents | nDCG@10 | nDCG@100 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| `bm25-engine` | 0.3118 | 0.3746 | 0.4800 | 0.2693 |
| `biencoder` (dense alone) | 0.2231 | 0.2933 | 0.3546 | 0.1935 |
| **`hybrid-biencoder`** (dense w=0.3) | **0.3212** | **0.3939** | **0.5159** | **0.2741** |

The fusion beats the engine row by **+0.0094 nDCG@10** and dense alone loses badly (−0.0887) — the same
shape as CQADupStack, smaller in size. But **NQ's recorded best is already 0.3264** (`bm25-title`), so the
new row does not become the member's `ours` value and **the summed deficit stays at 0.7085**. `nq` moves
from lexical-only to dense in §1.1 (the generated table above), which is a coverage gain and not an
arithmetic one — and that distinction is exactly what §1.2 is for.

CQADupStack moved the deficit because its best row was a plain `bm25-engine`; NQ's was already title-boosted
and 0.005 ahead of the fusion. **Two cells in, one moved the macro and one did not**, which is the useful
shape of the answer for the remaining six: the dense channel is worth ~+0.01–0.04 over the *engine* row,
so it moves the arithmetic only where the member has no title-boosted row ahead of it.

**The third member is running, and the progress instrument added for it is working in production.**
`signal1m` — **2,866,316 documents** — is embedding now (`--out eval/beir/signal1m-dense`,
`--lexical-backend ondisk`, `--embed-chunk 20000`). Two things worth recording from its first minutes:

Its **measured rate is 228 documents per second** — 3.5× nq's 66 — so the member is **~3.5 hours**,
where a document-count estimate gives ~12. The unit is doing the work again: I have now estimated
this harness's cost from *document counts* twice and been wrong by 4× (nq's ETA) and 3.5× (this one),
because the cost is per token. **Read the counter, and check what it is counting.**

* **the on-disk build reported as it worked** — `on-disk index: 1,000,000/2,866,316 documents (35%)`,
  then `2,500,000 (87%)`, then it was done. That line is the fix from round 70, and this is the first
  member larger than the one that motivated it to use it in anger: **before that fix the same phase
  printed one line and then nothing for half an hour**, which is what made a working run look hung;
* **Signal-1M is cheap to index because its documents are tweets.** The same build that took **29
  minutes for nq's 2.68M long documents** finished here in a few minutes for 2.87M short ones, and the
  cost is per token rather than per document — a reminder that "documents" is the wrong unit for
  estimating this harness, which every one of these estimates has quietly assumed.

**When a member's report lands**: `make channel-coverage` regenerates §1.1 (the gate will demand it —
CQADupStack's move took the lexical-only count 8 → 7 and the deficit 0.7314 → 0.7085), the consistency
check names every stale figure, and the deficit test is derived so it needs no edit.

**P4 arm 2** (`Qwen/Qwen3-Reranker-0.6B`, listwise). Weights are already in the writable cache, and the
path is validated:

```
HF_HOME=$PWD/eval/hf-rerank .venv/bin/python tools/engine_multihop.py \
    --out eval/benchmarks/multihop-rag/engine-p4-listwise \
    --rerank-model Qwen/Qwen3-Reranker-0.6B --limit 200 --sample-seed 0
```

`HF_HOME` matters: the shared `~/.cache/huggingface` is not writable, and the failure is a
`PermissionError` that looks like an intermittent cache lock. No threshold tuning is needed — the
multihop and beir runners set `LAYA_RAG_RELEVANCE_THRESHOLD=0.0` themselves. Compare against the
pointwise ceiling **0.3702** on the same subset, and re-take the subset for the base reranker in the
same session if the comparison matters.

**FEVER's BM25F row** (P2's falsifier, unrun branch):

```
HF_HOME=$PWD/eval/hf-rerank .venv/bin/python tools/beir.py --datasets fever \
    --retrievers bm25-engine,bm25-title --title-boost 2 --bm25f \
    --lexical-backend ondisk --out eval/beir/p2-fever-bm25f
```

`--bm25f` fuses the title-boosted channel, so `bm25-title` must be requested with it (the tool now
says so instead of raising `UnboundLocalError`). The index is cached under `eval/beir/cache/fever.bm25f`
and **reused** across `--title-weight` values, which is what makes the sweep affordable — run one
build, then sweep.

**The README's rerank-coverage figures are part of the re-take, and that is now a named job.**
`RERANK_COVERAGE_LAMBDA`'s default is justified in `README.md` by *"full-protocol MAP@10 0.2948 →
0.3165, Hits@4 0.7348 → 0.7694"*, all of it measured through the broken ordering path. Those two
columns are flagged as suspended in the README itself; **the run that reinstates or drops them is the
one that re-takes `RERANK_COVERAGE_LAMBDA` at 0.0 and 0.7 on MultiHop with the ordering fixed**. Until
it runs, the default stands as *unjustified* rather than *justified* — which is a different thing from
*wrong*, and the README now says which.

**The open question from last round is now answered by reading the reports, and the answer is not the
one either case predicted.** `README.md`'s `RERANK_ORDER` row says *"SciFact 0.654 (was 0.470
reordered)"*. Searching both report trees for the two halves:

* **0.654 reproduces exactly** as SciFact `bm25` nDCG@10 = **0.6539** — the plain lexical row, with
  `rerank_model: null`. So the first half is the recall order, as the row claims.
* **0.470 reproduces from nothing with a reranker configured.** No SciFact report with a non-empty
  `rerank_model` reads anywhere in 0.44–0.50. The only rows in that band are `adapted-runA.json`'s
  **`dense`** (0.4711) and `adapted-ensemble-AB.json`'s `biencoder` (0.4711) and `dense-ensemble`
  (0.4629) — **dense retrievers, not reranked lexical rankings**, which is a different claim wearing a
  similar number.

So the parenthetical's second half is **not reproducible as a reranking figure**, and the question was
never "λ=0.7 or λ=0" — it is whether the number was ever a reranking measurement at all. It is recorded
that way, with the two rows that do read 0.47 named so the next reader does not re-run this search. The
row's *conclusion* (`auto` keeps recall order unless a scorer is supplied) may still be right; its
*evidence* is a number that means something else.

**Which README rows the defect touches, and which it does not, because over-suspending has a cost.**
Three rounds of flags risked the opposite of the problem they fixed — a reader who finds every reranking
row marked "suspended" stops trusting the sound ones. So the scope is stated rather than implied:

| README row | affected? | why |
| --- | --- | --- |
| `RERANK_COVERAGE_LAMBDA`, `RERANK_COVERAGE_MIN_DOCS` | **yes** | both are differences *in ordering*, and the ordering was re-sorted by the first-stage score afterwards |
| `RERANK_THRESHOLD` (+0.0801 MAP@10) | **no** | a threshold **filters** — `above` is built from the cross-encoder's scores *before* anything is ordered, and the defect only re-sorted after that point |
| `RERANK_ORDER`'s second figure | **not a reranking number at all** | the 0.470 is a dense retriever's 0.4711, as the search above shows |
| everything else (BM25F, selector, fusion, extractive, MTEB) | **no** | no cross-encoder was configured, so the branch never ran |

**The re-takes beyond SciFact.** The other twelve reports that configured a reranker and a coverage λ
above zero (`eval/benchmarks/multihop-rag/rerank-*.json`) are the evidence §P4's premise rested on, and
they were taken before the ordering fix. Re-taking the full-protocol ones is the way to find out whether
"a pointwise cross-encoder loses" was ever measured.

### 10.7 The decision this document cannot make for you

§1–§6 and §8–§9 are **different kinds of work wearing the same document**. The first is *benchmark
placement*: it raises numbers on corpora that already exist and is measurable within weeks. The
second is *product capability*: it decides which corpora the system can accept at all, and it pays
off in a different currency — a paper-retrieval user, an enterprise deployment, a multilingual
corpus — none of which shows up on the BEIR-18 mean.

Both are defensible. They are not equally urgent, and the mix is not a technical question. **What
this document can say is the price of choosing:** if the answer is "score", do W0–W4 and stop, and
the system has a defensible public row in a quarter; if the answer is "product", start with C1 and
B2a, and the BEIR-18 mean does not move at all in that quarter. **Doing neither is the only option
this document argues against**, because the measurement machinery that makes either cheap is
already built and is currently idle.

---
