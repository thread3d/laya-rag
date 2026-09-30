# Where the retrieval stands, and the plan to move it

This is the working document behind the BEIR work in `tools/beir.py` and
`tools/beir_ablate.py`. It records the measured standing, the programme to improve it, and —
for the parts that cannot be finished in one sitting — what they concretely require.

Everything here is produced by `make beir` / `make beir-ablate`; the raw reports are in
`eval/beir/` (gitignored). Numbers are nDCG@10 unless stated.

## Standing

Published baselines are from the [BEIR paper](https://arxiv.org/abs/2104.08663) and the BAAI
BGE paper, compiled in an independent [BEIR matrix](https://github.com/ruvnet/ruflo/blob/56819391/docs/benchmarks/BEIR-MATRIX.md).
They are 2021–2023 numbers from other harnesses, so they are indicative rather than a
controlled head-to-head; our `bm25` control sitting near the published BM25 baseline is the
anchor that keeps the evaluator honest.

**Five datasets, nDCG@10:**

| dataset | us, shipped lexical | us, bi-encoder | us, bi-encoder + fused BM25 | BM25 (published) |
| --- | --- | --- | --- | --- |
| SciFact | 0.676 | 0.720 | **0.741** | 0.679 |
| NFCorpus | 0.324 | 0.339 | **0.367** | 0.325 |
| ArguAna | 0.313 | **0.429** | 0.414 | 0.397 |
| SciDocs | 0.150 | 0.197 | **0.199** | 0.158 |
| FiQA-2018 | 0.239 | **0.385** | 0.382 | 0.236 |
| **mean** | **0.340** | **0.414** | **0.421** | **0.359** |

**Where that places us.** On the four datasets where the published table carries every system
(SciFact, NFCorpus, ArguAna, SciDocs):

| system | 4-dataset mean |
| --- | --- |
| BGE-large-v1.5 (335M) | 0.491 |
| **us — best per dataset** | **0.434** |
| SPLADE++ | 0.433 |
| BM25 (Lucene) | 0.390 |

So: **third of four, level with SPLADE++, 0.057 behind a 335M BGE-large** — not the
"second of two datasets" the earlier two-dataset slice suggested.

After the tuning and model sweeps below, the four-dataset **best-per-dataset** mean is **0.448**:

| dataset | best configuration | nDCG@10 |
| --- | --- | --- |
| SciFact | `bge-large` + fused BM25, `k=5`, title boost | 0.750 |
| NFCorpus | `bge-large` + fused BM25 | 0.382 |
| ArguAna | `bge-small` dense on text-only documents | 0.459 |
| SciDocs | `bge-small` + fused BM25 | 0.199 |

That is **second of four**, above SPLADE++ (0.433) and 0.043 behind BGE-large (0.491) — the gap
narrowed from 0.057. The caveat is that it cherry-picks a configuration per dataset: a single
deployable configuration (`bge-small` + fused BM25) is 0.430, which is level with SPLADE++ rather
than ahead of it.

Broadening the evaluation *lowered* the standing, and that is the useful result. Two things it
exposed:

- **ArguAna is where dense quality dominates.** Published BGE-large scores 0.636 there against
  our 0.429 — a 0.21 gap on one dataset, which alone accounts for most of the distance between
  us and it. ArguAna is symmetric argument retrieval, so a stronger encoder is the lever.
- **Fusion does not generalise the way two datasets implied.** The tuned fusion beats the
  bi-encoder on SciFact (+0.021) and NFCorpus (+0.029) but loses on ArguAna (−0.015), FiQA
  (−0.003) and is flat on SciDocs (+0.002). It still wins on the five-dataset mean (0.421 vs
  0.414), but per-corpus A/B testing is not optional.

Caveats unchanged: five of eighteen datasets, no confidence intervals, `bm25` control slightly
below published Lucene (analyzer differences), and a score produced by `bge-small` + stemmed
BM25 + RRF rather than by Laya's encoder or reranker. Laya's contribution — calibrated
abstention — has no leaderboard counterpart.

## The programme

| # | work | expected | effort | status |
| --- | --- | --- | --- | --- |
| 1 | Fusion and BM25 tuning: RRF `k`, channel weights, title-boosted BM25F, passage indexing | +0.005–0.02 | hours | done (RRF `k=5`, title boost; passage indexing rejected) |
| 2 | Broaden evaluation to five datasets (add ArguAna, SciDocs, FiQA-2018) | — (credibility) | hours | in progress |
| 3 | Stronger dense model (`bge-base` measured; `bge-large` deferred) and reranker (`bge-reranker-base` cached, not measured) | +0.01–0.03 | hours–days | partly done |
| 4 | Passage-level indexing for long documents (TREC-COVID, Robust04) | needed there | days | harness ready, datasets deferred |
| 5 | Scale: inverted index (**done**), ANN index (**planned**) | prerequisite | days–weeks | partly done |
| 6 | Learned sparse (SPLADE) or document expansion (docT5query) as a third channel | +0.02–0.05 | weeks | planned |
| 7 | Domain adaptation / fine-tuning on in-domain pairs (no longer zero-shot) | +0.02–0.05 | weeks | planned |

### Scale work landed: an inverted index for BM25

Lexical scoring looped over every document for every query term — O(terms × documents) — which
is what made the five-dataset run stall before it started: ArguAna queries are ~190-word
arguments, so one query set was billions of iterations. `BM25Index` now keeps
`term -> [(row, frequency)]` postings and touches only the documents that contain each term,
which is the index Lucene keeps. ArguAna went from stalling past 50 minutes to 4.3 minutes for
the full query set, and the ranking is unchanged (pinned by a save/load equivalence test). The
ANN index for the dense channel is still outstanding, and is the remaining blocker for the
million-document datasets.

Items 5–7 are not "more of the same": each needs infrastructure or training that cannot be
done responsibly in a single session, and they are specified below rather than attempted
half-way.

### Model size — the measured headroom

Swapping `bge-small` (33M) for `bge-base` (110M), same tuned fusion:

| dataset | bge-small dense | bge-base dense | bge-small fusion | bge-base fusion |
| --- | --- | --- | --- | --- |
| SciFact | 0.720 | 0.738 | 0.741 | **0.747** |
| NFCorpus | 0.339 | 0.371 | 0.367 | **0.377** |
| ArguAna | 0.429 | 0.453 | 0.414 | **0.432** |
| 3-dataset mean (dense alone) | 0.496 | **0.520** | — | — |

A bigger encoder is worth **+0.025** on the three-dataset dense mean and every individual
dataset improves. It also changes fusion's value: with `bge-base` the dense channel is the
stronger one and fusion stops helping (0.520 dense against 0.519 fused), which is the same
"fusion is only as good as its weaker input" effect seen in the other direction earlier.

The ceiling is still visible above this. Published BGE-large scores 0.636 on ArguAna against
`bge-base`'s 0.453, and ArguAna alone accounts for most of the distance to the published
leaderboard, so **encoder size on ArguAna is the highest-value remaining item**. `bge-large`
(335M, ~2h to embed the five datasets here) was not run; `bge-reranker-base` was cached but not
measured, since the fast `ms-marco-MiniLM` cross-encoder had already been beaten by fusion.

### Experiments tried, measured, and rejected

Recorded so they are not retried from intuition:

| experiment | SciFact | NFCorpus | verdict |
| --- | --- | --- | --- |
| `bm25-title` (title repeated twice) | 0.682 | 0.325 | **kept** |
| `bm25-prf` (10 feedback docs, 15 terms ×2) | 0.424 | 0.274 | **rejected** — the expansion drifts the query badly |
| `bm25-passages` (600-char passages, max-pool) | 0.659 | 0.318 | rejected at this document size; kept for long docs |
| `hybrid-combsum` (score fusion, max-normalised) | 0.712 | 0.355 | rejected vs RRF's 0.741 / 0.367 |
| `bge-large` instead of `bge-base` | 0.750 | 0.382 | **no material gain** for 3× the embedding cost |

The PRF result is the starkest: a textbook technique, implemented plainly, cost 0.26 nDCG@10 on
SciFact. It assumes the first pass is mostly right and these queries are short enough that it is
not. CombSUM lost to RRF because BM25's and cosine's score scales do not normalise cleanly against
each other, which is exactly the problem RRF exists to avoid. And `bge-large` confirms the model
size *saturates* at `bge-base` on this task — ArguAna barely moved (0.4529 → 0.4547) for a model
3× the size, so bigger is not the lever it looked like.

### Document text is a per-channel choice, not a global one

ArguAna's `title` field is a shared topic keyword list (`animals environment general health …`)
repeated across a cluster, and BEIR's convention of indexing `title + text` puts those near-identical
terms into every document of the cluster:

| ArguAna representation | BM25 | dense (`bge-small`) |
| --- | --- | --- |
| `title + text` (BEIR convention) | **0.295** | 0.429 |
| text only | 0.270 | **0.459** |

The channels disagree: titles are high-precision lexical signal for BM25 and noise for a dense
encoder. That is why `--dense-text-only` exists — the dense channel embeds text alone while BM25
keeps `title + text`.

## Tuning results

RRF's constant `k` is normally left at 60, the value from the original RRF paper. Measured
here it is far from optimal — the two channels are of comparable strength, so weighting the
top ranks harder wins. Swept on SciFact and NFCorpus with the title-boosted lexical channel
and `bge-small` dense:

| RRF `k` | SciFact | NFCorpus | mean |
| --- | --- | --- | --- |
| 3 | 0.7402 | 0.3654 | 0.5528 |
| **5** | **0.7413** | **0.3673** | **0.5543** |
| 7 | 0.7404 | 0.3656 | 0.5530 |
| 10 | 0.7390 | 0.3648 | 0.5519 |
| 30 | 0.7330 | 0.3623 | 0.5477 |
| 60 (default) | 0.7310 | 0.3620 | 0.5465 |
| 100 | 0.7304 | 0.3609 | 0.5457 |

`k=5` is the optimum on both tuning datasets and is validated on the other three below.

Two BM25 variants were measured on the same pair:

| variant | SciFact | NFCorpus |
| --- | --- | --- |
| `bm25-engine` (stemming, stopwords dropped) | 0.6762 | 0.3244 |
| `bm25-title` (title repeated twice) | **0.6824** | **0.3248** |
| `bm25-passages` (600-char passages, max-pooled) | 0.6592 | 0.3183 |

Title weighting helps slightly and is now the fusion base. **Passage-level retrieval hurts at
this document size** (BEIR abstracts average ~230 words), which is expected: it is a
technique for genuinely long documents, and it stays in the harness for the datasets in item 4
rather than in the default path.

## Deferred work — what each item actually requires

**Item 4, passage-level retrieval for long documents.** The harness already supports it
(`--passages N`, max-pooled). It needs the long-document datasets to be meaningful:
TREC-COVID (171k docs) and Robust04 (528k docs), which are larger than anything run here.
Work: a passage budget that adapts to document length, score aggregation beyond max-pool
(sum or learned weights), and a check that title weighting survives passage splitting.

**Item 5, scale.** Exhaustive search is why only five datasets are measured. For the full
18-dataset BEIR average the infrastructure is the blocker: BioASQ has 14.9M documents, NQ
2.7M, FEVER 5.4M. Concretely:
- an ANN index for the dense channel (FAISS or hnswlib) with a recall check against the
  exhaustive scan at the sizes we can still afford;
- a real inverted index for the lexical channel — the pure-Python BM25 in `lexical.py` is
  already the slowest stage at 40k documents (~120 ms/query), so millions need Lucene or
  Pyserini, or at least an on-disk postings list with compressed term frequencies;
- streaming index construction, because `VectorIndex` currently holds every chunk's text and
  vector in memory.

**Item 6, learned sparse or expansion.** A third channel, not a tweak: SPLADE (`naver/splade-cocondenser-ensembledistil`)
or docT5query expansion. SPLADE measured 0.704/0.347 in the published table, so fusing it is
the largest remaining single gain. Work: model dependency, sparse-vector storage, and a
fusion weight fitted on held-out queries. docT5query is cheaper (a sequence-to-sequence
expansion pass at index time) and is the usual first step.

**Item 7, domain adaptation.** Not zero-shot, and must be reported separately. NFCorpus ships
a 110k-pair train split; fine-tuning `bge-small` on it (or generating synthetic queries
GenQ-style) typically buys 0.02–0.05 on that dataset. Work: a training script, hard-negative
mining, a held-out split that is never used for tuning, and a clear label on the resulting
row so a fine-tuned number is never presented as a zero-shot one.

## Method notes

- Documents are indexed as `title + text`, the concatenation BEIR's own baselines use.
- Only queries with qrels are scored, and queries without a relevant document are skipped,
  which is what `pytrec_eval`/BEIR do.
- Retrieval is exhaustive (no ANN), so runtime grows linearly with corpus size.
- `bm25` is the unstemmed control; `bm25-engine` is the shipped configuration (stemming on,
  query stopwords dropped) and feeds the fused and reranked rows.
