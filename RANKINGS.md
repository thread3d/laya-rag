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
| 4 | Passage-level indexing for long documents (TREC-COVID, Robust04) | needed there | days | measured: hurts; long-document datasets unavailable |
| 5 | Scale: inverted index (**done**), ANN index (**done**, `build_ann`) | prerequisite | days–weeks | done |
| 6 | Learned sparse (SPLADE) as a third channel | +0.02–0.05 | weeks | **done**: +0.004–0.013 fused, +0.03–0.10 standalone |
| 7 | Domain adaptation / fine-tuning on in-domain pairs (no longer zero-shot) | +0.02–0.05 | weeks | **demonstrated**: +0.0085 with 20k of 110k pairs |

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

### Item 5 — scale, measured

The lexical half is done (the inverted index above). For the dense half,
`tools/ann_bench.py` compares an IVF and an HNSW index against the exhaustive scan on
synthetic clustered vectors (the data is clustered on purpose; uniform vectors in 384
dimensions are near-orthogonal and make every ANN look perfect):

| vectors | exhaustive ms/query | IVF recall@10 (nprobe 1 / 8 / 32) | IVF ms (32) | HNSW recall@10 | HNSW ms |
| --- | --- | --- | --- | --- | --- |
| 10k | 0.046 | 0.52 / 0.66 / 0.84 | 0.030 | 0.933 | 0.006 |
| 100k | 1.15 | 1.00 / 1.00 / 1.00 | 0.174 | 0.865 | 0.006 |
| 1M | 11.6 | 1.00 / 1.00 / 1.00 | 0.724 | 0.744 | 0.016 |

Two honest readings. First, **exhaustive search is not the problem until well past a
million vectors** — 11.6 ms/query at 1M against 0.046 ms at 10k, linear, and a corpus that
size is already unusual for this engine. Second, when ANN is wanted, IVF is the better
trade here (recall 1.00 at 16x, HNSW 0.744 at 724x), but **recall on synthetic data is not
recall on real embeddings**, so the number to trust is the shape, not the digit.

`VectorIndex` now supports it (`build_ann`, `ANN`/`ANN_NPROBE`, persisted as `ann.faiss`),
with the exhaustive scan as the default and as the fallback when faiss is absent.

### Item 6 — a learned-sparse channel (SPLADE-style), measured

`tools/sparse.py` adds a third channel: a masked-LM encoder whose `log(1 + relu(logits))`
activations are top-k'd into a sparse vocabulary vector, retrieved by sparse dot product
over an inverted index. The checkpoint is OpenSearch's neural-sparse encoder because it
ships `safetensors`: the SPLADE checkpoints on the Hub are `.bin` only, and transformers
4.57 refuses to `torch.load` them on torch < 2.6 (CVE-2025-32434), which this project
cannot upgrade past on macOS Intel.

| dataset | BM25 | dense (`bge-small`) | **SPLADE** | 2-channel fusion | **3-channel fusion** |
| --- | --- | --- | --- | --- | --- |
| SciFact | 0.676 | 0.720 | **0.710** | 0.741 | **0.745** |
| NFCorpus | 0.324 | 0.339 | **0.356** | 0.367 | **0.372** |
| ArguAna | 0.313 | 0.429 | **0.408** | 0.414 | **0.426** |

Learned sparse beats BM25 on every dataset by +0.03 to +0.10, and it is the first channel
addition that improves the fusion consistently (+0.004 to +0.013). Combined with
`bge-large` it gives the best SciFact number measured (0.751) and lifts ArguAna to 0.443.
It does **not** close the ArguAna gap: published SPLADE scores 0.521 there against our
0.408, so the remaining distance is model quality, not the channel being missing.

### Item 4 — passage-level retrieval, measured

The harness supports `--passages N` (split each document, retrieve passages, aggregate to
documents by best passage). Measured at 600-character passages against document-level BM25:

| dataset | document-level | passages@600 |
| --- | --- | --- |
| SciFact | 0.682 | 0.659 |
| NFCorpus | 0.325 | 0.318 |
| TREC-COVID (40k-document subsample) | 0.306 | 0.245 |

**It hurt on all three**, so it stays out of the default path. The honest caveat is that
TREC-COVID is not a long-document collection as hoped: its abstracts average ~160 words,
*shorter* than NFCorpus's 232. The datasets that are genuinely long — Robust04 (466 words),
TREC-NEWS (635) — are not on the BEIR mirror this project uses (`robust04.zip` returns a
146-byte stub), so the long-document claim is **untested, not disproven**. Splitting 171k
documents into passages also proved intractable in-process: the pure-Python postings for
~half a million passages exceeded what the run could finish in well over an hour, which is
itself an argument for the on-disk postings store in item 5.

One large incidental result: on TREC-COVID, stemming is worth **+0.069** nDCG@10
(0.237 → 0.306) and title weighting a further +0.012 — the strongest evidence yet that the
`LEXICAL_STEM` default is the right one.

### Item 7 — domain adaptation, measured

`tools/finetune.py` fine-tunes the bi-encoder on a dataset's **train** split and evaluates
on its **test** split with `MultipleNegativesRankingLoss`. NFCorpus ships 110,575 train
pairs, and the 2,590 train queries are disjoint from the 323 test queries:

| stage | nDCG@10 |
| --- | --- |
| zero-shot `bge-small` | 0.3392 |
| adapted, 20k pairs, 1 epoch (15 min) | **0.3477** (+0.0085) |

Positive but small, and deliberately bounded — 20k of 110k pairs for one epoch. Published
domain-adaptation gains are 0.02–0.05 with the full split and more epochs, so this is a
floor, not a ceiling. The result is **adapted, not zero-shot**, and labelled that way in
the report it writes; it must never be compared as if it were a zero-shot number.

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

## What remains

Items 1–7 have all been attempted; these are the genuinely open threads.

**Long-document passage retrieval is untested, not disproven.** Passages hurt on SciFact,
NFCorpus and TREC-COVID (item 4), but TREC-COVID's abstracts are *shorter* than NFCorpus's
and Robust04/TREC-NEWS are not on the BEIR mirror this project uses. Reaching them means a
different source, an adaptive passage budget, and aggregation beyond max-pool (sum, or
learned weights).

**The full 18-dataset BEIR average needs the large datasets, and they need infrastructure.**
BioASQ is 14.9M documents, NQ 2.7M, FEVER 5.4M. The ANN index is in place; what is missing
is an on-disk postings store with compressed term frequencies (the in-memory Python
postings are what made splitting 171k documents intractable) and streaming index
construction, since `VectorIndex` still holds every chunk's text and vector in memory.

**Fine-tuning is a floor.** 20k of NFCorpus's 110k train pairs, one epoch, +0.0085. The
full split, more epochs and hard-negative mining are the next step, and the result must
keep the *adapted* label.

**ArguAna is a model-quality gap.** Published BGE-large scores 0.636 and SPLADE 0.521
there against our best 0.459; no configuration measured here closes it, and every channel
we added helped less there than elsewhere.

## Session state

- The engine and harness are committed at `6d489e0`; `make test` is 303 offline tests and
  `make test-real` is 4 against the real checkpoints.
- `eval/` is gitignored and regenerable. Duplicate dataset downloads and archives were
  removed (about 1 GB); what remains is the model cache, embedding/sparse caches, extracted
  corpora and reports, all rebuilt by `make eval` / `make beir` / `make beir-ablate`.
- Environment notes worth keeping: `transformers` is pinned at 4.57.6 and installing the
  fine-tuning stack silently upgraded it once; `torch` is pinned at 2.2.2 (macOS Intel), and
  that is why SPLADE checkpoints, which ship only `pytorch_model.bin`, cannot be loaded
  through transformers at all. Optional dependencies are declared in the `eval` extra.

## Method notes

- Documents are indexed as `title + text` by default, the concatenation BEIR's own baselines
  use; `--dense-text-only` lets the dense channel embed text alone, which helps datasets
  whose "titles" are shared topic keywords (ArguAna).
- Only queries with qrels are scored, and queries without a relevant document are skipped,
  which is what `pytrec_eval`/BEIR do.
- Dense retrieval is exhaustive by default; `ANN`/`ANN_NPROBE` switch on the IVF index, and
  `tools/ann_bench.py` measures the recall it costs.
- `bm25` is the unstemmed control; `bm25-engine` is the shipped configuration (stemming on,
  query stopwords dropped, title weighted) and feeds the fused and reranked rows.
