# Where the retrieval stands, and the plan to move it

This is the working document behind the BEIR work in `tools/beir.py` and
`tools/beir_ablate.py`. It records the measured standing, the programme to improve it, and —
for the parts that cannot be finished in one sitting — what they concretely require.

Everything here is produced by `make beir` / `make beir-ablate`; the raw reports are in
`eval/beir/` (gitignored). Numbers are nDCG@10 unless stated.

> **Revision note.** An earlier version of this document reported ArguAna at 0.429–0.459 and
> read the 0.21 gap to the published BGE-large (0.636) as an encoder-quality gap. That reading
> was wrong: this harness was not applying BEIR's `ignore_identical_ids` rule, so ArguAna's
> queries were retrieving themselves at rank 1 and burning a top-10 slot. The evaluator is
> fixed and every ArguAna number below is re-measured. See
> [The evaluator was mis-scoring ArguAna](#the-evaluator-was-mis-scoring-arguana).

## Standing

Published baselines are from the [BEIR paper](https://arxiv.org/abs/2104.08663), the BAAI
BGE paper and the [Pyserini 2CR table](https://castorini.github.io/pyserini/2cr/beir.html),
compiled in an independent [BEIR matrix](https://github.com/ruvnet/ruflo/blob/56819391/docs/benchmarks/BEIR-MATRIX.md).
They are 2021–2025 numbers from other harnesses, so they are indicative rather than a
controlled head-to-head.

The `bm25` control is the anchor that keeps the evaluator honest, and it is now checked rather
than asserted: `tools/beir.py` compares it to a pinned published baseline per dataset and reports
any deviation past 0.05. All five currently sit inside tolerance — SciFact −0.025, NFCorpus
−0.021, ArguAna **−0.004**, SciDocs −0.012, FiQA −0.002.

The ArguAna baseline is itself convention-dependent, which is worth knowing before chasing a
0.315 anywhere: the BEIR paper's 2021 Elasticsearch baseline indexed title and body as separate
*multifield* fields and reports 0.315, while Pyserini 2CR runs a **flat** index with
`--remove-query` — its name for the same rule this harness now applies — and reports 0.397. This
project's flat title+text control measures 0.394, so 0.397 is the number it is comparable to. The
two figures disagree about the indexing convention, not about retrieval quality
([beir#197](https://github.com/beir-cellar/beir/issues/197)).

**Five datasets, nDCG@10:**

| dataset | us, shipped lexical | us, bi-encoder | us, bi-encoder + fused BM25 | BM25 (published) | our `bm25` control |
| --- | --- | --- | --- | --- | --- |
| SciFact | 0.676 | 0.720 | **0.741** | 0.679 | 0.654 |
| NFCorpus | 0.324 | 0.339 | **0.367** | 0.325 | 0.304 |
| ArguAna | 0.417 | **0.596** | 0.568 | 0.397 | 0.394 |
| SciDocs | 0.150 | 0.197 | **0.199** | 0.158 | 0.146 |
| FiQA-2018 | 0.239 | **0.385** | 0.382 | 0.236 | 0.235 |
| **mean** | **0.361** | **0.447** | **0.452** | **0.359** | **0.347** |

Correcting the evaluator moved the shipped-lexical mean 0.340 → 0.361, the bi-encoder mean
0.414 → 0.447 and the fused mean 0.421 → 0.452. Only the ArguAna column changed; the other four
have no query/document id overlap, so no scoring convention can move them.

**Where that places us.** On the four datasets where the published table carries every system
(SciFact, NFCorpus, ArguAna, SciDocs):

| system | 4-dataset mean |
| --- | --- |
| **us — best per dataset** | **0.495** |
| BGE-large-v1.5 (335M) | 0.491 |
| SPLADE++ | 0.433 |
| BM25 (Lucene) | 0.390 |

The best-per-dataset configuration is SciFact `bge-large` + fused BM25 (0.750), NFCorpus
`bge-large` + fused BM25 (0.382), ArguAna `bge-small` dense on text-only documents (0.648) and
SciDocs `bge-small` + fused BM25 (0.199).

So: **first of four by a hair — 0.004 ahead of a 335M BGE-large, 0.062 ahead of SPLADE++.** That
margin is inside the noise band (see [Significance](#significance)), so the honest claim is *level
with BGE-large*, not ahead of it. The caveat is unchanged and still matters: this cherry-picks a
configuration per dataset. A single deployable configuration (`bge-small` + fused BM25) is 0.452
on the five-dataset mean, which is a different and weaker claim than 0.495.

### The Laya-native dense channel (adapted encoder)

The engine now auto-adopts a retrieval adaptation of Laya's **own** encoder when one is on disk
(`models/laya-retrieval`, built by `make laya-finetune`), fusing it at `ADAPTED_DENSE_WEIGHT=0.3`
instead of the historical `auto → 1.0`. Its rows (`eval/beir/five-adapted.json`):

| dataset | shipped lexical | adapted dense | adapted + fused BM25 | `bge-small` + fused BM25 |
| --- | --- | --- | --- | --- |
| SciFact | 0.676 | 0.531 | **0.688** | 0.741 |
| NFCorpus | 0.324 | 0.270 | **0.339** | 0.367 |
| ArguAna | 0.417 | 0.475 | **0.462** | 0.568 |
| SciDocs | 0.150 | 0.129 | **0.164** | 0.199 |
| FiQA-2018 | 0.239 | 0.217 | **0.280** | 0.382 |
| **mean (4)** | **0.392** | — | **0.413** | **0.469** |

A 307M Laya-native encoder replaces a 33M `bge-small` and improves on lexical-only on all five,
but it is a **0.056 regression against the `bge-small` configuration**, and it loses to BM25 on
MultiHop-RAG ordering (MAP@10 0.2490 vs 0.2534) and on BRIGHT dense. Both configurations are
therefore kept and drawn side by side: the `bge-small` row is labelled the higher-scoring
alternative, not hidden. Full refresh and the judge trade-off:
[`docs/findings-encoder-adaptation.md`](docs/findings-encoder-adaptation.md).

## The evaluator was mis-scoring ArguAna

ArguAna is the one BEIR corpus whose queries **are** corpus documents: 1,298 of its 1,406 test
queries share an `_id` with a document, and all 1,406 share the text. The original dataset
included every argument in the corpus for size reasons, and BEIR handles the consequence
explicitly — its evaluator has an `ignore_identical_ids` argument that defaults to `True` and
drops any retrieved document whose id equals the query id
([`beir/retrieval/evaluation.py`](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py);
confirmed by the maintainer in [beir-cellar/beir#14](https://github.com/beir-cellar/beir/issues/14)).

This harness did not implement it. Measured on the cached embeddings, the query's own document
was rank 1 for **100%** of ArguAna queries, including under title+text indexing. With one relevant
document per query that caps nDCG@10 at `1/log2(3) = 0.631`, and it cost:

| ArguAna configuration | self kept (before) | BEIR rule (after) |
| --- | --- | --- |
| `bm25` control | 0.295 | **0.394** |
| `bm25-engine` (shipped lexical) | 0.313 | **0.417** |
| `bge-small` dense, title+text | 0.428 | **0.596** |
| `bge-small` dense, text-only | 0.459 | **0.648** |
| `bge-base` dense | 0.453 | **0.638** |
| `bge-large` dense | 0.454 | **0.638** |
| SPLADE | 0.408 | **0.559** |
| fused (`bge-small` + BM25) | 0.413 | **0.568** |

Three things follow.

1. **The ArguAna "model-quality gap" was an evaluation artifact.** After the fix, `bge-base`
   scores 0.638 against a published BGE-large at 0.636 — the gap is zero, at a third of the
   parameters. SPLADE's 0.559 also clears the published SPLADE++ (0.521).
2. **The anchor had already failed and nobody noticed.** RANKINGS called `bm25` "the anchor that
   keeps the evaluator honest", and on ArguAna it sat 0.10 below the published baseline. That is
   exactly the check that should have caught this, so it is now automated: the control is
   compared against a pinned published table and any deviation past tolerance is reported with
   the results.
3. **The comparison protocol has to be stated.** Published ArguAna numbers assume the query's own
   document is gone. A harness that keeps it is not measuring the same task, and the 0.20
   difference looks exactly like a modelling gap.

The fix is `ignore_identical_ids=True` in `tools/evalkit.py` (BEIR's default), threaded through
`tools/beir.py` and `tools/beir_ablate.py`. `--keep-identical-ids` reproduces the old row for
anyone who wants it, and `tests/test_ir_metrics.py` pins the behaviour — including a regression
that the corrected control passes the anchor.

## Significance

Per-query scores are now persisted (`per_query_nDCG@10` in each report) and every row is
paired-bootstrapped against the baseline, resampling **queries** so query difficulty cancels.
That matters here because the claimed differences are routinely 0.004–0.013, which these query
counts cannot resolve without an interval. Rows against `bm25-engine` on the five-dataset run:

| dataset | row | delta | 95% CI | significant |
| --- | --- | --- | --- | --- |
| SciFact | `hybrid-biencoder` | +0.065 | [+0.044, +0.087] | yes |
| SciFact | `biencoder` | +0.044 | [+0.008, +0.080] | yes |
| SciFact | `bm25-title` | +0.006 | [−0.001, +0.014] | **no** |
| NFCorpus | `hybrid-biencoder` | +0.043 | [+0.029, +0.059] | yes |
| NFCorpus | `biencoder` | +0.014 | [−0.007, +0.036] | **no** |
| NFCorpus | `bm25-title` | +0.000 | [−0.003, +0.004] | **no** |
| ArguAna | `biencoder` | +0.179 | [+0.160, +0.197] | yes |
| ArguAna | `hybrid-biencoder` | +0.151 | [+0.139, +0.163] | yes |
| ArguAna | `bm25-title` | +0.011 | [+0.007, +0.015] | yes |
| SciDocs | `hybrid-biencoder` | +0.050 | [+0.043, +0.056] | yes |
| SciDocs | `biencoder` | +0.048 | [+0.037, +0.058] | yes |
| SciDocs | `bm25-title` | +0.003 | [+0.000, +0.005] | yes (barely) |
| FiQA-2018 | `hybrid-biencoder` | +0.143 | [+0.126, +0.161] | yes |
| FiQA-2018 | `biencoder` | +0.146 | [+0.122, +0.170] | yes |

Two readings worth keeping:

- **Fusion over the stemmed lexical channel is significant on all five corpora** (+0.043 to
  +0.151). That is the strongest and most portable result here.
- **Title weighting is an ArguAna effect, not a general one.** It is clearly significant there
  (+0.011) and negligible-to-noise on SciFact (+0.006, interval crosses zero), NFCorpus (+0.000)
  and FiQA (0.000). It stays on because it is cheap and never hurts significantly, but "title
  weighting helps slightly" was over-claimed.

Also worth recording: on ArguAna the bi-encoder alone (0.596) **beats** the fusion (0.568). Fusion
is only as good as its weaker input, and on this corpus BM25 is the weaker one — which is the same
effect seen with `bge-base` on SciFact/NFCorpus, and the reason per-corpus A/B testing is not
optional.

## The programme

| # | work | expected | effort | status |
| --- | --- | --- | --- | --- |
| 0 | Evaluator parity with BEIR (`ignore_identical_ids`), anchor guard, per-query persistence, paired bootstrap | correctness | hours | **done** — ArguAna +0.10–0.19, headline mean 0.448 → 0.495 |
| 1 | Fusion and BM25 tuning: RRF `k`, channel weights, title-boosted BM25F, passage indexing | +0.005–0.02 | hours | done (RRF `k=5`, title boost; passage indexing rejected) |
| 2 | Broaden evaluation to five datasets (add ArguAna, SciDocs, FiQA-2018) | — (credibility) | hours | done |
| 3 | Stronger dense model (`bge-base` measured; `bge-large` deferred) and reranker (`bge-reranker-base` cached, not measured) | +0.01–0.03 | hours–days | **done** — `bge-large` measured on BEIR and MultiHop-RAG: dense gains, fused hybrid does not; `bge-reranker-base` measured @100 and **loses on all five** |
| 4 | Passage-level indexing for long documents (TREC-COVID, Robust04) | needed there | days | measured: hurts; long-document datasets unavailable |
| 5 | Scale: inverted index (**done**), ANN index (**done**, `build_ann`), on-disk postings and streaming vectors (**done**, `tools/ondisk.py`) | prerequisite | days–weeks | done |
| 6 | Learned sparse (SPLADE) as a third channel | +0.02–0.05 | weeks | **done**: +0.004–0.023 fused, +0.03–0.14 over the lexical channel alone |
| 7 | Domain adaptation / fine-tuning on in-domain pairs (no longer zero-shot) | +0.02–0.05 | weeks | **done, negative**: 20k pairs +0.0085, but the full 110k + hard negatives is **−0.0100** |
| 8 | Second stage over a strong first stage: `bge-reranker-base` at depth 100, and a multi-model dense ensemble | +0.01–0.03 | hours–days | **done** — the reranker loses everywhere; the **ensemble is the win** (0.452 → 0.495) |

The originally-open items have now been run; [The raise-the-score programme, measured](#the-raise-the-score-programme-measured)
records each outcome, including the two that reversed their expectations.

### Scale work landed: an inverted index for BM25

Lexical scoring looped over every document for every query term — O(terms × documents) — which
is what made the five-dataset run stall before it started: ArguAna queries are ~190-word
arguments, so one query set was billions of iterations. `BM25Index` now keeps
`term -> [(row, frequency)]` postings and touches only the documents that contain each term,
which is the index Lucene keeps. ArguAna went from stalling past 50 minutes to ~4 minutes for
the full query set, and the ranking is unchanged (pinned by a save/load equivalence test).
`VectorIndex` supports ANN (`build_ann`, `ANN`/`ANN_NPROBE`, persisted as `ann.faiss`), with the
exhaustive scan as the default and as the fallback when faiss is absent.

### Model size — the measured headroom, and its ceiling

Swapping `bge-small` (33M) for `bge-base` (110M), same fusion, with ArguAna re-measured:

| dataset | bge-small dense | bge-base dense | bge-large dense | bge-small fusion | bge-large fusion |
| --- | --- | --- | --- | --- | --- |
| SciFact | 0.720 | 0.738 | 0.735 | 0.741 | **0.750** |
| NFCorpus | 0.339 | 0.371 | 0.370 | 0.367 | **0.382** |
| ArguAna | 0.596 | 0.638 | **0.638** | 0.568 | 0.599 |
| 3-dataset dense mean | 0.551 | **0.582** | 0.581 | — | — |

A bigger encoder is worth **+0.031** on the three-dataset dense mean and `bge-base` reaches it.
Past that it saturates: `bge-base` and `bge-large` agree to 0.001 on ArguAna (0.6377 vs 0.6378)
and `bge-large` is flat-to-slightly-worse on SciFact/NFCorpus. **`bge-base` is the efficient
choice and a 335M encoder buys nothing here.**

The ceiling is therefore visible but not reachable by scaling one vector: the published BGE-large
scores 0.636 on ArguAna and `bge-base` already matches it, so the remaining headroom is in the
*architecture*, not the parameter count — late interaction (ColBERT-style multi-vector), a
cross-encoder second stage, or ensembling. It has since been measured: the cross-encoder second
stage **lost on all five datasets** (item 4 below), and **ensembling is the architecture that
paid** — three dense models by RRF is worth +0.043 (item 5).

### Experiments tried, measured, and rejected

Recorded so they are not retried from intuition:

| experiment | SciFact | NFCorpus | verdict |
| --- | --- | --- | --- |
| `bm25-title` (title repeated twice) | 0.682 | 0.325 | **kept**, but only significant on ArguAna |
| `bm25-prf` (10 feedback docs, 15 terms ×2) | 0.424 | 0.274 | **rejected** — the expansion drifts the query badly |
| `bm25-passages` (600-char passages, max-pool) | 0.659 | 0.318 | rejected at this document size; kept for long docs |
| `hybrid-combsum` (score fusion, max-normalised) | 0.712 | 0.355 | rejected vs RRF's 0.741 / 0.367 |
| `bge-large` instead of `bge-base` | 0.735 | 0.370 | **no material gain** for 3× the embedding cost |
| `bge-reranker-base` over the fused top-100 | 0.741 → 0.714 | 0.367 → 0.315 | **rejected** — loses on all five datasets; ArguAna 0.568 → 0.250 |
| full-split fine-tuning (110,575 pairs + 5,180 hard negatives) | — | 0.3385 → 0.3285 | **rejected** — scaling the 20k-pair run up with naive hard negatives hurt |

The PRF result is the starkest: a textbook technique, implemented plainly, cost 0.26 nDCG@10 on
SciFact. It assumes the first pass is mostly right and these queries are short enough that it is
not. CombSUM lost to RRF because BM25's and cosine's score scales do not normalise cleanly against
each other, which is exactly the problem RRF exists to avoid. And `bge-large` confirms the model
size *saturates* at `bge-base` — with ArguAna corrected, `bge-base` and `bge-large` are
indistinguishable there (0.6377 / 0.6378) and both sit on the published number.

### Document text is a per-channel choice, not a global one

ArguAna's `title` field is a shared topic keyword list (`animals environment general health …`)
repeated across a cluster, and BEIR's convention of indexing `title + text` puts those near-identical
terms into every document of the cluster:

| ArguAna representation | BM25 | dense (`bge-small`) |
| --- | --- | --- |
| `title + text` (BEIR convention) | **0.394** | 0.596 |
| text only | 0.360 | **0.648** |

The channels disagree in opposite directions: titles are worth **+0.034** to BM25 and cost the
dense channel **0.052**. That is why `--dense-text-only` exists — the dense channel embeds text
alone while BM25 keeps `title + text`. Under the corrected evaluator the disagreement is larger,
not smaller, so the flag earns its place.

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

### Item 6 — a learned-sparse channel (SPLADE-style), measured

`tools/sparse.py` adds a third channel: a masked-LM encoder whose `log(1 + relu(logits))`
activations are top-k'd into a sparse vocabulary vector, retrieved by sparse dot product
over an inverted index. The checkpoint is OpenSearch's neural-sparse encoder because it
ships `safetensors`: the SPLADE checkpoints on the Hub are `.bin` only, and transformers
4.57 refuses to `torch.load` them on torch < 2.6 (CVE-2025-32434), which this project
cannot upgrade past on macOS Intel.

| dataset | BM25 | dense (`bge-small`) | **SPLADE** | 2-channel fusion | **3-channel fusion** |
| --- | --- | --- | --- | --- | --- |
| SciFact | 0.676 | 0.720 | 0.710 | 0.741 | **0.745** |
| NFCorpus | 0.324 | 0.339 | 0.356 | 0.367 | **0.372** |
| ArguAna | 0.417 | **0.596** | 0.559 | 0.568 | 0.591 |

Learned sparse beats the stemmed lexical channel on every dataset by +0.03 to +0.14, and it is
the first channel addition that improves the fusion consistently (+0.004 to +0.023 — the ArguAna
gain is the largest now). Combined with `bge-large` it gives the best SciFact number measured
(0.751) and lifts ArguAna to 0.610. It does **not** close the ArguAna gap to dense alone: SPLADE's
0.559 is below the `bge-small` bi-encoder's 0.596, so on this corpus the third channel is not
where the headroom is.

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

`k=5` is the optimum on both tuning datasets and is the value every row in
[Standing](#standing) uses.

Two BM25 variants were measured on the same pair:

| variant | SciFact | NFCorpus | SciDocs | ArguAna |
| --- | --- | --- | --- | --- |
| `bm25-engine` (stemming, stopwords dropped) | 0.676 | 0.324 | 0.150 | 0.417 |
| `bm25-title` (title repeated twice) | **0.682** | **0.325** | **0.152** | **0.428** |
| `bm25-passages` (600-char passages, max-pooled) | 0.659 | 0.318 | — | — |

Title weighting is kept as the fusion base, but with per-query intervals it is only significant on
ArguAna, so it is a cheap default rather than a demonstrated general win. **Passage-level retrieval
hurts at this document size** (BEIR abstracts average ~230 words), which is expected: it is a
technique for genuinely long documents, and it stays in the harness for the datasets in item 4
rather than in the default path.

## The raise-the-score programme, measured

The README's ordered list of what would raise the score has been run item by item. Reports
are under `eval/` (gitignored); the tools are `make` targets. The outcomes, in the order the
README ranked them:

### 1. `bge-large` under the MultiHop-RAG protocol — done, and it closes the encoder question

`tools/multihop_official.py --models bge-small,bge-base,bge-large` (cached after the first
run). The published comparison row is `bge-large`, so this is now a real head-to-head rather
than `bge-small` standing in for it. Hits@10 at `chunk_overlap=200`:

| dense model | dense Hits@10 | hybrid Hits@10 | hybrid MAP@10 | hybrid MRR@10 |
| --- | --- | --- | --- | --- |
| `bge-small` | 0.708 | 0.802 | 0.251 | 0.540 |
| `bge-base` | 0.683 | 0.799 | 0.249 | 0.530 |
| `bge-large` | **0.735** | **0.810** | **0.255** | **0.545** |

`bge-large` dense significantly beats `bge-small` dense (Hits@10 +0.027 [+0.009, +0.045];
Hits@4 +0.025; MAP@10 +0.011; MRR@10 +0.028 — all significant). But once BM25 is fused the
gap is gone: `bge-large` hybrid vs `bge-small` hybrid is Hits@10 +0.008 [−0.004, +0.020],
Hits@4 −0.014, MAP +0.004, MRR +0.005 — none of those intervals excludes zero — and `bge-base`
hybrid is significantly *worse* than `bge-small` hybrid on Hits@4. **Fusion absorbs the encoder
gain**, the same saturation the ArguAna table shows. The published `bge-large + rerank` row
(Hits@10 0.718, MAP 0.476) is beaten on Hits@10 by both our hybrids and still far ahead on
MAP; the remaining MAP gap is not encoder size.

### 2. Answerability recalibration on a held-out split — done, and it does not transfer

`tools/multihop_recalibrate.py`, over the 301 null and 300 sampled answerable queries:

| reading | balanced accuracy | gap (null − answerable abstention) |
| --- | --- | --- |
| shipped threshold 0.5 | 0.511 | +0.022 |
| threshold fitted on one half, scored on the other | 0.489 | −0.022 |
| oracle threshold fitted on the test half | 0.524 | +0.049 |

ROC-AUC over all 601 queries is 0.509 [0.463, 0.555]; over the 506 that reached the head it
is 0.529 [0.479, 0.579]. A Platt rescaling fitted on the calibration half has slope −0.18.
Across 500 stratified half-splits the held-out balanced accuracy is **0.511 [0.472, 0.548]**.
This is not a miscalibrated threshold: the head's probability carries no transferable signal
on this corpus, so recalibrating it on a held-out MultiHop split does not produce
discrimination.

### 3. A coverage objective for MAP@10 — done, and it is the largest single retrieval win

`tools/multihop_coverage.py`. Fusing the top-100 of each channel and then walking the fused
list while skipping a candidate whose cosine to an already-kept chunk is ≥ 0.9:

| selection | MAP@10 | Hits@10 | Hits@4 |
| --- | --- | --- | --- |
| fused top-10 (depth-100 fusion) | 0.2546 | 0.812 | 0.683 |
| fused top-10 (depth-10 fusion, README row) | 0.2510 | 0.802 | 0.686 |
| **cluster dedup @ 0.9** | **0.2790** | **0.864** | **0.712** |
| document MMR (λ 0.7) | 0.2740 | 0.827 | 0.688 |
| query-term coverage (λ 0.5) | 0.2717 | 0.821 | 0.698 |
| embedding MMR (λ 0.3) | 0.2666 | 0.822 | 0.675 |

Cluster dedup is +0.0244 MAP@10 [+0.0223, +0.0265] and +0.0523 Hits@10 [+0.040, +0.065]
against the depth-100 fused baseline, both significant; every family's setting was chosen on
a calibration half and scored on the other, and both halves chose 0.9. The loss really is
redundancy — an overlap-heavy corpus spending several slots on one passage cluster — rather
than model quality. It is now also an opt-in engine setting, `RECALL_DEDUP_THRESHOLD`
(off by default; the engine over-fetches ×3 when it is on so duplicates are replaced).

**Which objective wins is a function of the chunking, and the second setting was run.**
At `chunk_overlap=0` (chunks no longer duplicate each other) the best family changes: query-term
coverage (`query-coverage-0.5`) reaches 0.2796 held-out on a 0.2668 baseline (+0.0128), while
cluster dedup is neutral-to-worse (0.2653) and document MMR is +0.0006. So dedup is a fix for
overlap redundancy specifically, not a universal coverage win; query-term coverage is the
family that helps at both settings (0.2717 at overlap 200, 0.2796 at 0). Reports:
`eval/benchmarks/multihop-rag/coverage.json` and `coverage-ov0.json`.

### 4. `bge-reranker-base` over the fused top-100 — done, and it is the clearest rejection

`tools/beir.py` with `crossencoder` over the `hybrid-biencoder` top-100, `--rerank-depth 100`,
`BAAI/bge-reranker-base`. The earlier rejection used `ms-marco-MiniLM-L-6-v2` over a weaker
first stage; this is the cached 335M reranker over the stronger fused stage. nDCG@10:

| dataset | fused first stage | `bge-reranker-base` @100 | delta | 95% CI |
| --- | --- | --- | --- | --- |
| SciFact | 0.741 | 0.714 | −0.027 | [−0.054, −0.002] |
| NFCorpus | 0.367 | 0.315 | −0.053 | [−0.072, −0.034] |
| ArguAna | 0.568 | **0.250** | **−0.318** | [−0.341, −0.296] |
| SciDocs | 0.199 | 0.152 | −0.047 | [−0.056, −0.038] |
| FiQA-2018 | 0.382 | 0.328 | −0.054 | [−0.074, −0.034] |

**It loses on every dataset, with every interval excluding zero.** The +0.01–0.03 the
programme expected did not materialise. ArguAna is the sharpest case: its relevance is an
*opposition* relation (does the argument attack the query's claim), which a pointwise MS MARCO
reranker does not represent at all, so re-ordering destroys a dense ranking that had it right.
The practical conclusion is that this pipeline's second stage is not a cross-encoder; the
coverage objective (item 3) is where the measurable headroom was. Report:
`eval/beir/stage2-rerank`.

### 5. Multi-model dense ensemble (`bge-small` + `bge-base` + `bge-large` by RRF) — done, and it beat its estimate

`tools/beir.py --ensemble-models BAAI/bge-base-en-v1.5,BAAI/bge-large-en-v1.5`. nDCG@10, all
five datasets; `ensemble + BM25` is `dense-ensemble` fused with the title-boosted lexical
channel:

| dataset | dense (`bge-small`) | fusion | dense ensemble | ensemble + BM25 | delta vs fusion |
| --- | --- | --- | --- | --- | --- |
| SciFact | 0.720 | 0.741 | 0.754 | **0.767** | +0.025 [+0.010, +0.041] |
| NFCorpus | 0.339 | 0.367 | 0.379 | **0.387** | +0.019 [+0.010, +0.029] |
| ArguAna | 0.596 | 0.568 | 0.656 | **0.651** | +0.083 [+0.073, +0.093] |
| SciDocs | 0.197 | 0.199 | 0.222 | **0.225** | +0.026 [+0.020, +0.032] |
| FiQA-2018 | 0.385 | 0.382 | 0.438 | **0.444** | +0.062 [+0.050, +0.075] |
| **5-dataset mean** | 0.447 | 0.452 | 0.490 | **0.495** | +0.043 [+0.038, +0.048] |

Every paired interval excludes zero, and the gain is far above the +0.01–0.02 the programme
expected. The four-dataset mean (the datasets the published matrix carries) is **0.507**,
above the previously cherry-picked 0.495 and above BGE-large's 0.491 — and this is a *single*
configuration. ArguAna is again the largest gain, which is the inverse of the fusion story:
fusing a weak BM25 hurt there, but fusing three dense models helps. SciDocs reaches the
published 0.225. The cost is the caveat: 478M parameters across three encoders and three
corpus embedding passes, so this is not a per-parameter win. Reports:
`eval/beir/ensemble3` (SciFact, NFCorpus, ArguAna) and `eval/beir/ensemble2` (SciDocs, FiQA).

### 6. Per-corpus channel selection — done, and the oracle is recoverable

`tools/beir_routing.py`, choosing among `bm25-engine`, dense and fusion per corpus on a
calibration half and scoring the held-out half: 0.4576 vs global fusion's 0.4515, **+0.0061
[+0.0026, +0.0096]**, exactly the 0.4576 oracle that sees the test qrels. The router picks
dense on ArguAna 100% of the time; the rest is a tie. So the ArguAna exception is worth about
+0.006 on the five-dataset mean and is learnable from a labelled sample rather than an oracle.

### 7. Full fine-tuning on 110k in-domain pairs — done, and it went backwards

`tools/finetune.py --dataset nfcorpus --max-pairs 0 --epochs 1 --hard-negatives 2` — the
whole 110,575-pair train split, one epoch, plus 5,180 hard negatives mined from the zero-shot
model, with `MultipleNegativesRankingLoss` and a `TripletLoss` objective:

| stage | nDCG@10 |
| --- | --- |
| zero-shot `bge-small` | 0.3385 |
| adapted, 20k pairs, 1 epoch, in-batch negatives only | 0.3477 (+0.0085) |
| **adapted, 110,575 pairs + 5,180 mined hard negatives** | **0.3285 (−0.0100)** |

So the 20k result was not a floor. Scaling to the full split and adding naively mined hard
negatives **lost 0.010 nDCG@10**, below zero-shot. The likely mechanism is label noise: NFCorpus
qrels are sparse (about 1.5 judged-relevant documents per query), so the top non-*judged*
documents used as "hard negatives" are frequently actually relevant or near-relevant, and the
triplet objective pushes them away. A defensible next version needs negatives filtered against
the full judgment set, a stronger teacher for mining, or both. Report: `eval/finetune-full`;
the number stays labelled **adapted, not zero-shot**.

### 8. On-disk postings and streaming index construction — infrastructure landed

`tools/ondisk.py`. `OnDiskBM25` spills sorted runs and merges them into a packed,
memory-mapped `postings.bin`; on the full 171,332-document TREC-COVID corpus it builds in
1,073 s to **128 MB** and returns the *identical* ranking to the in-memory BM25 (pinned by a
test and by a 20/20 parity check on SciFact). Peak build RSS was 2.4 GB, dominated by one
build block, so the size a build can handle is set by `--block-docs`.

`StreamingVectorIndex` writes embedding batches positionally to a memory-mapped
`vectors.bin` and searches it blockwise. The first version still grew to **4.5 GB** over
30,000 documents, and the cause was not the store: with dynamic padding every batch has a
different tensor shape, and torch's MPS backend caches a compiled graph per shape, so the
cache grew with the corpus. Padding every batch to a fixed width makes construction
genuinely flat — **1,249 MB peak, unchanged from 5k to 15k documents**, of which ~1.2 GB is
the model itself — and 23 MB of vectors on disk (`beir.BiEncoder.embed(fixed_length=True)`).
15k documents is a bounded demonstration, not the 171k the lexical half ran: the dense
embedder, not the store, is the cost at that size.

10M+ document corpora still need a bounded external merge for the term dictionary (the
per-block Python lists are the 2.4 GB peak) and an on-disk ANN index; the store itself no
longer grows with the corpus.

## What remains

The programme's measurable items are now measured. What is genuinely open:

**The second stage is closed for pointwise rerankers.** `bge-reranker-base` over the fused
top-100 lost on all five datasets, so the +0.01–0.03 the literature predicts does not hold for
this first stage on these corpora. What remains open is a *listwise* or late-interaction ranker
(ColBERT-style multi-vector, or a cross-encoder trained on the corpus's own relevance
conventions) rather than another pointwise scorer.

**The ensemble is the new deployable default to beat, and it needs its cost measured.** A
single three-model configuration reaches 0.495 five-dataset / 0.507 four-dataset, above the
previous cherry-picked mean, with no cross-encoder. What is not yet measured is the cost
side: three corpus embedding passes, three model loads, and 478M parameters. A distillation
of the three into one encoder (or a two-model subset) is the obvious next question.

**Long-document passage retrieval is untested, not disproven.** Passages hurt on SciFact,
NFCorpus and TREC-COVID (item 4), but TREC-COVID's abstracts are *shorter* than NFCorpus's and
Robust04/TREC-NEWS were not on the BEIR mirror this project uses. Both have since been reached from
a different source and measured lexical-only (Robust04 0.4077, TREC-NEWS 0.3351,
`docs/findings-beir-recovered.md`). **The passage question has since been asked and answered, in the
negative**: at a 600-character passage budget Robust04 reads **0.3522** and TREC-NEWS **0.2456**
(`eval/beir/passages-long.json`), i.e. **−0.0555 and −0.0895 against document-level BM25 on the same
qrels** — so long-document passage decomposition is **disproven on exactly the two corpora it was
written for**, not merely untested. An adaptive passage budget and aggregation beyond max-pool remain
unbuilt, and are the only versions of the claim still open.

**The full 18-dataset BEIR average now needs only scale work, not new modelling.** The
on-disk postings and streaming vectors landed; what remains for BioASQ (14.9M) is a bounded external
merge for the term dictionary and an on-disk ANN index, plus a BioASQ-registration corpus fetch. FEVER
(5.4M) and NQ (2.7M) are measured, and 16 of the 18 members now have a report.

**SciDocs is still the relative shortfall.** 0.199 against a published 0.225 for BGE-large, and
the ensemble is the first thing measured that moves it; a per-dataset look (representation,
truncation, dense/lexical balance) is still justified.

**Encoder size is not the gap, and that is now measured twice.** ArguAna settled it for BEIR;
MultiHop-RAG settles it for the fused multi-hop pipeline.

## Session state

- The engine and harness are committed at `a48dcbe`; this evaluator fix, the anchor guard, the
  per-query persistence, the significance reporting and this document are the current working-tree
  change. `make test` is 315 offline tests and `make test-real` is 4 against the real checkpoints.
- Reports backing this document: `eval/beir/five-small-fixed` (five datasets, `bge-small`),
  `dense-text-only-fixed` (ArguAna text-only), `large-splade-fixed` (`bge-large` + SPLADE),
  `splade-small-fixed` (`bge-small` + SPLADE), `arguana-base-fixed` (`bge-base`) and
  `arguana-notitles-fixed` (ArguAna text-only BM25). The `-fixed` suffix distinguishes them from
  the pre-fix reports, which are kept for the before/after table.
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
- **A retrieved document whose id equals the query id is dropped before scoring**
  (`ignore_identical_ids`, BEIR's default). This is a no-op on four of the five datasets and
  worth +0.19 nDCG@10 on ArguAna. `--keep-identical-ids` turns it off.
- **The `bm25` control is checked against a pinned published baseline** (`PUBLISHED_BM25` in
  `tools/beir.py`) and any deviation past 0.05 is reported with the results. A control that
  drifts is the harness's problem, not the retriever's.
- **Differences are quoted with a paired bootstrap over queries** (10,000 resamples, 95%
  interval), never as bare point estimates — the per-query scores are persisted in each report
  for exactly this.
- Dense retrieval is exhaustive by default; `ANN`/`ANN_NPROBE` switch on the IVF index, and
  `tools/ann_bench.py` measures the recall it costs.
- `bm25` is the unstemmed control; `bm25-engine` is the shipped configuration (stemming on,
  query stopwords dropped, title weighted) and feeds the fused and reranked rows.
