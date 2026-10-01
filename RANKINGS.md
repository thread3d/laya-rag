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
| 3 | Stronger dense model (`bge-base` measured; `bge-large` deferred) and reranker (`bge-reranker-base` cached, not measured) | +0.01–0.03 | hours–days | partly done — **and the encoder-size lever is now exhausted** on ArguAna |
| 4 | Passage-level indexing for long documents (TREC-COVID, Robust04) | needed there | days | measured: hurts; long-document datasets unavailable |
| 5 | Scale: inverted index (**done**), ANN index (**done**, `build_ann`) | prerequisite | days–weeks | done |
| 6 | Learned sparse (SPLADE) as a third channel | +0.02–0.05 | weeks | **done**: +0.004–0.023 fused, +0.03–0.14 over the lexical channel alone |
| 7 | Domain adaptation / fine-tuning on in-domain pairs (no longer zero-shot) | +0.02–0.05 | weeks | **demonstrated**: +0.0085 with 20k of 110k pairs |
| 8 | Second stage over a strong first stage: `bge-reranker-base` at depth 100, and a multi-model dense ensemble | +0.01–0.03 | hours–days | open — the highest-value remaining item |

Items 5–8 are not "more of the same": each needs infrastructure or training that cannot be
done responsibly in a single session, and they are specified below rather than attempted
half-way.

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
cross-encoder second stage, or ensembling. `bge-reranker-base` was cached but never measured,
since the fast `ms-marco-MiniLM` cross-encoder had already been beaten by fusion; that verdict
was reached against a weaker first stage than the one that exists now.

### Experiments tried, measured, and rejected

Recorded so they are not retried from intuition:

| experiment | SciFact | NFCorpus | verdict |
| --- | --- | --- | --- |
| `bm25-title` (title repeated twice) | 0.682 | 0.325 | **kept**, but only significant on ArguAna |
| `bm25-prf` (10 feedback docs, 15 terms ×2) | 0.424 | 0.274 | **rejected** — the expansion drifts the query badly |
| `bm25-passages` (600-char passages, max-pool) | 0.659 | 0.318 | rejected at this document size; kept for long docs |
| `hybrid-combsum` (score fusion, max-normalised) | 0.712 | 0.355 | rejected vs RRF's 0.741 / 0.367 |
| `bge-large` instead of `bge-base` | 0.735 | 0.370 | **no material gain** for 3× the embedding cost |

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

## What remains

Items 1–7 have been attempted; item 8 is specified but not yet run. These are the genuinely open
threads, in the order they are worth doing.

**The second stage is the highest-value open item.** `bge-reranker-base` is cached and has never
been measured, and the earlier rejection used `ms-marco-MiniLM-L-6-v2` over a first stage that is
now much stronger. Re-ranking the fused top-100 and reporting a paired interval is the cheapest
credible gain left (+0.01–0.03 in the standard IR literature), and on ArguAna it should be tested
against the known risk that pointwise cross-encoders do not model *opposition* well.

**A multi-model dense ensemble is nearly free.** `bge-small` + `bge-base` + `bge-large` corpus
embeddings are already cached for these five datasets; ensembling them by RRF is mostly
query-side cost for a typical +0.01–0.02.

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

**SciDocs is the remaining relative shortfall.** 0.199 against a published 0.225 for BGE-large,
where the other three datasets are at or above the published number. It is the one place a
per-dataset look (representation, truncation, or the dense/lexical balance) is still justified.

**Encoder size is not the gap, and that is now measured rather than assumed.** ArguAna settles it:
`bge-base` matches both a published and a locally-run `bge-large`. Any further single-vector
scaling should be treated as unpromising until the second stage and ensembling are exhausted.

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
