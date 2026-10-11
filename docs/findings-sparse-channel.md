# Findings: the learned-sparse channel is now in the engine

BM25 matches terms that appear; a learned-sparse model *expands* the query and document into
weighted vocabulary terms, so it can match paraphrase while keeping an inverted index. The
BEIR harness measured this as **the first additional channel that improves the fusion
consistently** — which is why it now ships in the engine instead of living under `tools/`.

## What the harness measured (nDCG@10)

| dataset | BM25 | dense (`bge-small`) | **SPLADE** | 2-channel fusion | **3-channel fusion** |
| --- | --- | --- | --- | --- | --- |
| SciFact | 0.676 | 0.720 | 0.710 | 0.741 | **0.745** |
| NFCorpus | 0.324 | 0.339 | 0.356 | 0.367 | **0.372** |
| ArguAna | 0.417 | **0.596** | 0.559 | 0.568 | **0.591** |

Learned sparse beats the stemmed lexical channel on every dataset by **+0.03 to +0.14**, and
the third channel adds **+0.004 to +0.023** over the two-channel fusion (largest on ArguAna).
It does not close ArguAna's gap to dense alone, so it is a breadth gain, not a rescue.

## What shipped

* `laya_rag.sparse.SparseEncoder` — the OpenSearch neural-sparse checkpoint with the standard
  SPLADE activation `log(1 + relu(logits))`, max-pooled, top-`k` terms kept. Lazy: torch and
  transformers load only on first use, so the package import path is unchanged.
* `laya_rag.sparse.SparseIndex` — a pure-Python inverted index scored by sparse dot product,
  persisted as versioned `sparse.json` beside the other index files.
* `RagEngine` builds it during `index_paths` when `SPARSE_MODEL` is set, searches it per
  query, and fuses it as a **third RRF channel** (`sparse_hits`/`sparse_weight` in `_fuse`),
  with the channel named on each returned passage. Query vectors use the same bounded cache
  as the dense channel. `stats()` reports `sparse.configured/present/encoder`.

Config: `SPARSE_MODEL` (unset = engine behaves exactly as before), `SPARSE_WEIGHT` (1.0),
`SPARSE_TOP_K` (128), `SPARSE_BATCH_TEXTS` (16). Indexing encodes every chunk with the extra
model, so it is opt-in and rebuilds are slower; `SPARSE_MODEL` with the `ondisk` backend is
rejected with a clear error rather than silently ignored.

## Verification

- Unit: the sparse dot product, tie-breaking, top-k, JSON round trip, version rejection and
  encoder laziness (7 tests), plus engine tests that the channel is built, passed to `_fuse`,
  named on passages, reported in `stats()`, and absent by default.
- Real checkpoint: encoding three documents kept 64 terms each, a query 37, and the refund
  passage ranked first for "how long does a refund take" (17.17 vs 4.20 and 0.08) — the
  production encoder, not a stub.

## Interaction with the corpus-aware chunking default

The +0.0179 above was measured under the **fixed 1200/400** chunking that shipped at the
time. Round 35 made `CHUNK_CHARS_AUTO` the default (SciFact derives 2,000), and the two
improvements turn out to overlap — both recover part of the same chunked-representation loss:

| configuration | nDCG@10 | sparse's marginal gain |
| --- | --- | --- |
| fixed chunking, no sparse | 0.7264 | — |
| fixed chunking, sparse | 0.7443 | **+0.0179 [+0.0028, +0.0337]**, p=0.025 |
| corpus-aware chunking, no sparse | 0.7417 | — |
| **corpus-aware chunking + sparse** | **0.7465** | **+0.0048 [−0.0115, +0.0213]**, p=0.57 |

Both orders help and the combination is the best score measured on this dataset (0.7465), but
the increments do **not** compose: the sparse channel's marginal contribution falls from a
significant +0.0179 to a non-significant +0.0048 once documents are no longer split. That is
the expected shape — a channel that helps by supplying whole-document context has less left to
supply when the chunking already preserves it.

It does not change the shipping decision: the sparse channel is opt-in and never harmful, and
corpus-aware chunking is a default that helps whether or not sparse is installed. It does
change the *claim*: "sparse is worth +0.0179" should be read as "worth up to +0.0179, on a
fixed-chunking index".

## The row silently disappeared for a week

The row could not be produced at all between **2026-10-04** and the fix below, and no run said so.
`77630bd` made the harness's corpus embedding resumable and passed `chunk=args.embed_chunk`
through to `sparse.load_or_encode` at the same time — but that function has no `chunk` parameter,
so every `--splade-model` run raised `TypeError`. The call site catches `Exception` deliberately
("a missing model must not lose the run"), so the run **succeeded**, reported every other row, and
recorded `results["splade"]["error"]` — a row that is missing is not a row that is zero, and a run
whose whole purpose is that row reads as green in a log.

Found by trying to reproduce a report, not by a test: no test exercised the sparse path end to end.
The call site no longer passes the argument it cannot take, and a skipped row now prints to stderr
as well as into the report. **What is not fixed:** `load_or_encode` is still all-or-nothing, so a
sparse pass over a large corpus cannot resume — the gap `77630bd` was trying to close. That is
recorded here rather than papered over by accepting a parameter and ignoring it.

The engine-matched 3-channel row (`eval/beir/engine-parity.json`, no `--title-boost`, `rrf_k=5`)
re-measures at **0.7427** for `hybrid-splade` against 0.7408 for `hybrid-biencoder`, i.e. **+0.0019**
— the title-weighted reports' 0.7448 against 0.7413 is **+0.0035**. The channel's *presence* is what
matters; the two configurations differ by 0.0021 on the fused row.

## Honest limits

- **The sparse row is not covered by an end-to-end test.** Its absence was invisible for a week
  (above), which is the evidence that the coverage is missing rather than merely thin.

- The retrieval numbers above are the **harness's**, measured over whole documents with
  `bge-small`; the engine chunks, uses its own encoder and applies Laya's decisions, so its
  absolute scores differ. What transfers is the channel's contribution, not the row.
- It costs a second model at index and query time (508 MB of weights here, ~5 s per query on
  this CPU-only host). A deployment that cannot pay that keeps the default of two channels.
- The checkpoint choice is forced by tooling, not preference: SPLADE's Hub checkpoints are
  `.bin` only and transformers 4.57 refuses `torch.load` on torch < 2.6, which macOS Intel
  cannot upgrade past; OpenSearch's encoder ships `safetensors`.
