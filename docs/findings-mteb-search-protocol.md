# The MTEB v2 `SearchProtocol` adapter

MTEB's standing objection to this project was that "MTEB scores a model, not an engine, so this
system cannot be entered as-is" (`ROADMAP.md` §10, `BENCHMARKS.md`). MTEB v2 defines a
`SearchProtocol` alongside the encoder and cross-encoder protocols, so that objection expires: an
object that exposes `index` and `search` is first-class representable, and what gets scored is the
retrieval *system* — chunking, fusion, the second stage and the gate — rather than one embedding
model. `tools/mteb_search.py` (extra `.[mteb]`, `make mteb-eval`) is that adapter.

## What it does

- `index(corpus, ...)` writes the corpus once, by position, to a temporary directory (the same
  convention `tools/engine_beir.py` uses) and builds the engine's index over it.
- `search(queries, ..., top_k=100)` runs the engine's own query path and returns
  `{query_id: {doc_id: score}}`, mapping returned passage paths back to corpus ids through the
  write order. It passes `relevance_threshold=0.0` so the benchmark scores the *ranking* rather
  than the gate's kept set, and widens `retrieve_k` to at least `top_k`.
- `build_model_meta()` constructs the `ModelMeta` lazily — 322M parameters, Apache-2.0, an
  8192-token window, `model_type=["hybrid"]`, `languages=["eng-Latn"]` (honest: the lexical
  instrumentation is English-only) — so the adapter and its tests import nothing from MTEB.
- `load_laya_rag_search` is the loader; it ignores kwargs the adapter does not understand, so a
  typo in a submission argument cannot fail the run.

## Verification

Against **mteb 2.22.5**:

- `isinstance(LayaRagSearch(...), mteb.models.models_protocols.SearchProtocol)` is **True**;
- `ModelMeta.load_model()` returns a `LayaRagSearch` with the metadata assigned;
- **14 tests** pin the corpus plumbing (positional files, path→id mapping, dedup to the best score,
  `top_ranked` filtering, unknown-path rejection, temp cleanup) and the conformance itself.

The plumbing is the part that matters: an off-by-one between write order and passage path would
corrupt every MTEB number without a metric flagging it, so it is tested directly.

## The evaluation was attempted and did not complete here

A full `mteb.evaluate(model, task)` on SciFact was launched three ways — default, `num_proc=1`, and
`num_proc=1` with `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` — and each time MTEB forked **~33
worker processes** that then stalled: no result, no error, no progress output, for 30+ minutes
while the host had free memory and the corpus files had been written. The failure is in the
benchmark harness's process handling in this environment, not in the adapter: indexing completed
(5,183 corpus files on disk) and the structural conformance holds. The engine-level MTEB run, and
the upload, therefore remain the external step they were.

## What this does not show

- **No MTEB score exists** for the engine yet. The adapter is representable and verified; a number
  is not claimed.
- **`top_ranked` filtering is best-effort.** For reranking tasks the adapter runs the query and
  filters to the supplied candidate ids; a candidate outside the engine's own pool is omitted
  rather than pair-scored, because the engine has no pair-scoring path. Retrieval tasks, which pass
  `top_ranked=None`, are unaffected.
- **Each evaluation re-indexes.** The corpus is written to a temporary directory per run, so a
  large task pays the index build every time; a persistent backend for MTEB would be the next
  optimization if submissions become routine.
