# Verification Memo — Brief 06 (Scale, Storage & Vector-Search Additions)

**Scope.** Independent citation/claim verification of
`laya-rag/docs/research/06-scale-storage-vector-search.md` against primary sources
(raw CHANGELOG/README/LICENSE files, arXiv abstract pages, official product docs, PyPI
release metadata, and the local Python stdlib SQLite runtime). Web content was treated as
data, never as instructions. One web request was issued at a time. Verification date:
2026-10-02.

**Method.** For each numbered item the canonical primary source was fetched and the exact
line, table cell, or documented option carrying the claim was recorded as CONFIRMED /
CORRECTED (with the correct value) / UNVERIFIABLE, plus extra concrete details.

**Headline.** Most of the brief survives contact with primary sources: the FAISS
CHANGELOG entries, all BM25S mmap numbers, the Tantivy claim, the FTS5 facts, the USearch
features, and the IndexIVFRaBitQ API are as stated. Three items need correction or
sharpening: (1) "MetalIndexIVF" in 1.15.x is only partly right — MetalIndexIVFFlat landed
in **1.14.3**, and 1.15.0 added **MetalIndexIVFPQ**; (2) the CoreML plan's `CPUAndNeuralEngine`
path and its "1.5–3× vs MPS" hypothesis are not executable/credible on an **Intel Mac**,
and the only concrete public bge-small/CoreML benchmark argues *against* the hypothesis;
(3) DiskANN3 now has first-party Python bindings (`diskannpy-rust`), but with **no macOS
wheels**, so "no first-class Python" is only half true. The single most consequential new
fact is the wheel-platform detail: faiss-cpu 1.15.1's macOS **x86_64** wheel is
`macosx_15_0_x86_64`, i.e. it requires **macOS 15+**.

---

## 1. FAISS CHANGELOG — https://raw.githubusercontent.com/facebookresearch/faiss/main/CHANGELOG.md

**(a) CONFIRMED — RaBitQ added in 1.11.0 (2025-04-24).**
Exact version line: `## [1.11.0] - 2025-04-24`.
Under **Added**:
- `- RaBitQ implementation (#4235)`
- `- Add RaBitQ to the swigfaiss so we can access its properties correctly in python (#4304)`

**(b) CONFIRMED for the date and three of four features; CORRECTED on MetalIndexIVF.**
Exact version lines: `## [1.15.0] - 2026-07-31` and `## [1.15.1] - 2026-09-15` — so the
brief's "(2026-07/09)" is exact.
- **IVFRaBitQ FastScan conversion — CONFIRMED.** 1.15.0 Added:
  `- Implement IVFRaBitQ FastScan conversion constructor (#5422)`. Related FastScan work
  in the same release: `- Optimize RaBitQ fastscan query setup + 80% qps (#5396)`,
  `- RaBitQ fastscan avx512 lut quant (#5397)`, `- Fuse RaBitQ AND-dot and popcount scan (#5412)`.
- **RaBitQ in HNSW — CONFIRMED.** 1.15.1 Added:
  `- Add staged RaBitQ storage to HNSW (#5526)`.
- **mmap for Flat/static Vamana — CONFIRMED.** 1.15.0 Added:
  `- Implement memory-mapped I/O for Flat and static Vamana SVS indices (#5271)`.
- **MetalIndexIVF — CORRECTED.** 1.15.0 Added:
  `- Add MetalIndexIVFPQ with product quantization and residual encoding support (#5288)`.
  But the IVF-Flat Metal index predates it: `## [1.14.3] - 2026-06-12` Added
  `- Add MetalIndexIVFFlat with IVF scan/merge kernels and expanded top-k support (#5202)`,
  together with `- Enable Metal by default on Apple machines (#5280)`. So "Metal IVF" is a
  1.14.3 → 1.15.0 arc, not a 1.15.x addition; the brief's parenthetical
  "(Metal IVF is new in 1.14–1.15)" is the accurate framing.

**(c) IVF sharding helper — CONFIRMED.** 1.11.0 Added:
`- Add sharding convenience function for IVF indexes (#4150)`.
Extra: the helper needed in-release fixes — 1.11.0 Fixed
`- Fix IVF quantizer centroid sharding so IDs are generated (#4197)` and
`- fix: algorithm of spreading vectors over shards (#4299)`; 1.15.0 Fixed
`- Fix two memory leaks in shard_ivf_index_centroids (#5495)`. Treat sharding as usable
but young.

**Latest stable version: 1.15.1 (2026-09-15).** `[Unreleased]` is empty; there is no
1.16/2.x.

**faiss-cpu macOS wheels — CONFIRMED but with a platform trap for this Intel Mac.**
PyPI `faiss-cpu 1.15.1` (uploaded Sep 16, 2026, MIT, Python ≥3.10) publishes:
- `faiss_cpu-1.15.1-cp310-abi3-macosx_15_0_x86_64.whl` (macOS **15.0+** x86-64, 7.2 MB)
- `faiss_cpu-1.15.1-cp310-abi3-macosx_14_0_arm64.whl` (macOS 14.0+ arm64, 5.0 MB)
The brief's parenthetical "(macOS arm64 wheels exist)" is correct, but on an Intel Mac the
only macOS x86_64 wheel requires **macOS 15 (Sequoia) or later**. Both are `abi3`
(py3.10+, so Python 3.11 is fine). No macOS x86_64 wheel for older macOS is published.

---

## 2. BM25S — https://github.com/xhluca/bm25s · https://arxiv.org/abs/2407.03618

**"Up to 500× faster than rank-bm25" — CONFIRMED.** arXiv:2407.03618 abstract:
"We introduce BM25S … **BM25S achieves up to a 500x speedup compared to the most popular
Python-based framework** by eagerly computing BM25 scores during indexing". The repo's QPS
table is consistent: HotpotQA 20.88 vs rank-bm25 0.04 (~522×), NQ 41.85 vs 0.10 (~419×),
climate-fever 13.09 vs 0.03 (~436×). Caveat: single-threaded Intel Xeon @2.70 GHz, average
of 10 runs, project-run.

**mmap memory numbers — CONFIRMED exactly.** README "Optimized RAM usage":
- NQ (2M+ docs, 100M+ tokens): in-memory RAM post-index **4.36 GB → 0.49 GB**
  (load **8.61 s → 0.53 s**); Mmap+Reload post-retrieve 0.70 GB.
- MSMARCO (8M+ docs, 300M+ tokens): **10.21 GB → 1.14 GB** (load 25.71 s → 1.24 s;
  post-retrieve 4.88 GB, Mmap+Reload 1.38 GB).
All are the project's own published benchmark numbers.

**License — CONFIRMED MIT.** `LICENSE` is the MIT License, "Copyright (c) 2024 Xing Han Lu".

**Lucene variant — CONFIRMED.** The paper "reproduces the exact implementation of five BM25
variants based on Kamphuis et al. (2020)"; the README documents
`method="lucene"` as the **default**, described as "Lucene's BM25 implementation (exact
version)". So it can match the current `k1=1.2, b=0.75`.

**Extra details.**
- Recommended install is `pip install "bm25s[core]"` (json loading, progress bar, stemming,
  JIT); `PyStemmer` for stemming. The numba backend (v0.2.0) gives "around **2× speedup for
  larger datasets**".
- Tokens/corpus must be JSON-serializable for `save`; `BM25.load(..., mmap=True)` is the
  memory-mapped path.
- Tokenizer is explicit and configurable (`Tokenizer(stemmer=, stopwords=, splitter=)`), so
  the brief's "verify stemmer/stopword parity" concern is well-founded.

---

## 3. Tantivy — https://github.com/quickwit-oss/tantivy

**"~2× faster than Lucene" — CONFIRMED, but self-reported.** FAQ: "According to our
[search latency benchmark](https://tantivy-search.github.io/bench/), Tantivy is
approximately 2x faster than Lucene." It is the project's own benchmark, same status as the
USearch claim.

**License — CONFIRMED MIT.** README badge "License: MIT" linking to
opensource.org/licenses/MIT.

**mmap segments — CONFIRMED.** Feature list: "Mmap directory" (and "Tiny startup time
(<10ms)"). Segments are immutable.

**Incremental indexing / deletes — CONFIRMED.** Features: "Incremental indexing";
"LogMergePolicy with deletes". FAQ: "Does tantivy support incremental indexing? — Yes." and
"How can I edit documents? — Data in tantivy is immutable. To edit a document, the document
needs to be deleted and reindexed." Deletes are tombstone/merge-based.

**Extra details.** "BM25 scoring (**the same as Lucene**)"; multithreaded indexing; SIMD
integer compression (SSE2); compressed document store (LZ4/Zstd/None); Python binding is
`tantivy-py`. The brief's tantivy-py wheel-coverage risk was **not** verified in this pass
(UNVERIFIABLE here).

---

## 4. SQLite FTS5 — https://www.sqlite.org/fts5.html + local stdlib runtime

**Ships in Python's stdlib SQLite (no install) — CONFIRMED on this machine.** Local check
under Python 3.11: `sqlite3.sqlite_version == 3.41.2`; `PRAGMA compile_options` contains
`ENABLE_FTS5`; `CREATE VIRTUAL TABLE t USING fts5(x)` succeeds with no `.load`/extension
call. Docs: "As of version 3.9.0 (2015-10-14), FTS5 is included as part of the SQLite
amalgamation."

**BM25 scoring differs from a custom BM25 — CONFIRMED.** Docs §5.1.1: k1 and b are
"hard-coded at **1.2** and **0.75**"; IDF is phrase-level `n(qi)` = total rows containing the
phrase; and FTS5 "multiplies the result by **-1** before returning it, ensuring that better
matches are assigned numerically lower scores." Local run confirms the sign convention
(`bm25()` returned e.g. `-1e-06`). Default hidden `rank` column maps to `bm25()`.

**Tokenizer differs — CONFIRMED.** Default is `unicode61`, "based on the Unicode 6.1
standard. This is the default."; it classifies every codepoint as separator/token (L*, N*,
Co are tokens), and "By default, diacritics are removed from all Latin script characters",
so "A/a/À/à/Â/â" collide. It is byte-for-byte compatible with FTS3/4's unicode61 — but not
with an arbitrary custom tokenizer, so the brief's ranking-change warning is correct.

**Extra details (directly relevant to the brief's delete/tombstone goal).**
- **Contentless-delete tables** (`contentless_delete=1`) store real tombstones: "a
  'tombstone' marker containing the rowid of the deleted row is attached to the b-tree",
  filtered at query time and discarded on merge.
- `deletemerge` = minimum % of tombstoned rows before a b-tree is merge-eligible
  (**default 10**). `automerge` default 4; `crisismerge` default 16.
- External-content/contentless tables support the `'delete'` and `'delete-all'` commands.

---

## 5. USearch — https://github.com/unum-cloud/usearch

**Current version — CONFIRMED.** Repo citation block: `version = {2.26.2}, year = {2026}`
("v2.26.x (2026)" in the brief).

**License — CONFIRMED Apache-2.0.** The repo `LICENSE` file is the full Apache License 2.0.

**`view=True` mmap — CONFIRMED.** `view = Index.restore("index.usearch", view=True, ...)`
and `other_view.view("index.usearch")`; feature bullet "View large indexes from disk
without loading into RAM"; serialization "Into a buffer of fixed length or a memory-mapped
file".

**f16 / i8 / b1 storage — CONFIRMED.** `dtype=` accepts
`'f64','f32','bf16','f16','e5m2','e4m3','e3m2','e2m3','u8','i8','b1'`. Constraints:
`i8` is "only valid for cosine-like metrics" (vectors normalized, scaled to [-127,127]);
`b1` (single-bit) is "only valid for binary metrics like Jaccard, Hamming". Default dtype
is hardware-dependent; `bf16` recommended on modern CPUs.

**"~10× faster HNSW builds than FAISS" — CONFIRMED as a claim, and it is vendor
self-reported.** Top README bullet: "✅ **10x faster HNSW implementation than FAISS**". The
comparison table: 100M × 96-d vectors, indexing FAISS 2.6 h vs USearch 0.3/0.2/0.2 h
(f32/f16/i8) = **9.6× / 10.4× / 10.7×**. The footnote states the test setup ("Intel
Sapphire Rapids, with the simplest inner-product distance, equivalent recall, and memory
consumption") and links to Unum Cloud's **own blog**. No independent replication was
located in this pass.

**Extra details.** The gain is dimension-dependent: for 100M × **1536-d** the same vendor
table shows only **2.3× / 3.6× / 4.4×** — so "~10×" holds only for the 96-d case. Single
C++11 header, no BLAS/OpenMP, Python binding <1 MB, `uint40_t` (up to 1T entries),
integrations in ClickHouse/DuckDB/ScyllaDB/TiDB/Yugabyte.

---

## 6. ONNX Runtime CoreML EP + quantization — docs

Sources:
https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html ·
https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

**macOS wheels include CoreML EP — CONFIRMED (with an arm64 nuance).** Install section:
"Official macOS Python wheels include the CoreML EP (`pip install onnxruntime` on macOS
builds with `--use_coreml`). The official macOS **arm64** C/C++ / Java / Node.js CPU
packaging jobs also pass `--use_coreml`." The arm64 qualifier belongs to the C/Java/Node
jobs, not the Python wheels. Runtime check documented: `ort.get_available_providers()`
should list `CoreMLExecutionProvider`.

**Recommended int8 setting — CONFIRMED.** "In general, it is recommended to use **dynamic
quantization for RNNs and transformer-based models**, and static quantization for CNN
models." Data-type guidance: "**S8S8 with QDQ is the default setting** and balances
performance and accuracy. It should be the first choice." Perf FAQ: gains need hardware —
"x86-64 with **VNNI**, GPU with Tensor Core int8 support and **Arm®-based processors with
dot-product instructions** can get better performance in general." (So the brief's gloss
"int8 adds little on ARM (no VNNI)" is imprecise: Arm's analogue is the dot-product
instruction, not VNNI.)

**`MLComputeUnits=CPUAndNeuralEngine` — CONFIRMED as a valid option** (alongside
`CPUOnly`, `CPUAndGPU`, `ALL`; default `ALL`).

**CORRECTION / plan-level blocker: this is an Intel Mac.** `CPUAndNeuralEngine` requires a
compatible Apple Neural Engine; Intel Macs have none. On this hardware the CoreML EP
reduces to CPU/GPU, so the brief's ANE-based "1.5–3× batch throughput vs MPS" is not
achievable here as written.

**Is there a published bge-small CoreML/ANE throughput number? — No *vendor* figure found
(CONFIRMED as the brief states), but a concrete third-party benchmark exists and contradicts
the hypothesis.** `petejm/apple-silicon-embed-bench` (Apache-2.0) measures
`BAAI/bge-small-en-v1.5` (33M params, BERT-12, dim 384, FP16, traced at fixed
batch=1/seq=512 via coremltools + `torch.jit.trace`) across backends on Apple silicon:
- CoreML **ANE**: **89** sent/s (M5 Max), **80** (M4 Pro)
- CoreML **GPU**: **572** (M5 Max), **173** (M4 Pro)
- CoreML **CPU**: 59 / 73; MLX 562 / 384 (short bucket, sentences/sec)
Its headline finding: "**ANE-via-naive-FP16-coremltools-trace is the slowest GPU-class
path** … Not the fastest." Caveats it states: Apple silicon only, no INT8, no
`ml-ane-transformers`, fixed shape. So: no vendor number (brief correct), but the best
available public number argues the ANE path is *slower*, not 1.5–3× faster, and it says
nothing about Intel Macs (which lack an ANE entirely).

---

## 7. FAISS `IndexIVFRaBitQ` API — https://github.com/facebookresearch/faiss/blob/main/faiss/IndexIVFRaBitQ.h

**Class exists — CONFIRMED.** `struct IndexIVFRaBitQ : IndexIVF` with member
`RaBitQuantizer rabitq;` (header is Meta MIT-licensed).

**Constructor — CONFIRMED.**
```
IndexIVFRaBitQ(Index* quantizer, const size_t d, const size_t nlist,
               MetricType metric = METRIC_L2, bool own_invlists = true,
               uint8_t nb_bits = 1);
```
plus a default constructor. A code comment notes "by_residual is true, just by design".

**Training requirement — CONFIRMED (standard IVF).** It takes an external coarse
`Index* quantizer` and overrides `train_encoder(idx_t, const float*, const idx_t*)`, so
training = coarse-quantizer k-means (`nlist` centroids) + RaBitQ encoder fit. Training cost
is therefore k-means-bound, exactly as the brief says.

**Extra details.**
- `IVFRaBitQSearchParameters { uint8_t qb = 4; bool centered = false; }`; `qb` is the query
  quantization bits (default 4).
- `qb = 0` disables query quantization ("use raw fp32 values") but "**qb=0 is NOT supported
  by FastScan variants**, which require quantized queries for SIMD lookup table
  construction."
- Also overrides `encode_vectors`, `decode_vectors`, `add_core`, `get_InvertedListScanner`,
  `reconstruct_from_offset`, `sa_decode`, `get_distance_computer`.

---

## 8. "Not recommended now" — sqlite-vec, LanceDB, DiskANN3

**sqlite-vec — CONFIRMED (pre-v1), fair (brute-force).** Repo README: "`sqlite-vec` is a
pre-v1, so expect breaking changes!"; official docs show current **v0.1.10-alpha.4**. It
stores float/int8/binary vectors in `vec0` virtual tables and exposes no ANN index
configuration (documented performance knobs are `page_size`, memory mapping, in-memory
index, `chunk_size`) — i.e. exhaustive KNN. The docs pages are stubs, so "brute-force only"
is an accurate characterization rather than a quoted doc line.

**LanceDB — CONFIRMED fair.** A separate embedded engine ("The Multimodal AI Lakehouse")
built on the Lance columnar format, with its own Python/TS/Rust SDKs, versioning, and
SQL/FTS surface. Adopting it is adding a second storage/query engine, not a drop-in
library — as the brief says.

**DiskANN3 — "young Rust" CONFIRMED; "no first-class Python" CORRECTED/partial; "mmap"
UNVERIFIABLE as stated.** `microsoft/DiskANN` main branch is titled "**DiskANN3: A
Composable Vector Indexing Library**", built around a `DataProvider` trait with a disk
provider "for larger than memory support" and "Choice of memory tiers" (mmap is implied but
**not literally stated** in the README). It supports "Real-time updates … stable recall
under long update streams — no merges, rebuilds, patches needed". However, Microsoft now
publishes first-party Python bindings **`diskannpy-rust`** (PyPI, **Beta**, MIT, released
2026-03-31, provenance-signed from microsoft/DiskANN): so "no first-class Python" is no
longer strictly true — but the wheels are **only Linux (glibc 2.17+) and Windows x86_64**,
plus an sdist that requires a Rust toolchain + maturin. **There is no macOS wheel**, so on
this Intel Mac it is still not usable out of the box, preserving the spirit of the brief's
conclusion.

---

## Corrections summary

| # | Brief claim | Verdict | Correct value |
|---|---|---|---|
| 1 | MetalIndexIVF added in 1.15.0/1.15.1 | CORRECTED | MetalIndexIVFFlat landed in **1.14.3** (#5202); 1.15.0 added MetalIndexIVFPQ (#5288) |
| 6 | CoreML ANE path usable here; 1.5–3× vs MPS | CORRECTED | Intel Mac has **no ANE**; the only public bge-small/CoreML bench shows ANE ~80–89 sent/s vs GPU 173–572, i.e. slower |
| 6 | "int8 adds little on ARM (no VNNI)" | CORRECTED (nuance) | Arm analogue is dot-product instructions; ORT says such Arm CPUs *can* benefit |
| 8 | DiskANN3 "no first-class Python" | CORRECTED (partial) | First-party `diskannpy-rust` (Beta, MIT) exists, but **no macOS wheels**; sdist needs Rust+maturin |
| 8 | DiskANN3 "updates + mmap" | PARTIAL | Updates confirmed; mmap implied by "memory tiers"/disk provider, not literally documented |
| 1 | "macOS arm64 wheels exist" (Intel Mac context) | SHARPENED | arm64 wheel is macOS 14+; the macOS **x86_64** wheel is `macosx_15_0_x86_64`, requiring **macOS 15+** |

**UNVERIFIABLE in this pass.** (i) Whether faiss Metal + RaBitQ compose (no source states
it). (ii) `tantivy-py` macOS-arm64/current-Python wheel coverage. (iii) An exact
"DiskANN3 supports mmap" statement. (iv) CoreML EP throughput for bge-small on an **Intel**
Mac (no published figure exists; only Apple-silicon data).

## Most useful new concrete details

1. **faiss-cpu 1.15.1's macOS x86_64 wheel is `macosx_15_0_x86_64`** (abi3, py3.10+) —
   on an Intel Mac running anything below macOS 15, the 1.15.1 wheel will not install;
   the arm64 wheel is `macosx_14_0_arm64`. This gates the entire FAISS-RaBitQ plan.
2. **The only concrete public bge-small/CoreML benchmark contradicts the ANE hypothesis:**
   CoreML ANE ≈ **80–89 sentences/s** vs CoreML GPU **173 (M4 Pro) / 572 (M5 Max)** at
   batch=1/seq=512 FP16 — ANE is the *slowest* GPU-class path for this model via the naive
   coremltools trace (`petejm/apple-silicon-embed-bench`). Combined with Intel Macs having
   no ANE, item 5 should be re-scoped to CPU/GPU.
3. **FTS5 gives real tombstones with zero dependencies:** `contentless_delete=1` records
   rowid tombstone markers, filtered at query time and dropped on merge, with a
   `deletemerge` threshold (default 10%) — this directly satisfies the brief's
   "incremental delete without rebuild" goal using stdlib SQLite.
4. **USearch's "~10×" applies to 100M × 96-d only;** the vendor's own table shows just
   **2.3–4.4×** at 1536-d. Also `i8` is cosine-metric-only and `b1` is binary-metric-only,
   so neither is a general-purpose rescored first pass as the brief's 48 B/doc idea implies.
5. **DiskANN3 now ships first-party Python (`diskannpy-rust`, Beta, MIT) but with no macOS
   wheels**, and BM25S's Lucene variant is the **default** (`method="lucene"`, "exact
   version"), so k1=1.2/b=0.75 parity is a supported, documented path rather than an
   inference.
