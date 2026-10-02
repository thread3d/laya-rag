# Scale, Storage & Vector-Search Additions for laya-rag (2024–2026)

Ranked by leverage per risk. **[RP]** = ranking-preserving (same channel maths, different residency/engine; top-k should diff only on ties). **[Q]** = changes scores/recall — re-benchmark before defaulting.

## 1. [RP] FAISS 1.15 on-disk quantized ANN: IVF-RaBitQ FastScan + OPQ + mmap
- **What:** Replace `ann.faiss` IVFPQ with `IndexPreTransform(OPQ, IVFRaBitQ)`, memory-mapped. Metal IVF only if the platform supports it.
- **Gap & mechanism:** Closes the 25 GB fp32 dense residency (BioASQ) and the exact-mmap fallback's linear scan. RaBitQ encodes D dims → D bits (384 d → 48 B; 1 bit/dim, 4-bit query) with an unbiased estimator and an O(1/√D) error bound; OPQ rotates before quantisation; FastScan scans codes without decoding; mmap keeps codes out of RSS.
- **Evidence:** FAISS 1.11.0 (2025-04-24) added RaBitQ; 1.15.0/1.15.1 (2026-07/09) added IVFRaBitQ FastScan, RaBitQ-in-HNSW, mmap Flat/static Vamana. **Metal IVF is not a 1.15 addition:** `MetalIndexIVFFlat` landed in **1.14.3 (2026-06-12)** and `MetalIndexIVFPQ` in **1.15.0** — and Metal+RaBitQ composing is nowhere asserted. [CHANGELOG](https://raw.githubusercontent.com/facebookresearch/faiss/main/CHANGELOG.md) · [IndexIVFRaBitQ.h](https://github.com/facebookresearch/faiss/blob/main/faiss/IndexIVFRaBitQ.h) · [RaBitQ, SIGMOD'24](https://arxiv.org/abs/2405.12497) · [extended RaBitQ](https://arxiv.org/abs/2409.09913).
- **Cost:** `faiss-cpu>=1.11` (already an `eval` extra). **Verified locally (2026-10-02):** the installed `faiss-cpu 1.15.1` already exports `IndexIVFRaBitQ`, `IndexIVFRaBitQFastScan`, `IndexRaBitQFastScan` and `IndexHNSWRaBitQ`, so no upgrade is needed; the macOS **x86_64** wheel is tagged `macosx_15_0_x86_64` and this host reports **macOS 26.6.2**, so the wheel gate is satisfied. **`faiss-cpu` exposes no Metal classes**, and Metal+RaBitQ composing is unverified — drop "Metal IVF" from the plan. k-means-bound training: minutes at 1M vectors, hours at 10M on 56 cores.
- **Hypothesis:** BioASQ dense 25 GB → ~0.8–1.5 GB codes; recall@10 0.80–0.92 at 10–30× exhaustive speed (nprobe 32); query p50 <10 ms from page cache.
- **Falsify / scepticism:** Recall@10 vs `IndexFlatIP` on `tools/ann_bench.py` synthetics plus a 100k BEIR sample; reject <0.75 at 10×. Check the numpy<2 pin against current wheels and ARM SIMD maturity.

## 2. [RP] BM25S mmapped sparse lexical channel
- **What:** Swap `BM25Index._postings` (Python dicts) for BM25S eager sparse scoring with `mmap=True`.
- **Gap & mechanism:** Removes the Python graph scaling with vocabulary × postings. BM25S precomputes scores into SciPy sparse matrices and memory-maps them, so load is O(1) pages and query is a sparse matvec. It can reproduce the Lucene BM25 variant, so it can match the current `k1=1.2,b=0.75`.
- **Evidence:** MIT, active; paper claims up to 500× over `rank-bm25`. Published mmap: NQ 2M docs 4.36 GB → **0.49 GB** (load 8.61 s → 0.53 s); MSMARCO 8M docs 10.21 GB → **1.14 GB**. [repo](https://github.com/xhluca/bm25s) · [arXiv 2407.03618](https://arxiv.org/abs/2407.03618).
- **Cost:** `pip install bm25s` (numpy+scipy, optional numba). Keep `BM25Index` for tests/small corpora behind `INDEX_BACKEND`.
- **Hypothesis:** FEVER/NQ lexical build peak drops from multi-GB dicts to ~0.8–1.5 GB mmap; query latency holds or improves.
- **Falsify / scepticism:** Diff top-100 for 100 queries; eager scoring reorders float accumulation, so expect tie noise, not byte-identical. Verify stemmer/stopword parity (BM25S's tokenizer differs from `lexical.py`) and numba/scipy wheels against the pinned numpy/Python.

## 3. [RP] On-disk term dictionary + tombstones via SQLite FTS5 or Tantivy
- **What:** Replace the residual `terms.json` dict and chunk-id list with a disk-resident dictionary, plus incremental delete.
- **Gap & mechanism:** The documented 10M-doc wall is the term dict and id list. FTS5 (stdlib) keeps postings in a B-tree with incremental deletes; Tantivy adds mmap segments, compressed stores and explicit deletes/tombstones.
- **Evidence:** FTS5 ships with Python's SQLite and delivers **real tombstones with zero deps**: `contentless_delete=1` records rowid tombstone markers (filtered at query time, dropped on merge, `deletemerge` default 10%) — exactly the incremental-delete goal. Note its `bm25()` is **negated** (better = more negative; k1=1.2/b=0.75 hard-coded) and its `unicode61` tokenizer diacritic-folds. Tantivy is MIT, ~2× faster than Lucene, incremental + mmap; Pyserini/Lucene 10.5 is research-grade but needs JDK 21. [Tantivy](https://github.com/quickwit-oss/tantivy) · [Pyserini](https://github.com/castorini/pyserini).
- **Cost:** FTS5 hours; Tantivy days plus a Rust/`tantivy-py` wheel; Pyserini heavy JVM + PyTorch, not embedded-friendly.
- **Hypothesis:** Term-dict + id-list RSS (~0.5–1 GB at 14.9M docs) → near-zero resident; deletes O(docs) instead of requiring rebuild.
- **Falsify / scepticism:** FTS5's BM25 and `unicode61` tokenizer *change* rankings — only "ranking-preserving" if you keep your own scoring over its postings, else accept a measured diff. `tantivy-py` macOS-arm64/current-Python wheel coverage is the main risk.

## 4. [RP] Streaming BEIR ingestion + Parquet/SQLite artifacts + sharded builds
- **What:** Iterate `corpus.jsonl` into positional memmaps, score CE pairs per query batch, persist scores to Parquet/SQLite, shard the build across processes.
- **Gap & mechanism:** The harness holds whole corpus string+dict, full `.npy`, and all CE pairs. Streaming removes the 2–3× peak; sharding uses the 56 cores the single-process build ignores; columnar artifacts make 5M-row runs analysable.
- **Evidence:** PyArrow Parquet is the standard columnar path; FAISS 1.11 added a sharding convenience function for IVF. [Parquet](https://arrow.apache.org/docs/python/parquet.html) · [FAISS CHANGELOG](https://raw.githubusercontent.com/facebookresearch/faiss/main/CHANGELOG.md).
- **Cost:** days; `pyarrow` (~100 MB) or stdlib `sqlite3` (zero deps).
- **Hypothesis:** BEIR peak RSS halved at equal throughput; 8–16 shards cut build wall-time 4–8× on 56 cores.
- **Falsify / scepticism:** Compare nDCG@10 and bytes per shard count; pin tie ordering if merges shift it. Parquet may not be worth its weight over SQLite for ~5M rows.

## 5. [RP] GPU-free embedding inference: ONNX Runtime + CoreML EP / int8
- **What:** Export bge-small/base to ONNX and run `CoreMLExecutionProvider(ModelFormat=MLProgram, MLComputeUnits=CPUAndNeuralEngine)`, comparing int8 dynamic quantisation.
- **Gap & mechanism:** The embedding pass, not the index, makes 5.4M-doc builds take hours. CoreML uses ANE/GPU without CUDA; dynamic int8 quantises MatMul/Gather weights for CPU.
- **Evidence:** Official macOS arm64 ORT wheels include CoreML EP; docs recommend dynamic int8 for transformers and ANE hardware for best performance. **But this project's host is an Intel Mac**, so `CPUAndNeuralEngine` has no ANE to target and CoreML reduces to CPU/GPU. The only concrete public bge-small/CoreML bench (apple-silicon-embed-bench) shows ANE ≈80–89 sent/s vs GPU 173–572 — the ANE path is *slower*, not 1.5–3× faster than MPS. [CoreML EP](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) · [Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
- **Cost:** days; `onnxruntime` + `optimum`/`torch.onnx.export`; keep sentence-transformers as fallback.
- **Hypothesis (now weak):** 1.5–3× batch throughput vs MPS for short chunks on Apple Silicon; not achievable on this Intel host. int8 can help Arm CPUs via dot-product instructions (not only x86 VNNI), but the gain is unpredicted here.
- **Falsify / scepticism:** Benchmark tokens/s and nDCG@10 on SciFact/NFCorpus. ANE dispatch for dynamic-shape BERT is partial and CoreML may be slower than MPS for small batches; no vendor figure exists for this model. On this host, deprioritise below the other items.

## 6. [RP] USearch mmap-view HNSW (lighter ANN alternative)
- **What:** Optional HNSW with `Index.restore(..., view=True)` serving a multi-GB index from disk.
- **Gap & mechanism:** Single-header C++, no BLAS/OpenMP, `bf16`/`f16`/`i8`/`b1` storage, predicate filters, explicit view mode. Fills the gap where IVF-PQ tuning is too coarse and HNSW recall is wanted.
- **Evidence:** Apache-2.0, v2.26.x (2026), integrations in DuckDB/ClickHouse; project reports ~10× faster HNSW builds than FAISS — but **only at 96 d**; its own table shows **2.3–4.4× at 1536 d**. Also `i8` storage is cosine-only and `b1` is binary-metric-only, so `b1` is not a general 48 B/doc rescored pass. [repo](https://github.com/unum-cloud/usearch).
- **Cost:** days; one wheel, binding <1 MB, no BLAS.
- **Hypothesis:** ≥0.9 recall@10 at 1–5 ms with f16 ≈ 768 B/doc; `b1` ≈ 48 B/doc as a rescored first pass.
- **Falsify / scepticism:** Benchmark against existing IVFPQ on the same corpora; vendor comparisons are self-reported; compaction/delete semantics differ from a tombstone file; single-maintainer project.

## 7. [Q] Binary/int8 embedding quantisation with rescoring
- **What:** Store documents as binary (384 d → 48 B) or int8 (384 B), retrieve by Hamming/int8, then rescore a shortlist with the fp32 query.
- **Gap & mechanism:** Directly cuts the 1.5 KB/doc dense cost 8–32×; rescoring recovers most quality.
- **Evidence:** Binary + rescoring preserves **up to ~96%** of retrieval quality at **32×** memory/disk reduction and up to 32× speed (≈92.5% without rescoring) — but the figure is **model-specific**: 96.45% for mxbai-embed-large-v1, 93.79% (all-MiniLM-L6-v2), 87.7% (nomic-v1.5), 74.77% (e5-base-v2). [HF blog, 2024-03-22](https://huggingface.co/blog/embedding-quantization).
- **Cost:** days; `sentence_transformers.quantization.quantize_embeddings`, no new native deps.
- **Hypothesis:** BioASQ dense 25 GB → 0.8 GB (binary) / 5.7 GB (int8); nDCG@10 within 2–4% of fp32 after rescoring.
- **Falsify / scepticism:** It changes recall — measure nDCG@10 with a 10× rescore multiplier on all five BEIR sets first. Binary favours high-dim contrastive encoders (bge/mxbai); keep fp32 for the reranker.

## 8. [Q] Matryoshka encoder + dimension truncation
- **What:** Route to an MRL encoder (nomic-embed-text-v1.5, 768 d; mxbai-embed-large-v1, 1024 d) and truncate to 128/256 d for pass 1.
- **Gap & mechanism:** MRL trains coarse-to-fine so prefixes are usable; truncation shrinks storage and scan cost, and the full vector can rescore. bge-small-en-v1.5 is *not* MRL-trained, so this is a model swap.
- **Evidence:** MRL: up to 14× smaller embeddings at equal accuracy and 14× retrieval speedups. OpenAI text-embedding-3 retains 93.1% at 12× compression; nomic-v1.5 95.8% at 3×, 90% at 6×. [MRL v4, 2024](https://arxiv.org/abs/2205.13147) · [HF blog](https://huggingface.co/blog/embedding-quantization).
- **Cost:** days–weeks (re-embed + routing); new Apache-2.0 weights (~0.1–0.3 GB).
- **Hypothesis:** 384→128 d cuts dense bytes and scan ~3× at ≤1–2% nDCG; with int8, ~0.5 GB for 14.9M docs.
- **Falsify / scepticism:** Must beat bge-small at full dim on the five BEIR sets *before* truncation; MRL gains are corpus-dependent and this repo already measured encoder-size saturation. Re-embedding invalidates all cached vectors.

## Not recommended now
**sqlite-vec** (pre-v1, brute-force only) and **LanceDB** (a second engine) are fine for small/local cases but don't close the 5M-doc gap; **DiskANN3** now ships a first-party `diskannpy-rust` (Beta, MIT, 2026-03) but **only Linux/Windows x86_64 wheels** (no macOS) and its mmap support is implied, not documented; **Qdrant/Milvus/Chroma/pgvector** are server- or Postgres-shaped, wrong for a single-process local engine.
