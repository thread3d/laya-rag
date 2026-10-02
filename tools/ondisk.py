"""On-disk postings and streaming vector construction (item 8).

Every retrieval path in this repository holds the corpus in RAM: `BM25Index` keeps a
Python `term -> [(row, tf)]` dict, and `VectorIndex` keeps every chunk's text and vector
in one array. That is what puts BioASQ (14.9M documents), FEVER (5.4M) and NQ (2.7M) out
of reach — not the scoring math, the construction.

This module is the infrastructure route:

* :class:`OnDiskBM25` builds postings in blocks, spills each block to a sorted run file,
  then merges the runs into one packed ``postings.bin`` of ``(row, tf)`` int32 pairs with a
  term-offset table. Search memory-maps the postings and never materialises a Python dict of
  lists, so peak build memory is one block plus the term dictionary, not the whole index.

* :class:`StreamingVectorIndex` appends embedding batches straight to a memory-mapped
  ``.bin`` and searches it blockwise, so vectors are never all resident.

The CLI builds both for one BEIR dataset, reports wall time, on-disk size and peak RSS, and
checks the on-disk BM25 ranking against the in-memory one on the same queries — the disk
path has to produce the *same* ranking, not merely a plausible one.

    .venv/bin/python tools/ondisk.py --dataset trec-covid
"""
from __future__ import annotations

import argparse
import heapq
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import beir  # noqa: E402
import evalkit  # noqa: E402
from laya_rag.lexical import STOPWORDS, tokenize  # noqa: E402


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


# ── lexical: packed postings, mmap search ───────────────────────────────

class OnDiskBM25:
    """BM25 over a packed, memory-mapped postings file.

    Files under ``directory``: ``postings.bin`` (int32 ``(row, tf)`` pairs, grouped by
    term), ``offsets.npy`` (int64, ``n_terms + 1``), ``terms.json`` (term -> id),
    ``lengths.npy`` (int32 document lengths), ``meta.json`` (``k1``, ``b``, counts).
    """

    def __init__(
        self,
        directory: Path,
        k1: float = beir.BEIR_K1,
        b: float = beir.BEIR_B,
        drop_stopwords: bool = True,
        stem: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.k1 = float(k1)
        self.b = float(b)
        self.drop_stopwords = bool(drop_stopwords)
        self.stem = bool(stem)
        self.terms: dict[str, int] = {}
        self.offsets: np.ndarray | None = None
        self.lengths: np.ndarray | None = None
        self.avg_len = 0.0
        self.n_docs = 0
        self._postings: np.ndarray | None = None

    # -- build ----------------------------------------------------------
    def build(
        self,
        documents: Iterable[tuple[str, str]],
        block_docs: int = 50000,
        block_tokens: int = 1_000_000,
    ) -> None:
        """Stream ``(doc_id, text)`` through sorted runs, then merge them to packed form.

        A block flushes at ``block_docs`` documents or ``block_tokens`` tokens, whichever
        comes first. The token cap bounds the one build block that otherwise dominates
        peak RSS (a `term -> count` dict per document held in RAM); the merge is a heap
        over the sorted runs, so it holds one row per run rather than a run's column.
        """
        if block_docs < 1:
            raise ValueError(f"block_docs must be positive, got {block_docs}")
        if block_tokens < 1:
            raise ValueError(f"block_tokens must be positive, got {block_tokens}")
        self.directory.mkdir(parents=True, exist_ok=True)
        run_dir = self.directory / "runs"
        run_dir.mkdir(exist_ok=True)
        term_to_id: dict[str, int] = {}
        ids: list[str] = []
        lengths: list[int] = []
        doc_freq: list[int] = []
        runs: list[Path] = []
        block: list[tuple[str, Any]] = []
        block_start = 0
        block_token_count = 0

        def flush() -> None:
            if not block:
                return
            term_ids: list[int] = []
            rows: list[int] = []
            tfs: list[int] = []
            for offset, (doc_id, counts) in enumerate(block):
                row = block_start + offset
                ids.append(doc_id)
                lengths.append(sum(counts.values()))
                for term, tf in counts.items():
                    tid = term_to_id.get(term)
                    if tid is None:
                        tid = len(term_to_id)
                        term_to_id[term] = tid
                        doc_freq.append(0)
                    doc_freq[tid] += 1
                    term_ids.append(tid)
                    rows.append(row)
                    tfs.append(tf)
            arr = np.stack(
                [np.asarray(term_ids, np.int32), np.asarray(rows, np.int32), np.asarray(tfs, np.int32)],
                axis=1,
            )
            order = np.lexsort((arr[:, 1], arr[:, 0]))
            run_path = run_dir / f"run-{len(runs):04d}.npy"
            np.save(run_path, arr[order])
            runs.append(run_path)
            block.clear()

        for doc_id, text in documents:
            tokens = tokenize(text, stem=self.stem)
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            block.append((doc_id, counts))
            block_token_count += len(tokens)
            if len(block) >= block_docs or block_token_count >= block_tokens:
                flush()
                block_start = len(ids)
                block_token_count = 0
        flush()

        self.n_docs = len(ids)
        self.terms = term_to_id
        self.lengths = np.asarray(lengths, dtype=np.int32)
        self.avg_len = float(self.lengths.mean()) if len(self.lengths) else 0.0
        n_terms = len(term_to_id)
        df = np.asarray(doc_freq, dtype=np.int64)
        self._df = df
        offsets = np.zeros(n_terms + 1, dtype=np.int64)

        # K-way merge by (term id, run index). Popping the smallest run index first
        # drains a run's whole block for a term before the next run, reproducing the
        # previous per-run scan's order and therefore its ranking, while holding only
        # one head row per run.
        handles = [np.load(path, mmap_mode="r") for path in runs]
        positions = [0] * len(handles)
        heads: list[tuple[int, int, int, int]] = []
        for r, handle in enumerate(handles):
            if len(handle):
                row0 = handle[0]
                heapq.heappush(heads, (int(row0[0]), r, int(row0[1]), int(row0[2])))
                positions[r] = 1
        postings_path = self.directory / "postings.bin"
        written = 0
        closed_terms = 0
        with open(postings_path, "wb") as out:
            while heads:
                term, r, row, tf = heapq.heappop(heads)
                out.write(np.asarray([row, tf], dtype=np.int32).tobytes())
                written += 1
                handle = handles[r]
                position = positions[r]
                if position < len(handle):
                    nxt = handle[position]
                    heapq.heappush(heads, (int(nxt[0]), r, int(nxt[1]), int(nxt[2])))
                    positions[r] = position + 1
                if not heads or heads[0][0] != term:
                    for earlier in range(closed_terms, term + 1):
                        offsets[earlier + 1] = written
                    closed_terms = term + 1
        np.save(self.directory / "df.npy", df)
        np.save(self.directory / "offsets.npy", offsets)
        self.offsets = offsets
        self._postings = np.memmap(postings_path, dtype=np.int32, mode="r").reshape(-1, 2)
        (self.directory / "terms.json").write_text(json.dumps(term_to_id), encoding="utf-8")
        np.save(self.directory / "lengths.npy", self.lengths)
        (self.directory / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
        (self.directory / "meta.json").write_text(
            json.dumps({
                "version": 1,
                "n_docs": self.n_docs,
                "n_terms": n_terms,
                "n_postings": written,
                "avg_len": self.avg_len,
                "k1": self.k1,
                "b": self.b,
                "drop_stopwords": self.drop_stopwords,
                "stem": self.stem,
            }),
            encoding="utf-8",
        )
        for path in runs:
            path.unlink()
        run_dir.rmdir()
        for handle in handles:
            del handle

    # -- open / search --------------------------------------------------
    @classmethod
    def open(cls, directory: Path) -> "OnDiskBM25":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        index = cls(directory, k1=meta["k1"], b=meta["b"],
                    drop_stopwords=meta["drop_stopwords"], stem=meta["stem"])
        index.n_docs = meta["n_docs"]
        index.avg_len = meta["avg_len"]
        index.terms = json.loads((directory / "terms.json").read_text(encoding="utf-8"))
        index.offsets = np.load(directory / "offsets.npy")
        index.lengths = np.load(directory / "lengths.npy", mmap_mode="r")
        index._df = np.load(directory / "df.npy")
        index._postings = np.memmap(
            directory / "postings.bin", dtype=np.int32, mode="r"
        ).reshape(-1, 2)
        return index

    def _idf(self, term_id: int) -> float:
        import math

        df = int(self._df[term_id])
        if not self.n_docs or not df:
            return 0.0
        return max(math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5)), 0.0)

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        ids = self._ids
        terms = tokenize(query, stem=self.stem)
        if self.drop_stopwords:
            terms = [t for t in terms if t not in STOPWORDS]
        scores = np.zeros(self.n_docs, dtype=np.float64)
        for term in terms:
            term_id = self.terms.get(term)
            if term_id is None:
                continue
            idf = self._idf(term_id)
            if idf == 0.0:
                continue
            lo, hi = int(self.offsets[term_id]), int(self.offsets[term_id + 1])
            if hi <= lo:
                continue
            block = np.asarray(self._postings[lo:hi])
            rows = block[:, 0].astype(np.int64)
            tfs = block[:, 1].astype(np.float64)
            lengths = self.lengths[rows].astype(np.float64)
            denom = tfs + self.k1 * (1.0 - self.b + self.b * lengths / (self.avg_len or 1.0))
            scores[rows] += idf * (tfs * (self.k1 + 1.0)) / denom
        # Stable descending sort matches the in-memory implementation's tie-break
        # (equal scores stay in ascending row order), so the two rankings are identical.
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self._ids[int(i)], float(scores[int(i)])) for i in order if scores[int(i)] > 0.0]

    @property
    def _ids(self) -> list[str]:
        # Chunk ids live beside the postings in document-row order.
        if not hasattr(self, "_id_list"):
            self._id_list = json.loads((self.directory / "ids.json").read_text(encoding="utf-8"))
        return self._id_list

    def disk_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.directory.glob("*") if p.is_file())


# ── dense: streaming, memory-mapped vectors ─────────────────────────────

class StreamingVectorIndex:
    """Append-only dense index: vectors to a memmap, search blockwise."""

    def __init__(self, directory: Path, dim: int) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.dim = int(dim)
        self.count = 0
        self.ids: list[str] = []
        self._vectors: np.ndarray | None = None

    def add(self, chunk_ids: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of width {self.dim}, got {vectors.shape}")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-9, None)
        path = self.directory / "vectors.bin"
        # Plain positional writes, not a fresh memmap per batch: re-mapping a growing file
        # leaves its dirty pages charged to RSS, which made the "streaming" index grow
        # linearly anyway. Only the final index is mapped, read-only, for search.
        mode = "r+b" if path.exists() else "w+b"
        with open(path, mode) as handle:
            handle.seek(self.count * self.dim * 4)
            handle.write(vectors.tobytes())
        self.ids.extend(chunk_ids)
        self.count += len(vectors)

    def finalize(self) -> None:
        (self.directory / "ids.json").write_text(json.dumps(self.ids), encoding="utf-8")
        (self.directory / "meta.json").write_text(
            json.dumps({"version": 1, "dim": self.dim, "count": self.count}), encoding="utf-8"
        )

    @classmethod
    def open(cls, directory: Path) -> "StreamingVectorIndex":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        index = cls(directory, meta["dim"])
        index.count = meta["count"]
        index.ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
        index._vectors = np.memmap(
            directory / "vectors.bin", dtype=np.float32, mode="r", shape=(index.count, index.dim)
        )
        return index

    def search(self, query_vector: np.ndarray, k: int = 10, block: int = 200000) -> list[tuple[str, float]]:
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        query = query / max(float(np.linalg.norm(query)), 1e-9)
        best: list[tuple[float, int]] = []
        for start in range(0, self.count, block):
            stop = min(start + block, self.count)
            scores = self._vectors[start:stop] @ query
            take = min(k, len(scores))
            local = np.argpartition(-scores, take - 1)[:take]
            best.extend((float(scores[i]), start + int(i)) for i in local)
        best.sort(key=lambda item: -item[0])
        return [(self.ids[index], score) for score, index in best[:k]]

    def disk_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.directory.glob("*") if p.is_file())


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="ondisk.py")
    ap.add_argument("--dataset", default="trec-covid")
    ap.add_argument("--data-root", default=str(ROOT / "eval" / "beir"))
    ap.add_argument("--out", default=str(ROOT / "eval" / "beir" / "ondisk"))
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--limit-docs", type=int, default=None)
    ap.add_argument("--limit-queries", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--block-docs", type=int, default=50000)
    ap.add_argument("--block-tokens", type=int, default=1_000_000,
                    help="flush a build block at this many tokens (bounds peak RSS)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--verify-queries", type=int, default=50,
                    help="how many queries to check give the identical in-memory ranking")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--dense-only", action="store_true",
                    help="skip the BM25 build (use when the postings are already on disk)")
    ap.add_argument("--device", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dataset_dir = beir.ensure_dataset(args.dataset, Path(args.data_root))
    corpus = beir.load_corpus(dataset_dir)
    if args.limit_docs:
        corpus = dict(list(corpus.items())[: args.limit_docs])
    queries = beir.load_queries(dataset_dir)
    qrels = beir.load_qrels(dataset_dir)
    scored = {q: t for q, t in queries.items() if q in qrels}
    if args.limit_queries:
        scored = dict(list(scored.items())[: args.limit_queries])
    print(f"{args.dataset}: {len(corpus)} documents, {len(scored)} scored queries", flush=True)

    report: dict[str, Any] = {"settings": vars(args), "documents": len(corpus)}

    # -- lexical --------------------------------------------------------
    if not args.dense_only:
        bm25_dir = Path(args.out) / args.dataset / "bm25"
        started = time.perf_counter()
        ondisk = OnDiskBM25(bm25_dir)
        ondisk.build(
            ((doc_id, text) for doc_id, text in corpus.items()),
            block_docs=args.block_docs,
            block_tokens=args.block_tokens,
        )
        bm25_seconds = time.perf_counter() - started
        print(f"  on-disk BM25 built in {bm25_seconds:.0f}s, {ondisk.disk_bytes() / 1e6:.0f} MB, "
              f"peak RSS {peak_rss_mb():.0f} MB", flush=True)

        # Identical-ranking check against the in-memory implementation, on the same queries.
        # Optional: the in-memory build is the RAM-heavy path this module exists to avoid, so a
        # full-corpus run skips it rather than duplicating the whole index.
        mismatches: int | None = None
        if args.verify_queries:
            memory_index = beir.BM25Index(k1=beir.BEIR_K1, b=beir.BEIR_B, drop_stopwords=True, stem=True)
            memory_index.build(beir.make_chunks(corpus))
            mismatches = 0
            for query_id, text in list(scored.items())[: args.verify_queries]:
                disk = [doc_id for doc_id, _ in ondisk.search(text, k=args.top_k)]
                mem = [hit.chunk_id for hit in memory_index.search(text, k=args.top_k)]
                if disk != mem:
                    mismatches += 1
            print(f"  ranking parity: {args.verify_queries - mismatches}/{args.verify_queries} "
                  f"queries identical", flush=True)

        bm25_rankings = {
            query_id: [doc_id for doc_id, _ in ondisk.search(text, k=args.top_k)]
            for query_id, text in scored.items()
        }
        bm25_eval = evalkit.evaluate_rankings(bm25_rankings, qrels, k_values=(10, 100))
        report["bm25_ondisk"] = {
            "seconds": round(bm25_seconds, 1),
            "megabytes": round(ondisk.disk_bytes() / 1e6, 1),
            "ranking_mismatches": mismatches,
            "nDCG@10": bm25_eval["summary"]["nDCG@10"],
            "MRR@10": bm25_eval["summary"]["MRR@10"],
        }
        print(f"  on-disk BM25 nDCG@10 {bm25_eval['summary']['nDCG@10']:.4f}", flush=True)

    # -- dense ----------------------------------------------------------
    if not args.no_dense:
        enc = beir.BiEncoder(args.model, device=args.device, batch_size=args.batch_size)
        dense_dir = Path(args.out) / args.dataset / "dense"
        index = StreamingVectorIndex(dense_dir, dim=0)
        started = time.perf_counter()
        doc_ids = list(corpus)
        wrote = 0
        for start in range(0, len(doc_ids), 256):
            batch = doc_ids[start : start + 256]
            vectors = np.asarray(enc.embed([corpus[d] for d in batch], fixed_length=True), dtype=np.float32)
            if index.dim == 0:
                index.dim = int(vectors.shape[1])
            index.add(batch, vectors)
            wrote += len(batch)
            if wrote % 5120 == 0:
                print(f"    embedded {wrote}/{len(doc_ids)} (peak RSS {peak_rss_mb():.0f} MB)", flush=True)
        index.finalize()
        dense_seconds = time.perf_counter() - started
        print(f"  streaming dense built in {dense_seconds:.0f}s, {index.disk_bytes() / 1e6:.0f} MB, "
              f"peak RSS {peak_rss_mb():.0f} MB", flush=True)

        opened = StreamingVectorIndex.open(dense_dir)
        query_vectors = np.asarray(enc.embed_queries([text for text in scored.values()]), dtype=np.float32)
        dense_rankings = {
            query_id: [doc_id for doc_id, _ in opened.search(vector, k=args.top_k)]
            for (query_id, _), vector in zip(scored.items(), query_vectors)
        }
        dense_eval = evalkit.evaluate_rankings(dense_rankings, qrels, k_values=(10, 100))
        report["dense_streaming"] = {
            "seconds": round(dense_seconds, 1),
            "megabytes": round(index.disk_bytes() / 1e6, 1),
            "nDCG@10": dense_eval["summary"]["nDCG@10"],
            "MRR@10": dense_eval["summary"]["MRR@10"],
        }
        print(f"  streaming dense nDCG@10 {dense_eval['summary']['nDCG@10']:.4f}", flush=True)

    report["peak_rss_mb"] = round(peak_rss_mb(), 1)
    out = Path(args.out) / f"{args.dataset}.json"
    evalkit.write_json(out, report)
    print(f"\npeak RSS {report['peak_rss_mb']:.0f} MB; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
