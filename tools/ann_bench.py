#!/usr/bin/env python
"""Scale probe for the dense channel: approximate search versus the exhaustive scan.

`VectorIndex.search` multiplies the query against every stored vector, which is exact and
fast to a few hundred thousand chunks but linear in the corpus. This measures what an
approximate index costs in recall and buys in latency, on synthetic clustered vectors that
resemble an embedding space (uniform random vectors in 384 dimensions are near-orthogonal
and make every ANN look perfect).

    .venv/bin/python tools/ann_bench.py --sizes 10000,100000,1000000

faiss is an optional dependency; without it the tool reports that and exits.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import evalkit  # noqa: E402


def load_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "faiss is not installed; `pip install faiss-cpu` to run the ANN comparison"
        ) from exc
    return faiss


def make_centres(clusters: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    centres = rng.normal(size=(clusters, dim)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    return centres


def clustered_vectors(
    count: int, centres: np.ndarray, rng: np.random.Generator, noise_norm: float = 0.35
) -> np.ndarray:
    """Draw from a mixture of Gaussians around ``centres``, L2-normalised.

    Clustered rather than uniform: real embeddings form neighbourhoods, and uniform
    vectors in 384 dimensions are all roughly equidistant, which would flatter IVF.

    Documents **and queries** must be drawn from the same mixture. Drawing queries from a
    fresh set of centroids puts them off the document manifold, where a partition-based
    index cannot reach the true neighbours and recall collapses for reasons that have
    nothing to do with the index.
    """
    dim = centres.shape[1]
    assignment = rng.integers(0, len(centres), size=count)
    # Scale the per-dimension noise so the *norm* of the offset is `noise_norm`, which is
    # what decides whether clusters survive normalisation. A per-dimension scale of 0.35
    # in 384 dimensions is a norm of ~6.9 against a unit centre: that is uniform noise,
    # not a cluster, and every ANN looks broken on it.
    noise = rng.normal(scale=noise_norm / np.sqrt(dim), size=(count, dim)).astype(np.float32)
    vectors = centres[assignment] + noise
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors.astype(np.float32)


def time_queries(search, queries: np.ndarray, repeats: int = 3) -> tuple[np.ndarray, float]:
    """Run ``search(queries)`` and return its result plus median seconds per query."""
    samples: list[float] = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = search(queries)
        samples.append((time.perf_counter() - started) / len(queries))
    samples.sort()
    return result, samples[len(samples) // 2]


def recall_at_k(approximate: np.ndarray, exact: np.ndarray, k: int) -> float:
    hits = 0
    for row_a, row_e in zip(approximate[:, :k], exact[:, :k]):
        hits += len(set(row_a.tolist()) & set(row_e.tolist()))
    return hits / (len(exact) * k)


def rescored_search(index: Any, vectors: np.ndarray, queries: np.ndarray, k: int,
                    candidates: int) -> np.ndarray:
    """An ANN candidate list, re-ranked by exact inner product on the fp32 vectors.

    This is the shape the engine uses: a quantized index proposes candidates and the exact
    vectors — mmapped, so not resident — decide the order. Without the rescore the order is
    whatever the compressed codes said, which is where quantized search loses its recall.

    ``index`` is anything exposing faiss's ``search(queries, n) -> (scores, ids)``.
    """
    _, ids = index.search(queries, candidates)
    out = np.full((len(queries), k), -1, dtype=np.int64)
    for row_index, (query, candidate_ids) in enumerate(zip(queries, ids)):
        keep = np.asarray([i for i in candidate_ids if i >= 0], dtype=np.int64)
        if not len(keep):
            continue
        order = np.argsort(-(vectors[keep] @ query))
        selected = keep[order][:k]
        out[row_index, : len(selected)] = selected
    return out


def pq_index_mb(count: int, m: int, nlist: int, dim: int) -> float:
    """PQ codes plus the coarse quantiser's centroids — the index, not the vectors."""
    return round((count * m + nlist * dim * 4) / 1e6, 1)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ann_bench.py", description="ANN vs exhaustive at scale")
    parser.add_argument("--sizes", default="10000,100000,1000000")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=2000)
    parser.add_argument("--out", default=str(ROOT / "eval" / "ann-bench"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--pq-m", type=int, default=48,
                        help="PQ sub-quantisers: bytes per vector in the index")
    parser.add_argument("--pq-nbits", type=int, default=8)
    parser.add_argument("--pq-candidates", default="10,50,100",
                        help="candidate list sizes the fp32 rescore is measured at")
    parser.add_argument("--pq-nprobe", type=int, default=8,
                        help="IVF cells probed for the PQ arm; the candidate set's recall is "
                             "bounded by this, and the rescore can only reorder what it receives")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    faiss = load_faiss()
    rng = np.random.default_rng(args.seed)
    sizes = [int(value) for value in args.sizes.split(",") if value.strip()]

    rows: list[dict[str, Any]] = []
    centres = make_centres(args.clusters, args.dim, rng)
    for count in sizes:
        vectors = clustered_vectors(count, centres, rng)
        queries = clustered_vectors(args.queries, centres, rng)

        flat = faiss.IndexFlatIP(args.dim)
        started = time.perf_counter()
        flat.add(vectors)
        flat_build = time.perf_counter() - started
        exact, exact_seconds = time_queries(
            lambda q: flat.search(q, args.k)[1], queries
        )

        row: dict[str, Any] = {
            "vectors": count,
            "exhaustive_build_s": round(flat_build, 2),
            "exhaustive_ms_per_query": round(1000 * exact_seconds, 3),
            "index_mb": round(count * args.dim * 4 / 1e6, 1),
        }

        # IVF-Flat: k-means coarse quantiser plus an inverted list per cell.
        nlist = max(1, int(np.sqrt(count)))
        quantiser = faiss.IndexFlatIP(args.dim)
        ivf = faiss.IndexIVFFlat(quantiser, args.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        started = time.perf_counter()
        ivf.train(vectors)
        ivf.add(vectors)
        row["ivf_build_s"] = round(time.perf_counter() - started, 2)
        row["ivf_nlist"] = nlist
        for nprobe in (1, 8, 32):
            ivf.nprobe = nprobe
            result, seconds = time_queries(lambda q: ivf.search(q, args.k)[1], queries)
            row[f"ivf_recall@{args.k}_nprobe{nprobe}"] = round(recall_at_k(result, exact, args.k), 4)
            row[f"ivf_ms_nprobe{nprobe}"] = round(1000 * seconds, 3)
        row["ivf_speedup_vs_exhaustive"] = round(exact_seconds / max(seconds, 1e-9), 1)

        # HNSW: graph-based, usually better recall per unit of latency at this scale.
        hnsw = faiss.IndexHNSWFlat(args.dim, 32, faiss.METRIC_INNER_PRODUCT)
        started = time.perf_counter()
        hnsw.add(vectors)
        row["hnsw_build_s"] = round(time.perf_counter() - started, 2)
        result, seconds = time_queries(lambda q: hnsw.search(q, args.k)[1], queries)
        row[f"hnsw_recall@{args.k}"] = round(recall_at_k(result, exact, args.k), 4)
        row["hnsw_ms_per_query"] = round(1000 * seconds, 3)
        row["hnsw_speedup_vs_exhaustive"] = round(exact_seconds / max(seconds, 1e-9), 1)

        # IVF-PQ with an fp32 rescore: the index holds `pq_m`-byte codes instead of the
        # 1.5 KB vectors, and the exact order is restored from the vectors themselves --
        # which the engine keeps mmapped and therefore non-resident. This is the arm the
        # scale claim rests on: how much index shrinks, and what recall that costs.
        ivfpq = faiss.IndexIVFPQ(faiss.IndexFlatIP(args.dim), args.dim, nlist,
                                 args.pq_m, args.pq_nbits, faiss.METRIC_INNER_PRODUCT)
        started = time.perf_counter()
        ivfpq.train(vectors)
        ivfpq.add(vectors)
        row["ivfpq_build_s"] = round(time.perf_counter() - started, 2)
        row["ivfpq_m"] = args.pq_m
        row["ivfpq_index_mb"] = pq_index_mb(count, args.pq_m, nlist, args.dim)
        row["ivfpq_bytes_per_vector"] = args.pq_m
        row["ivfpq_nprobe"] = args.pq_nprobe
        ivfpq.nprobe = args.pq_nprobe
        for candidates in (int(v) for v in str(args.pq_candidates).split(",") if v.strip()):
            result, seconds = time_queries(
                lambda q: rescored_search(ivfpq, vectors, q, args.k, candidates), queries
            )
            row[f"ivfpq_rescore{candidates}_recall@{args.k}"] = round(
                recall_at_k(result, exact, args.k), 4)
            row[f"ivfpq_rescore{candidates}_ms"] = round(1000 * seconds, 3)
        rows.append(row)
        del vectors, flat, ivf, hnsw, ivfpq
        print(f"  {count} vectors done", flush=True)

    out = Path(args.out)
    evalkit.write_json(out.with_suffix(".json"), {"settings": vars(args), "rows": rows})
    evalkit.write_text(
        out.with_suffix(".md"),
        "# Approximate vs exhaustive dense search\n\n" + evalkit.markdown_table(rows) + "\n",
    )
    print()
    print(evalkit.markdown_table(rows))
    print(f"\nwrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
