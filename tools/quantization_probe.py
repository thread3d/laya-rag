"""Re-measure int8/fp16/PQ quantization on the encoder this project actually ships.

The external survey's brief asked for RaBitQ + OPQ + mmap and for binary/int8 quantization, and the
audit's tension 14 records that the question was **never re-measured here** — where the encoder is
`bge-small`, which is not MRL-trained, so the "~96% retained" figure the brief cites is itself
model-dependent (74.77–96.45%). This answers it on the real vectors rather than synthetic ones.

The probe is self-retrieval on a cached BEIR corpus embedding matrix: sample query vectors from the
corpus, take the exact `IndexFlatIP` top-10 as ground truth, and measure what each compressed index
keeps. No model, no network — the cache is already on disk.

    .venv/bin/python tools/quantization_probe.py --cache eval/beir/cache/scidocs.bge-small-en-v1.5.cls.npy
    .venv/bin/python tools/quantization_probe.py --cache eval/beir/cache/fiqa.bge-small-en-v1.5.cls.npy

faiss is an optional dependency; without it the tool says so and exits.
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

#: The compressed indexes, all IVF over the same coarse quantiser so the comparison isolates the
#: per-vector codec.
KINDS = ("ivf-flat", "ivf-pq", "ivf-sq8", "ivf-fp16")


def load_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "faiss is not installed; `pip install faiss-cpu` to run the quantization probe"
        ) from exc
    return faiss


def load_vectors(path: Path | str, *, limit: int | None = None, faiss: Any = None) -> np.ndarray:
    """Load a cached embedding matrix as L2-normalised float32 (so IP == cosine)."""
    vectors = np.load(Path(path)).astype(np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"{path} is {vectors.shape}, expected a 2-D matrix")
    if limit and limit < len(vectors):
        vectors = vectors[:limit]
    if faiss is not None:
        faiss.normalize_L2(vectors)
    return vectors


def recall_at_k(approximate: np.ndarray, exact: np.ndarray, k: int) -> float:
    """Mean overlap of the approximate and exact top-k, over queries."""
    hits = 0
    for approx_row, exact_row in zip(approximate[:, :k], exact[:, :k]):
        hits += len(set(approx_row.tolist()) & set(exact_row.tolist()))
    return hits / (len(approximate) * k) if len(approximate) else 0.0


def build_index(faiss: Any, kind: str, vectors: np.ndarray, *, nlist: int, m: int, nbits: int) -> Any:
    """One IVF index with the named per-vector codec, trained and populated."""
    dim = vectors.shape[1]
    quantiser = faiss.IndexFlatIP(dim)
    if kind == "ivf-flat":
        index = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    elif kind == "ivf-pq":
        index = faiss.IndexIVFPQ(quantiser, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    elif kind == "ivf-sq8":
        index = faiss.IndexIVFScalarQuantizer(
            quantiser, dim, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
        )
    elif kind == "ivf-fp16":
        index = faiss.IndexIVFScalarQuantizer(
            quantiser, dim, nlist, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT
        )
    else:
        raise ValueError(f"unknown index kind {kind!r}; expected one of {KINDS}")
    index.train(vectors)
    index.add(vectors)
    index.nprobe = min(32, nlist)
    return index


def probe(
    vectors: np.ndarray,
    *,
    kinds: Sequence[str] = KINDS,
    queries: int = 500,
    k: int = 10,
    nlist: int = 256,
    m: int = 48,
    nbits: int = 8,
    faiss: Any = None,
) -> dict[str, Any]:
    """Recall@k, index bytes and timings for each codec, against the exact scan."""
    faiss = faiss or load_faiss()
    count, dim = vectors.shape
    nlist = max(1, min(nlist, count))
    step = max(1, count // queries)
    sample = vectors[::step][:queries]

    flat = faiss.IndexFlatIP(dim)
    flat.add(vectors)
    started = time.perf_counter()
    _scores, exact = flat.search(sample, k)
    flat_seconds = time.perf_counter() - started

    rows: list[dict[str, Any]] = [{
        "kind": "flat-exact",
        "recall@10": 1.0,
        "index_bytes": int(count * dim * 4),
        "bytes_per_vector": dim * 4,
        "build_seconds": 0.0,
        "search_seconds": round(flat_seconds, 3),
    }]
    for kind in kinds:
        started = time.perf_counter()
        index = build_index(faiss, kind, vectors, nlist=nlist, m=m, nbits=nbits)
        build_seconds = time.perf_counter() - started
        started = time.perf_counter()
        _scores, approximate = index.search(sample, k)
        search_seconds = time.perf_counter() - started
        size = len(faiss.serialize_index(index))
        rows.append({
            "kind": kind,
            "recall@10": round(recall_at_k(approximate, exact, k), 4),
            "index_bytes": int(size),
            "bytes_per_vector": round(size / count, 1),
            "build_seconds": round(build_seconds, 2),
            "search_seconds": round(search_seconds, 3),
        })
    return {
        "_settings": {
            "vectors": int(count), "dim": int(dim), "nlist": int(nlist), "m": int(m),
            "nbits": int(nbits), "queries": int(len(sample)), "k": int(k),
        },
        "rows": rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="quantization_probe.py")
    ap.add_argument("--cache", required=True, help="a cached .npy corpus embedding matrix")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N vectors")
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nlist", type=int, default=256)
    ap.add_argument("--m", type=int, default=48)
    ap.add_argument("--nbits", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "eval" / "quantization-probe"))
    return ap.parse_args(argv)


def render_markdown(report: dict[str, Any]) -> str:
    settings = report["_settings"]
    lines = [
        "# Quantization probe (real encoder vectors, self-retrieval)",
        "",
        f"- {settings['vectors']} vectors of dim {settings['dim']}, {settings['queries']} sampled "
        f"queries, recall@{settings['k']}, nlist {settings['nlist']}",
        "",
        "| codec | recall@10 | index bytes | bytes/vector | build s | search s |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['kind']} | {row['recall@10']:.4f} | {row['index_bytes']} | "
            f"{row['bytes_per_vector']} | {row['build_seconds']} | {row['search_seconds']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from heavylock import heavy_job_guard

    heavy_job_guard("quantization_probe.py")
    faiss = load_faiss()
    vectors = load_vectors(args.cache, limit=args.limit, faiss=faiss)
    report = probe(
        vectors, queries=args.queries, k=args.k, nlist=args.nlist, m=args.m, nbits=args.nbits,
        faiss=faiss,
    )
    out = Path(args.out)
    evalkit.write_json(out.with_suffix(".json"), report)
    evalkit.write_text(out.with_suffix(".md"), render_markdown(report))
    print(render_markdown(report))
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
