"""The cost axis — ROADMAP 3.4, and the axis this project can actually win.

Quality without cost is half a claim. A CPU-only, 33M-encoder system cannot win an nDCG@10 column
against API-scale models, but "best retrieval per unit of compute" is measurable: this reports
per-query latency (mean, median, p95), throughput, peak resident memory, and index bytes per chunk
and per query, against an **existing** index so nothing is rebuilt and the retrieval half is the
system the quality numbers describe.

    .venv/bin/python tools/cost_report.py --label default
    .venv/bin/python tools/cost_report.py --embed-model BAAI/bge-small-en-v1.5 \
        --index-dir eval/benchmarks/multihop-rag/index/engine-hybrid --label hybrid

Peak RSS is the process high-water mark (`ru_maxrss`), so it covers the model *and* the index in
one number rather than attributing memory to a component. Queries run at the engine's shipped
defaults, gates included, so the latency is the latency a caller sees.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "eval" / "beir" / "hf-cache"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import evalkit  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"


def percentile(values: Sequence[float], q: float) -> float:
    """Linear interpolation between closest ranks — the definition numpy uses."""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (q / 100.0)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def index_bytes(directory: Path | str) -> int:
    """Total bytes on disk of an index, every artifact included."""
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def ru_maxrss_bytes() -> int:
    """Peak RSS in bytes; `ru_maxrss` is bytes on macOS and kilobytes on Linux."""
    usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return usage if sys.platform == "darwin" else usage * 1024


def summarize_latency(seconds: Sequence[float]) -> dict[str, float]:
    if not seconds:
        return {}
    total = sum(seconds)
    return {
        "mean": round(total / len(seconds), 4),
        "median": round(percentile(seconds, 50), 4),
        "p95": round(percentile(seconds, 95), 4),
        "min": round(min(seconds), 4),
        "max": round(max(seconds), 4),
        "queries_per_second": round(len(seconds) / total, 4) if total else 0.0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="cost_report.py")
    ap.add_argument("--index-dir", default=str(ROOT_BENCH / "index" / "engine-default"))
    ap.add_argument("--embed-model", default=None,
                    help="must match the encoder the index was built with")
    ap.add_argument("--records", default=str(ROOT_BENCH / "MultiHopRAG.json"))
    ap.add_argument("--queries", type=int, default=60, help="timed queries after warmup")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--label", default=None, help="defaults to the index directory name")
    ap.add_argument("--out", default=str(ROOT / "eval" / "cost-report"))
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    index_dir = Path(args.index_dir)
    os.environ["LAYA_RAG_INDEX_DIR"] = str(index_dir)
    if args.embed_model:
        os.environ["LAYA_RAG_EMBED_MODEL"] = args.embed_model
    else:
        os.environ.pop("LAYA_RAG_EMBED_MODEL", None)

    from laya_rag.config import EngineConfig
    from laya_rag.engine import RagEngine

    records = json.loads(Path(args.records).read_text(encoding="utf-8"))
    questions = [r["query"] for r in records if r["question_type"] != "null_query"]
    questions = questions[: args.warmup + args.queries]

    config = EngineConfig()
    engine = RagEngine(config)
    if not engine.has_index():
        raise SystemExit(f"no complete index at {index_dir}; build it first")
    chunks = len(engine.list_chunks())
    print(f"index {index_dir.name}: {chunks} chunks, dense="
          f"{config.resolved_dense_weight != 0.0}, embed={config.embed_model or 'engine-default'}",
          flush=True)

    for question in questions[: args.warmup]:
        engine.query(question, top_k=args.top_k)

    seconds: list[float] = []
    started = time.perf_counter()
    for question in questions[args.warmup :]:
        query_started = time.perf_counter()
        engine.query(question, top_k=args.top_k)
        seconds.append(time.perf_counter() - query_started)
    wall = time.perf_counter() - started

    on_disk = index_bytes(index_dir)
    report = {
        "label": args.label or index_dir.name,
        "index_dir": str(index_dir),
        "index_bytes": on_disk,
        "chunks": chunks,
        "bytes_per_chunk": round(on_disk / chunks, 1) if chunks else None,
        "bytes_per_query": round(on_disk / len(seconds), 1) if seconds else None,
        "peak_rss_bytes": ru_maxrss_bytes(),
        "timed_queries": len(seconds),
        "wall_seconds": round(wall, 2),
        "latency_seconds": summarize_latency(seconds),
        "settings": {
            "embed_model": config.embed_model or "engine-default",
            "dense_weight": config.resolved_dense_weight,
            "top_k": args.top_k,
            "warmup": args.warmup,
            "chunk_chars": config.chunk_chars,
        },
    }
    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), report)
    latency = report["latency_seconds"]
    print(f"{report['label']}: {latency.get('mean')}s mean, {latency.get('p95')}s p95, "
          f"{latency.get('queries_per_second')} q/s, peak RSS "
          f"{report['peak_rss_bytes'] / 2**20:.0f} MiB, index {on_disk / 2**20:.1f} MiB "
          f"({report['bytes_per_chunk']} B/chunk)")
    print("wrote", str(out) + ".json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
