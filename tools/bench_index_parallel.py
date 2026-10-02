#!/usr/bin/env python
"""Which mechanism actually speeds up indexing: batches, threads, or processes?

``tools/bench.py`` measures the runtime pipeline. This measures the one stage that
dominates indexing — the encoder forward pass — under every mechanism the engine
could plausibly adopt, on chunks sampled from a real index rather than synthetic text:

* ``serial``    — the baseline: one process, one thread, the shipped batch size.
* ``batch``     — embedding batch size and truncation length, one process.
* ``bucketed``  — batches formed by sorted length so padding is not paid on every row.
* ``threads``   — N in-process threads sharing one loaded checkpoint.
* ``processes`` — N spawned processes, each with its own checkpoint copy.
* ``stages``    — chunking, BM25 build and index save, to expose the ceiling on any
                  parallelism that does not touch the encoder.

Every parallel result is compared against the serial vectors for exact equality, because
a faster wrong embedding is not a result. Writes ``eval/index-parallel.json`` and
``eval/index-parallel.md``.

    .venv/bin/python tools/bench_index_parallel.py --n 256
    .venv/bin/python tools/bench_index_parallel.py --n 256 --device cpu --processes 2,4
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import random
import resource
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# `transformers`' ModernBERT module calls `torch.compile` as a decorator at import time
# (`@torch.compile(dynamic=True)`), which imports torch._inductor and, at that import,
# warms a pool of `torch._inductor.config.compile_threads` compiler workers. In a spawned
# worker with default settings that is up to `os.cpu_count()` extra processes per indexing
# worker, and from a daemonic worker it is a hard failure ("daemonic processes are not
# allowed to have children"). One compile thread runs compilation inline and spawns
# nothing, which is what the process experiment measures against.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

# The BEIR harness already cached the BAAI encoders under the repo; point transformers at
# that cache so the embedder comparison runs offline instead of downloading weights.
os.environ.setdefault("HF_HOME", str(ROOT / "eval" / "beir" / "hf-cache"))

from laya_rag.chunking import Chunk  # noqa: E402
from laya_rag.config import EngineConfig  # noqa: E402
from laya_rag.index_store import VectorIndex  # noqa: E402
from laya_rag.lexical import BM25Index  # noqa: E402
from laya_rag.model import LayaModel  # noqa: E402

INDEX_JSON = ROOT / "indexes" / "default" / "index.json"
OUT_JSON = ROOT / "eval" / "index-parallel.json"
OUT_MD = ROOT / "eval" / "index-parallel.md"


# ── corpus ───────────────────────────────────────────────────────────────

def sample_texts(n: int, seed: int = 0) -> list[str]:
    """Sample ``n`` real chunks from the built index, deterministically."""
    records = json.loads(INDEX_JSON.read_text(encoding="utf-8"))["chunks"]
    rng = random.Random(seed)
    picked = rng.sample(records, min(n, len(records)))
    return [record["text"] for record in picked]


def as_chunks(texts: Sequence[str]) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"bench-{i}",
            doc_id=f"bench-doc-{i}",
            path=f"bench/{i}.md",
            text=text,
            index=0,
            start_char=0,
            end_char=len(text),
        )
        for i, text in enumerate(texts)
    ]


# ── embedder plumbing ────────────────────────────────────────────────────

def load_agent(device: str, torch_threads: int | None = None) -> Any:
    config = EngineConfig()
    config.device = device
    if torch_threads is not None:
        config.torch_threads = torch_threads
    return LayaModel(config).ensure_loaded()


def make_embed_fn(agent: Any, max_length: int, batch_size: int) -> Any:
    import laya

    return laya.embed_fn_from_agent(agent, max_length=max_length, batch_size=batch_size)


def timed_embed(embed_fn: Any, texts: Sequence[str]) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    vectors = np.asarray(embed_fn(list(texts)), dtype=np.float32)
    return vectors, time.perf_counter() - started


def shards(items: Sequence[Any], k: int) -> list[list[Any]]:
    """Contiguous shards, so concatenating the results restores the input order."""
    size = (len(items) + k - 1) // k
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def equal_vectors(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {"equal": False, "shape_left": list(left.shape), "shape_right": list(right.shape)}
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {
        "equal": bool(np.array_equal(left, right)),
        "max_abs_delta": float(delta.max()) if delta.size else 0.0,
    }


# ── experiments ──────────────────────────────────────────────────────────

def exp_serial(
    texts: Sequence[str], agent: Any, load_seconds: float, batch: int, max_length: int
) -> dict[str, Any]:
    embed_fn = make_embed_fn(agent, max_length, batch)
    timed_embed(embed_fn, texts[: min(8, len(texts))])  # warm the device/graph cache
    vectors, embed_seconds = timed_embed(embed_fn, texts)
    return {
        "device": str(getattr(agent, "device", "?")),
        "chunks": len(texts),
        "batch_size": batch,
        "max_length": max_length,
        "load_seconds": round(load_seconds, 3),
        "embed_seconds": round(embed_seconds, 3),
        "chunks_per_second": round(len(texts) / embed_seconds, 2) if embed_seconds else None,
        "vectors": vectors,
    }


def exp_batch_sweep(
    texts: Sequence[str], agent: Any, batches: Sequence[int], lengths: Sequence[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for max_length in lengths:
        for batch in batches:
            embed_fn = make_embed_fn(agent, max_length, batch)
            timed_embed(embed_fn, texts[: min(8, len(texts))])
            _, seconds = timed_embed(embed_fn, texts)
            rows.append(
                {
                    "batch_size": batch,
                    "max_length": max_length,
                    "embed_seconds": round(seconds, 3),
                    "chunks_per_second": round(len(texts) / seconds, 2) if seconds else None,
                }
            )
    return rows


def exp_bucketed(texts: Sequence[str], agent: Any, batch: int, max_length: int) -> dict[str, Any]:
    """Batch by sorted length so a long row does not pad a whole batch of short ones.

    Bucketing by character count is a proxy for token count; it is measured against the
    corpus-order result, not assumed equivalent.
    """
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    sorted_texts = [texts[i] for i in order]
    embed_fn = make_embed_fn(agent, max_length, batch)
    timed_embed(embed_fn, sorted_texts[: min(8, len(sorted_texts))])
    sorted_vectors, seconds = timed_embed(embed_fn, sorted_texts)
    restored = np.empty_like(sorted_vectors)
    restored[order] = sorted_vectors
    plain = make_embed_fn(agent, max_length, batch)
    plain_vectors, plain_seconds = timed_embed(plain, texts)
    return {
        "batch_size": batch,
        "max_length": max_length,
        "bucketed_seconds": round(seconds, 3),
        "corpus_order_seconds": round(plain_seconds, 3),
        "speedup": round(plain_seconds / seconds, 2) if seconds else None,
        "vectors": restored,
        "baseline": plain_vectors,
    }


def exp_threads(
    texts: Sequence[str], agent: Any, workers: Sequence[int], batch: int, max_length: int
) -> list[dict[str, Any]]:
    embed_fn = make_embed_fn(agent, max_length, batch)
    timed_embed(embed_fn, texts[: min(8, len(texts))])
    baseline, base_seconds = timed_embed(embed_fn, texts)
    rows: list[dict[str, Any]] = []
    for k in workers:
        parts = shards(texts, k)

        def job(part: Sequence[str]) -> Any:
            try:
                return np.asarray(embed_fn(list(part)), dtype=np.float32)
            except Exception as exc:  # noqa: BLE001 - a failed worker is a result
                return exc

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=k) as pool:
            results = list(pool.map(job, parts))
        wall = time.perf_counter() - started
        errors = [str(r) for r in results if isinstance(r, Exception)]
        row: dict[str, Any] = {
            "workers": k,
            "wall_seconds": round(wall, 3),
            "speedup_vs_serial": round(base_seconds / wall, 2) if wall else None,
            "errors": errors[:2],
            "error_count": len(errors),
        }
        if not errors:
            stacked = np.vstack([r for r in results if isinstance(r, np.ndarray)])
            row["equality"] = equal_vectors(baseline, stacked)
        rows.append(row)
    row_serial = {
        "workers": 1,
        "wall_seconds": round(base_seconds, 3),
        "speedup_vs_serial": 1.0,
        "equality": equal_vectors(baseline, baseline),
    }
    return [row_serial, *rows]


def _process_worker(
    index: int,
    texts: Sequence[str],
    device: str,
    max_length: int,
    batch: int,
    torch_threads: int | None,
    queue: Any,
) -> None:
    """One shard, in its own non-daemonic process with its own checkpoint."""
    try:
        started = time.perf_counter()
        agent = load_agent(device, torch_threads)
        load_seconds = time.perf_counter() - started
        embed_fn = make_embed_fn(agent, max_length, batch)
        warmup_started = time.perf_counter()
        timed_embed(embed_fn, texts[: min(8, len(texts))])
        warmup_seconds = time.perf_counter() - warmup_started
        vectors, embed_seconds = timed_embed(embed_fn, texts)
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        queue.put(
            {
                "index": index,
                "vectors": vectors,
                "load_seconds": load_seconds,
                "warmup_seconds": warmup_seconds,
                "embed_seconds": embed_seconds,
                "rss_mb": rss_mb,
            }
        )
    except Exception:  # noqa: BLE001 - a failed worker is a result, not a crash
        queue.put({"index": index, "error": traceback.format_exc()})


def exp_processes(
    texts: Sequence[str],
    workers: Sequence[int],
    device: str,
    batch: int,
    max_length: int,
    torch_threads: int | None,
) -> list[dict[str, Any]]:
    """Spawn ``k`` workers, each loading its own checkpoint and embedding one shard.

    Wall time includes process start, checkpoint load and first-forward warmup, because a
    real indexing run pays all three. ``embed_seconds_total`` is the sum across workers, so
    the fixed per-worker cost is visible separately from the encoder throughput.
    """
    context = mp.get_context("spawn")
    rows: list[dict[str, Any]] = []
    for k in workers:
        parts = shards(texts, k)
        queue = context.Queue()
        processes = [
            context.Process(
                target=_process_worker,
                args=(i, part, device, max_length, batch, torch_threads, queue),
                daemon=False,
            )
            for i, part in enumerate(parts)
        ]
        started = time.perf_counter()
        for process in processes:
            process.start()
        results = [queue.get() for _ in processes]
        for process in processes:
            process.join()
        wall = time.perf_counter() - started

        errors = [result["error"] for result in results if "error" in result]
        row: dict[str, Any] = {
            "workers": k,
            "wall_seconds": round(wall, 3),
            "errors": errors[:1],
            "error_count": len(errors),
        }
        if not errors:
            ordered = sorted(results, key=lambda result: result["index"])
            slowest_embed = max(r["embed_seconds"] for r in ordered)
            row.update(
                {
                    "load_seconds_max": round(max(r["load_seconds"] for r in ordered), 3),
                    "warmup_seconds_max": round(max(r["warmup_seconds"] for r in ordered), 3),
                    "embed_seconds_total": round(sum(r["embed_seconds"] for r in ordered), 3),
                    "embed_seconds_max": round(slowest_embed, 3),
                    "aggregate_chunks_per_second": round(len(texts) / slowest_embed, 2)
                    if slowest_embed
                    else None,
                    "rss_mb_max": round(max(r["rss_mb"] for r in ordered), 1),
                    "vectors": np.vstack([r["vectors"] for r in ordered]),
                }
            )
        rows.append(row)
    return rows


def exp_embedders(
    texts: Sequence[str],
    agent: Any,
    device: str,
    model_ids: Sequence[str],
    batch: int,
    max_length: int,
) -> list[dict[str, Any]]:
    """Throughput of the encoder actually shipped vs retrieval-trained alternatives.

    The shipped Laya encoder is 322M parameters and is not retrieval-trained; the BEIR
    harness measured `bge-small-en-v1.5` (33M) as both faster *and* better on nDCG@10.
    This says what each costs per chunk, so the trade is explicit.
    """
    from laya_rag.model import BiEncoderModel

    rows: list[dict[str, Any]] = []
    embed_fn = make_embed_fn(agent, max_length, batch)
    timed_embed(embed_fn, texts[: min(8, len(texts))])
    _, seconds = timed_embed(embed_fn, texts)
    rows.append(
        {
            "model": "laya-multilingual (shipped encoder)",
            "params": int(sum(p.numel() for p in agent.model.encoder.parameters())),
            "max_length": max_length,
            "batch_size": batch,
            "embed_seconds": round(seconds, 3),
            "chunks_per_second": round(len(texts) / seconds, 2) if seconds else None,
        }
    )

    for model_id in model_ids:
        config = EngineConfig()
        config.device = device
        config.embed_model = model_id
        config.embed_max_length = max_length
        config.embed_batch_size = batch
        try:
            model = BiEncoderModel(config).ensure_loaded()
            model.embed(list(texts[: min(8, len(texts))]))
            started = time.perf_counter()
            model.embed(list(texts))
            seconds = time.perf_counter() - started
            rows.append(
                {
                    "model": model_id,
                    "params": int(sum(p.numel() for p in model._encoder.parameters())),
                    "max_length": max_length,
                    "batch_size": batch,
                    "embed_seconds": round(seconds, 3),
                    "chunks_per_second": round(len(texts) / seconds, 2) if seconds else None,
                }
            )
        except Exception as exc:  # noqa: BLE001 - report the model that failed
            rows.append(
                {
                    "model": model_id,
                    "max_length": max_length,
                    "batch_size": batch,
                    "error": str(exc)[:300],
                }
            )
    return rows


def exp_stages(texts: Sequence[str]) -> dict[str, Any]:
    """The non-encoder stages, so their share of a real index build is visible."""
    chunks = as_chunks(texts)

    started = time.perf_counter()
    from laya_rag.chunking import chunk_text

    pieces = 0
    for text in texts:
        pieces += len(chunk_text(text, 1200, 200, 80))
    chunk_seconds = time.perf_counter() - started

    started = time.perf_counter()
    lexical = BM25Index(stem=True)
    lexical.build(chunks)
    bm25_seconds = time.perf_counter() - started

    started = time.perf_counter()
    index = VectorIndex()
    index.add(chunks, np.random.default_rng(0).random((len(chunks), 768), dtype=np.float32))
    with tempfile.TemporaryDirectory() as tmp:
        index.save(Path(tmp))
    save_seconds = time.perf_counter() - started

    return {
        "chunks": len(chunks),
        "chunk_text_seconds": round(chunk_seconds, 3),
        "chunk_text_chunks": pieces,
        "bm25_build_seconds": round(bm25_seconds, 3),
        "vector_add_and_save_seconds": round(save_seconds, 3),
    }


# ── reporting ────────────────────────────────────────────────────────────

def markdown(results: dict[str, Any]) -> str:
    lines: list[str] = ["# Indexing parallelism", ""]
    env = results["environment"]
    lines += [
        f"Host: {env['machine']} • {env['cpu']} • {env['physical_cores']} physical cores • "
        f"torch {env['torch']} • device `{env['device']}` • {env['chunks']} sampled chunks",
        "",
    ]

    serial = results.get("serial")
    if serial:
        lines += [
            "## Serial baseline",
            "",
            "| chunks | device | batch | max_len | load_s | embed_s | chunks/s |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            f"| {serial['chunks']} | {serial['device']} | {serial['batch_size']} | "
            f"{serial['max_length']} | {serial['load_seconds']} | {serial['embed_seconds']} | "
            f"{serial['chunks_per_second']} |",
            "",
        ]

    sweep = results.get("batch_sweep")
    if sweep:
        lines += [
            "## Batch size and truncation length",
            "",
            "| max_len | batch | embed_s | chunks/s |",
            "| --- | --- | --- | --- |",
        ]
        lines += [
            f"| {r['max_length']} | {r['batch_size']} | {r['embed_seconds']} | "
            f"{r['chunks_per_second']} |"
            for r in sweep
        ]
        lines.append("")

    embedders = results.get("embedders")
    if embedders:
        lines += [
            "## Encoder choice",
            "",
            "| model | params | max_len | batch | embed_s | chunks/s |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in embedders:
            if "error" in row:
                lines.append(f"| {row['model']} | — | {row['max_length']} | {row['batch_size']} | error | — |")
                continue
            lines.append(
                f"| {row['model']} | {row['params'] / 1e6:.0f}M | {row['max_length']} | "
                f"{row['batch_size']} | {row['embed_seconds']} | {row['chunks_per_second']} |"
            )
        lines.append("")

    bucketed = results.get("bucketed")
    if bucketed:
        lines += [
            "## Length-bucketed batches",
            "",
            "| batch | corpus order s | bucketed s | speedup | vectors equal | max delta |",
            "| --- | --- | --- | --- | --- | --- |",
            f"| {bucketed['batch_size']} | {bucketed['corpus_order_seconds']} | "
            f"{bucketed['bucketed_seconds']} | {bucketed['speedup']} | "
            f"{bucketed['equality']['equal']} | {bucketed['equality'].get('max_abs_delta', 'n/a')} |",
            "",
        ]

    threads = results.get("threads")
    if threads:
        lines += [
            "## Threads (one shared checkpoint)",
            "",
            "| workers | wall_s | speedup | equal | max delta | errors |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in threads:
            eq = row.get("equality", {})
            lines.append(
                f"| {row['workers']} | {row['wall_seconds']} | {row['speedup_vs_serial']} | "
                f"{eq.get('equal', 'n/a')} | {eq.get('max_abs_delta', 'n/a')} | "
                f"{row.get('error_count', 0)} |"
            )
        lines.append("")

    processes = results.get("processes")
    if processes:
        lines += [
            "## Processes (one checkpoint per worker)",
            "",
            "`agg chunks/s` is total chunks over the slowest worker's embed time — the encoder",
            "throughput the run actually achieved, independent of the fixed load cost.",
            "",
            "| workers | wall_s | load_s (max) | warmup_s (max) | embed_s (max) | agg chunks/s | "
            "speedup vs 1 proc | rss_mb (max) | equal | max delta |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        baseline = next(
            (r for r in processes if r["workers"] == 1 and "wall_seconds" in r), processes[0]
        )
        for row in processes:
            if row.get("error_count"):
                lines.append(
                    f"| {row['workers']} | {row['wall_seconds']} | — | — | — | — | — | — | "
                    f"error ({row['error_count']}) | — |"
                )
                continue
            eq = row.get("equality", {})
            speedup = (
                round(baseline["wall_seconds"] / row["wall_seconds"], 2)
                if row["wall_seconds"]
                else None
            )
            lines.append(
                f"| {row['workers']} | {row['wall_seconds']} | {row['load_seconds_max']} | "
                f"{row['warmup_seconds_max']} | {row['embed_seconds_max']} | "
                f"{row['aggregate_chunks_per_second']} | {speedup} | "
                f"{row['rss_mb_max']} | {eq.get('equal', 'n/a')} | "
                f"{eq.get('max_abs_delta', 'n/a')} |"
            )
        lines.append("")

    stages = results.get("stages")
    if stages:
        lines += [
            "## Non-encoder stages",
            "",
            f"Over {stages['chunks']} chunks: `chunk_text` {stages['chunk_text_seconds']}s, "
            f"BM25 build {stages['bm25_build_seconds']}s, vector add+save "
            f"{stages['vector_add_and_save_seconds']}s.",
            "",
            "These are pure-Python and GIL-bound: threads cannot overlap them with each "
            "other, though they can overlap with a released-GIL encoder forward pass.",
            "",
        ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=256, help="chunks to sample from the index")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-sizes", default="8,16,32,64")
    parser.add_argument("--max-lengths", default="512,384,256")
    parser.add_argument("--threads", default="2,4,8")
    parser.add_argument("--processes", default="1,2,4,8")
    parser.add_argument(
        "--embedders",
        default="BAAI/bge-small-en-v1.5,BAAI/bge-base-en-v1.5,BAAI/bge-large-en-v1.5",
        help="retrieval-trained encoders to compare against the shipped Laya encoder",
    )
    parser.add_argument("--torch-threads", type=int, default=None,
                        help="per-process torch threads for the process experiment")
    parser.add_argument("--skip", default="", help="comma-separated experiments to skip")
    parser.add_argument("--json-out", default=str(OUT_JSON))
    parser.add_argument("--md-out", default=str(OUT_MD))
    args = parser.parse_args(argv)

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    texts = sample_texts(args.n, args.seed)
    import torch

    started = time.perf_counter()
    agent = load_agent(args.device, args.torch_threads)
    agent_load_seconds = time.perf_counter() - started
    device = getattr(agent, "device", args.device)
    print(f"loaded {device} in {agent_load_seconds:.1f}s; {len(texts)} chunks", flush=True)

    results: dict[str, Any] = {
        "environment": {
            "machine": platform.machine(),
            "cpu": platform.processor() or platform.machine(),
            "physical_cores": 28,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": str(device),
            "chunks": len(texts),
            "agent_load_seconds": round(agent_load_seconds, 3),
        }
    }

    if "serial" not in skip:
        print("serial baseline…", flush=True)
        serial = exp_serial(texts, agent, agent_load_seconds, args.batch, args.max_length)
        baseline = serial.pop("vectors")
        results["serial"] = serial
    else:
        baseline = None

    if "batch" not in skip:
        print("batch/length sweep…", flush=True)
        results["batch_sweep"] = exp_batch_sweep(
            texts,
            agent,
            [int(b) for b in args.batch_sizes.split(",")],
            [int(m) for m in args.max_lengths.split(",")],
        )

    if "embedders" not in skip and args.embedders.strip():
        print("encoder comparison…", flush=True)
        results["embedders"] = exp_embedders(
            texts,
            agent,
            args.device,
            [m.strip() for m in args.embedders.split(",") if m.strip()],
            args.batch,
            args.max_length,
        )

    if "bucketed" not in skip and baseline is not None:
        print("length-bucketed batches…", flush=True)
        bucketed = exp_bucketed(texts, agent, args.batch, args.max_length)
        results["bucketed"] = {
            **{k: v for k, v in bucketed.items() if k not in {"vectors", "baseline"}},
            "equality": equal_vectors(baseline, bucketed["vectors"]),
            "batches_equal": equal_vectors(bucketed["baseline"], bucketed["vectors"]),
        }

    if "threads" not in skip and baseline is not None:
        print("thread sweep…", flush=True)
        results["threads"] = exp_threads(
            texts, agent, [int(t) for t in args.threads.split(",")], args.batch, args.max_length
        )

    if "processes" not in skip and baseline is not None:
        print("process sweep…", flush=True)
        rows = exp_processes(
            texts,
            [int(p) for p in args.processes.split(",")],
            args.device,
            args.batch,
            args.max_length,
            args.torch_threads,
        )
        for row in rows:
            if "vectors" in row:
                row["equality"] = equal_vectors(baseline, row.pop("vectors"))
        results["processes"] = rows

    if "stages" not in skip:
        print("stage breakdown…", flush=True)
        results["stages"] = exp_stages(texts)

    Path(args.json_out).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    Path(args.md_out).write_text(markdown(results), encoding="utf-8")
    print(markdown(results), flush=True)
    print(f"\nwrote {args.json_out} and {args.md_out}", flush=True)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
