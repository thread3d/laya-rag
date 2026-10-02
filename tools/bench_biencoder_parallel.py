#!/usr/bin/env python
"""Does parallelism help a small bi-encoder, or is MPS saturated at any model size?

``tools/bench_index_parallel.py`` shows the shipped 307M Laya encoder saturates Metal by
itself: threads do not help and process sharding only adds per-worker load cost. A 33M
retrieval-trained bi-encoder is the faster encoder this project already recommends, and it
might leave device headroom for parallelism. This measures that directly on real chunks:
serial at two batch sizes, then thread and process sharding with an exact-vector check.

    .venv/bin/python tools/bench_biencoder_parallel.py --n 512
    .venv/bin/python tools/bench_biencoder_parallel.py --model BAAI/bge-base-en-v1.5
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "eval" / "beir" / "hf-cache"))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def sample(n: int, seed: int = 0) -> list[str]:
    records = json.loads((ROOT / "indexes" / "default" / "index.json").read_text())["chunks"]
    return [record["text"] for record in random.Random(seed).sample(records, min(n, len(records)))]


def load(model_id: str, batch: int, max_length: int, device: str = "auto") -> Any:
    from laya_rag.config import EngineConfig
    from laya_rag.model import BiEncoderModel

    config = EngineConfig()
    config.device = device
    config.embed_model = model_id
    config.embed_batch_size = batch
    config.embed_max_length = max_length
    return BiEncoderModel(config).ensure_loaded()


def shards(items: Sequence[Any], k: int) -> list[list[Any]]:
    size = (len(items) + k - 1) // k
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _worker(index: int, texts: Sequence[str], model_id: str, batch: int, max_length: int,
            queue: Any) -> None:
    try:
        started = time.perf_counter()
        model = load(model_id, batch, max_length)
        load_seconds = time.perf_counter() - started
        model.embed(list(texts[:1]))
        started = time.perf_counter()
        vectors = model.embed(list(texts))
        queue.put(
            {
                "index": index,
                "vectors": np.asarray(vectors, dtype=np.float32),
                "load_seconds": load_seconds,
                "embed_seconds": time.perf_counter() - started,
            }
        )
    except Exception:  # noqa: BLE001 - a failed worker is a result
        queue.put({"index": index, "error": traceback.format_exc()})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--threads", default="2,4,8")
    parser.add_argument("--processes", default="1,2,4")
    args = parser.parse_args(argv)

    texts = sample(args.n, args.seed)
    model = load(args.model, args.batch, args.max_length)
    print(f"loaded {args.model}; {len(texts)} chunks", flush=True)

    for batch in (args.batch, args.batch * 4):
        model.config.embed_batch_size = batch
        model.embed(texts[: min(8, len(texts))])
        started = time.perf_counter()
        model.embed(texts)
        seconds = time.perf_counter() - started
        print(
            f"serial batch={batch}: {seconds:.2f}s  {len(texts) / seconds:.1f} chunks/s",
            flush=True,
        )

    model.config.embed_batch_size = args.batch
    model.embed(texts[: min(8, len(texts))])
    started = time.perf_counter()
    baseline = np.asarray(model.embed(texts), dtype=np.float32)
    serial_seconds = time.perf_counter() - started
    for k in (int(t) for t in args.threads.split(",")):
        parts = shards(texts, k)

        def run(part: Sequence[str]) -> np.ndarray:
            return np.asarray(model.embed(list(part)), dtype=np.float32)

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=k) as pool:
            results = list(pool.map(run, parts))
        wall = time.perf_counter() - started
        print(
            f"threads k={k}: wall {wall:.2f}s  speedup {serial_seconds / wall:.2f}x  "
            f"equal={np.array_equal(baseline, np.vstack(results))}",
            flush=True,
        )

    context = mp.get_context("spawn")
    for k in (int(p) for p in args.processes.split(",")):
        queue = context.Queue()
        processes = [
            context.Process(
                target=_worker,
                args=(i, part, args.model, args.batch, args.max_length, queue),
                daemon=False,
            )
            for i, part in enumerate(shards(texts, k))
        ]
        started = time.perf_counter()
        for process in processes:
            process.start()
        results = [queue.get() for _ in processes]
        for process in processes:
            process.join()
        wall = time.perf_counter() - started
        errors = [result for result in results if "error" in result]
        if errors:
            print(f"processes k={k}: ERROR {errors[0]['error'].splitlines()[-1]}", flush=True)
            continue
        stacked = np.vstack([r["vectors"] for r in sorted(results, key=lambda r: r["index"])])
        embed_max = max(r["embed_seconds"] for r in results)
        load_max = max(r["load_seconds"] for r in results)
        print(
            f"processes k={k}: wall {wall:.2f}s  load(max) {load_max:.2f}s  "
            f"embed(max) {embed_max:.2f}s  agg {len(texts) / embed_max:.1f} chunks/s  "
            f"equal={np.array_equal(baseline, stacked)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
