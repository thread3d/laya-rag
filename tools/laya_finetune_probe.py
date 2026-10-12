#!/usr/bin/env python
"""Feasibility probe for fine-tuning the Laya encoder on this host.

Answers the two questions `docs/study-laya-finetuning.md` depends on, without training a
real model: can the vendored Laya encoder be trained on this device at a usable step
rate, and can an adapted checkpoint be saved and re-opened the way `laya.load` expects?

Both are cheap. The throughput probe runs one cold + three warm forward/backward steps of
a symmetric InfoNCE objective over the checkpoint encoder and reports seconds/step and MPS
memory. Batches are padded to `--query-len` / `--doc-len` regardless of the sample text,
so the step rate is a **worst-case** bound rather than a function of the dummy strings; the
round-trip probe writes an fp16 checkpoint and reloads it through `laya.load`, asking it
the engine's own relevance question.

    .venv/bin/python tools/laya_finetune_probe.py
    .venv/bin/python tools/laya_finetune_probe.py --checkpoint multilingual --device mps
    .venv/bin/python tools/laya_finetune_probe.py --skip-roundtrip

This is a probe, not a benchmark: the step rate depends on the batch shape, and the study
projects real run lengths from the printed number.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "laya"))

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="laya_finetune_probe.py")
    ap.add_argument("--checkpoint", default="multilingual",
                    choices=("laya", "english", "multilingual", "typed-decisions"))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--device", default="auto", help="auto picks MPS when available")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--query-len", type=int, default=192)
    ap.add_argument("--doc-len", type=int, default=256)
    ap.add_argument("--steps", type=int, default=3, help="warm steps to average")
    ap.add_argument("--skip-roundtrip", action="store_true")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    return ap.parse_args(argv)


def subfolder(checkpoint: str) -> str | None:
    return None if checkpoint in {"laya", "english"} else checkpoint


def pick_device(torch: Any, requested: str) -> str:
    if requested in ("", "auto", None):
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return requested


def mean_pool(hidden: Any, mask: Any) -> Any:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


def encode_batch(
    tok: Any,
    texts: Sequence[str],
    max_length: int,
    device: str,
    *,
    pad_to_max: bool = False,
) -> dict[str, Any]:
    encoded = tok(
        list(texts),
        padding="max_length" if pad_to_max else True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def throughput_probe(agent: Any, args: argparse.Namespace, torch: Any, device: str) -> dict[str, Any]:
    encoder = agent.model.encoder
    encoder.train()
    parameters = sum(p.numel() for p in encoder.parameters())
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-5)
    size = args.batch_size
    queries = [f"query about refund timing number {i}" for i in range(size)]
    documents = [f"refunds are processed within five business days, document {i}" for i in range(size)]
    query_batch = encode_batch(agent.tok, queries, args.query_len, device, pad_to_max=True)
    doc_batch = encode_batch(agent.tok, documents, args.doc_len, device, pad_to_max=True)
    labels = torch.arange(size, device=device)
    step_seconds: list[float] = []
    losses: list[float] = []
    for _ in range(args.steps + 1):
        started = time.perf_counter()
        query_vectors = mean_pool(encoder(**query_batch).last_hidden_state, query_batch["attention_mask"])
        doc_vectors = mean_pool(encoder(**doc_batch).last_hidden_state, doc_batch["attention_mask"])
        query_vectors = torch.nn.functional.normalize(query_vectors, dim=1)
        doc_vectors = torch.nn.functional.normalize(doc_vectors, dim=1)
        logits = query_vectors @ doc_vectors.t() / 0.05
        loss = (
            torch.nn.functional.cross_entropy(logits, labels)
            + torch.nn.functional.cross_entropy(logits.t(), labels)
        ) / 2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        # The first step pays the backend's graph-compile cost; it is reported separately.
        step_seconds.append(time.perf_counter() - started)
        losses.append(float(loss.item()))
    cold = step_seconds[0]
    warm = step_seconds[1:]
    allocated = None
    if device == "mps":
        allocated = round(torch.mps.current_allocated_memory() / 1e6, 1)
    return {
        "device": device,
        "encoder_parameters": parameters,
        "batch_size": size,
        "query_len": args.query_len,
        "doc_len": args.doc_len,
        "cold_step_seconds": round(cold, 3),
        "warm_step_seconds_mean": round(sum(warm) / len(warm), 3) if warm else None,
        "warm_step_seconds": [round(s, 3) for s in warm],
        "loss_first": round(losses[0], 4),
        "loss_last": round(losses[-1], 4),
        "mps_allocated_mb": allocated,
    }


def roundtrip_probe(agent: Any, torch: Any) -> dict[str, Any]:
    from laya.train import save_checkpoint

    directory = Path(tempfile.mkdtemp(prefix="laya-checkpoint-probe-"))
    target = directory / "checkpoint"
    try:
        started = time.perf_counter()
        save_checkpoint(agent.model, agent.tok, agent.cfg, str(target))
        save_seconds = time.perf_counter() - started
        files = sorted(p.name for p in target.iterdir())
        import laya

        reloaded = laya.load(str(target), device="cpu")
        result = reloaded.predict(
            {"query": "how long do refunds take?", "passage": "Refunds take five business days."},
            {"relevant": {"type": "noul", "instructions": "Does `passage` help answer `query`?"}},
        )
        probability = float(result["answers"]["relevant"]["noul"])
        return {
            "save_seconds": round(save_seconds, 3),
            "files": files,
            "reload_probability": probability,
            "reload_ok": 0.0 <= probability <= 1.0,
        }
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    device = pick_device(torch, args.device)
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    spec = Path(args.models_dir) / "laya"
    started = time.perf_counter()
    import laya

    agent = laya.load(str(spec), device=device, subfolder=subfolder(args.checkpoint))
    report: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "models_dir": str(spec),
        "load_seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(report, indent=2), flush=True)
    report["throughput"] = throughput_probe(agent, args, torch, device)
    print(json.dumps(report["throughput"], indent=2), flush=True)
    if not args.skip_roundtrip:
        report["roundtrip"] = roundtrip_probe(agent, torch)
        print(json.dumps(report["roundtrip"], indent=2), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
