#!/usr/bin/env python
"""Fine-tune the Laya checkpoint encoder for retrieval (Bet 3 / study I1).

Laya's encoder is a decision encoder: mean-pooled cosine over its embeddings scores
nDCG@10 ~0.014 on SciFact where BM25 scores ~0.65, which is why the engine ships with the
dense channel weighted to zero. This trains that encoder with a contrastive objective on
the retrieval training splits already on disk, freezes the decision head, and writes the
adapted encoder back into a checkpoint `laya.load` reads — so `EMBED_MODEL=""` can become a
real dense channel without a `bge` dependency.

Two supervision modes:

* **gold pairs** (default): `(query, positive document)` from a dataset's `train` qrels;
* **teacher distillation** (`--teacher-models`): the positives are the teachers' RRF-fused
  top documents, exactly the recipe `tools/finetune.py` uses for a `bge-small` student
  (measured +0.0120 nDCG@10 on NFCorpus), with Laya's encoder as the student.

Both towers share one encoder, so the exported artefact is a normal Laya checkpoint and the
engine keeps its single-encoder invariant. The objective is symmetric InfoNCE over in-batch
negatives at `--temperature`.

A run checkpoints itself: encoder + optimiser + step state every `--checkpoint-steps`
steps, keyed by a fingerprint of the settings, so an interrupted run resumes where it
stopped and a changed setting is refused rather than silently reused.

    # smoke: 20 steps, no export
    .venv/bin/python tools/laya_encoder_finetune.py --dataset nfcorpus --limit-steps 20

    # the real run
    .venv/bin/python tools/laya_encoder_finetune.py \
        --datasets nfcorpus,fiqa --max-pairs 150000 --epochs 1 \
        --checkpoint-steps 500 --save-model eval/laya-encoder-model
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "laya"))

import beir  # noqa: E402
import evalkit  # noqa: E402

CHECKPOINTS: dict[str, str | None] = {
    "laya": None,
    "english": None,
    "multilingual": "multilingual",
    "typed-decisions": "typed-decisions",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="laya_encoder_finetune.py")
    ap.add_argument("--checkpoint", default="multilingual", choices=sorted(CHECKPOINTS))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--datasets", default="nfcorpus",
                    help="comma list of datasets with a local train split")
    ap.add_argument("--data-root", default=str(ROOT / "eval" / "beir"))
    ap.add_argument("--max-pairs", type=int, default=150000, help="0 means every pair")
    ap.add_argument("--per-dataset-cap", type=int, default=60000,
                    help="cap each dataset so one large training set cannot dominate")
    ap.add_argument("--dataset-caps", default="",
                    help="per-dataset overrides, e.g. 'msmarco:150000,hotpotqa:40000'; "
                         "unlisted datasets use --per-dataset-cap")
    ap.add_argument("--init-from", default="",
                    help="warm-start the encoder from a Laya checkpoint directory (encoder/* "
                         "tensors) or a plain HF encoder directory, e.g. a previous run")
    ap.add_argument("--teacher-models", default="",
                    help="comma list of bi-encoders to distil from; their RRF-fused top "
                         "documents become the positives instead of the train qrels")
    ap.add_argument("--teacher-positives", type=int, default=3)
    ap.add_argument("--teacher-top-k", type=int, default=100,
                    help="ranking depth the teachers are asked for before fusion")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--mask-false-negatives", action="store_true",
                    help="mask in-batch documents that are known positives of another query "
                         "(matters for datasets with many judgements per query)")
    ap.add_argument("--query-len", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--dynamic-padding", action="store_true",
                    help="pad each batch to its own longest sequence. Fastest per step in "
                         "theory, but on MPS torch caches a graph per shape and the cache grows "
                         "until the run OOMs (measured at 35 GB after ~150 steps at batch 64)")
    ap.add_argument("--pad-bucket", type=int, default=32,
                    help="pad each batch up to the next multiple of this many tokens: a handful "
                         "of shapes (bounded MPS graph cache) without paying for tokens the batch "
                         "does not have. 0 pads to the exact batch maximum; ignored with "
                         "--pad-to-max")
    ap.add_argument("--pad-to-max", action="store_true",
                    help="pad every batch to --max-len/--query-len (the previous default; "
                         "correct but wastes the difference on short passages)")
    ap.add_argument("--empty-cache-steps", type=int, default=50,
                    help="release the MPS allocator every N optimiser steps (0 disables)")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit-steps", type=int, default=0, help="stop after N optimiser steps (smoke)")
    ap.add_argument("--checkpoint-steps", type=int, default=500,
                    help="save resumable state every N optimiser steps")
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--resume-checkpoint", action="store_true",
                    help="resume encoder + optimiser + step from the latest state in --checkpoint-dir")
    ap.add_argument("--save-model", default=str(ROOT / "eval" / "laya-encoder-model"),
                    help="directory for the servable Laya checkpoint (empty string skips it)")
    ap.add_argument("--export-hf", default=str(ROOT / "eval" / "laya-encoder-hf"),
                    help="directory for a plain HF encoder loadable via EMBED_MODEL (empty skips)")
    ap.add_argument("--out", default=str(ROOT / "eval" / "laya-encoder-finetune"))
    return ap.parse_args(argv)


# ── data ─────────────────────────────────────────────────────────────────

def gold_pairs(
    queries: dict[str, str],
    qrels: dict[str, dict[str, float]],
    corpus: dict[str, str],
    cap: int,
) -> list[tuple[str, str]]:
    """``(query, positive document)`` pairs from a train split, capped per dataset."""
    pairs: list[tuple[str, str]] = []
    for query_id, relevant in qrels.items():
        query = queries.get(query_id)
        if not query:
            continue
        for doc_id, score in relevant.items():
            if score <= 0 or doc_id not in corpus:
                continue
            pairs.append((query, corpus[doc_id]))
            if cap and len(pairs) >= cap:
                return pairs
    return pairs


def positive_texts(dataset_dir: Path, needed: set[str], include_titles: bool = True) -> dict[str, str]:
    """``_id -> "title text"`` for only ``needed`` ids, streaming ``corpus.jsonl``.

    Gold-pair training needs the judged documents, not the collection: MS MARCO's corpus is
    8.8M documents, and materialising all of it to look up 533k positives wastes several GB
    on a host that has crashed under memory pressure before. `tools/beir.py`'s
    ``load_corpus`` builds the identical string, so training text matches the dense channel.
    """
    out: dict[str, str] = {}
    wanted = {str(doc_id) for doc_id in needed}
    if not wanted:
        return out
    with (dataset_dir / "corpus.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row["_id"])
            if doc_id not in wanted:
                continue
            title = (row.get("title") or "").strip()
            text = (row.get("text") or "").strip()
            out[doc_id] = f"{title} {text}".strip() if include_titles else text
            if len(out) == len(wanted):
                break
    return out


def load_dataset_pairs(
    dataset: str,
    data_root: Path,
    cap: int,
    teacher_models: Sequence[str],
    teacher_positives: int,
    top_k: int,
    seed: int,
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """Training pairs for one dataset, from gold qrels or teacher-fused positives."""
    dataset_dir = beir.ensure_dataset(dataset, data_root, download=False)
    queries = beir.load_queries(dataset_dir)
    train_qrels = beir.load_qrels(dataset_dir, split="train")
    train_queries = {q: t for q, t in queries.items() if q in train_qrels}
    info: dict[str, Any] = {
        "dataset": dataset,
        "train_queries": len(train_queries),
        "train_judgements": sum(len(r) for r in train_qrels.values()),
    }
    if teacher_models:
        import finetune

        corpus = beir.load_corpus(dataset_dir)
        info["documents"] = len(corpus)
        rankings: dict[str, dict[str, list[str]]] = {}
        for model_id in teacher_models:
            teacher = beir.BiEncoder(model_id)
            cache = data_root / "cache" / f"{dataset}.{teacher.name}"
            vectors = beir.cached_embed(teacher, corpus, cache)
            rankings[model_id] = beir.dense_rankings(
                corpus, train_queries, teacher, vectors, top_k, fixed_length=True
            )
            print(f"  teacher {model_id} ranked {len(train_queries)} train queries", flush=True)
        pairs = finetune.ensemble_positive_pairs(corpus, train_queries, rankings, teacher_positives)
        info["teacher_models"] = list(teacher_models)
        info["supervision"] = "teacher"
    else:
        needed = {
            str(doc_id)
            for relevant in train_qrels.values()
            for doc_id, score in relevant.items()
            if score > 0
        }
        corpus = positive_texts(dataset_dir, needed)
        info["documents"] = len(corpus)
        info["positives_found"] = len(corpus)
        info["positives_missing"] = len(needed) - len(corpus)
        pairs = gold_pairs(queries, train_qrels, corpus, 0)
        info["supervision"] = "gold"
    # Shuffle before capping so a cap samples the split instead of taking the earliest
    # queries in qrels order, which is correlated with query id and therefore with topic.
    random.Random(seed).shuffle(pairs)
    if cap:
        pairs = pairs[:cap]
    info["pairs"] = len(pairs)
    return pairs, info


# ── checkpointing ────────────────────────────────────────────────────────

def parse_dataset_caps(spec: str) -> dict[str, int]:
    """``"msmarco:150000,hotpotqa:40000"`` -> ``{"msmarco": 150000, ...}``."""
    caps: dict[str, int] = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        name, _, value = part.partition(":")
        if not name or not value.strip().isdigit():
            raise SystemExit(f"--dataset-caps: expected name:count, got {part!r}")
        caps[name.strip()] = int(value)
    return caps


def init_encoder_from(encoder: Any, source: str, torch: Any) -> None:
    """Warm-start the encoder from a Laya checkpoint directory or a plain HF directory.

    Staged training is the cheap way to scale: a first run learns retrieval generally, a
    second run sharpens it on a different mix (or at a lower LR) without paying for the
    first pass again. A Laya checkpoint stores the whole decision model with ``encoder.``
    prefixes; an HF directory stores the encoder alone.
    """
    from safetensors.torch import load_file

    path = Path(source)
    if (path / "rl_agent_config.json").is_file():
        state = load_file(str(path / "model.safetensors"))
        state = {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")}
        kind = "Laya checkpoint"
    elif (path / "model.safetensors").is_file():
        state = load_file(str(path / "model.safetensors"))
        kind = "HF encoder"
    else:
        raise SystemExit(f"--init-from: no model.safetensors under {path}")
    if not state:
        raise SystemExit(f"--init-from: {path} yielded no encoder tensors")
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(f"--init-from: unexpected keys in {path}: {sorted(unexpected)[:5]}")
    print(f"  warm-started encoder from {path} ({kind}: {len(state)} tensors, "
          f"{len(missing)} missing)", flush=True)


def fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    """The settings a resumable state belongs to. Changing any refuses the resume."""
    keys = [
        "checkpoint", "datasets", "max_pairs", "per_dataset_cap", "dataset_caps", "init_from",
        "teacher_models", "teacher_positives", "epochs", "batch_size", "grad_accum", "lr",
        "temperature", "query_len", "max_len", "seed",
    ]
    return {key: getattr(args, key) for key in keys}


def fingerprint_id(payload: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def checkpoint_dir(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir)
    return ROOT / "eval" / "laya-encoder-checkpoints" / args.datasets.replace(",", "-")


def save_state(
    directory: Path,
    encoder: Any,
    optimizer: Any,
    scheduler: Any,
    step: int,
    epoch: int,
    args: argparse.Namespace,
    fp: str,
    torch: Any,
) -> Path:
    """Write the resumable state atomically (torch.save then replace)."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "latest.pt"
    temporary = directory / ".latest.pt.tmp"
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "encoder": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "fingerprint": fp,
            "settings": fingerprint(args),
        },
        temporary,
    )
    os.replace(temporary, target)
    return target


def load_state(path: Path, expected: str, torch: Any) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu")
    if state.get("fingerprint") != expected:
        raise SystemExit(
            f"{path} belongs to settings fingerprint {state.get('fingerprint')!r}, "
            f"this run is {expected!r}; refusing to resume"
        )
    return state


# ── training ─────────────────────────────────────────────────────────────

def mean_pool(hidden: Any, mask: Any) -> Any:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


def batch_vectors(
    torch: Any,
    encoder: Any,
    tok: Any,
    texts: list[str],
    max_length: int,
    device: str,
    pad_to_max: bool = True,
    bucket: int = 32,
) -> Any:
    """Mean-pooled embeddings for one batch.

    Padding width is chosen to keep torch's MPS graph cache bounded **without** wasting
    compute. The backend caches a compiled graph per tensor shape, so fully dynamic padding
    grows that cache without bound until the run dies with an MPS OOM (measured: 35 GB of
    "other allocations" after ~150 steps at batch 64 while thrashing). Padding every batch to
    the full ``max_length`` fixes the cache but pays for tokens that are not there: MS MARCO
    passages average ~50 tokens, so a 128-wide pad doubles the work for nothing.

    The default is therefore to pad up to the next multiple of ``bucket`` — at most
    ``max_length`` padded positions are rounded up to the next bucket, which gives a handful of
    distinct shapes (bounded cache) and close to the batch's real width (little waste).
    ``bucket=0`` pads to the exact batch maximum, and ``pad_to_max`` restores full-width padding.
    Padded positions are masked and excluded from the mean, so the three modes are numerically
    equivalent.
    """
    encoded = tok(
        list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    width = int(encoded["input_ids"].shape[1])
    if pad_to_max:
        target = max_length
    elif bucket and bucket > 1:
        target = min(max_length, max(bucket, ((width + bucket - 1) // bucket) * bucket))
    else:
        target = width
    if target > width:
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        encoded["input_ids"] = torch.nn.functional.pad(
            encoded["input_ids"], (0, target - width), value=pad_id
        )
        encoded["attention_mask"] = torch.nn.functional.pad(
            encoded["attention_mask"], (0, target - width), value=0
        )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    hidden = encoder(**encoded).last_hidden_state
    return mean_pool(hidden, encoded["attention_mask"])


def info_nce(torch: Any, query: Any, document: Any, temperature: float, block: Any = None) -> Any:
    """Symmetric InfoNCE, optionally masking in-batch false negatives.

    ``block[i, j]`` true means document ``j`` is a *known positive* for query ``i`` other than
    the diagonal, so it must not be pushed away. Datasets with many judgements per query
    (NFCorpus averages ~42) put several true positives in one batch; without the mask MNRL
    penalises them as negatives, which inflates the loss and teaches a narrower notion of
    relevance. The transpose is the mask for the document-to-query direction.
    """
    query = torch.nn.functional.normalize(query, dim=1)
    document = torch.nn.functional.normalize(document, dim=1)
    logits = query @ document.t() / temperature
    labels = torch.arange(query.size(0), device=query.device)
    forward = logits
    backward = logits.t()
    if block is not None:
        forward = logits.masked_fill(block, -1e4)
        backward = logits.t().masked_fill(block.t(), -1e4)
    return (
        torch.nn.functional.cross_entropy(forward, labels)
        + torch.nn.functional.cross_entropy(backward, labels)
    ) / 2


def false_negative_blocks(
    torch: Any,
    queries: Sequence[str],
    documents: Sequence[str],
    positives: set[tuple[str, str]],
    device: str,
) -> Any:
    """``block[i, j]`` for a batch: document ``j`` is another known positive of query ``i``."""
    size = len(queries)
    rows = [
        [index != other and (queries[index], documents[other]) in positives for other in range(size)]
        for index in range(size)
    ]
    return torch.tensor(rows, dtype=torch.bool, device=device)


def pick_device(torch: Any, requested: str) -> str:
    if requested in ("", "auto", None):
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return requested


def export_checkpoint(agent: Any, encoder: Any, out: Path, torch: Any) -> None:
    """Write the training weights back into a checkpoint `laya.load` reads."""
    from laya.train import save_checkpoint

    agent.model.encoder.load_state_dict(encoder.state_dict())
    if out.exists():
        shutil.rmtree(out)
    save_checkpoint(agent.model, agent.tok, agent.cfg, str(out))
    print(f"  wrote Laya checkpoint {out}", flush=True)


def export_hf(agent: Any, encoder: Any, out: Path, metadata: dict[str, Any] | None = None) -> None:
    """Write a plain HF encoder directory so `EMBED_MODEL=<dir>` works with no engine change.

    The `laya_retrieval.json` marker is what lets the engine tell a retrieval adaptation of
    Laya's own encoder from a stock bi-encoder: they want different RRF weights (0.3 against
    1.0), and getting that wrong silently costs several points of nDCG.
    """
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    encoder.save_pretrained(out)
    agent.tok.save_pretrained(out)
    pooling = out / "1_Pooling"
    pooling.mkdir(parents=True, exist_ok=True)
    (pooling / "config.json").write_text(
        json.dumps({"pooling_mode_mean_tokens": True, "pooling_mode_cls_token": False}, indent=2),
        encoding="utf-8",
    )
    marker = {
        "kind": "laya-retrieval-adaptation",
        "source_checkpoint": getattr(agent, "model_id", None),
        "note": "Marked so the engine fuses it at the tuned retrieval weight, not a stock "
                "bi-encoder's 1.0. Mean pooling, 768 dimensions.",
    }
    if metadata:
        marker.update(metadata)
    (out / "laya_retrieval.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
    print(f"  wrote HF encoder {out} (mean pooling, marked as a Laya retrieval adaptation)",
          flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # Padding policy: bucketed by default (bounded MPS graph cache, little waste); `--pad-to-max`
    # restores full-width padding, `--dynamic-padding` the unbucketed batch maximum.
    if args.dynamic_padding:
        args.pad_bucket = 0
    from heavylock import heavy_job_guard

    heavy_job_guard("laya_encoder_finetune.py")
    random.seed(args.seed)
    np.random.seed(args.seed)

    import torch

    torch.manual_seed(args.seed)
    device = pick_device(torch, args.device)
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    teachers = [m.strip() for m in (args.teacher_models or "").split(",") if m.strip()]
    if not datasets:
        raise SystemExit("--datasets is empty")

    started = time.perf_counter()
    import laya

    agent = laya.load(
        str(Path(args.models_dir) / "laya"), device=device, subfolder=CHECKPOINTS[args.checkpoint]
    )
    encoder = agent.model.encoder
    for parameter in encoder.parameters():
        parameter.requires_grad_(True)
    print(f"loaded {args.checkpoint} on {device} in {time.perf_counter() - started:.1f}s; "
          f"encoder {sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M params", flush=True)
    if args.init_from:
        init_encoder_from(encoder, args.init_from, torch)

    dataset_caps = parse_dataset_caps(args.dataset_caps)
    pairs: list[tuple[str, str]] = []
    dataset_info: list[dict[str, Any]] = []
    for dataset in datasets:
        found, info = load_dataset_pairs(
            dataset, Path(args.data_root), dataset_caps.get(dataset, args.per_dataset_cap),
            teachers, args.teacher_positives, args.teacher_top_k, args.seed,
        )
        pairs.extend(found)
        dataset_info.append(info)
        print(f"  {dataset}: {info['pairs']} pairs ({info['supervision']})", flush=True)
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    if len(pairs) < args.batch_size:
        raise SystemExit(f"only {len(pairs)} pairs; need at least --batch-size")
    print(f"training on {len(pairs)} pairs", flush=True)
    positives = set(pairs) if args.mask_false_negatives else None
    if positives is not None:
        print(f"  masking in-batch false negatives ({len(positives)} known positive pairs)",
              flush=True)

    fp = fingerprint_id(fingerprint(args))
    directory = checkpoint_dir(args)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(pairs) // (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_frac))

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    # LambdaLR multiplies the base lr by this factor, so the schedule must be returned as a
    # ratio. Passing an absolute learning rate here silently trains at lr**2 (measured: a
    # 2e-5 schedule stepped to 4e-12 and the loss barely moved).
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_at(s) / args.lr)
    step = 0
    epoch = 0
    if args.resume_checkpoint:
        latest = directory / "latest.pt"
        if not latest.is_file():
            raise SystemExit(f"--resume-checkpoint: no state at {latest}")
        state = load_state(latest, fp, torch)
        encoder.load_state_dict(state["encoder"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") and scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        step, epoch = int(state["step"]), int(state["epoch"])
        print(f"resumed encoder + optimiser at step {step} (epoch {epoch})", flush=True)

    encoder.train()
    losses: list[float] = []
    train_started = time.perf_counter()
    stop = False
    for epoch in range(epoch, args.epochs):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        cursor = 0
        while cursor + args.batch_size <= len(order):
            if args.limit_steps and step >= args.limit_steps:
                stop = True
                break
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            for _ in range(args.grad_accum):
                if cursor + args.batch_size > len(order):
                    break
                indices = order[cursor : cursor + args.batch_size]
                cursor += args.batch_size
                queries = [pairs[i][0] for i in indices]
                documents = [pairs[i][1] for i in indices]
                query_vectors = batch_vectors(
                    torch, encoder, agent.tok, queries, args.query_len, device,
                    args.pad_to_max, args.pad_bucket,
                )
                doc_vectors = batch_vectors(
                    torch, encoder, agent.tok, documents, args.max_len, device,
                    args.pad_to_max, args.pad_bucket,
                )
                block = (
                    false_negative_blocks(torch, queries, documents, positives, device)
                    if positives is not None
                    else None
                )
                loss = info_nce(torch, query_vectors, doc_vectors, args.temperature, block) / args.grad_accum
                loss.backward()
                epoch_loss += float(loss.item())
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            step += 1
            losses.append(epoch_loss)
            if step % 50 == 0 or step == 1:
                elapsed = time.perf_counter() - train_started
                print(f"  step {step}/{total_steps} loss {epoch_loss:.4f} "
                      f"lr {optimizer.param_groups[0]['lr']:.2e} "
                      f"{elapsed / step:.2f}s/step", flush=True)
            if args.checkpoint_steps and step % args.checkpoint_steps == 0:
                saved = save_state(directory, encoder, optimizer, scheduler, step, epoch, args, fp, torch)
                print(f"  checkpoint at step {step} -> {saved}", flush=True)
            if args.empty_cache_steps and step % args.empty_cache_steps == 0 and device == "mps":
                # Even at a fixed width the allocator holds freed blocks; releasing them keeps
                # the trainer's footprint flat over thousands of steps.
                torch.mps.empty_cache()
        if stop:
            break
    train_seconds = time.perf_counter() - train_started
    print(f"trained {step} steps in {train_seconds:.1f}s "
          f"({train_seconds / max(1, step):.2f}s/step)", flush=True)

    # Always keep a final resumable state so a later run can resume this one.
    if args.checkpoint_steps:
        save_state(directory, encoder, optimizer, scheduler, step, epoch, args, fp, torch)

    report: dict[str, Any] = {
        "settings": vars(args),
        "device": device,
        "datasets": dataset_info,
        "pairs": len(pairs),
        "steps": step,
        "train_seconds": round(train_seconds, 1),
        "seconds_per_step": round(train_seconds / max(1, step), 3),
        "loss_first": round(losses[0], 4) if losses else None,
        "loss_last": round(losses[-1], 4) if losses else None,
        "fingerprint": fp,
        "trainable_parameters": sum(p.numel() for p in encoder.parameters()),
    }
    if args.save_model:
        export_checkpoint(agent, encoder, Path(args.save_model), torch)
        report["save_model"] = args.save_model
    if args.export_hf:
        export_hf(
            agent, encoder, Path(args.export_hf),
            {
                "datasets": datasets,
                "pairs": len(pairs),
                "steps": step,
                "init_from": args.init_from or None,
                "fingerprint": fp,
            },
        )
        report["export_hf"] = args.export_hf
    evalkit.write_json(Path(args.out).with_suffix(".json"), report)
    print(f"wrote {Path(args.out).with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
