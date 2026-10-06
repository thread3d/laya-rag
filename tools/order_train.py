"""Train a cross-encoder on MuSiQue's own relevance conventions — ROADMAP Phase 2.2.

The two Phase 2 ordering heads (2.1 late interaction, 2.3 relational NLI) both failed, and both
borrowed a model trained on something else. This is the alternative the roadmap calls the real
lever: train on **this task's** conventions — a multi-hop query and the paragraph that answers part
of it — rather than on MS MARCO points or topical similarity.

Pairs are mined from the MuSiQue **train** split (`tools/fetch_musique.py --split train`), which
ships gold supporting paragraphs, so no retrieval and no LLM is needed:

* positive — a paragraph the record marks `is_supporting`;
* negative — a sampled non-supporting paragraph from the *same record's* candidate list, which is
  a same-topic distractor rather than a random corpus paragraph.

The trainer is a plain PyTorch loop over the cross-encoder's own HF model, so step checkpoints can
save optimiser state and a killed run resumes rather than restarting the epoch. The final model is
written in `sentence_transformers` format, so `tools/order_eval.py` (and anything else) loads it
with `CrossEncoder(path)`.

    .venv/bin/python tools/order_train.py --max-queries 4000   # bounded adaptation
    .venv/bin/python tools/order_train.py --epochs 1 --checkpoint-steps 500
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
# Project-local model cache, following tools/beir.py and tools/bright.py. The global cache is not
# written to: it may hold another user's or an interrupted download's lock.
os.environ.setdefault("HF_HOME", str(ROOT / "eval" / "order-hf-cache"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import evalkit, jobkit  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "musique"
DEFAULT_BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def mine_pairs(records: Mapping[str, Any], *, negatives: int = 3, seed: int = 17,
               max_queries: int = 0) -> tuple[list[tuple[str, str, float]], list[str]]:
    """`(query, paragraph, label)` pairs, plus the query ids the pairs came from.

    MuSiQue records carry their own candidate paragraphs with an `is_supporting` flag, so the
    negatives are same-topic distractors. A record with no supporting paragraph is skipped rather
    than contributing an all-negative group.
    """
    rng = random.Random(seed)
    pairs: list[tuple[str, str, float]] = []
    used: list[str] = []
    for query_id, record in records.items():
        if max_queries and len(used) >= max_queries:
            break
        question = str(record.get("question") or "")
        paragraphs = record.get("paragraphs") or []
        positives = [p for p in paragraphs if p.get("is_supporting") and p.get("paragraph_text")]
        if not question or not positives:
            continue
        others = [p for p in paragraphs if not p.get("is_supporting") and p.get("paragraph_text")]
        for paragraph in positives:
            pairs.append((question, str(paragraph["paragraph_text"]), 1.0))
        for paragraph in rng.sample(others, min(negatives, len(others))):
            pairs.append((question, str(paragraph["paragraph_text"]), 0.0))
        used.append(query_id)
    return pairs, used


def split_pairs(
    pairs: Sequence[tuple[str, str, float]], holdout: float, seed: int
) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float]]]:
    """Hold out a fraction of the *pairs* for a training-time sanity check.

    The real adapted-vs-zero-shot comparison is `tools/order_eval.py` on MuSiQue validation; this
    split only exists so the trainer can report a held-out loss it did not fit.
    """
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1.0 - holdout))
    return shuffled[:cut], shuffled[cut:]


def checkpoint_path(directory: Path | str, step: int) -> Path:
    return Path(directory) / f"step-{step:06d}.pt"


def latest_checkpoint(directory: Path | str) -> Path | None:
    directory = Path(directory)
    if not directory.is_dir():
        return None
    found = sorted((p for p in directory.glob("step-*.pt") if p.stem.split("-")[-1].isdigit()),
                   key=lambda p: int(p.stem.split("-")[-1]))
    return found[-1] if found else None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="order_train.py")
    ap.add_argument("--split-file", default=str(ROOT_BENCH / "train.json"))
    ap.add_argument("--base", default=DEFAULT_BASE, help="cross-encoder to adapt")
    ap.add_argument("--max-queries", type=int, default=4000, help="0 = the whole split")
    ap.add_argument("--negatives", type=int, default=3, help="distractors per query")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--holdout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default=None)
    ap.add_argument("--checkpoint-steps", type=int, default=500)
    ap.add_argument("--checkpoint-dir", default=None,
                    help="default: eval/order-ce-checkpoints/<tag>")
    ap.add_argument("--fresh", action="store_true",
                    help="discard any existing step checkpoints for this tag")
    ap.add_argument("--save-model", default=str(ROOT / "eval" / "order-ce-musique"))
    ap.add_argument("--tag", default="musique")
    ap.add_argument("--out", default=str(ROOT / "eval" / "order-train"))
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from heavylock import heavy_job_guard

    heavy_job_guard("order_train.py")
    import torch
    from sentence_transformers import CrossEncoder

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir or (ROOT / "eval" / "order-ce-checkpoints" / args.tag))

    raw = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    records = {str(record.get("id")): record for record in raw}
    pairs, query_ids = mine_pairs(records, negatives=args.negatives, seed=args.seed,
                                  max_queries=args.max_queries)
    if not pairs:
        raise SystemExit(f"no training pairs mined from {args.split_file}")
    train_pairs, holdout_pairs = split_pairs(pairs, args.holdout, args.seed)
    print(f"mined {len(pairs)} pairs from {len(query_ids)} queries "
          f"({len(train_pairs)} train / {len(holdout_pairs)} holdout)", flush=True)

    model = CrossEncoder(args.base, num_labels=1, max_length=args.max_length, device=device)
    tokenizer, hf = model.tokenizer, model.model
    optimizer = torch.optim.AdamW(hf.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # A step checkpoint is only meaningful for the run that wrote it. Without this, a smoke run's
    # checkpoint is silently resumed by a full run and the adaptation starts from the wrong data —
    # which happened, and is why the fingerprint exists.
    fingerprint = jobkit.config_fingerprint(
        tool="order_train", base=args.base, split_file=Path(args.split_file).name,
        negatives=args.negatives, max_queries=args.max_queries, seed=args.seed,
        max_length=args.max_length, epochs=args.epochs,
    )
    if args.fresh and checkpoint_dir.is_dir():
        for stale in checkpoint_dir.glob("step-*.pt"):
            stale.unlink()
        print(f"  discarded checkpoints in {checkpoint_dir}", flush=True)

    start_step = 0
    resume = latest_checkpoint(checkpoint_dir)
    if resume is not None:
        state = torch.load(resume, map_location=device)
        if state.get("fingerprint") != fingerprint:
            raise SystemExit(
                f"{resume} was written by a different run (fingerprint "
                f"{str(state.get('fingerprint'))[:12]} != {fingerprint[:12]}); rerun with --fresh "
                "to discard it"
            )
        hf.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        print(f"  resumed from {resume.name} at step {start_step}", flush=True)

    def encode(batch):
        queries = [row[0] for row in batch]
        passages = [row[1] for row in batch]
        labels = torch.tensor([row[2] for row in batch], dtype=torch.float, device=device)
        enc = tokenizer(queries, passages, padding=True, truncation=True,
                        max_length=args.max_length, return_tensors="pt")
        return {key: value.to(device) for key, value in enc.items()}, labels

    def held_out_loss() -> float | None:
        if not holdout_pairs:
            return None
        torch_hf = hf.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(holdout_pairs), args.batch_size):
                enc, labels = encode(holdout_pairs[start : start + args.batch_size])
                logits = torch_hf(**enc).logits.view(-1).float()
                total += float(loss_fn(logits, labels)) * len(labels)
                count += len(labels)
        hf.train()
        return total / max(count, 1)

    started = time.perf_counter()
    step = start_step
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(train_pairs)
        for start in range(0, len(train_pairs), args.batch_size):
            batch = train_pairs[start : start + args.batch_size]
            enc, labels = encode(batch)
            optimizer.zero_grad()
            logits = hf(**enc).logits.view(-1).float()
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hf.parameters(), 1.0)
            optimizer.step()
            step += 1
            if args.checkpoint_steps and step % args.checkpoint_steps == 0:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"model": hf.state_dict(), "optimizer": optimizer.state_dict(),
                            "step": step, "fingerprint": fingerprint},
                           checkpoint_path(checkpoint_dir, step))
                print(f"    step {step}: loss {float(loss):.4f} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)

    holdout = held_out_loss()
    Path(args.save_model).mkdir(parents=True, exist_ok=True)
    model.save(args.save_model)

    report = {
        "_settings": vars(args),
        "queries": len(query_ids),
        "pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "holdout_pairs": len(holdout_pairs),
        "steps": step - start_step,
        "resumed_from_step": start_step,
        "holdout_loss": None if holdout is None else round(holdout, 4),
        "seconds": round(time.perf_counter() - started, 1),
        "saved_model": args.save_model,
    }
    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), report)
    print(f"adapted {args.base} on {len(train_pairs)} pairs, {step - start_step} steps, "
          f"holdout loss {report['holdout_loss']} in {report['seconds']}s", flush=True)
    print("saved", args.save_model)
    print("wrote", str(out) + ".json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
