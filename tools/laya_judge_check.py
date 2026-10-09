#!/usr/bin/env python
"""Did adapting the encoder damage the decision head it shares weights with?

The decision head reads the encoder's hidden states, so fine-tuning the encoder
(`tools/laya_encoder_finetune.py`) changes what the head sees even though the head's own
weights are frozen. This measures the engine's **relevance** decision — the exact question
`rerank.py` asks — on a labelled set of `(query, positive)` and `(query, unjudged)` pairs
from a dataset's test split, and reports AUC, accuracy at the engine's 0.5 threshold and
ECE. Run it once against the shipped checkpoint and once against the adapted one: the
difference is the judge damage, if any.

    .venv/bin/python tools/laya_judge_check.py --dataset scifact --out eval/judge-original
    LAYA_RAG_CHECKPOINT=eval/laya-encoder-runA \
        .venv/bin/python tools/laya_judge_check.py --dataset scifact --out eval/judge-adapted

Negatives are sampled deterministically from documents the qrels never judge for that query,
so the labels are honest and the run is reproducible. AUC is computed directly (rank statistic
with tie-corrected ranks) so the tool adds no dependency.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "laya"))

import beir  # noqa: E402
import evalkit  # noqa: E402
from laya_rag.config import EngineConfig  # noqa: E402
from laya_rag.model import LayaModel  # noqa: E402
from laya_rag.rerank import LayaReranker  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="laya_judge_check.py")
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--data-root", default=str(ROOT / "eval" / "beir"))
    ap.add_argument("--negatives", type=int, default=2, help="unjudged documents per query")
    ap.add_argument("--limit", type=int, default=0, help="cap the number of queries (smoke)")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "eval" / "judge-check"))
    return ap.parse_args(argv)


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Tie-corrected AUC via the Mann-Whitney rank statistic."""
    y = np.asarray(labels, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    positives = y == 1
    n_pos = int(positives.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_scores = s[order]
    index = 0
    while index < len(s):
        end = index
        while end + 1 < len(s) and sorted_scores[end + 1] == sorted_scores[index]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        ranks[order[index : end + 1]] = average
        index = end + 1
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ece(labels: Sequence[int], scores: Sequence[float], bins: int = 10) -> float:
    y = np.asarray(labels, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (s >= low) & (s < high if high < 1.0 else s <= high)
        if not mask.any():
            continue
        total += mask.mean() * abs(y[mask].mean() - s[mask].mean())
    return float(total)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_dir = beir.ensure_dataset(args.dataset, Path(args.data_root), download=False)
    corpus = beir.load_corpus(dataset_dir)
    queries = beir.load_queries(dataset_dir)
    test_qrels = beir.load_qrels(dataset_dir, split="test")
    rng = random.Random(args.seed)
    doc_ids = list(corpus)
    judged = {q: {d for d, s in rel.items() if s > 0} for q, rel in test_qrels.items()}

    pairs: list[tuple[str, str]] = []
    labels: list[int] = []
    for query_id, relevant in test_qrels.items():
        query = queries.get(query_id)
        positives = [d for d, s in relevant.items() if s > 0 and d in corpus]
        if not query or not positives:
            continue
        pairs.append((query, corpus[rng.choice(positives)]))
        labels.append(1)
        # An unjudged document is a negative under the pooled-judgement assumption. Sampling
        # by index keeps this O(1) per query and reproducible; a sampled doc that happens to
        # be judged for this query is rejected and re-drawn.
        drawn = 0
        attempts = 0
        while drawn < args.negatives and attempts < 50:
            attempts += 1
            candidate = doc_ids[rng.randrange(len(doc_ids))]
            if candidate in judged[query_id]:
                continue
            pairs.append((query, corpus[candidate]))
            labels.append(0)
            drawn += 1
        if args.limit and len(labels) >= args.limit * (1 + args.negatives):
            break

    config = EngineConfig()
    model = LayaModel(config)
    reranker = LayaReranker(model)
    print(f"{args.dataset}: scoring {len(pairs)} pairs from {len(pairs) - sum(labels)} negatives "
          f"with checkpoint {config.checkpoint!r}", flush=True)
    scores, _usage = reranker.score_relevance_pairs(pairs, batch_size=args.batch_size)
    report: dict[str, Any] = {
        "dataset": args.dataset,
        "checkpoint": str(config.checkpoint),
        "pairs": len(pairs),
        "positives": int(sum(labels)),
        "negatives": int(len(labels) - sum(labels)),
        "auc": round(roc_auc(labels, scores), 4),
        "accuracy_at_0.5": round(float(np.mean((np.asarray(scores) >= 0.5) == np.asarray(labels))), 4),
        "ece": round(ece(labels, scores), 4),
        "mean_probability_positive": round(float(np.mean([s for s, l in zip(scores, labels) if l])), 4),
        "mean_probability_negative": round(float(np.mean([s for s, l in zip(scores, labels) if not l])), 4),
    }
    print(json.dumps(report, indent=2), flush=True)
    evalkit.write_json(Path(args.out).with_suffix(".json"), report)
    print(f"wrote {Path(args.out).with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
