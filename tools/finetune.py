#!/usr/bin/env python
"""Domain-adapted fine-tuning (item 7): train a bi-encoder on in-domain pairs.

Everything else in this project is zero-shot. This intentionally is not: it fine-tunes
`bge-small` on a dataset's **train** split and evaluates on its **test** split, so the
number it produces must be labelled adapted, never compared as if it were zero-shot.

NFCorpus ships 110,575 train pairs and is the standard demonstration for this. The first
run here used 20k pairs, one epoch, and in-batch negatives only, for +0.0085 — a floor.
This version supports the whole split (`--max-pairs 0`), several epochs, and mined hard
negatives: for each training query it retrieves the non-relevant documents the zero-shot
model scores highest and adds them as explicit triplets, which is the part in-batch
negatives cannot see.

    .venv/bin/python tools/finetune.py --dataset nfcorpus --max-pairs 0 --epochs 2 \
        --hard-negatives 1 --hard-negative-pairs 40000 --save-model eval/finetuned-full

Uses sentence-transformers' `MultipleNegativesRankingLoss` (in-batch negatives) plus a
`TripletLoss` objective over the mined hard negatives.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import beir  # noqa: E402
import evalkit  # noqa: E402
from laya_rag.index_store import VectorIndex  # noqa: E402
from laya_rag.lexical import reciprocal_rank_fusion  # noqa: E402


def ensemble_positive_pairs(
    corpus: dict[str, str],
    queries: dict[str, str],
    teacher_rankings: dict[str, dict[str, list[str]]],
    per_query: int,
) -> list[tuple[str, str]]:
    """``(query, document)`` pairs from the teachers' RRF-fused top documents.

    Distillation supervision: instead of the train qrels, the student learns to retrieve
    what an ensemble of teachers already ranks highly. Each teacher is one dense channel;
    fusing them by rank needs no score calibration between models. Deterministic.
    """
    teachers = list(teacher_rankings)
    pairs: list[tuple[str, str]] = []
    for query_id, text in queries.items():
        channels = [teacher_rankings[name].get(query_id, []) for name in teachers]
        if not any(channels):
            continue
        merged = reciprocal_rank_fusion(channels, k=5, limit=per_query, weights=[1.0] * len(channels))
        for doc_id, _score in merged:
            if doc_id in corpus:
                pairs.append((text, corpus[doc_id]))
    return pairs


def training_pairs(
    queries: dict[str, str],
    qrels: dict[str, dict[str, float]],
    corpus: dict[str, str],
    max_pairs: int,
) -> list[tuple[str, str]]:
    """``(query, positive document)`` pairs from a train split."""
    pairs: list[tuple[str, str]] = []
    for query_id, documents in qrels.items():
        query = queries.get(query_id)
        if not query:
            continue
        for doc_id, score in documents.items():
            if score <= 0 or doc_id not in corpus:
                continue
            pairs.append((query, corpus[doc_id]))
            if max_pairs and len(pairs) >= max_pairs:
                return pairs
    return pairs


def mine_hard_negatives(
    encoder: Any,
    corpus: dict[str, str],
    queries: dict[str, str],
    qrels: dict[str, dict[str, float]],
    limit: int,
    negatives_per_query: int,
    batch_size: int = 64,
) -> list[tuple[str, str, str]]:
    """Triplets ``(query, positive, hard negative)`` mined with the current encoder.

    One corpus embedding pass, then each training query's highest-scoring non-relevant
    documents. Bounded by ``limit`` so the triplet objective stays a useful fraction of
    the training time rather than doubling it.
    """
    doc_ids = list(corpus)
    doc_vectors = np.asarray(
        encoder.encode([corpus[d] for d in doc_ids], normalize_embeddings=True, batch_size=batch_size),
        dtype=np.float32,
    )
    query_ids = [qid for qid in qrels if qid in queries]
    query_vectors = np.asarray(
        encoder.encode([queries[q] for q in query_ids], normalize_embeddings=True, batch_size=batch_size),
        dtype=np.float32,
    )
    triplets: list[tuple[str, str, str]] = []
    for qid, qvec in zip(query_ids, query_vectors):
        relevant = {d for d, s in qrels[qid].items() if s > 0 and d in corpus}
        if not relevant:
            continue
        positive = next(iter(relevant))
        scores = doc_vectors @ qvec
        order = np.argsort(-scores)
        chosen = 0
        for index in order:
            doc_id = doc_ids[int(index)]
            if doc_id in relevant:
                continue
            triplets.append((queries[qid], corpus[positive], corpus[doc_id]))
            chosen += 1
            if chosen >= negatives_per_query:
                break
        if len(triplets) >= limit:
            break
    return triplets


def rank_with_encoder(
    encoder: Any, corpus: dict[str, str], queries: dict[str, str], top_k: int
) -> dict[str, list[str]]:
    """Rank documents with a sentence-transformers encoder (CLS pooling per its config)."""
    doc_ids = list(corpus)
    started = time.perf_counter()
    doc_vectors = encoder.encode(
        [corpus[doc_id] for doc_id in doc_ids], normalize_embeddings=True, batch_size=64
    )
    print(f"  embedded {len(doc_ids)} documents in {time.perf_counter() - started:.1f}s", flush=True)
    index = VectorIndex()
    index.add(beir.make_chunks(corpus), np.asarray(doc_vectors, dtype=np.float32))
    query_vectors = encoder.encode(
        list(queries.values()), normalize_embeddings=True, batch_size=64
    )
    return {
        query_id: [hit.chunk.chunk_id for hit in index.search(vector, k=top_k)]
        for (query_id, _), vector in zip(queries.items(), query_vectors)
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="finetune.py", description="In-domain fine-tuning")
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--data-root", default=str(ROOT / "eval" / "beir"))
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--max-pairs", type=int, default=60000,
                        help="0 means every training pair in the split")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hard-negatives", type=int, default=0,
                        help="mined hard negatives per training query (0 = in-batch only)")
    parser.add_argument("--hard-negative-pairs", type=int, default=40000,
                        help="cap on mined triplets")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--out", default=str(ROOT / "eval" / "finetune"))
    parser.add_argument("--save-model", default=str(ROOT / "eval" / "finetuned-model"))
    parser.add_argument("--resume", default=None, help="checkpoint to continue training from")
    parser.add_argument("--teacher-models", default="",
                        help="comma list of bi-encoders to distil from: their RRF-fused top "
                             "documents become the training positives instead of the qrels")
    parser.add_argument("--teacher-positives", type=int, default=3,
                        help="positive documents per train query taken from the teachers")
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    dataset_dir = beir.ensure_dataset(args.dataset, Path(args.data_root))
    corpus = beir.load_corpus(dataset_dir)
    queries = beir.load_queries(dataset_dir)
    train_qrels = beir.load_qrels(dataset_dir, split="train")
    test_qrels = beir.load_qrels(dataset_dir, split="test")
    test_queries = {qid: text for qid, text in queries.items() if qid in test_qrels}
    all_pairs = training_pairs(queries, train_qrels, corpus, 0)
    train_queries = {qid: text for qid, text in queries.items() if qid in train_qrels}
    print(
        f"{args.dataset}: {len(corpus)} documents, {len(all_pairs)} train pairs, "
        f"{len(test_queries)} test queries",
        flush=True,
    )

    # Optional distillation: mine the teachers' fused top documents as positives. The
    # document embeddings for the shipped teachers are already cached, so this is one
    # query-embedding pass per teacher, not a corpus re-embed.
    teacher_models = [m.strip() for m in (args.teacher_models or "").split(",") if m.strip()]
    teacher_rankings: dict[str, dict[str, list[str]]] = {}
    for model_id in teacher_models:
        teacher = beir.BiEncoder(model_id)
        cache = ROOT / "eval" / "beir" / "cache" / f"{args.dataset}.{teacher.name}"
        vectors = beir.cached_embed(teacher, corpus, cache)
        teacher_rankings[model_id] = beir.dense_rankings(
            corpus, train_queries, teacher, vectors, args.top_k, fixed_length=True
        )
        print(f"  teacher {model_id} ranked {len(train_queries)} train queries", flush=True)

    import torch
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    report: dict[str, Any] = {
        "settings": vars(args),
        "dataset": args.dataset,
        "device": device,
        "available_train_pairs": len(all_pairs),
    }

    # If resuming, skip the zero-shot baseline (it is already in the previous report and
    # re-embedding is the expensive part).
    if args.resume:
        model = SentenceTransformer(args.resume, device=device)
        zero_shot_metrics = None
    else:
        model = SentenceTransformer(args.model, device=device)
        zero_shot = rank_with_encoder(model, corpus, test_queries, args.top_k)
        zero_shot_metrics = evalkit.evaluate_rankings(zero_shot, test_qrels, k_values=(10, 100))["summary"]
        report["zero_shot"] = zero_shot_metrics
        print(f"  zero-shot nDCG@10 {zero_shot_metrics['nDCG@10']:.4f}", flush=True)

    if teacher_rankings:
        pairs = ensemble_positive_pairs(
            corpus, train_queries, teacher_rankings, args.teacher_positives
        )
        print(f"  distilling from {len(teacher_rankings)} teachers -> {len(pairs)} pairs", flush=True)
    else:
        pairs = training_pairs(queries, train_qrels, corpus, args.max_pairs)
    examples = [InputExample(texts=[query, document]) for query, document in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    objectives: list[tuple[Any, Any]] = [(loader, losses.MultipleNegativesRankingLoss(model))]

    triplets: list[tuple[str, str, str]] = []
    if args.hard_negatives > 0:
        started = time.perf_counter()
        triplets = mine_hard_negatives(
            model, corpus, queries, train_qrels,
            limit=args.hard_negative_pairs,
            negatives_per_query=args.hard_negatives,
        )
        print(f"  mined {len(triplets)} hard negatives in {time.perf_counter() - started:.1f}s", flush=True)
        if triplets:
            triplet_examples = [InputExample(texts=[q, p, n]) for q, p, n in triplets]
            triplet_loader = DataLoader(triplet_examples, shuffle=True, batch_size=args.batch_size)
            objectives.append((
                triplet_loader,
                losses.TripletLoss(
                    model,
                    distance_metric=losses.TripletDistanceMetric.COSINE,
                    triplet_margin=0.3,
                ),
            ))
    report["train_pairs"] = len(pairs)
    report["hard_negatives"] = len(triplets)
    report["teacher_models"] = teacher_models

    started = time.perf_counter()
    model.fit(
        train_objectives=objectives,
        epochs=args.epochs,
        warmup_steps=max(1, sum(len(loader) for loader, _ in objectives) // 20),
        show_progress_bar=False,
    )
    report["train_seconds"] = round(time.perf_counter() - started, 1)
    print(f"  trained on {len(pairs)} pairs + {len(triplets)} triplets, "
          f"{args.epochs} epoch(s), in {report['train_seconds']}s", flush=True)

    Path(args.save_model).mkdir(parents=True, exist_ok=True)
    model.save(args.save_model)

    adapted = rank_with_encoder(model, corpus, test_queries, args.top_k)
    report["adapted"] = evalkit.evaluate_rankings(adapted, test_qrels, k_values=(10, 100))["summary"]
    if zero_shot_metrics is not None:
        report["delta_nDCG@10"] = round(report["adapted"]["nDCG@10"] - zero_shot_metrics["nDCG@10"], 4)
        print(
            f"  adapted   nDCG@10 {report['adapted']['nDCG@10']:.4f} "
            f"({report['delta_nDCG@10']:+.4f})",
            flush=True,
        )
    else:
        print(f"  adapted   nDCG@10 {report['adapted']['nDCG@10']:.4f}", flush=True)

    out = Path(args.out)
    evalkit.write_json(out.with_suffix(".json"), report)
    rows = []
    if zero_shot_metrics is not None:
        rows.append({"stage": "zero-shot", **zero_shot_metrics})
    rows.append({"stage": "adapted", **report["adapted"]})
    evalkit.write_text(
        out.with_suffix(".md"),
        f"# In-domain fine-tuning on {args.dataset} (adapted, not zero-shot)\n\n"
        f"{len(pairs)} train pairs, {len(triplets)} mined hard negatives, {args.epochs} epoch(s).\n\n"
        + evalkit.markdown_table(rows)
        + "\n",
    )
    print(f"\nwrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
