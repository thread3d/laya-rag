#!/usr/bin/env python
"""Domain-adapted fine-tuning (item 7): train a bi-encoder on in-domain pairs.

Everything else in this project is zero-shot. This intentionally is not: it fine-tunes
`bge-small` on a dataset's **train** split and evaluates on its **test** split, so the
number it produces must be labelled adapted, never compared as if it were zero-shot.

NFCorpus ships 110k train pairs and is the standard demonstration for this, so:

    .venv/bin/python tools/finetune.py --dataset nfcorpus --max-pairs 60000

Uses sentence-transformers' `MultipleNegativesRankingLoss`, which treats the other
positives in a batch as negatives and needs no explicit negative mining.
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
sys.path.insert(0, str(ROOT / "src"))

import beir  # noqa: E402
import evalkit  # noqa: E402
from laya_rag.index_store import VectorIndex  # noqa: E402


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
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


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
    parser.add_argument("--max-pairs", type=int, default=60000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--out", default=str(ROOT / "eval" / "finetune"))
    parser.add_argument("--save-model", default=str(ROOT / "eval" / "finetuned-model"))
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_dir = beir.ensure_dataset(args.dataset, Path(args.data_root))
    corpus = beir.load_corpus(dataset_dir)
    queries = beir.load_queries(dataset_dir)
    train_qrels = beir.load_qrels(dataset_dir, split="train")
    test_qrels = beir.load_qrels(dataset_dir, split="test")
    test_queries = {qid: text for qid, text in queries.items() if qid in test_qrels}
    print(
        f"{args.dataset}: {len(corpus)} documents, {len(training_pairs(queries, train_qrels, corpus, 10**9))} "
        f"train pairs, {len(test_queries)} test queries",
        flush=True,
    )

    import torch
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    report: dict[str, Any] = {"settings": vars(args), "dataset": args.dataset, "device": device}

    # Zero-shot baseline with the same encoder and evaluator, so the delta is the training.
    zero_shot_model = SentenceTransformer(args.model, device=device)
    zero_shot = rank_with_encoder(zero_shot_model, corpus, test_queries, args.top_k)
    report["zero_shot"] = evalkit.evaluate_rankings(zero_shot, test_qrels, k_values=(10, 100))["summary"]
    print(f"  zero-shot nDCG@10 {report['zero_shot']['nDCG@10']:.4f}", flush=True)

    pairs = training_pairs(queries, train_qrels, corpus, args.max_pairs)
    examples = [InputExample(texts=[query, document]) for query, document in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    loss = losses.MultipleNegativesRankingLoss(zero_shot_model)
    started = time.perf_counter()
    zero_shot_model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=max(1, len(loader) // 20),
        show_progress_bar=False,
    )
    report["train_seconds"] = round(time.perf_counter() - started, 1)
    report["train_pairs"] = len(pairs)
    print(f"  trained on {len(pairs)} pairs in {report['train_seconds']}s", flush=True)

    Path(args.save_model).mkdir(parents=True, exist_ok=True)
    zero_shot_model.save(args.save_model)

    adapted = rank_with_encoder(zero_shot_model, corpus, test_queries, args.top_k)
    report["adapted"] = evalkit.evaluate_rankings(adapted, test_qrels, k_values=(10, 100))["summary"]
    report["delta_nDCG@10"] = round(report["adapted"]["nDCG@10"] - report["zero_shot"]["nDCG@10"], 4)
    print(
        f"  adapted   nDCG@10 {report['adapted']['nDCG@10']:.4f} "
        f"({report['delta_nDCG@10']:+.4f})",
        flush=True,
    )

    out = Path(args.out)
    evalkit.write_json(out.with_suffix(".json"), report)
    evalkit.write_text(
        out.with_suffix(".md"),
        f"# In-domain fine-tuning on {args.dataset} (adapted, not zero-shot)\n\n"
        + evalkit.markdown_table([
            {"stage": "zero-shot", **report["zero_shot"]},
            {"stage": "adapted", **report["adapted"]},
        ])
        + "\n",
    )
    print(f"\nwrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
