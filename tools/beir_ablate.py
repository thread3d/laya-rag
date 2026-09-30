#!/usr/bin/env python
"""Ablations that search for a better retrieval configuration, measured on BEIR.

The BEIR run showed the shipped hybrid recall is ~43% below plain BM25 because
equal-weight RRF lets a near-chance dense channel dilute a strong lexical one. This
script measures the candidate fixes over the same corpora and qrels, so the choice is
made on evidence rather than intuition:

* **fusion weight** — how much the dense channel should contribute, including none;
* **Laya decision-head rerank** — whether the calibrated cross-encoder recovers what the
  mean-pooled encoder loses, reranking BM25 or fused candidates;
* **BM25 query stopwords** — the engine drops them, the published baseline does not.

Protocol: tune on SciFact, confirm the chosen configuration on NFCorpus with
``--datasets nfcorpus``. Both are the official BEIR test qrels; nothing here tunes on the
confirmation set.

    .venv/bin/python tools/beir_ablate.py --datasets scifact
    .venv/bin/python tools/beir_ablate.py --datasets nfcorpus --depths 20,50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import beir  # noqa: E402  (also sets HF_HOME inside the workspace)
import evalkit  # noqa: E402
from laya_rag.config import EngineConfig  # noqa: E402
from laya_rag.lexical import reciprocal_rank_fusion  # noqa: E402
from laya_rag.rerank import LayaReranker  # noqa: E402

DENSE_WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)
BIENCODER_WEIGHTS = (0.5, 0.75, 1.0, 1.5)


def fuse(
    dense: Mapping[str, Sequence[str]],
    lexical: Mapping[str, Sequence[str]],
    dense_weight: float,
    lexical_weight: float,
    top_k: int,
) -> dict[str, list[str]]:
    """RRF of two channels at the given weights, the same call the engine makes."""
    fused: dict[str, list[str]] = {}
    for query_id, dense_ids in dense.items():
        merged = reciprocal_rank_fusion(
            [list(dense_ids)[:top_k], list(lexical.get(query_id, []))[:top_k]],
            limit=top_k,
            weights=[dense_weight, lexical_weight],
        )
        fused[query_id] = [doc_id for doc_id, _ in merged]
    return fused


def laya_rerank(
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    base: Mapping[str, Sequence[str]],
    reranker: LayaReranker,
    depth: int,
    top_k: int,
    batch_size: int = 16,
) -> tuple[dict[str, list[str]], float]:
    """Rerank each query's top ``depth`` with Laya's calibrated relevance decision.

    Every ``(query, candidate)`` pair for every query is scored in one batched pass,
    which is how the engine's own ``score_relevance_pairs`` avoids a per-query call.
    """
    spans: list[tuple[str, list[str]]] = []
    pairs: list[tuple[str, str]] = []
    for query_id, text in queries.items():
        docs = [doc_id for doc_id in base.get(query_id, [])[:depth]]
        spans.append((query_id, docs))
        pairs.extend((text, corpus[doc_id]) for doc_id in docs)

    started = time.perf_counter()
    scores, _ = reranker.score_relevance_pairs(pairs, batch_size=batch_size)
    elapsed = time.perf_counter() - started

    reranked: dict[str, list[str]] = {}
    cursor = 0
    for query_id, docs in spans:
        block = scores[cursor : cursor + len(docs)]
        cursor += len(docs)
        order = sorted(range(len(docs)), key=lambda index: -block[index])
        reranked[query_id] = [docs[index] for index in order[:top_k]]
    return reranked, elapsed


def evaluate(rankings: Mapping[str, Sequence[str]], qrels: Mapping[str, Any]) -> dict[str, float]:
    return evalkit.evaluate_rankings(rankings, qrels, k_values=(10, 100))["summary"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="beir_ablate.py", description="Search for a better retrieval config")
    parser.add_argument("--datasets", default="scifact")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--depths", default="20", help="comma list of rerank depths to test")
    parser.add_argument("--data-root", default=str(ROOT / "eval" / "beir"))
    parser.add_argument("--out", default=str(ROOT / "eval" / "beir" / "ablation"))
    parser.add_argument("--biencoder-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--skip-biencoder", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root)
    config = EngineConfig()
    if args.device:
        config.device = args.device
    depths = [int(value) for value in args.depths.split(",") if value.strip()]

    report: dict[str, Any] = {"datasets": {}}
    for name in [value.strip() for value in args.datasets.split(",") if value.strip()]:
        print(f"[{name}]", flush=True)
        dataset_dir = beir.ensure_dataset(name, data_root)
        corpus = beir.load_corpus(dataset_dir)
        all_queries = beir.load_queries(dataset_dir)
        qrels = beir.load_qrels(dataset_dir)
        queries = {qid: text for qid, text in all_queries.items() if qid in qrels}

        # BM25 at the published settings (the control) and with the engine's stopword drop.
        bm25 = beir.bm25_rankings(corpus, queries, args.top_k)
        bm25_engine = beir.bm25_rankings(corpus, queries, args.top_k, drop_stopwords=True)

        laya_embedder = beir.LayaEmbedder(config)
        dense = beir.dense_rankings(
            corpus,
            queries,
            laya_embedder,
            beir.cached_embed(laya_embedder, corpus, data_root / "cache" / f"{name}.{laya_embedder.name}"),
            args.top_k,
        )

        biencoder_rankings: dict[str, list[str]] | None = None
        if not args.skip_biencoder:
            try:
                biencoder = beir.BiEncoder(args.biencoder_model, device=args.device)
                biencoder_rankings = beir.dense_rankings(
                    corpus,
                    queries,
                    biencoder,
                    beir.cached_embed(biencoder, corpus, data_root / "cache" / f"{name}.{biencoder.name}"),
                    args.top_k,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  bi-encoder unavailable: {type(exc).__name__}: {exc}", flush=True)

        variants: dict[str, dict[str, list[str]]] = {
            "bm25 (published settings)": bm25,
            "bm25 (engine stopword drop)": bm25_engine,
        }
        for weight in DENSE_WEIGHTS:
            variants[f"rrf dense w={weight}"] = fuse(dense, bm25, weight, 1.0, args.top_k)
        if biencoder_rankings is not None:
            for weight in BIENCODER_WEIGHTS:
                variants[f"rrf bi-encoder w={weight}"] = fuse(
                    biencoder_rankings, bm25, weight, 1.0, args.top_k
                )

        reranker = LayaReranker(laya_embedder._model) if not args.skip_rerank else None
        timings: dict[str, float] = {}
        if reranker is not None:
            for depth in depths:
                reranked, seconds = laya_rerank(corpus, queries, bm25, reranker, depth, args.top_k)
                variants[f"laya rerank bm25 depth={depth}"] = reranked
                timings[f"laya rerank bm25 depth={depth}"] = round(seconds, 1)
                if biencoder_rankings is not None:
                    reranked_bi, seconds_bi = laya_rerank(
                        corpus, queries, biencoder_rankings, reranker, depth, args.top_k
                    )
                    variants[f"laya rerank bi-encoder depth={depth}"] = reranked_bi
                    timings[f"laya rerank bi-encoder depth={depth}"] = round(seconds_bi, 1)

        results: dict[str, Any] = {}
        baseline = evaluate(bm25, qrels)["nDCG@10"]
        for label, ranking in variants.items():
            summary = evaluate(ranking, qrels)
            summary["delta_vs_bm25"] = round(summary["nDCG@10"] - baseline, 4)
            if label in timings:
                summary["rerank_seconds"] = timings[label]
            results[label] = summary
            print(
                f"  {label:<34} nDCG@10 {summary['nDCG@10']:.4f} "
                f"({summary['delta_vs_bm25']:+.4f})  MRR@10 {summary['MRR@10']:.4f}",
                flush=True,
            )
        report["datasets"][name] = {"documents": len(corpus), "queries": len(queries), "variants": results}

    out = Path(args.out)
    evalkit.write_json(out.with_suffix(".json"), report)
    evalkit.write_text(out.with_suffix(".md"), render(report))
    print(f"\nwrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


def render(report: dict[str, Any]) -> str:
    parts = ["# Retrieval configuration ablations", ""]
    for name, dataset in report["datasets"].items():
        rows = [
            {"variant": label, **summary}
            for label, summary in sorted(
                dataset["variants"].items(), key=lambda item: -item[1]["nDCG@10"]
            )
        ]
        parts += [
            f"## {name} ({dataset['documents']} documents, {dataset['queries']} queries)",
            "",
            evalkit.markdown_table(rows),
            "",
        ]
    return "\n".join(parts)


if __name__ == "__main__":
    sys.exit(main())
