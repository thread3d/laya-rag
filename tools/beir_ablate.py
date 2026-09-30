#!/usr/bin/env python
"""Ablations that search for a better retrieval configuration, measured on BEIR.

The BEIR run showed the shipped hybrid recall was ~43% below plain BM25, and the first
round of ablations turned the dense channel off and moved the rerank order to recall.
This round looks for gains on top of that, following what the BAAI model card and normal
retrieval practice recommend:

* **pooling** — BGE models are served with the ``[CLS]`` token's hidden state, not mean
  pooling, so the shipped bi-encoder was mismatched to its own model;
* **query instruction** — optional for bge-*-v1.5; measured both ways;
* **cross-encoder rerank** — BAAI's own recommended second stage: retrieve wide, then
  re-rank the top candidates with a model that attends over query and passage together;
* **fusion weight** — re-checked now that the components changed.

Protocol: tune on SciFact, confirm the chosen configuration on NFCorpus.

    .venv/bin/python tools/beir_ablate.py --datasets scifact
    .venv/bin/python tools/beir_ablate.py --datasets nfcorpus --include-laya-rerank
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import beir  # noqa: E402  (also sets HF_HOME inside the workspace)
import evalkit  # noqa: E402
from laya_rag.config import EngineConfig  # noqa: E402
from laya_rag.lexical import reciprocal_rank_fusion  # noqa: E402

DENSE_WEIGHTS = (0.0, 0.25, 0.5, 1.0)


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


def evaluate(rankings: Mapping[str, Sequence[str]], qrels: Mapping[str, Any]) -> dict[str, float]:
    return evalkit.evaluate_rankings(rankings, qrels, k_values=(10, 100))["summary"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="beir_ablate.py", description="Search for a better retrieval config")
    parser.add_argument("--datasets", default="scifact")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--data-root", default=str(ROOT / "eval" / "beir"))
    parser.add_argument("--out", default=str(ROOT / "eval" / "beir" / "ablation"))
    parser.add_argument("--biencoder-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--ce-depths", default="50", help="comma list of cross-encoder rerank depths")
    parser.add_argument("--dense-weights", default=",".join(str(w) for w in DENSE_WEIGHTS))
    parser.add_argument("--include-laya-rerank", action="store_true",
                        help="also time Laya's decision head as a reranker (slow)")
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root)
    config = EngineConfig()
    if args.device:
        config.device = args.device
    ce_depths = [int(value) for value in args.ce_depths.split(",") if value.strip()]
    dense_weights = [float(value) for value in args.dense_weights.split(",") if value.strip()]

    report: dict[str, Any] = {"settings": vars(args), "datasets": {}}
    for name in [value.strip() for value in args.datasets.split(",") if value.strip()]:
        print(f"[{name}]", flush=True)
        dataset_dir = beir.ensure_dataset(name, data_root)
        corpus = beir.load_corpus(dataset_dir)
        all_queries = beir.load_queries(dataset_dir)
        qrels = beir.load_qrels(dataset_dir)
        queries = {qid: text for qid, text in all_queries.items() if qid in qrels}
        cache_dir = data_root / "cache"

        # BM25: the published settings and the engine's stopword behaviour.
        bm25 = beir.bm25_rankings(corpus, queries, args.top_k)
        variants: dict[str, dict[str, list[str]]] = {
            "bm25 (published settings)": bm25,
            "bm25 (engine stopword drop)": beir.bm25_rankings(
                corpus, queries, args.top_k, drop_stopwords=True
            ),
        }
        timings: dict[str, float] = {}

        # Bi-encoder: pooling and query-instruction variants. Documents are cached per
        # (model, pooling) because the instruction is query-side only.
        dense_best: dict[str, list[str]] | None = None
        for pooling in ("cls", "mean"):
            for instruction in (True, False):
                if pooling == "mean" and not instruction:
                    continue  # mean pooling has no trained instruction convention
                encoder = beir.BiEncoder(
                    args.biencoder_model,
                    device=args.device,
                    pooling=pooling,
                    query_instruction=instruction,
                )
                vectors = beir.cached_embed(
                    encoder, corpus, cache_dir / f"{name}.{encoder.name}"
                )
                started = time.perf_counter()
                dense = beir.dense_rankings(corpus, queries, encoder, vectors, args.top_k)
                label = f"dense {pooling}{'+instr' if instruction else '+noinstr'}"
                variants[label] = dense
                timings[label] = round(time.perf_counter() - started, 1)
                if pooling == "cls" and instruction:
                    dense_best = dense
                del encoder

        assert dense_best is not None
        for weight in dense_weights:
            variants[f"rrf dense w={weight}"] = fuse(dense_best, bm25, weight, 1.0, args.top_k)
        rrf_one = variants["rrf dense w=1.0"]

        # Cross-encoder: rerank BM25, the bi-encoder, and the fusion.
        if ce_depths:
            cross_encoder = beir.CrossEncoder(args.rerank_model, device=args.device)
            for depth in ce_depths:
                for base_label, base in (("bm25", bm25), ("dense", dense_best), ("rrf", rrf_one)):
                    reranked, seconds = beir.crossencoder_rankings(
                        corpus, queries, base, cross_encoder, depth, args.top_k
                    )
                    label = f"ce {base_label} depth={depth}"
                    variants[label] = reranked
                    timings[label] = round(seconds, 1)
            del cross_encoder

        if args.include_laya_rerank:
            from laya_rag.rerank import LayaReranker

            laya_embedder = beir.LayaEmbedder(config)
            reranker = LayaReranker(laya_embedder._model)
            for depth in ce_depths or [20]:
                reranked, seconds = _laya_rerank(corpus, queries, bm25, reranker, depth, args.top_k)
                label = f"laya rerank bm25 depth={depth}"
                variants[label] = reranked
                timings[label] = round(seconds, 1)
            del laya_embedder, reranker

        results: dict[str, Any] = {}
        baseline = evaluate(bm25, qrels)["nDCG@10"]
        for label, ranking in variants.items():
            summary = evaluate(ranking, qrels)
            summary["delta_vs_bm25"] = round(summary["nDCG@10"] - baseline, 4)
            if label in timings:
                summary["seconds"] = timings[label]
            results[label] = summary
            print(
                f"  {label:<32} nDCG@10 {summary['nDCG@10']:.4f} "
                f"({summary['delta_vs_bm25']:+.4f})  MRR@10 {summary['MRR@10']:.4f}",
                flush=True,
            )
        report["datasets"][name] = {
            "documents": len(corpus), "queries": len(queries), "variants": results
        }

    out = Path(args.out)
    evalkit.write_json(out.with_suffix(".json"), report)
    evalkit.write_text(out.with_suffix(".md"), render(report))
    print(f"\nwrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


def _laya_rerank(
    corpus: Mapping[str, str],
    queries: Mapping[str, str],
    base: Mapping[str, Sequence[str]],
    reranker: Any,
    depth: int,
    top_k: int,
) -> tuple[dict[str, list[str]], float]:
    spans: list[tuple[str, list[str]]] = []
    pairs: list[tuple[str, str]] = []
    for query_id, text in queries.items():
        docs = list(base.get(query_id, []))[:depth]
        spans.append((query_id, docs))
        pairs.extend((text, corpus[doc_id]) for doc_id in docs)
    started = time.perf_counter()
    scores, _ = reranker.score_relevance_pairs(pairs, batch_size=16)
    elapsed = time.perf_counter() - started
    out: dict[str, list[str]] = {}
    cursor = 0
    for query_id, docs in spans:
        block = scores[cursor : cursor + len(docs)]
        cursor += len(docs)
        order = sorted(range(len(docs)), key=lambda index: -block[index])
        out[query_id] = [docs[index] for index in order[:top_k]]
    return out, elapsed


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
