"""Evaluate the engine itself on a BEIR dataset (not the ``tools/beir.py`` replica).

Every BEIR number in this project comes from `tools/beir.py`, which reimplements the
engine's channels rather than calling the engine. That is exactly how the engine fused at
`k=60` while the harness used `k=5` for several rounds. This runs the real `RagEngine` over a
dataset's test queries, with the decision thresholds set to zero so nothing is filtered and
the returned order is the retrieval order, and scores nDCG@10.

    # two runs, to isolate the sparse channel's contribution at the engine level
    .venv/bin/python tools/engine_beir.py --dataset scifact --tag base
    .venv/bin/python tools/engine_beir.py --dataset scifact --tag sparse --sparse
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import beir, evalkit  # noqa: E402

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_SPARSE = "opensearch-project/opensearch-neural-sparse-encoding-v1"


def write_corpus(corpus: Mapping[str, str], docs_dir: Path) -> list[str]:
    """One ``.txt`` per document, named by position; returns the id order."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    doc_ids = list(corpus)
    for index, doc_id in enumerate(doc_ids):
        (docs_dir / f"{index:06d}.txt").write_text(corpus[doc_id], encoding="utf-8")
    return doc_ids


def ranking_from_passages(passages: Sequence[Mapping[str, Any]], doc_ids: Sequence[str]) -> list[str]:
    """Map returned passage paths back to document ids, de-duplicated, order preserved."""
    ranking: list[str] = []
    seen: set[str] = set()
    for passage in passages:
        path = str(passage.get("path") or "")
        stem = Path(path).stem
        if not stem.isdigit():
            continue
        index = int(stem)
        if not (0 <= index < len(doc_ids)):
            continue
        doc_id = doc_ids[index]
        if doc_id not in seen:
            seen.add(doc_id)
            ranking.append(doc_id)
    return ranking


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="engine_beir.py")
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--data-root", default=str(ROOT / "eval" / "beir"))
    ap.add_argument("--embed-model", default=DEFAULT_MODEL)
    ap.add_argument("--sparse", action="store_true", help="also index the sparse channel")
    ap.add_argument("--sparse-model", default=DEFAULT_SPARSE)
    ap.add_argument("--tag", default="base", help="names the index directory and outputs")
    ap.add_argument("--min-docs", type=int, default=0,
                    help="promote the best passage of this many distinct documents first")
    ap.add_argument("--rerank-model", default=None,
                    help="cross-encoder used as the second stage")
    ap.add_argument("--rerank-coverage-lambda", type=float, default=None,
                    help="explicit coverage weight; omit to use the engine's auto default")
    ap.add_argument("--k1", type=float, default=None, help="BM25 k1 (harness baseline 0.9)")
    ap.add_argument("--b", type=float, default=None, dest="b_bm25",
                    help="BM25 b (harness baseline 0.4)")
    ap.add_argument("--chunk-chars", type=int, default=None,
                    help="chunk size; the default 1200 splits short abstracts the replica "
                         "embeds whole")
    ap.add_argument("--chunk-overlap", type=int, default=None)
    ap.add_argument("--min-chunk-chars", type=int, default=None)
    ap.add_argument("--chunk-merge-short", action="store_true",
                    help="keep documents of at most chunk_chars+overlap as one chunk")
    ap.add_argument("--retrieve-k", type=int, default=None,
                    help="candidates considered before rerank; chunking packs several "
                         "chunks per document into this budget")
    ap.add_argument("--context-prefix", action="store_true",
                    help="index a structural breadcrumb (path + title line) on every chunk, "
                         "so later chunks keep the document's context")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="cap the number of test queries")
    ap.add_argument("--out", default=str(ROOT / "eval" / "engine-beir"))
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    index_dir = out / "index" / f"{args.dataset}-{args.tag}"
    docs_dir = out / "docs" / args.dataset
    os.environ["LAYA_RAG_INDEX_DIR"] = str(index_dir)
    os.environ["LAYA_RAG_EMBED_MODEL"] = args.embed_model
    # Nothing may be filtered, so the returned order is the engine's retrieval order.
    os.environ["LAYA_RAG_RELEVANCE_THRESHOLD"] = "0.0"
    os.environ["LAYA_RAG_ANSWERABILITY_THRESHOLD"] = "0.0"
    if args.min_docs:
        os.environ["LAYA_RAG_RERANK_MIN_DOCS"] = str(args.min_docs)
    else:
        os.environ.pop("LAYA_RAG_RERANK_MIN_DOCS", None)
    if args.rerank_model:
        os.environ["LAYA_RAG_RERANK_MODEL"] = args.rerank_model
    else:
        os.environ.pop("LAYA_RAG_RERANK_MODEL", None)
    if args.rerank_coverage_lambda is not None:
        os.environ["LAYA_RAG_RERANK_COVERAGE_LAMBDA"] = str(args.rerank_coverage_lambda)
    else:
        os.environ.pop("LAYA_RAG_RERANK_COVERAGE_LAMBDA", None)
    if args.k1 is not None:
        os.environ["LAYA_RAG_BM25_K1"] = str(args.k1)
    else:
        os.environ.pop("LAYA_RAG_BM25_K1", None)
    if args.b_bm25 is not None:
        os.environ["LAYA_RAG_BM25_B"] = str(args.b_bm25)
    else:
        os.environ.pop("LAYA_RAG_BM25_B", None)
    for name, value in (
        ("CHUNK_CHARS", args.chunk_chars),
        ("CHUNK_OVERLAP_CHARS", args.chunk_overlap),
        ("MIN_CHUNK_CHARS", args.min_chunk_chars),
    ):
        if value is not None:
            os.environ[f"LAYA_RAG_{name}"] = str(value)
        else:
            os.environ.pop(f"LAYA_RAG_{name}", None)
    os.environ["LAYA_RAG_CHUNK_MERGE_SHORT"] = "1" if args.chunk_merge_short else "0"
    if args.retrieve_k is not None:
        os.environ["LAYA_RAG_RETRIEVE_K"] = str(args.retrieve_k)
    else:
        os.environ.pop("LAYA_RAG_RETRIEVE_K", None)
    os.environ["LAYA_RAG_CONTEXT_PREFIX"] = "1" if args.context_prefix else "0"
    if args.sparse:
        os.environ["LAYA_RAG_SPARSE_MODEL"] = args.sparse_model
    else:
        os.environ.pop("LAYA_RAG_SPARSE_MODEL", None)

    from laya_rag.config import EngineConfig
    from laya_rag.engine import RagEngine

    dataset_dir = beir.ensure_dataset(args.dataset, Path(args.data_root), download=False)
    corpus = beir.load_corpus(dataset_dir, include_titles=True)
    queries = beir.load_queries(dataset_dir)
    qrels = beir.load_qrels(dataset_dir, split="test")
    test_queries = {qid: text for qid, text in queries.items() if qid in qrels}
    if args.limit:
        test_queries = dict(list(test_queries.items())[: args.limit])
    doc_ids = write_corpus(corpus, docs_dir)

    config = EngineConfig()
    engine = RagEngine(config)
    if not engine.has_index():
        started = time.perf_counter()
        report = engine.index_paths([docs_dir], rebuild=True)
        print(f"indexed {report.documents} docs / {report.chunks} chunks in "
              f"{time.perf_counter() - started:.0f}s", flush=True)
    print(f"sparse configured: {config.sparse_enabled}", flush=True)

    rankings: dict[str, list[str]] = {}
    started = time.perf_counter()
    for position, (query_id, text) in enumerate(test_queries.items(), start=1):
        result = engine.query(text, top_k=args.top_k)
        rankings[query_id] = ranking_from_passages(result.passages, doc_ids)
        if position % 50 == 0:
            print(f"  {position}/{len(test_queries)} in {time.perf_counter() - started:.0f}s", flush=True)
    seconds = time.perf_counter() - started

    evaluation = evalkit.evaluate_rankings(rankings, qrels, k_values=(10,), ignore_identical_ids=True)
    payload = {
        "_settings": vars(args),
        "documents": len(corpus),
        "queries": len(test_queries),
        "sparse": config.sparse_enabled,
        "seconds": round(seconds, 1),
        "summary": evaluation["summary"],
        "per_query": evalkit.per_query_metric(evaluation["per_query"], "nDCG@10"),
    }
    evalkit.write_json(out / f"{args.dataset}-{args.tag}.json", payload)
    print(f"{args.dataset} {args.tag}: nDCG@10 {evaluation['summary']['nDCG@10']:.4f} "
          f"over {len(test_queries)} queries in {seconds:.0f}s")
    print("wrote", out / f"{args.dataset}-{args.tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
