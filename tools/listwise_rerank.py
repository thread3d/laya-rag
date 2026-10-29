#!/usr/bin/env python
"""The listwise arm R5 named: `jina-reranker-v3` over the same ArguAna pool as the pointwise arms.

This exists because R5 was blocked on an interface, not on a measurement. The pointwise models the rest of
this document measured score one `(query, document)` pair at a time, so they cannot compare candidates; a
listwise model scores the **whole list in one pass**, which is the interface ArguAna's opposition relation
would need. `jinaai/jina-reranker-v3` is 596M — the same size class as `bge-reranker-v2-m3` — and its
`rerank(query=…, documents=[…])` is that interface.

It reuses `tools/beir.py`'s own first stage (`channel_score_maps` + `hybrid_rankings`) rather than
re-implementing fusion, so the pool a listwise model sees is the pool the pointwise arms saw. The prediction
and both falsifiers are registered in `docs/findings-reranker-generation.md` before this runs.

**The licence bounds the conclusion, not the measurement.** The model is CC-BY-NC-4.0: a research
instrument, not something this engine could ship. That is stated wherever the number is quoted.

    python tools/listwise_rerank.py --limit-queries 400 --query-seed 17 --out eval/r1/arguana-listwise
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("HF_HOME", str(ROOT / "eval" / "r1" / "hf-cache"))

MODEL = "jinaai/jina-reranker-v3"


def first_stage(corpus, queries, *, top_k: int, depth: int) -> dict[str, list[str]]:
    """Build the pool exactly as `tools/beir.py` builds its `hybrid-biencoder` row.

    Reusing the harness's own functions rather than re-implementing fusion is the whole point: a listwise
    model that reranked a *different* pool would answer a different question. `bge-small` is the encoder
    every pointwise arm in this document used.
    """
    import beir

    encoder = beir.BiEncoder(model_id="BAAI/bge-small-en-v1.5")
    # Cache beside the run, not in `/tmp`: a corpus embedding is the long part of an evaluation, `/tmp` is
    # not durable here, and a re-run that re-embedded ArguAna to prove a number was already measured would
    # be paying twice for the same vectors.
    cache = ROOT / "eval" / "r1" / "listwise-dense-cache"
    cache.parent.mkdir(parents=True, exist_ok=True)
    vectors = beir.cached_embed(encoder, corpus, cache, use_cache=True)
    dense = beir.dense_rankings(corpus, queries, encoder, vectors, top_k)
    lexical = beir.bm25_rankings(corpus, queries, top_k, k1=beir.BEIR_K1, b=beir.BEIR_B)
    fused = beir.hybrid_rankings(dense, lexical, top_k)
    return {qid: ids[:depth] for qid, ids in fused.items()}


def score_listwise(model, tokenizer, queries, corpus_text, pools, top_k: int) -> dict[str, list[str]]:
    """One listwise pass per query; the model sees every candidate at once, which is the whole point."""
    ranked: dict[str, list[str]] = {}
    for query_id, doc_ids in pools.items():
        documents = [corpus_text[doc_id] for doc_id in doc_ids]
        result = model.rerank(query=queries[query_id], documents=documents, top_n=len(documents))
        order = [doc_ids[int(item["index"])] for item in result]
        ranked[query_id] = order[:top_k]
    return ranked


def main(argv: Sequence[str] | None = None) -> int:
    from heavylock import heavy_job_guard

    ap = argparse.ArgumentParser(prog="listwise_rerank.py", description=__doc__)
    ap.add_argument("--dataset", default="arguana")
    ap.add_argument("--data-root", default=str(ROOT / "eval" / "r1" / "data"))
    ap.add_argument("--out", default=str(ROOT / "eval" / "r1" / "arguana-listwise"))
    ap.add_argument("--limit-queries", type=int, default=400)
    ap.add_argument("--query-seed", type=int, default=17)
    ap.add_argument("--depth", type=int, default=50, help="candidates the listwise pass sees, per query")
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args(argv)
    heavy_job_guard("listwise_rerank.py")

    import beir

    dataset_dir = beir.ensure_dataset(args.dataset, Path(args.data_root), download=False)
    corpus = beir.load_corpus(dataset_dir, include_titles=True)
    queries = beir.load_queries(dataset_dir)
    qrels = beir.load_qrels(dataset_dir)
    query_ids = sorted(queries)
    random.Random(args.query_seed).shuffle(query_ids)
    query_ids = query_ids[: args.limit_queries]
    queries = {qid: queries[qid] for qid in query_ids}
    # The harness's own rule: score only the queries that were sampled, or every unsampled query counts
    # as a zero and the summary becomes a fraction of the sampler's denominator. The 5-query smoke test
    # reported nDCG 0.0028 over 1,406 queries before this line existed.
    qrels = beir.restrict_qrels(qrels, queries)

    import torch
    from transformers import AutoModel, AutoTokenizer

    pools = first_stage(corpus, queries, top_k=args.top_k, depth=args.depth)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.float32).eval()

    started = time.time()
    ranked = score_listwise(model, tokenizer, queries, corpus, pools, args.top_k)
    seconds = time.time() - started

    import evalkit

    evaluation = evalkit.evaluate_rankings(ranked, qrels, k_values=(10, 100))
    report = {
        "dataset": args.dataset,
        "model": MODEL,
        "licence": "cc-by-nc-4.0",
        "interface": "listwise",
        "depth": args.depth,
        "queries": len(query_ids),
        "query_seed": args.query_seed,
        "seconds": round(seconds, 2),
        "summary": evaluation["summary"],
        "per_query_nDCG@10": evalkit.per_query_metric(evaluation["per_query"], "nDCG@10"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_query"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
