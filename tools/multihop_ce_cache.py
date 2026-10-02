"""Cache cross-encoder scores for the deterministic MultiHop-RAG subset.

Every second-stage experiment on MultiHop-RAG — blending, a rank-1 guardrail,
CE-plus-coverage selection, per-query routing — needs the same expensive thing: the
cross-encoder's score for each candidate in the fused pool. Scoring it once and caching
it makes all of those experiments cheap and exactly reproducible, instead of re-running a
335M-parameter model per idea.

The subset is the one `tools/multihop_official.py --limit-queries 400 --sample-seed 17`
uses, so its cross-encoder row (MAP@10 0.2986) is the bar.

    .venv/bin/python tools/multihop_ce_cache.py --limit-queries 400

Writes ``<out>.json`` (gitignored) with, per query: the fused pool ids in order, the
cross-encoder score for each, the atom ids each candidate covers, the global atom
weights, and the gold facts. Nothing here is a metric; see
``tools/multihop_rerank_experiment.py`` for those.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import beir  # noqa: E402
from multihop_coverage import atom_index, split_atoms  # noqa: E402
from multihop_official import chunk_text  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"
MODELS = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="multihop_ce_cache.py")
    ap.add_argument("--model", default="bge-small")
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--depth", type=int, default=100, help="candidates scored per query")
    ap.add_argument("--rrf-k", type=int, default=5)
    ap.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit-queries", type=int, default=400)
    ap.add_argument("--sample-seed", type=int, default=17)
    ap.add_argument("--out", default=str(ROOT_BENCH / "ce-scores"))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    model_id = MODELS.get(args.model, args.model)

    raw = json.load(open(ROOT_BENCH / "corpus.json"))
    items_all = json.load(open(ROOT_BENCH / "MultiHopRAG.json"))
    ev_items = [(i, x) for i, x in enumerate(items_all) if x["question_type"] != "null_query"]
    total = len(ev_items)
    if args.limit_queries and args.limit_queries < total:
        sampled = sorted(
            random.Random(args.sample_seed).sample(range(total), args.limit_queries)
        )
        ev_items = [ev_items[i] for i in sampled]
    golds = [[e["fact"] for e in x["evidence_list"]] for _, x in ev_items]

    corpus: dict[str, str] = {}
    titles: dict[str, str] = {}
    for a in raw:
        for j, c in enumerate(chunk_text(a["title"] + " " + a["body"], 256, args.overlap)):
            cid = f"{a['url']}#{j}"
            corpus[cid] = c
            titles[cid] = a["title"]
    queries = {str(i): x["query"] for i, x in ev_items}
    print(f"[overlap={args.overlap}] {len(corpus)} chunks, {len(queries)} queries", flush=True)

    enc = beir.BiEncoder(model_id)
    vectors = beir.cached_embed(
        enc, corpus, ROOT_BENCH / "cache" / f"multihop-chunks{args.overlap}.{enc.name}"
    )
    print("  lexical ranking...", flush=True)
    lexical = beir.bm25_rankings(
        corpus, queries, args.depth, stem=True, drop_stopwords=True,
        titles=titles, title_boost=2,
    )
    print("  dense ranking...", flush=True)
    dense = beir.dense_rankings(corpus, queries, enc, vectors, args.depth, fixed_length=True)
    fused = beir.multi_channel_fusion([dense, lexical], [1.0, 1.0], args.depth, rrf_k=args.rrf_k)

    # Atoms are query-independent, so build them once over the whole corpus.
    print("  atomising...", flush=True)
    atoms_by_doc = {doc_id: split_atoms(text) for doc_id, text in corpus.items()}
    atom_ids_of, atom_weight, _ = atom_index(list(atoms_by_doc.values()))
    atom_ids_by_doc = dict(zip(atoms_by_doc, atom_ids_of))

    cross_encoder = beir.CrossEncoder(args.rerank_model, batch_size=args.batch_size)
    print(f"  scoring with {args.rerank_model}...", flush=True)
    started = time.perf_counter()
    per_query: dict[str, dict] = {}
    for n, (q, text) in enumerate(queries.items()):
        pool = list(fused[q])[: args.depth]
        pairs = [(text, corpus[doc_id]) for doc_id in pool]
        scores = cross_encoder.score(pairs, fixed_length=True)
        per_query[q] = {
            "pool": pool,
            "ce": [round(float(s), 4) for s in scores],
            "atoms": [list(atom_ids_by_doc[doc_id]) for doc_id in pool],
        }
        if n and n % 50 == 0:
            print(f"    {n}/{len(queries)} ({time.perf_counter() - started:.0f}s)", flush=True)

    payload = {
        "_settings": {
            "model": model_id,
            "overlap": args.overlap,
            "depth": args.depth,
            "rrf_k": args.rrf_k,
            "rerank_model": args.rerank_model,
            "limit_queries": args.limit_queries,
            "sample_seed": args.sample_seed,
            "corpus_chunks": len(corpus),
            "atom_weight": atom_weight,
            "seconds": round(time.perf_counter() - started, 1),
        },
        "golds": {str(i): gold for (i, _), gold in zip(ev_items, golds)},
        "queries": per_query,
    }
    out = Path(str(args.out) + ".json")
    evalkit_write(out, payload)
    print(f"wrote {out} in {payload['_settings']['seconds']}s", flush=True)
    return 0


def evalkit_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
