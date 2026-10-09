#!/usr/bin/env python
"""Is the *query* side the linear bottleneck? A query-only adapter probe.

`tools/projection_head.py` fits one affine map from Laya's document embeddings into a
teacher's space and applies the same map to Laya's queries. That conflates two things.
This probe isolates the query side: it fits a map on the **train queries**
(Laya query embedding -> teacher query embedding) and then evaluates, on the test
queries, how much retrieval a linearly-adapted Laya query recovers against a *fixed*
document side.

The comparison that matters is `qadapter_teacher_docs` against `teacher_docs_laya_q_proj`
(the projection tool's 0.1327 control) and `teacher_full` (the teacher's own ceiling). If a
query-only linear map closes most of the gap to the teacher, the query representation is
linearly recoverable and an encoder fine-tune has room; if all the query arms sit near the
control, the frozen representation is the limit.

    .venv/bin/python tools/laya_query_adapter.py --dataset scifact
    .venv/bin/python tools/laya_query_adapter.py --dataset nfcorpus --l2 1e-2

Requires the dataset's Laya and teacher document caches, which `tools/beir.py` writes;
this tool never re-embeds a corpus. `--fresh` recomputes the query embeddings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import beir  # noqa: E402
import evalkit  # noqa: E402
import projection_head  # noqa: E402
from laya_rag.config import EngineConfig  # noqa: E402

LinearProjection = projection_head.LinearProjection
retrieval_rankings = projection_head.retrieval_rankings


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="laya_query_adapter.py")
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--data-root", default=str(ROOT / "eval" / "beir"))
    ap.add_argument("--teacher", default=projection_head.DEFAULT_TEACHER)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--query-cache", default=None,
                    help="where query embeddings go (default: <data-root>/cache/queries-<dataset>-<teacher>")
    ap.add_argument("--out", default=None, help="output stem for .json/.md")
    return ap.parse_args(argv)


def query_vectors(
    embedder: Any,
    queries: Mapping[str, str],
    cache: Path,
) -> dict[str, np.ndarray]:
    """Embed queries once, cached per query id so a re-run is free."""
    ids_file = Path(str(cache) + ".ids.json")
    vectors_file = Path(str(cache) + ".npy")
    ids = sorted(queries)
    if vectors_file.is_file() and ids_file.is_file():
        if json.loads(ids_file.read_text(encoding="utf-8")) == ids:
            print(f"  reusing cached {embedder.name} query embeddings ({vectors_file.name})",
                  flush=True)
            block = np.load(vectors_file)
            return {query_id: block[index] for index, query_id in enumerate(ids)}
    print(f"  embedding {len(ids)} queries with {embedder.name}", flush=True)
    block = np.asarray(embedder.embed_queries([queries[q] for q in ids]), dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(vectors_file, block)
    ids_file.write_text(json.dumps(ids), encoding="utf-8")
    return {query_id: block[index] for index, query_id in enumerate(ids)}


def matrix_for(
    vectors: Mapping[str, np.ndarray],
    qrels: Mapping[str, Mapping[str, float]],
) -> tuple[list[str], np.ndarray]:
    ids = [q for q in qrels if q in vectors]
    if len(ids) != len(qrels):
        raise SystemExit(
            f"{len(qrels) - len(ids)} labelled queries have no embedding "
            f"(first: {sorted(set(qrels) - set(vectors))[:3]})"
        )
    return ids, np.vstack([np.asarray(vectors[q], dtype=np.float64) for q in ids])


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root)
    dataset_dir = beir.ensure_dataset(args.dataset, data_root, download=False)
    corpus = beir.load_corpus(dataset_dir)
    queries = beir.load_queries(dataset_dir)
    train_qrels = beir.load_qrels(dataset_dir, split="train")
    test_qrels = beir.load_qrels(dataset_dir, split="test")
    train_queries = {q: t for q, t in queries.items() if q in train_qrels}
    test_queries = {q: t for q, t in queries.items() if q in test_qrels}
    print(f"{args.dataset}: {len(corpus)} docs, {len(train_queries)} train / "
          f"{len(test_queries)} test queries", flush=True)

    from laya_rag.model import LayaModel

    laya = beir.LayaEmbedder(EngineConfig())
    teacher = beir.BiEncoder(args.teacher)
    cache_dir = data_root / "cache"
    laya_docs = beir.cached_embed(laya, corpus, cache_dir / f"{args.dataset}.{laya.name}")
    teacher_docs = beir.cached_embed(teacher, corpus, cache_dir / f"{args.dataset}.{teacher.name}")

    query_cache = Path(args.query_cache) if args.query_cache else (
        cache_dir / f"queries-{args.dataset}-{laya.name}-{teacher.name}"
    )
    all_queries = {**train_queries, **test_queries}
    laya_query = query_vectors(laya, all_queries, Path(str(query_cache) + f".{laya.name}"))
    teacher_query = query_vectors(teacher, all_queries, Path(str(query_cache) + f".{teacher.name}"))

    doc_ids = list(corpus)
    _, laya_train = matrix_for(laya_query, train_qrels)
    _, teacher_train = matrix_for(teacher_query, train_qrels)
    test_ids, laya_test = matrix_for(laya_query, test_qrels)
    _, teacher_test = matrix_for(teacher_query, test_qrels)

    # Document-side map (the existing projection) and the query-only map this probe adds.
    doc_map = LinearProjection.fit(laya_docs, teacher_docs, l2=args.l2)
    query_map = LinearProjection.fit(laya_train, teacher_train, l2=args.l2)

    projected_docs = doc_map.transform_normalized(laya_docs)
    projected_queries = doc_map.transform_normalized(laya_test)
    adapted_queries = query_map.transform_normalized(laya_test)
    # Pass query vectors as {id: vector} so alignment never depends on qrels iteration order.
    laya_test_map = {q: v for q, v in zip(test_ids, laya_test)}
    projected_map = {q: v for q, v in zip(test_ids, projected_queries)}
    adapted_map = {q: v for q, v in zip(test_ids, adapted_queries)}
    teacher_map = {q: v for q, v in zip(test_ids, teacher_test)}

    arms: dict[str, dict[str, list[str]]] = {
        "raw_laya": retrieval_rankings(laya_docs, laya_test_map, doc_ids, test_qrels, args.top_k),
        "docproj_shared_map": retrieval_rankings(
            projected_docs, projected_map, doc_ids, test_qrels, args.top_k
        ),
        "teacher_docs_laya_q_proj": retrieval_rankings(
            teacher_docs, projected_map, doc_ids, test_qrels, args.top_k
        ),
        "qadapter_teacher_docs": retrieval_rankings(
            teacher_docs, adapted_map, doc_ids, test_qrels, args.top_k
        ),
        "qadapter_docproj": retrieval_rankings(
            projected_docs, adapted_map, doc_ids, test_qrels, args.top_k
        ),
        "teacher_full": retrieval_rankings(
            teacher_docs, teacher_map, doc_ids, test_qrels, args.top_k
        ),
    }
    summary = {
        name: evalkit.evaluate_rankings(rankings, test_qrels)["summary"]
        for name, rankings in arms.items()
    }
    report = {
        "dataset": args.dataset,
        "teacher": args.teacher,
        "l2": args.l2,
        "documents": len(doc_ids),
        "train_queries": len(laya_train),
        "test_queries": len(test_ids),
        "settings": vars(args),
        "arms": summary,
    }
    print(json.dumps({k: v.get("nDCG@10") for k, v in summary.items()}, indent=2), flush=True)
    out = Path(args.out) if args.out else data_root / f"query-adapter-{args.dataset}"
    evalkit.write_json(out.with_suffix(".json"), report)
    rows = [{"arm": name, **metrics} for name, metrics in summary.items()]
    evalkit.write_text(
        out.with_suffix(".md"),
        f"# Query-side linear adapter — {args.dataset}\n\n"
        f"Ridge (l2={args.l2}) fitted on {len(laya_train)} train queries from Laya's query "
        f"embeddings to `{args.teacher}`'s; document side frozen and cached.\n\n"
        + evalkit.markdown_table(rows)
        + "\n",
    )
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
