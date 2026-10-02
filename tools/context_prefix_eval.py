"""Measure the structural context prefix on a labelled local corpus.

The engine can attach a per-chunk breadcrumb (document path and first heading) that the
lexical and dense channels index while the returned evidence keeps only the document's own
words. This compares retrieval with and without it, **lexically and model-free**, on the
vendored Laya documentation, so it is a retrieval-unit measurement rather than a model
measurement.

    .venv/bin/python tools/context_prefix_eval.py --corpus docs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import evalkit  # noqa: E402

from laya_rag.chunking import chunk_documents, load_documents  # noqa: E402
from laya_rag.lexical import BM25Index  # noqa: E402


def evaluate(
    documents: Sequence[Any],
    queries: Sequence[Any],
    *,
    context_prefix: bool,
    top_k: int = 20,
    k: int = 10,
    chunk_chars: int = 1200,
    overlap_chars: int = 400,
) -> dict[str, Any]:
    """Retrieval summary for one prefix setting, over the answerable queries."""
    chunks = chunk_documents(
        documents, chunk_chars=chunk_chars, overlap_chars=overlap_chars,
        context_prefix=context_prefix,
    )
    index = BM25Index(stem=True, drop_stopwords=True)
    index.build(chunks)
    path_of = {chunk.chunk_id: chunk.path for chunk in chunks}

    ranks: list[int | None] = []
    per_query: list[dict[str, Any]] = []
    for query in queries:
        if not getattr(query, "answerable", True):
            continue
        hits = index.search(query.query, k=top_k)
        paths = [path_of[hit.chunk_id] for hit in hits]
        rank = evalkit.first_match_rank(paths, query.gold)
        ranks.append(rank)
        per_query.append({"id": query.id, "rank": rank})
    summary = evalkit.rank_summary(ranks, k=k)
    summary["chunks"] = len(chunks)
    summary["per_query"] = per_query
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="context_prefix_eval.py")
    ap.add_argument("--corpus", default="docs", help="docs (vendored Laya docs) or a directory")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--k", type=int, default=10, help="rank cutoff for the reported metrics")
    ap.add_argument("--out", default=str(ROOT / "eval" / "context-prefix"))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.corpus == "docs":
        root = ROOT / "vendor" / "laya" / "docs"
        queries = evalkit.docs_queries()
        label = "docs"
    else:
        root = Path(args.corpus)
        queries = evalkit.docs_queries()
        label = str(root)
    loaded = load_documents([root])
    print(f"[{label}] {len(loaded.documents)} documents, "
          f"{sum(1 for q in queries if getattr(q, 'answerable', True))} answerable queries",
          flush=True)

    off = evaluate(loaded.documents, queries, context_prefix=False, top_k=args.top_k, k=args.k)
    on = evaluate(loaded.documents, queries, context_prefix=True, top_k=args.top_k, k=args.k)
    payload = {
        "_settings": {**vars(args), "corpus": label, "documents": len(loaded.documents)},
        "without_prefix": off,
        "with_prefix": on,
        "delta": {
            key: (round(on[key] - off[key], 4) if isinstance(off.get(key), (int, float)) else None)
            for key in ("recall", "mrr", f"precision@{args.k}")
        },
    }
    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), payload)
    evalkit.write_text(Path(str(out) + ".md"), render_markdown(payload))
    print(render_markdown(payload))
    print("wrote", str(out) + ".json and .md")
    return 0


def render_markdown(payload: Mapping[str, Any]) -> str:
    off, on = payload["without_prefix"], payload["with_prefix"]
    k = payload["_settings"]["k"]
    lines = [
        "# Structural context prefix — lexical retrieval",
        "",
        f"- corpus: {payload['_settings']['corpus']} "
        f"({payload['_settings']['documents']} documents, {off['n']} answerable queries)",
        f"- chunks: {off['chunks']} without prefix, {on['chunks']} with prefix",
        "",
        "| metric | without prefix | with prefix | delta |",
        "| --- | --- | --- | --- |",
    ]
    for key in ("recall", f"precision@{k}", "mrr", "median_rank", "worst_rank"):
        left, right = off.get(key), on.get(key)
        delta = payload["delta"].get(key)
        lines.append(
            f"| {key} | {left} | {right} | "
            f"{'—' if delta is None else f'{delta:+.4f}'} |"
        )
    lines += ["", "Higher is better for recall/precision/MRR; lower is better for ranks.", ""]
    off_by_id = {row["id"]: row["rank"] for row in off.get("per_query", [])}
    on_by_id = {row["id"]: row["rank"] for row in on.get("per_query", [])}
    if off_by_id:
        lines += ["## Per query (gold rank)", "", "| query | without | with |", "| --- | --- | --- |"]
        for query_id in off_by_id:
            lines.append(f"| {query_id} | {off_by_id[query_id]} | {on_by_id.get(query_id)} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
