"""Measure the two lexical backends: ranking parity, size and query latency.

The eager backend precomputes each posting's BM25 weight and scores with NumPy slice-adds;
the historical backend keeps per-document term frequencies and loops over postings in
Python. They must rank identically, so the interesting numbers are size and latency.

Size is reported as a deterministic object-graph estimate rather than process RSS: RSS is
dominated by the interpreter and by whichever index was built first, so it cannot attribute
memory to one structure. The estimate walks the containers the plain index holds
(``Counter`` per document, the postings dict of tuples) and, for the eager index, sums the
``nbytes`` of its arrays.

    .venv/bin/python tools/bench_lexical.py --docs 20000 --queries 200
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import evalkit  # noqa: E402

from laya_rag.lexical import BM25Index, EagerBM25Index  # noqa: E402


def estimate_size(obj, seen: set[int] | None = None) -> int:
    """Approximate deep size of a Python object graph, in bytes.

    Arrays report ``nbytes``; containers recurse; strings and numbers use ``sys.getsizeof``.
    Shared objects are counted once, so the postings sharing no state are summed correctly.
    """
    seen = seen if seen is not None else set()
    identity = id(obj)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for key, value in obj.items():
            size += estimate_size(key, seen) + estimate_size(value, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += estimate_size(item, seen)
    elif hasattr(obj, "nbytes"):
        size += int(obj.nbytes)
    elif hasattr(obj, "__dict__"):
        # A plain object (the index itself) holds everything in its instance dict.
        for value in vars(obj).values():
            size += estimate_size(value, seen)
    return size


def make_corpus(n_docs: int, vocab: int, length: int, seed: int = 17) -> list[SimpleNamespace]:
    """A deterministic Zipf-ish corpus, so term frequencies are realistic, not uniform."""
    import random

    rng = random.Random(seed)
    weights = [1.0 / (i + 1) for i in range(vocab)]
    docs = []
    for doc in range(n_docs):
        tokens = rng.choices(range(vocab), weights=weights, k=length)
        docs.append(SimpleNamespace(chunk_id=f"d{doc}", text=" ".join(f"w{t}" for t in tokens)))
    return docs


def make_queries(vocab: int, count: int, seed: int = 23) -> list[str]:
    import random

    rng = random.Random(seed)
    weights = [1.0 / (i + 1) for i in range(vocab)]
    return [" ".join(f"w{t}" for t in rng.choices(range(vocab), weights=weights, k=rng.randint(2, 4)))
            for _ in range(count)]


def measure(index, queries, k: int, repeats: int) -> dict:
    started = time.perf_counter()
    for _ in range(repeats):
        for query in queries:
            index.search(query, k=k)
    elapsed = time.perf_counter() - started
    per_query_ms = elapsed / (len(queries) * repeats) * 1000.0
    return {"ms_per_query": round(per_query_ms, 4)}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="bench_lexical.py")
    ap.add_argument("--docs", type=int, default=20000)
    ap.add_argument("--vocab", type=int, default=5000)
    ap.add_argument("--length", type=int, default=120, help="tokens per document")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--stem", action="store_true")
    ap.add_argument("--multihop", action="store_true",
                    help="use the real MultiHop-RAG corpus instead of a synthetic one")
    ap.add_argument("--out", default=str(ROOT / "eval" / "bench-lexical"))
    return ap.parse_args(argv)


def multihop_corpus(limit_queries: int) -> tuple[list[SimpleNamespace], list[str]]:
    """The real MultiHop-RAG chunks and queries, for a parity check on real text."""
    import json

    sys.path.insert(0, str(ROOT / "tools"))
    from multihop_official import chunk_text

    bench = ROOT / "eval" / "benchmarks" / "multihop-rag"
    raw = json.loads((bench / "corpus.json").read_text(encoding="utf-8"))
    items = [
        x for x in json.loads((bench / "MultiHopRAG.json").read_text(encoding="utf-8"))
        if x["question_type"] != "null_query"
    ][:limit_queries]
    docs: list[SimpleNamespace] = []
    for article in raw:
        for position, text in enumerate(chunk_text(article["title"] + " " + article["body"], 256, 200)):
            docs.append(SimpleNamespace(chunk_id=f"{article['url']}#{position}", text=text))
    return docs, [x["query"] for x in items]


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.multihop:
        docs, queries = multihop_corpus(args.queries)
    else:
        docs = make_corpus(args.docs, args.vocab, args.length)
        queries = make_queries(args.vocab, args.queries)
    print(f"{len(docs)} docs, {len(queries)} queries", flush=True)

    results: dict[str, dict] = {}
    rankings: dict[str, list[list[str]]] = {}
    for name, cls in (("bm25", BM25Index), ("eager", EagerBM25Index)):
        index = cls(stem=args.stem)
        started = time.perf_counter()
        index.build(docs)
        build_seconds = time.perf_counter() - started
        rankings[name] = [[hit.chunk_id for hit in index.search(q, k=args.k)] for q in queries]
        size = estimate_size(index)
        results[name] = {
            "build_seconds": round(build_seconds, 2),
            "index_bytes": int(size),
            "postings": int(len(getattr(index, "_postings", {}))),
            **measure(index, queries, args.k, args.repeats),
        }
        print(f"  {name}: build {results[name]['build_seconds']}s, "
              f"{size / 1e6:.1f} MB, {results[name]['ms_per_query']} ms/query", flush=True)

    parity = all(a == b for a, b in zip(rankings["bm25"], rankings["eager"]))
    identical_queries = sum(1 for a, b in zip(rankings["bm25"], rankings["eager"]) if a == b)
    payload = {
        "_settings": {
            **vars(args),
            "documents": len(docs),
            "corpus": "multihop" if args.multihop else "synthetic",
        },
        "parity": {
            "identical_queries": identical_queries,
            "queries": len(queries),
            "top_k_identical": parity,
        },
        "results": results,
        "speedup": round(
            results["bm25"]["ms_per_query"] / results["eager"]["ms_per_query"], 2
        ) if results["eager"]["ms_per_query"] else None,
        "size_ratio": round(
            results["bm25"]["index_bytes"] / results["eager"]["index_bytes"], 2
        ) if results["eager"]["index_bytes"] else None,
    }
    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), payload)
    evalkit.write_text(Path(str(out) + ".md"), render_markdown(payload))
    print(render_markdown(payload))
    print("wrote", str(out) + ".json and .md")
    return 0


def render_markdown(payload) -> str:
    r = payload["results"]
    settings = payload["_settings"]
    lines = [
        "# Lexical backend benchmark",
        "",
        f"- corpus: {settings.get('corpus', 'synthetic')}, "
        f"{settings.get('documents', settings.get('docs'))} documents, "
        f"{settings['queries']} queries, top-{settings['k']}",
        f"- ranking parity: {payload['parity']['identical_queries']}/"
        f"{payload['parity']['queries']} queries identical "
        f"({'PASS' if payload['parity']['top_k_identical'] else 'FAIL'})",
        "",
        "| backend | build (s) | index (MB) | ms/query |",
        "| --- | --- | --- | --- |",
    ]
    for name, row in r.items():
        lines.append(
            f"| {name} | {row['build_seconds']} | {row['index_bytes'] / 1e6:.1f} | "
            f"{row['ms_per_query']} |"
        )
    lines += [
        "",
        f"- eager speedup: **{payload['speedup']}x**",
        f"- size ratio (bm25/eager): **{payload['size_ratio']}x**",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
