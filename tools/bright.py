"""BRIGHT: reasoning-intensive retrieval, measured retrieval-only.

BRIGHT is the benchmark that tests whether a retriever can find documents that require
*reasoning* to connect to the query, where BM25 is competitive and dense-only retrieval is
weak. Its official score is nDCG@10 over official gold ids — retrieval only, no generator —
so it is a legitimate row for this engine.

The dataset ships on Hugging Face as parquet splits. This downloads the requested subsets
once, caches them as JSONL under ``eval/bright/<subset>/`` so later runs are offline, and
evaluates the engine's channels with the repo's own metric path.

    .venv/bin/python tools/bright.py --subsets pony,theoremqa_theorems --retrievers bm25,dense,hybrid
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

# Keep the download inside the workspace; the default ~/.cache is outside the sandbox.
os.environ.setdefault("HF_HOME", str(ROOT / "eval" / "bright-hf-cache"))

import beir, evalkit  # noqa: E402

#: Subsets that fit on a CPU-only host. leetcode (414k) and aops/theoremqa_questions (188k)
#: are excluded by default; pass them explicitly if there is disk and time.
DEFAULT_SUBSETS = "pony,theoremqa_theorems"


def parse_gold(value: Any) -> list[str]:
    """BRIGHT's gold field is a list, but some exports serialise it as a Python repr."""
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    text = str(value or "").strip()
    if not text or text == "N/A":
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, (list, tuple)) else []


def build_dataset(
    examples_rows: Sequence[Mapping[str, Any]],
    document_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    """``(corpus, queries, qrels)`` from BRIGHT's example and document rows."""
    corpus = {str(row["id"]): str(row.get("content") or "") for row in document_rows}
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    for row in examples_rows:
        query_id = str(row["id"])
        queries[query_id] = str(row.get("query") or "")
        gold = {doc_id: 1 for doc_id in parse_gold(row.get("gold_ids")) if doc_id in corpus}
        if gold:
            qrels[query_id] = gold
    return corpus, queries, qrels


def read_cache(subset: str, data_root: Path) -> dict[str, Any] | None:
    directory = data_root / subset
    corpus_path = directory / "corpus.jsonl"
    queries_path = directory / "queries.jsonl"
    qrels_path = directory / "qrels.json"
    if not (corpus_path.is_file() and queries_path.is_file() and qrels_path.is_file()):
        return None
    corpus = {
        json.loads(line)["id"]: json.loads(line)["content"]
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    queries = {
        json.loads(line)["id"]: json.loads(line)["query"]
        for line in queries_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    return {"corpus": corpus, "queries": queries, "qrels": qrels}


def write_cache(subset: str, data_root: Path, payload: Mapping[str, Any]) -> None:
    directory = data_root / subset
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "corpus.jsonl").write_text(
        "\n".join(json.dumps({"id": k, "content": v}, ensure_ascii=False)
                  for k, v in payload["corpus"].items()),
        encoding="utf-8",
    )
    (directory / "queries.jsonl").write_text(
        "\n".join(json.dumps({"id": k, "query": v}, ensure_ascii=False)
                  for k, v in payload["queries"].items()),
        encoding="utf-8",
    )
    (directory / "qrels.json").write_text(json.dumps(payload["qrels"]), encoding="utf-8")


def download(subset: str) -> dict[str, Any]:
    from datasets import load_dataset

    print(f"  downloading BRIGHT/{subset}...", flush=True)
    examples = load_dataset("xlangai/BRIGHT", "examples", split=subset)
    documents = load_dataset("xlangai/BRIGHT", "documents", split=subset)
    corpus, queries, qrels = build_dataset(list(examples), list(documents))
    return {"corpus": corpus, "queries": queries, "qrels": qrels}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="bright.py")
    ap.add_argument("--subsets", default=DEFAULT_SUBSETS)
    ap.add_argument("--data-root", default=str(ROOT / "eval" / "bright"))
    ap.add_argument("--retrievers", default="bm25,dense,hybrid")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--stem", action="store_true", help="stem the lexical channel (off by default: code/math)")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--refresh", action="store_true", help="re-download even if cached")
    ap.add_argument("--out", default=str(ROOT / "eval" / "bright" / "report"))
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from heavylock import heavy_job_guard

    heavy_job_guard("bright.py")
    data_root = Path(args.data_root)
    retrievers = {name.strip() for name in args.retrievers.split(",") if name.strip()}
    enc = beir.BiEncoder(args.model) if "dense" in retrievers or "hybrid" in retrievers else None
    payload: dict[str, Any] = {"_settings": vars(args), "subsets": {}}

    for subset in [s.strip() for s in args.subsets.split(",") if s.strip()]:
        cached = None if args.refresh else read_cache(subset, data_root)
        if cached is None:
            cached = download(subset)
            write_cache(subset, data_root, cached)
        else:
            print(f"  using cached BRIGHT/{subset}", flush=True)
        corpus, queries, qrels = cached["corpus"], cached["queries"], cached["qrels"]
        print(f"\n[{subset}] {len(corpus)} docs, {len(queries)} queries, {len(qrels)} with gold",
              flush=True)

        rankings: dict[str, dict[str, list[str]]] = {}
        if "bm25" in retrievers:
            rankings["bm25"] = beir.bm25_rankings(
                corpus, queries, args.top_k, stem=args.stem, drop_stopwords=False
            )
        if "dense" in retrievers:
            vectors = beir.cached_embed(
                enc, corpus, data_root / subset / "cache" / f"{enc.name}"
            )
            rankings["dense"] = beir.dense_rankings(
                corpus, queries, enc, vectors, args.top_k, fixed_length=True
            )
        if "hybrid" in retrievers:
            if "dense" not in rankings:
                vectors = beir.cached_embed(
                    enc, corpus, data_root / subset / "cache" / f"{enc.name}"
                )
                rankings["dense"] = beir.dense_rankings(
                    corpus, queries, enc, vectors, args.top_k, fixed_length=True
                )
            if "bm25" not in rankings:
                rankings["bm25"] = beir.bm25_rankings(
                    corpus, queries, args.top_k, stem=args.stem, drop_stopwords=False
                )
            rankings["hybrid"] = beir.multi_channel_fusion(
                [rankings["dense"], rankings["bm25"]], [1.0, 1.0], args.top_k, rrf_k=args.rrf_k
            )

        summary: dict[str, dict[str, float]] = {}
        per_query: dict[str, dict[str, float]] = {}
        for name, ranking in rankings.items():
            evaluation = evalkit.evaluate_rankings(
                ranking, qrels, k_values=(10, 100), ignore_identical_ids=True
            )
            summary[name] = evaluation["summary"]
            per_query[name] = evalkit.per_query_metric(evaluation["per_query"], "nDCG@10")
        comparison = {}
        if "hybrid" in per_query and "bm25" in per_query:
            comparison["hybrid vs bm25"] = evalkit.paired_bootstrap(
                per_query["bm25"], per_query["hybrid"], samples=2000
            )
        if "dense" in per_query and "bm25" in per_query:
            comparison["dense vs bm25"] = evalkit.paired_bootstrap(
                per_query["bm25"], per_query["dense"], samples=2000
            )
        payload["subsets"][subset] = {
            "documents": len(corpus),
            "queries": len(queries),
            "scored_queries": summary[next(iter(summary))].get("queries") if summary else 0,
            "summary": summary,
            "comparisons": comparison,
        }
        for name, row in summary.items():
            print(f"  {name:8s} nDCG@10 {row['nDCG@10']:.4f} (nDCG@100 {row.get('nDCG@100', 0):.4f})",
                  flush=True)
        for key, value in comparison.items():
            print(f"  {key}: {value['delta']:+.4f} [{value['ci_low']:+.4f}, {value['ci_high']:+.4f}]",
                  flush=True)

    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), payload)
    evalkit.write_text(Path(str(out) + ".md"), render_markdown(payload))
    print("\n" + render_markdown(payload))
    print("wrote", str(out) + ".json and .md")
    return 0


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# BRIGHT — reasoning-intensive retrieval (retrieval only)",
        "",
        "nDCG@10 over official gold ids. No generator is involved, so these are the engine's",
        "numbers; the published BRIGHT leaderboard's top nDCG@10 is ~21, and BM25 is a strong",
        "baseline there, so a dense-only win is not expected.",
        "",
        "| subset | docs | queries | retriever | nDCG@10 | nDCG@100 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for subset, row in payload["subsets"].items():
        for name, metrics in row["summary"].items():
            lines.append(
                f"| {subset} | {row['documents']} | {row['queries']} | {name} | "
                f"{metrics['nDCG@10']:.4f} | {metrics.get('nDCG@100', 0.0):.4f} |"
            )
    lines += ["", "## Paired comparisons (bootstrap 95% CI, nDCG@10)", ""]
    for subset, row in payload["subsets"].items():
        for key, value in row.get("comparisons", {}).items():
            lines.append(
                f"- {subset} {key}: {value['delta']:+.4f} "
                f"[{value['ci_low']:+.4f}, {value['ci_high']:+.4f}] "
                f"({'significant' if value['significant'] else 'not significant'})"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
