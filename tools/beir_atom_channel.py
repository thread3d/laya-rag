"""Does an atomic lexical channel help BEIR fusion?

The MultiHop experiment found that atom-level BM25 alone is worse than chunk BM25, but
fusing the two significantly improved Hits@4. BEIR is the project's headline retrieval
benchmark, so this asks the same question there: fuse the shipped dense+BM25 hybrid with an
atom-level BM25 channel and measure nDCG@10 per dataset against the two-channel baseline.

It reuses the cached `bge-small` document embeddings, so no document is re-embedded; only
the queries are embedded, and the atom index is pure BM25.

    .venv/bin/python tools/beir_atom_channel.py --datasets scifact,nfcorpus
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import beir, evalkit  # noqa: E402
from multihop_atom_index import atom_hits, build_atoms, parent_rankings  # noqa: E402

from laya_rag.lexical import BM25Index  # noqa: E402

DEFAULT_DATASETS = "scifact,nfcorpus,arguana,scidocs,fiqa"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="beir_atom_channel.py")
    ap.add_argument("--datasets", default=DEFAULT_DATASETS)
    ap.add_argument("--data-root", default=str(beir.DEFAULT_DATA_ROOT))
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--top-atoms", type=int, default=100)
    ap.add_argument("--atom-min-chars", type=int, default=40)
    ap.add_argument("--atom-weight", type=float, default=1.0, help="RRF weight of the atom channel")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "eval" / "beir-atom-channel"))
    return ap.parse_args(argv)


def atom_channel(corpus: Mapping[str, str], queries: Mapping[str, str], args) -> dict[str, list[str]]:
    atoms, parent = build_atoms(corpus, min_chars=args.atom_min_chars)
    index = BM25Index(k1=beir.BEIR_K1, b=beir.BEIR_B, drop_stopwords=True, stem=True)
    index.build(atoms)
    return parent_rankings(atom_hits(index, queries, args.top_atoms), parent, args.top_k)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root)
    enc = beir.BiEncoder(args.model)
    payload: dict[str, Any] = {"_settings": vars(args), "datasets": {}}

    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        dataset_dir = beir.ensure_dataset(name, data_root, download=False)
        corpus = beir.load_corpus(dataset_dir, include_titles=True)
        queries = beir.load_queries(dataset_dir)
        qrels = beir.load_qrels(dataset_dir)
        print(f"\n[{name}] {len(corpus)} docs, {len(queries)} queries", flush=True)

        vectors = beir.cached_embed(
            enc, corpus, ROOT / "eval" / "beir" / "cache" / f"{name}.{enc.name}"
        )
        dense = beir.dense_rankings(corpus, queries, enc, vectors, args.top_k, fixed_length=True)
        lexical = beir.bm25_rankings(corpus, queries, args.top_k, stem=True, drop_stopwords=True)
        print("  atom channel...", flush=True)
        atoms = atom_channel(corpus, queries, args)

        rankings = {
            "dense": dense,
            "lexical": lexical,
            "atom": atoms,
            "hybrid": beir.multi_channel_fusion([dense, lexical], [1.0, 1.0], args.top_k, rrf_k=args.rrf_k),
            "hybrid+atom": beir.multi_channel_fusion(
                [dense, lexical, atoms], [1.0, 1.0, args.atom_weight], args.top_k, rrf_k=args.rrf_k
            ),
        }
        per_query: dict[str, dict[str, float]] = {}
        summary: dict[str, dict[str, float]] = {}
        for retriever, ranking in rankings.items():
            evaluation = evalkit.evaluate_rankings(ranking, qrels, k_values=(10,), ignore_identical_ids=True)
            summary[retriever] = evaluation["summary"]
            per_query[retriever] = evalkit.per_query_metric(evaluation["per_query"], "nDCG@10")
        comparison = {
            "bootstrap": evalkit.paired_bootstrap(
                per_query["hybrid"], per_query["hybrid+atom"], samples=args.samples
            ),
            "permutation": evalkit.permutation_test(
                per_query["hybrid"], per_query["hybrid+atom"], samples=args.samples
            ),
        }
        payload["datasets"][name] = {
            "summary": summary,
            "queries": len(per_query["hybrid"]),
            "hybrid+atom vs hybrid": comparison,
        }
        for retriever, row in summary.items():
            print(f"  {retriever:14s} nDCG@10 {row['nDCG@10']:.4f}", flush=True)
        print(f"  hybrid+atom vs hybrid: {comparison['bootstrap']['delta']:+.4f} "
              f"[{comparison['bootstrap']['ci_low']:+.4f}, {comparison['bootstrap']['ci_high']:+.4f}] "
              f"p={comparison['permutation']['p_value']:.4f}", flush=True)

    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), payload)
    evalkit.write_text(Path(str(out) + ".md"), render_markdown(payload))
    print("\n" + render_markdown(payload))
    print("wrote", str(out) + ".json and .md")
    return 0


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# BEIR: does an atomic lexical channel help fusion?",
        "",
        "nDCG@10 per dataset; `hybrid` is the shipped dense+BM25 fusion, `hybrid+atom` adds an",
        "atom-level BM25 channel by RRF.",
        "",
        "| dataset | queries | dense | lexical | atom | hybrid | hybrid+atom | delta |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, row in payload["datasets"].items():
        s = row["summary"]
        delta = row["hybrid+atom vs hybrid"]["bootstrap"]["delta"]
        lines.append(
            f"| {name} | {row['queries']} | {s['dense']['nDCG@10']:.4f} | "
            f"{s['lexical']['nDCG@10']:.4f} | {s['atom']['nDCG@10']:.4f} | "
            f"{s['hybrid']['nDCG@10']:.4f} | {s['hybrid+atom']['nDCG@10']:.4f} | {delta:+.4f} |"
        )
    lines += ["", "## Paired comparison (hybrid+atom vs hybrid, nDCG@10)", ""]
    for name, row in payload["datasets"].items():
        c = row["hybrid+atom vs hybrid"]
        lines.append(
            f"- {name}: {c['bootstrap']['delta']:+.4f} "
            f"[{c['bootstrap']['ci_low']:+.4f}, {c['bootstrap']['ci_high']:+.4f}] "
            f"p={c['permutation']['p_value']:.4f} "
            f"({'significant' if c['bootstrap']['significant'] else 'not significant'})"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
