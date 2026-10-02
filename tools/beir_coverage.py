"""Does the coverage selection help BEIR, or only MultiHop?

`RECALL_COVERAGE_LAMBDA` is off by default even though it is the best measured pass on
MultiHop-RAG (MAP@10 0.2546 -> 0.2850), because the selection objective targets a *set* of
facts and BEIR is largely single-fact: one abstract answers a query. This measures the same
objective on BEIR with the cached `bge-small` embeddings, so the default is a decision backed
by evidence on both sides rather than an assumption.

    .venv/bin/python tools/beir_coverage.py --datasets scifact,nfcorpus,arguana
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
from multihop_coverage import atom_index, facility_location_select, split_atoms  # noqa: E402

DEFAULT_DATASETS = "scifact,nfcorpus,arguana"
LAMBDAS = (0.5, 0.7)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="beir_coverage.py")
    ap.add_argument("--datasets", default=DEFAULT_DATASETS)
    ap.add_argument("--data-root", default=str(beir.DEFAULT_DATA_ROOT))
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--k", type=int, default=10, help="final depth the family keeps")
    ap.add_argument("--rrf-k", type=int, default=5)
    ap.add_argument("--atom-min-chars", type=int, default=40)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "eval" / "beir-coverage"))
    return ap.parse_args(argv)


def coverage_rankings(
    fused_scored: Mapping[str, Sequence[tuple[str, float]]],
    vectors,
    row_of: Mapping[str, int],
    atom_ids_by_doc: Mapping[str, Sequence[int]],
    atom_weight: Sequence[float],
    k: int,
) -> dict[str, dict[str, list[str]]]:
    """Fused baseline plus coverage-selected lists at each lambda, with/without redundancy."""
    names = ["fused"] + [f"coverage-{lam}" for lam in LAMBDAS] \
        + [f"coverage-nored-{lam}" for lam in LAMBDAS]
    out: dict[str, dict[str, list[str]]] = {name: {} for name in names}
    import numpy as np

    for query_id, scored in fused_scored.items():
        cand = [doc_id for doc_id, _score in scored]
        out["fused"][query_id] = cand[:k]
        scores = np.asarray([score for _doc_id, score in scored], dtype=float)
        rel = (scores - scores.min()) / (scores.max() - scores.min()) if len(scores) > 1 else scores
        vecs = np.asarray(vectors[[row_of[d] for d in cand]], dtype=np.float32)
        gram = vecs @ vecs.T
        atom_ids = [list(atom_ids_by_doc[d]) for d in cand]
        for lam in LAMBDAS:
            for suffix, use_redundancy in (("", True), ("-nored", False)):
                selected = facility_location_select(
                    rel, atom_ids, atom_weight, gram, k, lam, use_redundancy=use_redundancy
                )
                out[f"coverage{suffix}-{lam}"][query_id] = [cand[i] for i in selected]
    return out


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
        rows = list(corpus)
        row_of = {doc_id: i for i, doc_id in enumerate(rows)}
        vectors = beir.cached_embed(
            enc, corpus, ROOT / "eval" / "beir" / "cache" / f"{name}.{enc.name}"
        )
        dense = beir.dense_rankings(corpus, queries, enc, vectors, args.top_k, fixed_length=True)
        lexical = beir.bm25_rankings(corpus, queries, args.top_k, stem=True, drop_stopwords=True)
        from laya_rag.lexical import reciprocal_rank_fusion

        fused_scored = {
            query_id: reciprocal_rank_fusion(
                [dense[query_id], lexical[query_id]],
                k=args.rrf_k, limit=args.top_k, weights=[1.0, 1.0],
            )
            for query_id in queries
        }

        atoms_by_doc = {doc_id: split_atoms(text, min_chars=args.atom_min_chars)
                        for doc_id, text in corpus.items()}
        atom_ids_of, atom_weight, _ = atom_index(list(atoms_by_doc.values()))
        atom_ids_by_doc = dict(zip(atoms_by_doc, atom_ids_of))

        rankings = coverage_rankings(
            fused_scored, vectors, row_of, atom_ids_by_doc, atom_weight, args.k
        )
        summary: dict[str, dict[str, float]] = {}
        per_query: dict[str, dict[str, float]] = {}
        for retriever, ranking in rankings.items():
            evaluation = evalkit.evaluate_rankings(ranking, qrels, k_values=(10,), ignore_identical_ids=True)
            summary[retriever] = evaluation["summary"]
            per_query[retriever] = evalkit.per_query_metric(evaluation["per_query"], "nDCG@10")
        comparisons = {}
        for retriever in rankings:
            if retriever == "fused":
                continue
            comparisons[retriever] = {
                "bootstrap": evalkit.paired_bootstrap(
                    per_query["fused"], per_query[retriever], samples=args.samples
                ),
            }
        payload["datasets"][name] = {
            "documents": len(corpus),
            "summary": summary,
            "comparisons": comparisons,
        }
        for retriever, row in summary.items():
            print(f"[{name}] {retriever:20s} nDCG@10 {row['nDCG@10']:.4f}", flush=True)

    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), payload)
    evalkit.write_text(Path(str(out) + ".md"), render_markdown(payload))
    print("\n" + render_markdown(payload))
    print("wrote", str(out) + ".json and .md")
    return 0


def render_markdown(payload: Mapping[str, Any]) -> str:
    names = ["fused", "coverage-0.5", "coverage-nored-0.5", "coverage-0.7", "coverage-nored-0.7"]
    lines = [
        "# Does the coverage selection help BEIR?",
        "",
        "nDCG@10 per dataset; `fused` is the shipped dense+BM25 hybrid at RRF k=5.",
        "",
        "| dataset | fused | " + " | ".join(n for n in names[1:]) + " |",
        "| --- | " + " | ".join("---" for _ in names) + " |",
    ]
    for dataset, row in payload["datasets"].items():
        s = row["summary"]
        cells = [f"{s[n]['nDCG@10']:.4f}" if n in s else "—" for n in names]
        lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
    lines += ["", "## Coverage vs fused (paired bootstrap, nDCG@10)", ""]
    for dataset, row in payload["datasets"].items():
        for name, value in row["comparisons"].items():
            boot = value["bootstrap"]
            lines.append(
                f"- {dataset} {name}: {boot['delta']:+.4f} "
                f"[{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] "
                f"({'significant' if boot['significant'] else 'not significant'})"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
