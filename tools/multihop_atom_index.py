"""Atomic-unit lexical retrieval with parent-window return, on MultiHop-RAG.

RESEARCH.md Tier 1.5. The engine indexes whole chunks; a fact split across a chunk
boundary, or a short answer buried in a long chunk, competes with the chunk's length
normalisation rather than on its own. This experiment indexes sentence-like *atoms*
instead, retrieves atoms, and returns the parent chunk of each — so the ranking is decided
by the best-matching sentence, while the evidence handed back is still a whole chunk.

It is deliberately lexical and model-free, so it runs in minutes with no encoder and no
network, and it isolates the retrieval-unit question from representation quality. The
comparison is against the same chunk BM25 the engine ships, and against a rank fusion of
the two.

    .venv/bin/python tools/multihop_atom_index.py --limit-queries 400
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
from multihop_coverage import split_atoms  # noqa: E402
from multihop_official import chunk_text, official_metrics  # noqa: E402

from laya_rag.lexical import BM25Index, reciprocal_rank_fusion  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"
METRICS = ("Hits@10", "Hits@4", "MAP@10", "MRR@10")


class _Atom:
    """The two attributes ``BM25Index`` needs from a unit."""

    __slots__ = ("chunk_id", "text")

    def __init__(self, chunk_id: str, text: str) -> None:
        self.chunk_id = chunk_id
        self.text = text


# ── pure helpers (unit-tested offline) ───────────────────────────────────

def build_atoms(
    corpus: Mapping[str, str], min_chars: int = 40
) -> tuple[list[_Atom], dict[str, str]]:
    """Sentence atoms with a stable id per (chunk, position) and an atom -> chunk map."""
    atoms: list[_Atom] = []
    parent: dict[str, str] = {}
    for chunk_id, text in corpus.items():
        for position, atom in enumerate(split_atoms(text, min_chars=min_chars)):
            atom_id = f"{chunk_id}::{position}"
            atoms.append(_Atom(atom_id, atom))
            parent[atom_id] = chunk_id
    return atoms, parent


def parent_rankings(
    hits: Mapping[str, Sequence[tuple[str, float]]],
    parent: Mapping[str, str],
    top_k: int,
) -> dict[str, list[str]]:
    """Map each query's atom hits to parent chunks, keeping the best atom per parent.

    Hits arrive in score order, so a parent's first appearance is its best atom; that
    also makes this the parent-window return — the chunk is the evidence, the atom chose
    it.
    """
    out: dict[str, list[str]] = {}
    for query_id, ranked in hits.items():
        seen: set[str] = set()
        ordered: list[str] = []
        for atom_id, _score in ranked:
            chunk_id = parent.get(atom_id)
            if chunk_id is None or chunk_id in seen:
                continue
            seen.add(chunk_id)
            ordered.append(chunk_id)
            if len(ordered) >= top_k:
                break
        out[query_id] = ordered
    return out


def atom_hits(
    index: BM25Index, queries: Mapping[str, str], top_atoms: int
) -> dict[str, list[tuple[str, float]]]:
    return {
        query_id: [(hit.chunk_id, float(hit.score)) for hit in index.search(text, k=top_atoms)]
        for query_id, text in queries.items()
    }


def score_strategies(
    corpus: Mapping[str, str],
    ranking: Mapping[str, Sequence[str]],
    items: Sequence[tuple[int, Any]],
    golds: Sequence[Sequence[str]],
    metrics: Sequence[str] = METRICS,
) -> dict[str, dict[str, float]]:
    """Per-query metric rows for one ranking.

    ``official_metrics`` takes a list of *per-query ranked lists*; handing it one flat
    list of strings slices a string into characters and scores everything zero. This
    wrapper is the single place that contract is applied, and it is unit-tested.
    """
    rows: dict[str, dict[str, float]] = {metric: {} for metric in metrics}
    for i, gold in zip((i for i, _ in items), golds):
        qid = str(i)
        summary, _ = official_metrics([[corpus[d] for d in ranking.get(qid, [])]], [gold])
        for metric in metrics:
            rows[metric][qid] = summary[metric]
    return rows


# ── experiment ───────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="multihop_atom_index.py")
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--top-atoms", type=int, default=200, help="atoms retrieved per query")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--rrf-k", type=int, default=5)
    ap.add_argument("--atom-min-chars", type=int, default=40)
    ap.add_argument("--limit-queries", type=int, default=None)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT_BENCH / "atom-index"))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    raw = json.load(open(ROOT_BENCH / "corpus.json"))
    items_all = json.load(open(ROOT_BENCH / "MultiHopRAG.json"))
    items = [(i, x) for i, x in enumerate(items_all) if x["question_type"] != "null_query"]
    if args.limit_queries:
        items = items[: args.limit_queries]
    golds = [[e["fact"] for e in x["evidence_list"]] for _, x in items]

    corpus: dict[str, str] = {}
    titles: dict[str, str] = {}
    for a in raw:
        for j, c in enumerate(chunk_text(a["title"] + " " + a["body"], 256, args.overlap)):
            cid = f"{a['url']}#{j}"
            corpus[cid] = c
            titles[cid] = a["title"]
    queries = {str(i): x["query"] for i, x in items}
    print(f"[overlap={args.overlap}] {len(corpus)} chunks, {len(queries)} queries", flush=True)

    # Chunk baseline: the same lexical channel the engine ships.
    chunk_rankings = beir.bm25_rankings(
        corpus, queries, args.top_k, stem=True, drop_stopwords=True,
        titles=titles, title_boost=2,
    )

    print("  building atom index...", flush=True)
    atoms, parent = build_atoms(corpus, min_chars=args.atom_min_chars)
    index = BM25Index(k1=beir.BEIR_K1, b=beir.BEIR_B, drop_stopwords=True, stem=True)
    index.build(atoms)
    print(f"  {len(atoms)} atoms from {len(corpus)} chunks", flush=True)

    atom_rankings = parent_rankings(
        atom_hits(index, queries, args.top_atoms), parent, args.top_k
    )
    hybrid_rankings: dict[str, list[str]] = {}
    for query_id in queries:
        merged = reciprocal_rank_fusion(
            [chunk_rankings.get(query_id, []), atom_rankings.get(query_id, [])],
            k=args.rrf_k, limit=args.top_k, weights=[1.0, 1.0],
        )
        hybrid_rankings[query_id] = [doc_id for doc_id, _ in merged]

    strategies = {
        "chunk-bm25": chunk_rankings,
        "atom-bm25": atom_rankings,
        "chunk+atom-rrf": hybrid_rankings,
    }
    per_query: dict[str, dict[str, dict[str, float]]] = {}
    summary: dict[str, dict[str, float]] = {}
    for name, ranking in strategies.items():
        rows = score_strategies(corpus, ranking, items, golds)
        per_query[name] = rows
        summary[name] = {m: round(sum(rows[m].values()) / len(rows[m]), 4) for m in METRICS}

    comparisons = {}
    for name in ("atom-bm25", "chunk+atom-rrf"):
        for metric in ("MAP@10", "Hits@4", "Hits@10"):
            comparisons[f"{name} vs chunk-bm25 {metric}"] = {
                "bootstrap": evalkit.paired_bootstrap(
                    per_query["chunk-bm25"][metric], per_query[name][metric], samples=args.samples
                ),
                "permutation": evalkit.permutation_test(
                    per_query["chunk-bm25"][metric], per_query[name][metric], samples=args.samples
                ),
            }

    result = {
        "_settings": vars(args) | {"chunks": len(corpus), "atoms": len(atoms)},
        "summary": summary,
        "comparisons": comparisons,
        "per_query": {name: per_query[name]["MAP@10"] for name in strategies},
    }
    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), result)
    evalkit.write_text(Path(str(out) + ".md"), render_markdown(result))
    print(render_markdown(result))
    print("wrote", str(out) + ".json and .md")
    return 0


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Atomic-unit lexical retrieval with parent-window return",
        "",
        f"- {result['_settings']['chunks']} chunks -> {result['_settings']['atoms']} atoms, "
        f"top-{result['_settings']['top_atoms']} atoms mapped to top-{result['_settings']['top_k']} chunks",
        "",
        "| strategy | Hits@10 | Hits@4 | MAP@10 | MRR@10 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, row in result["summary"].items():
        lines.append(
            f"| {name} | {row['Hits@10']:.4f} | {row['Hits@4']:.4f} | "
            f"{row['MAP@10']:.4f} | {row['MRR@10']:.4f} |"
        )
    lines += ["", "## Paired comparisons vs chunk BM25", ""]
    for key, value in sorted(result["comparisons"].items()):
        boot = value["bootstrap"]
        perm = value["permutation"]
        lines.append(
            f"- `{key}`: {boot['delta']:+.4f} [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] "
            f"p={perm['p_value']:.4f} ({'sig' if boot['significant'] else 'ns'})"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
