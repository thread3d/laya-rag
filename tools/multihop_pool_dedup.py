"""Does duplication in the candidate pool cost retrieval slots?

`laya-rag integrity` found 690 exact-duplicate clusters (1,095 redundant chunks) in this
repository's own index. A duplicate is not merely untidy: two copies of one passage occupy
two of the ten slots a selector has, so a MAP@10 that rewards *distinct facts* is paying for
them. The engine already suppresses near-duplicates *at query time* by embedding cosine
(`RECALL_DEDUP_THRESHOLD`) and the coverage pass penalises redundancy through sentence atoms,
so whether *content*-level dedup adds anything on top is an empirical question, not an obvious
yes.

This measures it over the cached pool — no model, no network, exact and deterministic:

* how much duplication the MultiHop-RAG pool actually contains;
* whether removing exact duplicates, or near-duplicates at a shingle-Jaccard threshold,
  changes MAP@10 for a plain cross-encoder order and for the coverage-plus-quota rule;
* with a paired bootstrap interval on every delta, because the effect is not assumed to be
  large enough to see.

    .venv/bin/python tools/multihop_pool_dedup.py
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
import evalkit  # noqa: E402
import numpy as np  # noqa: E402
from multihop_official import chunk_text  # noqa: E402
from multihop_rerank_experiment import (  # noqa: E402
    coverage_doc_quota_order,
    document_of,
    gold_matches_for_pool,
    metrics_from_matches,
)

from laya_rag.chunking import Chunk  # noqa: E402
from laya_rag.integrity import fingerprint, jaccard, shingles  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"
METRICS = ("Hits@10", "Hits@4", "MAP@10", "MRR@10")


def load_corpus(overlap: int) -> dict[str, str]:
    raw = json.loads((ROOT_BENCH / "corpus.json").read_text(encoding="utf-8"))
    corpus: dict[str, str] = {}
    for article in raw:
        for j, text in enumerate(chunk_text(article["title"] + " " + article["body"], 256, overlap)):
            corpus[f"{article['url']}#{j}"] = text
    return corpus


def as_chunks(pool: Sequence[str], corpus: Mapping[str, str]) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=doc_id, doc_id=doc_id.split("#")[0], path=doc_id.split("#")[0],
            text=corpus[doc_id], index=0, start_char=0, end_char=len(corpus[doc_id]),
        )
        for doc_id in pool
    ]


def dedup_exact(pool: Sequence[str], corpus: Mapping[str, str], order: Sequence[int]) -> list[int]:
    """Drop candidates whose text is byte-identical to a better-ranked one already kept."""
    seen: set[str] = set()
    kept: list[int] = []
    for index in order:
        key = fingerprint(corpus[pool[index]])
        if key in seen:
            continue
        seen.add(key)
        kept.append(index)
    return kept


def dedup_near(
    pool: Sequence[str],
    corpus: Mapping[str, str],
    order: Sequence[int],
    threshold: float,
    shingle_size: int,
) -> list[int]:
    """Drop candidates whose shingle Jaccard to a better-ranked kept candidate is >= threshold."""
    sketches = {index: shingles(corpus[pool[index]], shingle_size) for index in order}
    kept: list[int] = []
    kept_sketches: list[frozenset[str]] = []
    for index in order:
        sketch = sketches[index]
        if any(jaccard(sketch, other) >= threshold for other in kept_sketches):
            continue
        kept.append(index)
        kept_sketches.append(sketch)
    return kept


def rank_by_ce(pool: Sequence[str], scores: Sequence[float]) -> list[int]:
    return sorted(range(len(pool)), key=lambda i: (-float(scores[i]), i))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="multihop_pool_dedup.py")
    ap.add_argument("--ce-cache", default=str(ROOT_BENCH / "ce-scores.json"))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--near-threshold", type=float, default=0.85)
    ap.add_argument("--shingle-size", type=int, default=5)
    ap.add_argument("--coverage-lambda", type=float, default=0.7)
    ap.add_argument("--min-docs", type=int, default=3)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT_BENCH / "pool-dedup"))
    args = ap.parse_args(argv)

    cache = json.loads(Path(str(args.ce_cache)).read_text(encoding="utf-8"))
    overlap = int(cache["_settings"]["overlap"])
    corpus = load_corpus(overlap)
    atom_weight = cache["_settings"]["atom_weight"]

    per_query: dict[str, dict[str, dict[str, float]]] = {}
    prevalence = {
        "queries": 0,
        "with_exact_duplicate": 0,
        "with_near_duplicate": 0,
        "exact_redundant_chunks": 0,
        "near_redundant_chunks": 0,
        "pool_chunks": 0,
        "exact_in_top10": 0,
        "near_in_top10": 0,
    }

    def record(strategy: str, q: str, ranked: Sequence[int], matches, gold_count: int) -> None:
        metrics = metrics_from_matches(list(ranked), matches, gold_count, args.top_k)
        for metric in METRICS:
            per_query.setdefault(strategy, {}).setdefault(metric, {})[q] = metrics[metric]

    for q, payload in cache["queries"].items():
        pool = list(payload["pool"])
        scores = payload["ce"]
        atoms = payload["atoms"]
        matches = gold_matches_for_pool(corpus, pool, cache["golds"][q])
        gold_count = len(cache["golds"][q])
        ce_order = rank_by_ce(pool, scores)

        exact = dedup_exact(pool, corpus, ce_order)
        near = dedup_near(pool, corpus, ce_order, args.near_threshold, args.shingle_size)
        docs = [document_of(doc_id) for doc_id in pool]
        cov = coverage_doc_quota_order(
            scores, atoms, atom_weight, docs, args.top_k, args.coverage_lambda, args.min_docs
        )
        cov_exact = dedup_exact(pool, corpus, list(cov))
        cov_near = dedup_near(pool, corpus, list(cov), args.near_threshold, args.shingle_size)

        record("ce", q, ce_order, matches, gold_count)
        record("ce-dedup-exact", q, exact, matches, gold_count)
        record("ce-dedup-near", q, near, matches, gold_count)
        record("covquota", q, list(cov), matches, gold_count)
        record("covquota-dedup-exact", q, cov_exact, matches, gold_count)
        record("covquota-dedup-near", q, cov_near, matches, gold_count)

        prevalence["queries"] += 1
        prevalence["pool_chunks"] += len(pool)
        exact_removed = len(ce_order) - len(exact)
        near_removed = len(ce_order) - len(near)
        prevalence["exact_redundant_chunks"] += exact_removed
        prevalence["near_redundant_chunks"] += near_removed
        if exact_removed:
            prevalence["with_exact_duplicate"] += 1
        if near_removed:
            prevalence["with_near_duplicate"] += 1
        top10 = ce_order[: args.top_k]
        prevalence["exact_in_top10"] += len(top10) - len([i for i in top10 if i in set(exact)])
        prevalence["near_in_top10"] += len(top10) - len([i for i in top10 if i in set(near)])

    summary = {
        strategy: {metric: round(float(np.mean(list(rows.values()))), 4) for metric, rows in block.items()}
        for strategy, block in per_query.items()
    }
    comparisons: dict[str, dict[str, Any]] = {}
    for base, candidate in (
        ("ce", "ce-dedup-exact"),
        ("ce", "ce-dedup-near"),
        ("covquota", "covquota-dedup-exact"),
        ("covquota", "covquota-dedup-near"),
    ):
        for metric in ("MAP@10", "Hits@4", "Hits@10"):
            comparisons[f"{candidate} vs {base} {metric}"] = {
                "bootstrap": evalkit.paired_bootstrap(
                    per_query[base][metric], per_query[candidate][metric], samples=args.samples
                ),
                "permutation": evalkit.permutation_test(
                    per_query[base][metric], per_query[candidate][metric], samples=args.samples
                ),
            }

    result = {
        "_settings": {
            "ce_cache": Path(str(args.ce_cache)).name,
            "queries": len(cache["queries"]),
            "top_k": args.top_k,
            "near_threshold": args.near_threshold,
            "shingle_size": args.shingle_size,
            "coverage_lambda": args.coverage_lambda,
        },
        "prevalence": prevalence,
        "summary": summary,
        "comparisons": comparisons,
        "per_query": per_query,
    }
    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), result)
    evalkit.write_text(Path(str(out) + ".md"), render_markdown(result))
    print(render_markdown(result))
    print("wrote", str(out) + ".json and .md")
    return 0


def render_markdown(result: Mapping[str, Any]) -> str:
    settings = result["_settings"]
    prevalence = result["prevalence"]
    queries = max(prevalence["queries"], 1)
    lines = [
        "# Duplication in the candidate pool, and what removing it changes",
        "",
        f"- {settings['queries']} queries, top-k {settings['top_k']}, near-duplicate threshold "
        f"{settings['near_threshold']} (shingle {settings['shingle_size']})",
        "",
        "## How much duplication the pool holds",
        "",
        f"- pools with an exact duplicate: **{prevalence['with_exact_duplicate']} / {queries}** "
        f"({prevalence['with_exact_duplicate'] / queries:.1%})",
        f"- pools with a near-duplicate: **{prevalence['with_near_duplicate']} / {queries}** "
        f"({prevalence['with_near_duplicate'] / queries:.1%})",
        f"- redundant chunks across the pools: exact **{prevalence['exact_redundant_chunks']}**, "
        f"near **{prevalence['near_redundant_chunks']}** of {prevalence['pool_chunks']}",
        f"- duplicate slots inside the cross-encoder top-10: exact "
        f"**{prevalence['exact_in_top10']}**, near **{prevalence['near_in_top10']}**",
        "",
        "## Metrics",
        "",
        "| strategy | Hits@10 | Hits@4 | MAP@10 | MRR@10 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, row in result["summary"].items():
        lines.append(
            f"| {name} | {row['Hits@10']:.4f} | {row['Hits@4']:.4f} | "
            f"{row['MAP@10']:.4f} | {row['MRR@10']:.4f} |"
        )
    lines += ["", "## Paired comparisons (bootstrap 95% CI, permutation p)", ""]
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
