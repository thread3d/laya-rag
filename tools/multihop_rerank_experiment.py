"""Second-stage selection experiments on cached MultiHop-RAG cross-encoder scores.

The published comparison this project lost was MAP@10, and the diagnostics
(`tools/multihop_diagnostics.py`) show the loss is selection, not recall. This sweep is
where the candidate second stages are compared on the *same* pool, so the only thing that
differs is the selection rule:

* ``fused`` — the first-stage order (the baseline);
* ``ce`` — full cross-encoder re-sort, the historical (and partly damaging) behaviour;
* ``blend-b`` — RRF over the first-stage order and the cross-encoder order, weight ``b``;
* ``guardrail-n`` — the cross-encoder order with the first ``n`` first-stage candidates
  pinned, so a second stage can improve the tail without demoting a confident rank-1;
* ``coverage-l`` — sentence-atom max-coverage selection using the cross-encoder score as
  the relevance term, so the top-10 spends its slots on distinct facts.

It reads the cache written by ``tools/multihop_ce_cache.py`` (no model, no network) and
reports paired bootstrap intervals and permutation p-values against both the fused
baseline and the full cross-encoder.

    .venv/bin/python tools/multihop_ce_cache.py --limit-queries 400
    .venv/bin/python tools/multihop_rerank_experiment.py
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
from multihop_official import official_metrics  # noqa: E402

from laya_rag.lexical import reciprocal_rank_fusion  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"
DEFAULT_CACHE = ROOT_BENCH / "ce-scores"
METRICS = ("Hits@10", "Hits@4", "MAP@10", "MRR@10")


# ── pure selection rules (unit-tested offline) ───────────────────────────

def ce_order(pool: Sequence[str], scores: Sequence[float]) -> list[str]:
    """Pool ids in descending cross-encoder score; ties break to the pool order."""
    return [pool[i] for i in sorted(range(len(pool)), key=lambda i: (-float(scores[i]), i))]


def blend_order(
    pool: Sequence[str], scores: Sequence[float], blend: float, rrf_k: int = 60
) -> list[str]:
    """RRF over the first-stage order and the cross-encoder order.

    ``blend`` in ``[0, 1]`` is the weight on the cross-encoder, so 0 keeps the first
    stage and 1 replaces it. This is the same shape ``beir.crossencoder_rankings`` uses
    when a second stage nudges rather than replaces.
    """
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"blend must be in [0, 1], got {blend!r}")
    if blend >= 1.0:
        return ce_order(pool, scores)
    if blend <= 0.0:
        return list(pool)
    merged = reciprocal_rank_fusion(
        [list(pool), ce_order(pool, scores)],
        k=rrf_k,
        limit=len(pool),
        weights=[1.0 - blend, blend],
    )
    return [doc_id for doc_id, _ in merged]


def guardrail_order(
    pool: Sequence[str], scores: Sequence[float], pin: int = 1
) -> list[str]:
    """The cross-encoder order with the first ``pin`` first-stage ids pinned in place.

    A second stage that may not touch a confident rank-1 keeps most of its upside on the
    tail while bounding the top-heavy flips that cost the most nDCG on BEIR.
    """
    order = list(range(len(pool)))
    order.sort(key=lambda i: (-float(scores[i]), i))
    for position in range(max(0, min(pin, len(pool)))):
        current = order.index(position)
        if current != position:
            order.pop(current)
            order.insert(position, position)
    return [pool[i] for i in order]


def coverage_order(
    scores: Sequence[float],
    atoms: Sequence[Sequence[int]],
    atom_weight: Sequence[float],
    top_k: int,
    lam: float = 0.5,
) -> list[int]:
    """Greedy max-coverage over sentence atoms, with the CE score as relevance.

    Mirrors ``multihop_coverage.facility_location_select`` but without the dense-gram
    redundancy term: the atom coverage already prevents re-selecting the same fact, and
    the gram matrix is not in the cache. Marginal gain is normalised by the best available
    gain at each step so it is comparable to the normalized relevance.
    """
    rel = [float(value) for value in scores]
    lo, hi = (min(rel), max(rel)) if rel else (0.0, 0.0)
    normalized = [(value - lo) / (hi - lo) if hi > lo else 0.0 for value in rel]
    selected: list[int] = []
    covered: set[int] = set()
    candidates = list(range(len(rel)))
    while candidates and len(selected) < top_k:
        gains = {
            i: float(sum(atom_weight[a] for a in atoms[i] if a not in covered))
            for i in candidates
        }
        scale = max(gains.values()) if gains else 0.0
        scale = scale if scale > 0.0 else 1.0
        best_index, best_score = None, -1e18
        for i in candidates:
            score = lam * normalized[i] + (1.0 - lam) * (gains[i] / scale)
            if score > best_score:
                best_index, best_score = i, score
        selected.append(best_index)
        covered.update(atoms[best_index])
        candidates.remove(best_index)
    return selected


def strategy_rankings(cache: Mapping[str, Any], top_k: int, rrf_k: int) -> dict[str, dict[str, list[str]]]:
    """Every strategy's top-k ids, keyed by strategy then query id."""
    atom_weight = cache["_settings"]["atom_weight"]
    names = ["fused", "ce", "blend-0.25", "blend-0.5", "blend-0.75",
             "guardrail-1", "guardrail-2",
             "coverage-0.3", "coverage-0.5", "coverage-0.7"]
    out: dict[str, dict[str, list[str]]] = {name: {} for name in names}
    for q, payload in cache["queries"].items():
        pool = payload["pool"]
        scores = payload["ce"]
        atoms = payload["atoms"]
        out["fused"][q] = list(pool[:top_k])
        out["ce"][q] = ce_order(pool, scores)[:top_k]
        for blend in (0.25, 0.5, 0.75):
            out[f"blend-{blend}"][q] = blend_order(pool, scores, blend, rrf_k)[:top_k]
        for pin in (1, 2):
            out[f"guardrail-{pin}"][q] = guardrail_order(pool, scores, pin)[:top_k]
        for lam in (0.3, 0.5, 0.7):
            indices = coverage_order(scores, atoms, atom_weight, top_k, lam)
            out[f"coverage-{lam}"][q] = [pool[i] for i in indices]
    return out


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="multihop_rerank_experiment.py")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT_BENCH / "rerank-experiment"))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cache_path = Path(str(args.cache) + ".json")
    if not cache_path.is_file():
        raise SystemExit(
            f"{cache_path} is missing; run tools/multihop_ce_cache.py first"
        )
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    golds = cache["golds"]
    order = list(cache["queries"])

    rankings = strategy_rankings(cache, args.top_k, args.rrf_k)
    per_query: dict[str, dict[str, dict[str, float]]] = {}
    summary: dict[str, dict[str, float]] = {}
    for name, ranking in rankings.items():
        rows: dict[str, dict[str, float]] = {metric: {} for metric in METRICS}
        for q in order:
            retrieved = ranking[q]
            gold = golds[q]
            metrics, _ = official_metrics([retrieved], [gold])
            for metric in METRICS:
                rows[metric][q] = metrics[metric]
        per_query[name] = rows
        summary[name] = {m: round(sum(rows[m].values()) / len(rows[m]), 4) for m in METRICS}

    comparisons: dict[str, dict[str, Any]] = {}
    for baseline in ("fused", "ce"):
        for name in rankings:
            if name == baseline:
                continue
            for metric in ("MAP@10", "Hits@4"):
                key = f"{name} vs {baseline} {metric}"
                comparisons[key] = {
                    "bootstrap": evalkit.paired_bootstrap(
                        per_query[baseline][metric], per_query[name][metric], samples=args.samples
                    ),
                    "permutation": evalkit.permutation_test(
                        per_query[baseline][metric], per_query[name][metric], samples=args.samples
                    ),
                }

    result = {
        "_settings": {**cache["_settings"], "top_k": args.top_k, "rrf_k": args.rrf_k},
        "summary": summary,
        "comparisons": comparisons,
        "per_query": {name: per_query[name]["MAP@10"] for name in rankings},
    }
    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), result)
    evalkit.write_text(Path(str(out) + ".md"), render_markdown(result))
    print(render_markdown(result))
    print("wrote", str(out) + ".json and .md")
    return 0


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Second-stage selection experiment (MultiHop-RAG subset)",
        "",
        f"- cache: {result['_settings']['limit_queries']} queries x "
        f"depth {result['_settings']['depth']}, reranker {result['_settings']['rerank_model']}",
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
