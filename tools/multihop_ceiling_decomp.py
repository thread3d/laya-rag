"""Where is the multi-hop headroom: the scoring signal or the set-selection algorithm?

The corrected reorder ceiling (0.609 at depth 100) is an upper bound that uses gold labels, so
it cannot say which of two very different failures is binding:

* the **signal** — the scorer does not rank gold candidates highly enough, so no selector can
  reach them, or
* the **algorithm** — the scorer's list contains the right candidates but the rule that turns
  a ranked list into a *set* of ten cannot find them.

This separates them by restricting the oracle to the cross-encoder's own top-N. The oracle over
a *prefix* of the scorer's ranking is the best any selector could do if it were handed that
prefix, so the gap between it and the achieved score is **algorithm headroom**, and the gap
between the full-pool oracle and the prefix oracle is **signal headroom**.

    .venv/bin/python tools/multihop_ceiling_decomp.py

No model, no network: pure NumPy over the cached pool.
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

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"


def oracle_ceiling(matches: Sequence[set], gold_count: int, top_k: int = 10) -> float:
    """Best MAP@10 any ordering of these candidates could reach, gold labels assumed.

    Greedy by marginal new facts, divided by ``min(gold_count, top_k)`` — the same
    normalisation as ``official_metrics`` and ``metrics_from_matches``. Using the pool's own
    matched-fact count here instead of the query's true gold count inflates the ceiling on
    exactly the queries whose recall is failing; see ``multihop_coverage.pool_ceiling_map``.
    """
    sets = [set(m) for m in matches]
    if not sets or top_k <= 0:
        return 0.0
    found: set = set()
    picked: list[int] = []
    candidates = list(range(len(sets)))
    while candidates and len(picked) < top_k:
        best_index, best_new = None, -1
        for i in candidates:
            new = len(sets[i] - found)
            if new > best_new:
                best_index, best_new = i, new
        picked.append(best_index)
        found |= sets[best_index]
        candidates.remove(best_index)
    ap = 0.0
    found = set()
    for rank, i in enumerate(picked, start=1):
        newly = sets[i] - found
        ap += len(newly) / rank
        found |= newly
    return ap / min(max(int(gold_count), 1), top_k)


def restrict(matches: Sequence[set], order: Sequence[int], n: int) -> list[set]:
    """The match sets of the first ``n`` candidates of a ranking."""
    return [set(matches[i]) for i in order[:n]]


def load_corpus(overlap: int) -> dict[str, str]:
    raw = json.loads((ROOT_BENCH / "corpus.json").read_text(encoding="utf-8"))
    corpus: dict[str, str] = {}
    for article in raw:
        for j, text in enumerate(chunk_text(article["title"] + " " + article["body"], 256, overlap)):
            corpus[f"{article['url']}#{j}"] = text
    return corpus


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="multihop_ceiling_decomp.py")
    ap.add_argument("--ce-cache", default=str(ROOT_BENCH / "ce-scores-full.json"))
    ap.add_argument("--prefixes", default="5,10,20,50,100",
                    help="cross-encoder top-N prefixes to evaluate the oracle over")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--coverage-lambda", type=float, default=0.7)
    ap.add_argument("--min-docs", type=int, default=3)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT_BENCH / "ceiling-decomp"))
    args = ap.parse_args(argv)

    cache = json.loads(Path(args.ce_cache).read_text(encoding="utf-8"))
    corpus = load_corpus(int(cache["_settings"]["overlap"]))
    prefixes = [int(v) for v in str(args.prefixes).split(",") if v.strip()]

    rows: dict[str, dict[str, float]] = {str(n): {} for n in prefixes}
    rows["pool"] = {}
    achieved: dict[str, float] = {}
    achieved_covquota: dict[str, float] = {}
    ce_order_map: dict[str, list[int]] = {}

    for q, payload in cache["queries"].items():
        pool = list(payload["pool"])
        gold_count = len(cache["golds"][q])
        matches = gold_matches_for_pool(corpus, pool, cache["golds"][q])
        rows["pool"][q] = oracle_ceiling(matches, gold_count, args.top_k)
        order = sorted(range(len(pool)), key=lambda i: (-float(payload["ce"][i]), i))
        ce_order_map[q] = order
        for n in prefixes:
            rows[str(n)][q] = oracle_ceiling(restrict(matches, order, n), gold_count, args.top_k)
        achieved[q] = metrics_from_matches(order, matches, gold_count, args.top_k)["MAP@10"]
        atoms = payload["atoms"]
        docs = [document_of(c) for c in pool]
        cov = coverage_doc_quota_order(
            payload["ce"], atoms, cache["_settings"]["atom_weight"], docs,
            args.top_k, args.coverage_lambda, args.min_docs,
        )
        achieved_covquota[q] = metrics_from_matches(cov, matches, gold_count, args.top_k)["MAP@10"]

    summary = {
        "ce order (achieved)": round(float(np.mean(list(achieved.values()))), 4),
        f"covquota-{args.min_docs}-{args.coverage_lambda} (achieved)":
            round(float(np.mean(list(achieved_covquota.values()))), 4),
        "oracle over the full pool": round(float(np.mean(list(rows["pool"].values()))), 4),
    }
    for n in prefixes:
        summary[f"oracle over CE top-{n}"] = round(float(np.mean(list(rows[str(n)].values()))), 4)

    best_achieved = max(float(np.mean(list(achieved.values()))),
                        float(np.mean(list(achieved_covquota.values()))))
    full = summary["oracle over the full pool"]
    # Three distinct quantities, and collapsing them into one number hides the decision:
    #   * reordering the *same* candidate set the scorer already returns (prefix = top_k);
    #   * choosing a better *subset* out of a slightly deeper scorer list (2 * top_k);
    #   * what only a better *scorer* could reach (everything beyond that).
    # A tighter reference prefix makes the first look small and the last look huge, which is a
    # statement about the reference, not about the system.
    def at(n: int) -> float:
        key = f"oracle over CE top-{n}"
        if key in summary:
            return float(summary[key])
        # Fall back to the nearest measured prefix at or below n, so a caller who passes an odd
        # prefix list still gets a defined, stated reference.
        lower = [p for p in prefixes if p <= n]
        return float(summary[f"oracle over CE top-{max(lower, default=min(prefixes))}"])

    reference = args.top_k
    deeper = 2 * args.top_k
    decomposition = {
        "best_achieved": round(best_achieved, 4),
        "ordering_headroom": round(at(reference) - best_achieved, 4),
        "selection_headroom": round(at(deeper) - at(reference), 4),
        "signal_headroom": round(full - at(deeper), 4),
        "reference_prefix": reference,
        "deeper_prefix": deeper,
        "read": (
            "ordering_headroom is available from ordering the *same* candidate set better; "
            "selection_headroom from choosing a better subset of the scorer's slightly deeper "
            "list; signal_headroom is what only a better scorer reaches. The first two are "
            "algorithm work, the third is model work."
        ),
    }
    comparisons = {
        f"covquota vs ce order MAP@10": {
            "bootstrap": evalkit.paired_bootstrap(achieved, achieved_covquota, samples=args.samples),
            "permutation": evalkit.permutation_test(achieved, achieved_covquota, samples=args.samples),
        }
    }
    result = {
        "_settings": {
            "ce_cache": Path(args.ce_cache).name,
            "queries": len(cache["queries"]),
            "prefixes": prefixes,
            "top_k": args.top_k,
            "coverage_lambda": args.coverage_lambda,
            "min_docs": args.min_docs,
        },
        "summary": summary,
        "decomposition": decomposition,
        "comparisons": comparisons,
    }
    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), result)
    evalkit.write_text(Path(str(out) + ".md"), render_markdown(result))
    print(render_markdown(result))
    print("wrote", str(out) + ".json and .md")
    return 0


def render_markdown(result: Mapping[str, Any]) -> str:
    settings = result["_settings"]
    lines = [
        "# Signal or algorithm? Ceiling decomposition on MultiHop-RAG",
        "",
        f"- {settings['queries']} queries, top-k {settings['top_k']}, "
        f"coverage lambda {settings['coverage_lambda']}, document quota {settings['min_docs']}",
        f"- **pool**: the cross-encoder's own depth-100 list from the CE cache. "
        "`tools/multihop_diagnostics.py` reports the *fused* depth-100 pool separately "
        "(0.6088); the two artifacts are built differently, so the figures are complementary "
        "rather than interchangeable and must not be averaged or added",
        "",
        "| level | MAP@10 |",
        "| --- | --- |",
    ]
    for name, value in result["summary"].items():
        lines.append(f"| {name} | {value:.4f} |")
    d = result["decomposition"]
    lines += [
        "",
        "## Decomposition",
        "",
        f"- best achieved: **{d['best_achieved']:.4f}**",
        f"- **ordering headroom** (oracle over CE top-{d['reference_prefix']} − best achieved): "
        f"**{d['ordering_headroom']:+.4f}**",
        f"- **selection headroom** (oracle over CE top-{d['deeper_prefix']} − "
        f"oracle over CE top-{d['reference_prefix']}): **{d['selection_headroom']:+.4f}**",
        f"- **signal headroom** (full-pool oracle − oracle over CE top-{d['deeper_prefix']}): "
        f"**{d['signal_headroom']:+.4f}**",
        f"- {d['read']}",
        "",
        "## Paired comparisons (bootstrap 95% CI, permutation p)",
        "",
    ]
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
