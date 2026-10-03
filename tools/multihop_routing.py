"""Per-question-type routing of the selection rule, under a nested held-out protocol.

MultiHop-RAG labels every query `inference_query`, `comparison_query` or `temporal_query`,
and the selection rule's hyperparameters (the coverage weight and the document quota) are
currently single global constants chosen on a held-out half. A global constant is the right
answer only if the best rule is the same for every question type; if it is not, per-type
routing is free accuracy.

The discipline matters more than the idea, because selecting a rule per type from ~25
candidates doubles the ways to overfit. This uses the two-direction nested protocol the
coverage sweep already uses:

* split the queries in half, seeded;
* on one half, pick the best strategy **per question type**, and score those types' queries
  in the other half with the pick;
* repeat with the halves reversed, so every query is scored by a rule chosen without it;
* report the pooled out-of-sample score against (a) the single globally-selected strategy,
  chosen the same way on the same halves, and (b) the fixed shipped rule.

It also reports each type's picks in both directions, because a route that chooses
differently on each half is not a route.

    .venv/bin/python tools/multihop_routing.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import evalkit  # noqa: E402

ROOT_BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"


def question_types() -> dict[str, str]:
    """``{cache query id: question_type}`` — the id is the record index, as everywhere else."""
    items = json.loads((ROOT_BENCH / "MultiHopRAG.json").read_text(encoding="utf-8"))
    return {str(i): x.get("question_type") or "unknown" for i, x in enumerate(items)}


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def route_by_type(
    per_query: Mapping[str, Mapping[str, float]],
    types: Mapping[str, str],
    *,
    seed: int = 17,
) -> dict[str, Any]:
    """Two-direction nested per-type routing over the strategies a report already holds.

    ``per_query`` is ``{strategy: {query_id: metric}}``. Returns the pooled out-of-sample
    per-query scores for the routed rule and for the globally-selected one, plus the picks.
    """
    strategies = [name for name, rows in per_query.items() if rows]
    if not strategies:
        return {"routed": {}, "global": {}, "picks": {}, "strategies": []}
    shared = sorted(set.intersection(*(set(per_query[name]) for name in strategies)))
    order = list(shared)
    random.Random(seed).shuffle(order)
    half = len(order) // 2
    halves = (order[:half], order[half:])

    routed: dict[str, float] = {}
    chosen_global: dict[str, float] = {}
    picks: dict[str, list[str]] = {}
    global_picks: list[str] = []

    for calib, test in (halves, halves[::-1]):
        calib_types = {types.get(q, "unknown") for q in calib}
        for qtype in sorted(calib_types):
            calib_qs = [q for q in calib if types.get(q, "unknown") == qtype]
            test_qs = [q for q in test if types.get(q, "unknown") == qtype]
            if not calib_qs or not test_qs:
                continue
            scores = {
                name: mean([per_query[name][q] for q in calib_qs]) for name in strategies
            }
            best = max(scores, key=lambda name: (scores[name], name))
            picks.setdefault(qtype, []).append(best)
            for q in test_qs:
                routed[q] = float(per_query[best][q])
        global_scores = {name: mean([per_query[name][q] for q in calib]) for name in strategies}
        best_global = max(global_scores, key=lambda name: (global_scores[name], name))
        global_picks.append(best_global)
        for q in test:
            chosen_global[q] = float(per_query[best_global][q])

    return {
        "routed": routed,
        "global": chosen_global,
        "picks": picks,
        "global_picks": global_picks,
        "strategies": strategies,
    }


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="multihop_routing.py")
    ap.add_argument("--report", default=str(ROOT_BENCH / "rerank-experiment-full.json"),
                    help="a report from tools/multihop_rerank_experiment.py")
    ap.add_argument("--fixed", default="covquota-3-0.7",
                    help="the shipped rule to compare against, when the report has it")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT_BENCH / "routing"))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    per_query = report.get("per_query") or {}
    if not per_query:
        raise SystemExit(f"{args.report} has no per_query scores")
    types = question_types()
    result = route_by_type(per_query, types, seed=args.seed)
    routed, global_selected = result["routed"], result["global"]
    if not routed:
        raise SystemExit("no queries were routed; check that the report and query ids agree")

    counts = Counter(types.get(q, "unknown") for q in routed)
    summary = {
        "routed_MAP@10": round(mean(list(routed.values())), 4),
        "globally_selected_MAP@10": round(mean(list(global_selected.values())), 4),
    }
    if args.fixed in per_query:
        fixed_rows = {q: float(per_query[args.fixed][q]) for q in routed}
        summary[f"fixed_{args.fixed}_MAP@10"] = round(mean(list(fixed_rows.values())), 4)
    else:
        fixed_rows = {}
    comparisons: dict[str, Any] = {}
    for label, other in (
        ("global", global_selected),
        ("fixed", fixed_rows),
    ):
        if not other:
            continue
        comparisons[f"routed vs {label} MAP@10"] = {
            "bootstrap": evalkit.paired_bootstrap(other, routed, samples=args.samples),
            "permutation": evalkit.permutation_test(other, routed, samples=args.samples),
        }
        for qtype in sorted(counts):
            qs = [q for q in routed if types.get(q, "unknown") == qtype]
            if len(qs) < 30:
                continue
            comparisons[f"routed vs {label} MAP@10 ({qtype})"] = {
                "bootstrap": evalkit.paired_bootstrap(
                    {q: other[q] for q in qs}, {q: routed[q] for q in qs}, samples=args.samples
                ),
                "permutation": evalkit.permutation_test(
                    {q: other[q] for q in qs}, {q: routed[q] for q in qs}, samples=args.samples
                ),
            }

    payload = {
        "_settings": {
            "report": Path(args.report).name,
            "fixed": args.fixed,
            "seed": args.seed,
            "queries": len(routed),
        },
        "summary": summary,
        "type_counts": dict(counts),
        "picks_by_type": {k: v for k, v in sorted(result["picks"].items())},
        "global_picks": result["global_picks"],
        "comparisons": comparisons,
    }
    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), payload)
    evalkit.write_text(Path(str(out) + ".md"), render_markdown(payload))
    print(render_markdown(payload))
    print("wrote", str(out) + ".json and .md")
    return 0


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Per-question-type routing (two-direction nested selection)",
        "",
        f"- {payload['_settings']['queries']} queries from "
        f"`{payload['_settings']['report']}`, seed {payload['_settings']['seed']}",
        "",
        "| rule | out-of-sample MAP@10 |",
        "| --- | --- |",
    ]
    for name, value in payload["summary"].items():
        lines.append(f"| {name} | {value:.4f} |")
    lines += ["", "## Which strategy each type picked, per direction", ""]
    for qtype, picks in payload["picks_by_type"].items():
        lines.append(f"- `{qtype}` ({payload['type_counts'].get(qtype, 0)} queries): {picks}")
    lines.append(f"- globally selected, per direction: {payload['global_picks']}")
    lines += ["", "## Paired comparisons (bootstrap 95% CI, permutation p)", ""]
    for key, value in sorted(payload["comparisons"].items()):
        boot, perm = value["bootstrap"], value["permutation"]
        lines.append(
            f"- `{key}`: {boot['delta']:+.4f} [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] "
            f"p={perm['p_value']:.4f} ({'sig' if boot['significant'] else 'ns'})"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
