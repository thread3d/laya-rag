"""Permutation tests and Benjamini-Hochberg FDR over a saved comparison grid.

`tools/multihop_official.py` (and the other harnesses) persist a ``_per_query`` row per
system, which is what a paired test needs. This reads such a report, compares every
declared row against a baseline on each metric, and reports two things the harness did
not: an exact/approximate **paired permutation** p-value beside the bootstrap interval,
and **FDR control** across the whole family of cells.

Dozens of (system x metric) cells are tested in one table, so a per-cell 5% level is not
a 5% family error rate. Benjamini-Hochberg controls the expected proportion of false
discoveries instead. Both tests need exchangeable paired deltas, so a cell whose
configuration was chosen using the *same* queries is not valid evidence; declare the
family before looking.

    .venv/bin/python tools/eval_significance.py \
        --report eval/benchmarks/multihop-rag/official.json \
        --baseline "bge-small hybrid@ov200"
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

DEFAULT_REPORT = ROOT / "eval" / "benchmarks" / "multihop-rag" / "official.json"
DEFAULT_METRICS = ("MAP@10", "Hits@10", "Hits@4", "MRR@10")

#: Rows that are diagnostic rather than compared systems, excluded from a default family.
NON_SYSTEM_ROWS = {"_settings"}


def compared_rows(report: Mapping[str, Any]) -> list[str]:
    """Row names that carry a per-query series and can therefore be paired."""
    return [
        name for name, payload in report.items()
        if name not in NON_SYSTEM_ROWS
        and isinstance(payload, dict)
        and isinstance(payload.get("_per_query"), list)
        and payload["_per_query"]
    ]


def per_query_series(
    report: Mapping[str, Any], row: str, metric: str
) -> dict[str, float]:
    """``query_id -> metric`` for one row, keyed positionally.

    The harness persists per-query rows in one fixed evaluation order and without ids, so
    the index *is* the query identity; pairing by index is what keeps the test paired.
    """
    payload = report.get(row)
    if not isinstance(payload, dict) or not isinstance(payload.get("_per_query"), list):
        raise KeyError(f"{row!r} has no per-query series")
    series: dict[str, float] = {}
    for index, entry in enumerate(payload["_per_query"]):
        if metric in entry:
            series[str(index)] = float(entry[metric])
    if not series:
        raise KeyError(f"{row!r} has no {metric!r} values")
    return series


def choose_baseline(report: Mapping[str, Any], requested: str | None) -> str:
    """The requested row, or the fused hybrid, or the first comparable row."""
    rows = compared_rows(report)
    if not rows:
        raise SystemExit("the report has no per-query rows to compare")
    if requested:
        if requested not in rows:
            raise SystemExit(f"baseline {requested!r} is not a comparable row; have {rows}")
        return requested
    for candidate in rows:
        if "hybrid" in candidate and "rerank" not in candidate:
            return candidate
    return rows[0]


def comparison_cells(
    report: Mapping[str, Any],
    baseline: str,
    family: Sequence[str],
    metrics: Sequence[str],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """One paired permutation test + bootstrap interval per (row, metric) cell."""
    base_series = {metric: per_query_series(report, baseline, metric) for metric in metrics}
    cells: list[dict[str, Any]] = []
    for row in family:
        for metric in metrics:
            candidate = per_query_series(report, row, metric)
            permutation = evalkit.permutation_test(base_series[metric], candidate, samples=samples, seed=seed)
            bootstrap = evalkit.paired_bootstrap(base_series[metric], candidate, samples=samples, seed=seed)
            cells.append({
                "row": row,
                "metric": metric,
                "delta": permutation["delta"],
                "p_value": permutation["p_value"],
                "method": permutation["method"],
                "ci_low": bootstrap["ci_low"],
                "ci_high": bootstrap["ci_high"],
                "bootstrap_significant": bootstrap["significant"],
                "queries": permutation["queries"],
            })
    return cells


def apply_fdr(cells: Sequence[dict[str, Any]], q: float) -> dict[str, Any]:
    """Attach BH-adjusted p-values and the rejected set to the cells in place."""
    pvalues = {f"{cell['row']}|{cell['metric']}": cell["p_value"] for cell in cells}
    adjusted = evalkit.benjamini_hochberg(pvalues, q=q)
    for cell in cells:
        key = f"{cell['row']}|{cell['metric']}"
        cell["p_adjusted"] = adjusted["adjusted"][key]
        cell["rejected_fdr"] = key in adjusted["rejected"]
    return adjusted


def summarize(cells: Sequence[dict[str, Any]], adjusted: Mapping[str, Any]) -> dict[str, Any]:
    before = sum(1 for cell in cells if cell["p_value"] <= adjusted["q"])
    after = sum(1 for cell in cells if cell["rejected_fdr"])
    return {
        "cells": len(cells),
        "significant_uncorrected": before,
        "significant_fdr": after,
        "q": adjusted["q"],
        "false_discoveries_expected_uncorrected": round(before * adjusted["q"], 2),
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Permutation tests and FDR over a comparison grid",
        "",
        f"- report: `{result['report']}`",
        f"- baseline: `{result['baseline']}`",
        f"- family: **{result['summary']['cells']}** cells across "
        f"{len(result['family'])} rows",
        f"- significant uncorrected (p <= {result['summary']['q']}): "
        f"**{result['summary']['significant_uncorrected']}**",
        f"- significant after BH-FDR: **{result['summary']['significant_fdr']}**",
        "",
        "| row | metric | delta | 95% CI | perm p | BH p | FDR |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in sorted(result["cells"], key=lambda c: (c["p_adjusted"], c["row"], c["metric"])):
        lines.append(
            f"| {cell['row']} | {cell['metric']} | {cell['delta']:+.4f} | "
            f"[{cell['ci_low']:+.4f}, {cell['ci_high']:+.4f}] | {cell['p_value']:.4f} | "
            f"{cell['p_adjusted']:.4f} | {'yes' if cell['rejected_fdr'] else 'no'} |"
        )
    lines += [
        "",
        "`perm p` is the paired sign-flip permutation test; `BH p` is the Benjamini-Hochberg",
        "adjusted value across the declared family; `FDR` is rejected at the stated q.",
        "The CI is the paired bootstrap over queries, reported beside the permutation test",
        "because the two answer complementary questions.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="eval_significance.py")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--baseline", default=None, help="row to compare against")
    parser.add_argument("--family", default=None,
                        help="comma list of rows; default: every comparable row but the baseline")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--q", type=float, default=0.05, help="FDR level")
    parser.add_argument("--samples", type=int, default=2000,
                        help="permutation/bootstrap draws (exact below 12 queries)")
    parser.add_argument("--seed", type=int, default=20210617)
    parser.add_argument("--out", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    baseline = choose_baseline(report, args.baseline)
    rows = compared_rows(report)
    if args.family:
        family = [r.strip() for r in str(args.family).split(",") if r.strip()]
        missing = [r for r in family if r not in rows]
        if missing:
            raise SystemExit(f"family rows not in the report: {missing}")
    else:
        family = [r for r in rows if r != baseline]

    print(f"baseline {baseline!r}; {len(family)} rows x {len(metrics)} metrics", flush=True)
    cells = comparison_cells(
        report, baseline, family, metrics, samples=args.samples, seed=args.seed
    )
    adjusted = apply_fdr(cells, args.q)
    result = {
        "report": str(report_path),
        "baseline": baseline,
        "family": family,
        "metrics": metrics,
        "samples": args.samples,
        "summary": summarize(cells, adjusted),
        "cells": cells,
    }
    out = Path(args.out) if args.out else report_path.with_name(report_path.stem + "-significance")
    evalkit.write_json(Path(str(out) + ".json"), result)
    evalkit.write_text(Path(str(out) + ".md"), render_markdown(result))
    print(render_markdown(result))
    print("wrote", str(out) + ".json and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
