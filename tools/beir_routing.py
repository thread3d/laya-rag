"""Per-corpus channel selection: what a single global configuration leaves on the table.

Fusion helps on four of the five BEIR datasets but loses on ArguAna, where the dense
channel alone is worth +0.028 nDCG@10 (0.596 vs 0.568). A single deployable configuration
therefore pays for ArguAna in every dataset's average.

This asks whether that is recoverable *without* the test qrels: split each dataset's queries
in half, pick the best channel on one half, and score it on the other. A router calibrated
this way is compared against always-fusion on the same held-out halves, with a paired
bootstrap over queries.

    .venv/bin/python tools/beir_routing.py --report eval/beir/five-small-fixed.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import evalkit  # noqa: E402

METRIC = "nDCG@10"


def load_dataset_scores(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """``{dataset: {retriever: {query_id: score}}}`` from one report."""
    report = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, dict[str, float]]] = {}
    for dataset, block in report["datasets"].items():
        per_query = block.get("per_query_nDCG@10") or {}
        out[dataset] = {
            name: {q: float(v) for q, v in scores.items()}
            for name, scores in per_query.items()
        }
    return out


def mean(scores: dict[str, float]) -> float:
    return float(np.mean(list(scores.values()))) if scores else float("nan")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="beir_routing.py")
    ap.add_argument("--report", default=str(ROOT / "eval" / "beir" / "five-small-fixed.json"))
    ap.add_argument("--extra", default="",
                    help="optional second report merged per dataset (e.g. the ArguAna text-only run)")
    ap.add_argument("--rows", default="bm25-engine,biencoder,hybrid-biencoder",
                    help="candidate channels the router may choose between")
    ap.add_argument("--global-row", default="hybrid-biencoder",
                    help="the single configuration compared against routing")
    ap.add_argument("--splits", type=int, default=500)
    ap.add_argument("--bootstrap-samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=str(ROOT / "eval" / "beir" / "routing"))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from heavylock import heavy_job_guard

    heavy_job_guard("beir_routing.py")
    rows = [r.strip() for r in args.rows.split(",") if r.strip()]
    scores = load_dataset_scores(Path(args.report))
    if args.extra:
        extra = load_dataset_scores(Path(args.extra))
        for dataset, block in extra.items():
            scores.setdefault(dataset, {}).update(block)

    datasets = list(scores)
    per_dataset_oracle: dict[str, dict[str, float]] = {}
    for dataset in datasets:
        means = {row: mean(scores[dataset][row]) for row in rows if row in scores[dataset]}
        per_dataset_oracle[dataset] = means

    # Fixed global configurations, for the standing table.
    fixed = {}
    for row in rows:
        values = [mean(scores[d][row]) for d in datasets if row in scores[d]]
        fixed[row] = round(float(np.mean(values)), 4)
    oracle_values = [max(per_dataset_oracle[d].values()) for d in datasets]
    oracle = round(float(np.mean(oracle_values)), 4)

    # Held-out routing: choose per dataset on a calibration half, score the other half.
    # Both directions are used so every query is scored out of sample. The headline is the
    # macro mean (per-dataset mean, then average across datasets — the README convention),
    # and the interval is a paired bootstrap over queries *within* each dataset.
    rng = random.Random(args.seed)
    routed_by_dataset: dict[str, dict[str, float]] = {d: {} for d in datasets}
    global_by_dataset: dict[str, dict[str, float]] = {d: {} for d in datasets}
    chosen_counts: dict[tuple[str, str], int] = {}
    for _ in range(args.splits):
        for dataset in datasets:
            available = [row for row in rows if row in scores[dataset]]
            shared = sorted(set.intersection(*(set(scores[dataset][row]) for row in available)))
            if not shared:
                continue
            rng.shuffle(shared)
            half = len(shared) // 2
            for calib, test in ((shared[:half], shared[half:]), (shared[half:], shared[:half])):
                calib_means = {row: mean({q: scores[dataset][row][q] for q in calib}) for row in available}
                pick = max(calib_means, key=calib_means.get)
                chosen_counts[(dataset, pick)] = chosen_counts.get((dataset, pick), 0) + 1
                for q in test:
                    routed_by_dataset[dataset][q] = scores[dataset][pick][q]
                    global_by_dataset[dataset][q] = scores[dataset][args.global_row][q]

    per_dataset_routed = {d: mean(routed_by_dataset[d]) for d in datasets}
    per_dataset_global = {d: mean(global_by_dataset[d]) for d in datasets}
    routed_mean = round(float(np.mean(list(per_dataset_routed.values()))), 4)
    global_mean = round(float(np.mean(list(per_dataset_global.values()))), 4)
    delta = round(routed_mean - global_mean, 4)

    # Paired hierarchical bootstrap: resample queries within each dataset, keep the pairing
    # between routed and global, then average the per-dataset deltas equally.
    rng_np = np.random.default_rng(args.seed)
    per_dataset_deltas = [
        np.array([routed_by_dataset[d][q] - global_by_dataset[d][q] for q in sorted(routed_by_dataset[d])])
        for d in datasets
    ]
    draws = []
    for _ in range(args.bootstrap_samples):
        dataset_means = []
        for deltas in per_dataset_deltas:
            idx = rng_np.integers(0, len(deltas), len(deltas))
            dataset_means.append(float(deltas[idx].mean()))
        draws.append(float(np.mean(dataset_means)))
    draws_arr = np.sort(np.asarray(draws))
    comparison = {
        "queries": int(sum(len(v) for v in routed_by_dataset.values())),
        "delta": delta,
        "ci_low": round(float(np.percentile(draws_arr, 2.5)), 4),
        "ci_high": round(float(np.percentile(draws_arr, 97.5)), 4),
        "significant": bool(
            float(np.percentile(draws_arr, 2.5)) > 0.0 or float(np.percentile(draws_arr, 97.5)) < 0.0
        ),
        "samples": args.bootstrap_samples,
    }

    report = {
        "rows": rows,
        "global_row": args.global_row,
        "datasets": datasets,
        "per_dataset_means": {
            d: {k: round(v, 4) for k, v in per_dataset_oracle[d].items()} for d in datasets
        },
        "fixed_global_means": fixed,
        "oracle_best_per_dataset": oracle,
        "held_out_routing": {
            "routed_mean": routed_mean,
            "global_mean": global_mean,
            "delta": delta,
            "per_dataset": {
                d: {"routed": round(per_dataset_routed[d], 4), "global": round(per_dataset_global[d], 4)}
                for d in datasets
            },
            "paired_ci": comparison,
            "choice_distribution": {
                f"{dataset}/{row}": round(count / (2 * args.splits), 4)
                for (dataset, row), count in sorted(chosen_counts.items())
            },
        },
    }
    out = Path(args.out)
    evalkit.write_json(out.with_suffix(".json"), report)
    print("per-dataset nDCG@10 by channel:")
    for d in datasets:
        cells = "  ".join(f"{k}={v:.4f}" for k, v in sorted(per_dataset_oracle[d].items(), key=lambda kv: -kv[1]))
        print(f"  {d:10s} {cells}")
    print(f"\nfixed global: {fixed}")
    print(f"oracle best-per-dataset: {oracle}")
    print(f"held-out routing: {routed_mean} vs {global_mean} ({delta:+.4f}), "
          f"CI [{comparison['ci_low']}, {comparison['ci_high']}], significant={comparison['significant']}")
    print("router choices:", json.dumps(report["held_out_routing"]["choice_distribution"], indent=1))
    print("\nwrote", out.with_suffix(".json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
