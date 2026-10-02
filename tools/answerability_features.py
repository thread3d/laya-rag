"""Does feature engineering improve the answerability gate? An offline comparison.

The shipped gate is a logistic model over nine retrieval/evidence signals
(`laya_rag.answerability.FEATURE_NAMES`). A linear model over nine raw columns may miss
simple interactions the signals imply — a relevance *margin*, whether recall and
relevance agree, whether coverage multiplies a confident top score. This fits the same
`LogisticGate` on the raw nine and on the raw nine plus derived columns, over the same
stratified splits, and reports the held-out AUC and selective-prediction metrics for both.

It does not touch the shipped artefact; it answers whether extending `FEATURE_NAMES` is
worth the breaking format change, using only the recorded features.

    .venv/bin/python tools/answerability_features.py \
        --features eval/benchmarks/multihop-rag/gate-features.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import evalkit  # noqa: E402
import answerability_train as train  # noqa: E402
from laya_rag.answerability import FEATURE_NAMES, LogisticGate, evaluate_gate  # noqa: E402

#: ``name -> (function of the named base columns)``. Kept small and interpretable: each
#: column is a relationship the raw nine imply but a linear model cannot form itself.
DERIVED: dict[str, Any] = {
    "relevance_margin": lambda c: c["max_relevance"] - c["mean_relevance"],
    "relevance_spread": lambda c: c["max_relevance"] - c["min_relevance"],
    "recall_relevance_agreement": lambda c: c["mean_recall"] - c["mean_relevance"],
    "coverage_times_max_relevance": lambda c: c["query_term_coverage"] * c["max_relevance"],
    "distinct_documents": lambda c: c["distinct_document_ratio"] * c["passage_count"],
    "chars_per_passage": lambda c: c["log_evidence_chars"] - np.log1p(c["passage_count"]),
    "max_relevance_squared": lambda c: c["max_relevance"] ** 2,
    "mean_relevance_squared": lambda c: c["mean_relevance"] ** 2,
    "max_recall_times_max_relevance": lambda c: c["max_recall"] * c["max_relevance"],
}


def augment(base: np.ndarray) -> np.ndarray:
    """Append the :data:`DERIVED` columns to a base matrix in ``FEATURE_NAMES`` order.

    The derived columns are functions of the **nine score features**, so only those are
    named; a 12-wide base (with the evidence-shape features) still derives from its first
    nine columns.
    """
    base = np.asarray(base, dtype=np.float64)
    score_names = FEATURE_NAMES[:9]
    columns = {name: base[:, index] for index, name in enumerate(score_names)}
    extra = np.column_stack([function(columns) for function in DERIVED.values()])
    return np.column_stack([base, extra])


def load_extras(path: Path) -> np.ndarray:
    """The recorded ``extra`` columns, in ``EXTRA_NAMES`` order and record order.

    ``multihop_gate.py`` writes these beside the shipped nine, so they can be compared
    without re-running Laya. Record order matches ``answerability_train.load_records``:
    every ``null`` row, then every ``positive`` row.
    """
    from gate_extra_features import EXTRA_NAMES

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: list[list[float]] = []
    for group in ("null", "positive"):
        for row in payload.get(group) or []:
            recorded = row.get("extra") or {}
            rows.append([float(recorded.get(name, 0.0)) for name in EXTRA_NAMES])
    return np.asarray(rows, dtype=np.float64)


def selective(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    probabilities = [float(v) for v in scores]
    outcomes = [int(v) for v in labels]
    sweep = evalkit.ece_sweep(probabilities, outcomes)
    return {
        "auc": float(evaluate_gate(labels, scores, threshold=threshold)["auc"]),
        "balanced_accuracy": float(
            evaluate_gate(labels, scores, threshold=threshold)["balanced_accuracy"]
        ),
        "aurc": evalkit.aurc(probabilities, outcomes),
        "augrc": evalkit.augrc(probabilities, outcomes),
        "risk@80": evalkit.risk_at_coverage(probabilities, outcomes, 0.80),
        "ece": sweep["ece"],
    }


def evaluate_splits(X: np.ndarray, y: np.ndarray, splits: int, seed: int) -> list[dict[str, float]]:
    """One reading per stratified split, so two feature sets can be compared pairwise.

    Both arms use the same split seeds, so the per-split deltas are paired and a
    permutation test over them is meaningful.
    """
    readings: list[dict[str, float]] = []
    for offset in range(splits):
        train_idx, test_idx = train.stratified_split(y, seed + offset)
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        gate = LogisticGate().fit(X[train_idx], y[train_idx])
        threshold, _ = train.best_threshold(y[train_idx], gate.predict_proba(X[train_idx]))
        readings.append(selective(y[test_idx], gate.predict_proba(X[test_idx]), threshold))
    return readings


def evaluate(X: np.ndarray, y: np.ndarray, splits: int, seed: int) -> dict[str, dict[str, float]]:
    readings = evaluate_splits(X, y, splits, seed)
    keys = sorted(readings[0]) if readings else []
    return {key: train.summarize([reading[key] for reading in readings]) for key in keys}


def paired_across_splits(
    base_readings: Sequence[Mapping[str, float]],
    augmented_readings: Sequence[Mapping[str, float]],
    samples: int = 5000,
) -> dict[str, dict[str, float]]:
    """Paired bootstrap over the per-split readings of the two arms."""
    base = {str(i): row for i, row in enumerate(base_readings)}
    augmented = {str(i): row for i, row in enumerate(augmented_readings)}
    metrics = sorted(set(base_readings[0]) & set(augmented_readings[0])) if base_readings else []
    return {
        metric: evalkit.paired_bootstrap(
            {key: float(row[metric]) for key, row in base.items()},
            {key: float(row[metric]) for key, row in augmented.items()},
            samples=samples,
        )
        for metric in metrics
    }


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="answerability_features.py")
    ap.add_argument("--features", default=str(
        ROOT / "eval" / "benchmarks" / "multihop-rag" / "gate-features.json"
    ))
    ap.add_argument("--extras", action="store_true",
                    help="compare against the recorded evidence-shape extras from "
                         "gate_extra_features.py instead of the derived columns")
    ap.add_argument("--prefix", type=int, default=None,
                    help="base arm is the first N columns and augmented is every column; "
                         "use 9 to compare the nine score features against the full "
                         "production vector")
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=str(ROOT / "eval" / "answerability-features"))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    records = train.load_records(Path(args.features))
    X, y, _head = train.build_matrix(records)
    if args.prefix:
        if not 0 < args.prefix < X.shape[1]:
            raise SystemExit(f"--prefix must be inside 1..{X.shape[1] - 1}, got {args.prefix}")
        X_base = X[:, : args.prefix]
        X_aug = X
        column_names = list(FEATURE_NAMES[args.prefix :])
    elif args.extras:
        from gate_extra_features import EXTRA_NAMES

        X_base = X
        extras = load_extras(Path(args.features))
        X_aug = np.column_stack([X, extras])
        column_names = list(EXTRA_NAMES)
    else:
        X_base = X
        X_aug = augment(X)
        column_names = list(DERIVED)
    print(f"{len(records)} queries; {X_base.shape[1]} base features, {X_aug.shape[1]} augmented",
          flush=True)

    base_readings = evaluate_splits(X_base, y, args.splits, args.seed)
    augmented_readings = evaluate_splits(X_aug, y, args.splits, args.seed)
    keys = sorted(base_readings[0]) if base_readings else []
    base = {key: train.summarize([row[key] for row in base_readings]) for key in keys}
    augmented = {key: train.summarize([row[key] for row in augmented_readings]) for key in keys}
    deltas = {
        key: round(augmented[key]["mean"] - base[key]["mean"], 4)
        for key in base
        if key in augmented
    }
    paired = paired_across_splits(base_readings, augmented_readings)
    payload = {
        "_settings": vars(args),
        "base": base,
        "augmented": augmented,
        "derived_names": column_names,
        "delta": deltas,
        "paired": paired,
        "n": len(records),
    }
    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), payload)
    evalkit.write_text(Path(str(out) + ".md"), render_markdown(payload))
    print(render_markdown(payload))
    print("wrote", str(out) + ".json and .md")
    return 0


def render_markdown(payload: Mapping[str, Any]) -> str:
    base, aug = payload["base"], payload["augmented"]
    lines = [
        "# Answerability gate: raw vs extended features",
        "",
        f"- {payload['n']} labelled queries, {payload['_settings']['splits']} stratified splits",
        f"- added columns: {', '.join(payload['derived_names'])}",
        "",
        "| metric | raw nine | + added | delta | paired 95% CI | p |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    paired = payload.get("paired", {})
    for key in sorted(base):
        if key not in aug:
            continue
        row = paired.get(key) or {}
        ci = (f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]" if row else "—")
        p_value = f"{row.get('p_value', float('nan')):.4f}" if row else "—"
        lines.append(
            f"| {key} | {base[key]['mean']:.4f} [{base[key]['lo']:.4f}, {base[key]['hi']:.4f}] | "
            f"{aug[key]['mean']:.4f} [{aug[key]['lo']:.4f}, {aug[key]['hi']:.4f}] | "
            f"{payload['delta'][key]:+.4f} | {ci} | {p_value} |"
        )
    lines += [
        "",
        "Higher is better for `auc`/`balanced_accuracy`; lower is better for "
        "`aurc`/`augrc`/`risk@80`/`ece`. The CI and p are a paired bootstrap/permutation over "
        "the per-split readings, since both arms use the same splits.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
