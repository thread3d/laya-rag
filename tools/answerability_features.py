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
    """Append the :data:`DERIVED` columns to a base matrix in ``FEATURE_NAMES`` order."""
    base = np.asarray(base, dtype=np.float64)
    columns = {name: base[:, index] for index, name in enumerate(FEATURE_NAMES)}
    extra = np.column_stack([function(columns) for function in DERIVED.values()])
    return np.column_stack([base, extra])


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


def evaluate(X: np.ndarray, y: np.ndarray, splits: int, seed: int) -> dict[str, dict[str, float]]:
    readings: list[dict[str, float]] = []
    for offset in range(splits):
        train_idx, test_idx = train.stratified_split(y, seed + offset)
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        gate = LogisticGate().fit(X[train_idx], y[train_idx])
        threshold, _ = train.best_threshold(y[train_idx], gate.predict_proba(X[train_idx]))
        readings.append(selective(y[test_idx], gate.predict_proba(X[test_idx]), threshold))
    keys = sorted(readings[0]) if readings else []
    return {key: train.summarize([reading[key] for reading in readings]) for key in keys}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="answerability_features.py")
    ap.add_argument("--features", default=str(
        ROOT / "eval" / "benchmarks" / "multihop-rag" / "gate-features.json"
    ))
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=str(ROOT / "eval" / "answerability-features"))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    records = train.load_records(Path(args.features))
    X, y, _head = train.build_matrix(records)
    X_aug = augment(X)
    print(f"{len(records)} queries; {X.shape[1]} base features, {X_aug.shape[1]} augmented",
          flush=True)

    base = evaluate(X, y, args.splits, args.seed)
    augmented = evaluate(X_aug, y, args.splits, args.seed)
    deltas = {
        key: round(augmented[key]["mean"] - base[key]["mean"], 4)
        for key in base
        if key in augmented
    }
    payload = {
        "_settings": vars(args),
        "base": base,
        "augmented": augmented,
        "derived_names": list(DERIVED),
        "delta": deltas,
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
        "# Answerability gate: raw vs engineered features",
        "",
        f"- {payload['n']} labelled queries, {payload['_settings']['splits']} stratified splits",
        f"- derived columns: {', '.join(payload['derived_names'])}",
        "",
        "| metric | raw nine | + derived | delta |",
        "| --- | --- | --- | --- |",
    ]
    for key in sorted(base):
        if key not in aug:
            continue
        lines.append(
            f"| {key} | {base[key]['mean']:.4f} [{base[key]['lo']:.4f}, {base[key]['hi']:.4f}] | "
            f"{aug[key]['mean']:.4f} [{aug[key]['lo']:.4f}, {aug[key]['hi']:.4f}] | "
            f"{payload['delta'][key]:+.4f} |"
        )
    lines += [
        "",
        "Higher is better for `auc`/`balanced_accuracy`; lower is better for "
        "`aurc`/`augrc`/`risk@80`/`ece`.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
