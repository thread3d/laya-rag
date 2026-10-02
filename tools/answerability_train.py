"""Fit and evaluate the feature-based answerability gate on a labelled run.

`tools/multihop_gate.py` records, per query, Laya's scalar `p` **and** the retrieval
signals the engine already produced (`max_relevance`, `mean_recall`, source diversity,
query-term coverage, ...). This script answers the question the recalibration test left
open: when the scalar ranks the classes at chance, is there a *combination* of the other
signals that does not?

It fits `laya_rag.answerability.LogisticGate` on a calibration half and scores the held-out
half, repeated across many stratified splits, and reports the gate against the scalar
baseline on the same queries. The final gate is fit on all the data and persisted for
`LAYA_RAG_ANSWERABILITY_GATE_PATH`.

    .venv/bin/python tools/answerability_train.py \
        --features eval/benchmarks/multihop-rag/gate.json \
        --out eval/benchmarks/multihop-rag/answerability-gate.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laya_rag.answerability import (  # noqa: E402
    FEATURE_NAMES,
    LogisticGate,
    evaluate_gate,
    roc_auc,
)


def load_records(path: Path) -> list[dict[str, Any]]:
    """Flatten the gate report's ``null``/``positive`` groups into one labelled list."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for group, default_label in (("null", 0), ("positive", 1)):
        for row in payload.get(group, []) or []:
            if "features" not in row:
                raise ValueError(
                    f"{path} has no recorded features; re-run tools/multihop_gate.py "
                    "with this build so the gate can be trained"
                )
            records.append({**row, "label": int(row.get("label", default_label))})
    if not records:
        raise ValueError(f"{path} contains no records")
    return records


def build_matrix(records: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X, y, head_scores)`` in :data:`FEATURE_NAMES` order."""
    X = np.asarray(
        [[float(row["features"].get(name, 0.0)) for name in FEATURE_NAMES] for row in records],
        dtype=np.float64,
    )
    y = np.asarray([int(row["label"]) for row in records], dtype=np.float64)
    head = np.asarray([float(row.get("p") or 0.0) for row in records], dtype=np.float64)
    return X, y, head


def best_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """The threshold maximising balanced accuracy, ties nearest 0.5."""
    candidates = sorted({float(v) for v in scores})
    candidates += [i / 200.0 for i in range(201)]
    best_value, best_threshold = -1.0, 0.5
    for threshold in candidates:
        predictions = scores >= threshold
        positive = y == 1
        negative = ~positive
        tpr = float(np.mean(predictions[positive])) if positive.any() else 0.0
        tnr = float(np.mean(~predictions[negative])) if negative.any() else 0.0
        value = 0.5 * (tpr + tnr)
        if value > best_value + 1e-12 or (
            abs(value - best_value) <= 1e-12 and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_value, best_threshold = value, threshold
    return best_threshold, best_value


def stratified_split(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic 50/50 split keeping the class ratio in both halves."""
    order = list(range(len(y)))
    random.Random(seed).shuffle(order)
    halves: list[list[int]] = [[], []]
    for label in (0, 1):
        indices = [index for index in order if int(y[index]) == label]
        for position, index in enumerate(indices):
            halves[position % 2].append(index)
    return np.asarray(sorted(halves[0])), np.asarray(sorted(halves[1]))


def evaluate_split(
    X: np.ndarray, y: np.ndarray, head: np.ndarray, seed: int
) -> dict[str, float]:
    """Fit on one half, score the other, and compare against the scalar baseline."""
    train, test = stratified_split(y, seed)
    if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
        return {}
    gate = LogisticGate().fit(X[train], y[train])
    gate_scores = gate.predict_proba(X[test])
    threshold, _ = best_threshold(y[train], gate.predict_proba(X[train]))
    gate_report = evaluate_gate(y[test], gate_scores, threshold=threshold)
    head_report = evaluate_gate(y[test], head[test], threshold=0.5)
    return {
        "gate_auc": float(gate_report["auc"]),
        "gate_balanced_accuracy": float(gate_report["balanced_accuracy"]),
        "head_auc": float(head_report["auc"]),
        "head_balanced_accuracy": float(head_report["balanced_accuracy"]),
        "threshold": float(threshold),
    }


def summarize(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(array.mean()), 4),
        "lo": round(float(np.percentile(array, 2.5)), 4),
        "hi": round(float(np.percentile(array, 97.5)), 4),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="answerability_train.py")
    parser.add_argument(
        "--features", default=str(ROOT / "eval" / "benchmarks" / "multihop-rag" / "gate.json")
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "eval" / "benchmarks" / "multihop-rag" / "answerability-gate.json"),
    )
    parser.add_argument("--splits", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = load_records(Path(args.features))
    X, y, head = build_matrix(records)
    print(f"{len(records)} labelled queries, {X.shape[1]} features", flush=True)

    readings = [
        evaluate_split(X, y, head, seed=args.seed + offset) for offset in range(args.splits)
    ]
    readings = [reading for reading in readings if reading]
    reached = ~np.isclose(head, 0.0)
    report = {
        "n": len(records),
        "class_counts": {str(int(label)): int((y == label).sum()) for label in np.unique(y)},
        "features": list(FEATURE_NAMES),
        "held_out": {
            key: summarize([reading[key] for reading in readings])
            for key in (
                "gate_auc",
                "gate_balanced_accuracy",
                "head_auc",
                "head_balanced_accuracy",
                "threshold",
            )
        },
        "full_data": {
            "head_auc": round(float(roc_auc(y, head)), 4),
            "head_auc_reached": round(
                float(roc_auc(y[reached], head[reached])), 4
            )
            if reached.any()
            else None,
            "reached_head": int(reached.sum()),
        },
        "source": str(Path(args.features)),
    }

    final_gate = LogisticGate().fit(X, y)
    out_path = Path(args.out)
    gate_path = out_path.with_suffix(".json.gate")
    final_gate.save(gate_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("held-out (mean [2.5%, 97.5%] over splits):")
    for key, value in report["held_out"].items():
        print(f"  {key:26s} {value['mean']:.4f} [{value['lo']:.4f}, {value['hi']:.4f}]")
    print(f"  head AUC (all)             {report['full_data']['head_auc']}")
    print(f"  head AUC (reached head)    {report['full_data']['head_auc_reached']}")
    print(f"wrote {out_path}")
    print(f"wrote {gate_path}  (set LAYA_RAG_ANSWERABILITY_GATE_PATH to use it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
