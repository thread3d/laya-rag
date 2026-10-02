"""Is a learned decision boundary or a fitted temperature worth adding to the gate?

The gate module warns that its class-weighted intercept makes 0.5 non-optimal and that a
threshold should be chosen on held-out data; the engine hard-codes 0.5 and reports the raw
probability. This measures both on the labelled queries `tools/multihop_gate.py` records:
for each stratified split it fits the gate on one half, learns a threshold and a temperature
on that half, and scores the other half.

    .venv/bin/python tools/gate_calibration.py \
        --features eval/benchmarks/multihop-rag/gate-shape.json
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import evalkit  # noqa: E402
import answerability_train as train  # noqa: E402
from laya_rag.answerability import LogisticGate, evaluate_gate  # noqa: E402

TEMPERATURE_GRID = tuple(round(0.5 + 0.05 * i, 2) for i in range(51))


def logit(probabilities: Any) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(values / (1.0 - values))


def sigmoid(logits: Any) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits, dtype=np.float64), -60.0, 60.0)))


def negative_log_likelihood(labels: Any, probabilities: Any) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-9, 1 - 1e-9)
    return -float(np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_temperature(
    labels: Any,
    probabilities: Any,
    grid: Sequence[float] = TEMPERATURE_GRID,
) -> float:
    """The single temperature minimising NLL, by grid search. ``1.0`` means no change."""
    z = logit(probabilities)
    best_temperature, best_nll = 1.0, float("inf")
    for temperature in grid:
        nll = negative_log_likelihood(labels, sigmoid(z / float(temperature)))
        if nll < best_nll:
            best_temperature, best_nll = float(temperature), nll
    return best_temperature


def measure(
    X: np.ndarray, y: np.ndarray, splits: int, seed: int
) -> dict[str, dict[str, float]]:
    thresholds: list[float] = []
    fixed_accuracy: list[float] = []
    learned_accuracy: list[float] = []
    fixed_coverage: list[float] = []
    learned_coverage: list[float] = []
    raw_ece: list[float] = []
    scaled_ece: list[float] = []
    raw_brier: list[float] = []
    scaled_brier: list[float] = []
    raw_nll: list[float] = []
    scaled_nll: list[float] = []
    temperatures: list[float] = []

    for offset in range(splits):
        train_idx, test_idx = train.stratified_split(y, seed + offset)
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        gate = LogisticGate().fit(X[train_idx], y[train_idx])
        train_scores = gate.predict_proba(X[train_idx])
        test_scores = gate.predict_proba(X[test_idx])
        threshold, _ = train.best_threshold(y[train_idx], train_scores)
        thresholds.append(float(threshold))
        fixed_accuracy.append(evaluate_gate(y[test_idx], test_scores, threshold=0.5)["balanced_accuracy"])
        learned_accuracy.append(
            evaluate_gate(y[test_idx], test_scores, threshold=threshold)["balanced_accuracy"]
        )
        fixed_coverage.append(float(np.mean(test_scores >= 0.5)))
        learned_coverage.append(float(np.mean(test_scores >= threshold)))

        temperature = fit_temperature(y[train_idx], train_scores)
        temperatures.append(temperature)
        z_test = logit(test_scores)
        scaled = sigmoid(z_test / temperature)
        labels = [int(v) for v in y[test_idx]]
        raw = [float(v) for v in test_scores]
        scaled_values = [float(v) for v in scaled]
        raw_ece.append(evalkit.ece_sweep(raw, labels)["ece"])
        scaled_ece.append(evalkit.ece_sweep(scaled_values, labels)["ece"])
        raw_brier.append(evalkit.brier_score(raw, labels))
        scaled_brier.append(evalkit.brier_score(scaled_values, labels))
        raw_nll.append(evalkit.negative_log_likelihood(raw, labels))
        scaled_nll.append(evalkit.negative_log_likelihood(scaled_values, labels))

    def summary(values: Sequence[float]) -> dict[str, float]:
        return {
            "mean": round(statistics.mean(values), 4),
            "sd": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        }

    return {
        "threshold": {
            "mean": round(statistics.mean(thresholds), 4),
            "sd": round(statistics.stdev(thresholds), 4),
            "min": round(min(thresholds), 4),
            "max": round(max(thresholds), 4),
        },
        "temperature": summary(temperatures),
        "fixed_accuracy": summary(fixed_accuracy),
        "learned_accuracy": summary(learned_accuracy),
        "fixed_coverage": summary(fixed_coverage),
        "learned_coverage": summary(learned_coverage),
        "raw_ece": summary(raw_ece),
        "scaled_ece": summary(scaled_ece),
        "raw_brier": summary(raw_brier),
        "scaled_brier": summary(scaled_brier),
        "raw_nll": summary(raw_nll),
        "scaled_nll": summary(scaled_nll),
        "splits": {"mean": float(len(thresholds)), "sd": 0.0},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="gate_calibration.py")
    ap.add_argument("--features", default=str(
        ROOT / "eval" / "benchmarks" / "multihop-rag" / "gate-shape.json"
    ))
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=str(ROOT / "eval" / "gate-calibration"))
    return ap.parse_args(argv)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Gate threshold and calibration",
        "",
        f"- {int(report['splits']['mean'])} stratified splits",
        f"- learned threshold: mean {report['threshold']['mean']}, "
        f"sd {report['threshold']['sd']}, range {report['threshold']['min']}–{report['threshold']['max']}",
        f"- fitted temperature: mean {report['temperature']['mean']}",
        "",
        "| | fixed 0.5 / raw | learned / scaled |",
        "| --- | --- | --- |",
        f"| balanced accuracy | {report['fixed_accuracy']['mean']} | {report['learned_accuracy']['mean']} |",
        f"| coverage | {report['fixed_coverage']['mean']} | {report['learned_coverage']['mean']} |",
        f"| ECE (lower better) | {report['raw_ece']['mean']} | {report['scaled_ece']['mean']} |",
        f"| Brier (lower better) | {report['raw_brier']['mean']} | {report['scaled_brier']['mean']} |",
        f"| NLL (lower better) | {report['raw_nll']['mean']} | {report['scaled_nll']['mean']} |",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = train.load_records(Path(args.features))
    X, y, _head = train.build_matrix(records)
    report = measure(X, y, args.splits, args.seed)
    payload = {"_settings": vars(args), "n": len(records), **report}
    out = Path(args.out)
    evalkit.write_json(Path(str(out) + ".json"), payload)
    evalkit.write_text(Path(str(out) + ".md"), render_markdown(payload))
    print(render_markdown(payload))
    print("wrote", str(out) + ".json and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
