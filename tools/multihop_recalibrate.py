"""MultiHop-RAG answerability recalibration: does a threshold fit on one split transfer?

The project's own verdict is that the answerability gate abstains at ~31% on null
queries and ~29% on answerable ones — a 0.022 gap against a standard error of ~0.037,
i.e. no discrimination. That measurement used the shipped threshold (0.5) and leaves the
central question unanswered: is the *head* carrying no signal on this corpus, or is the
*threshold* simply wrong for it?

This runs the test that distinguishes the two. It treats MultiHop-RAG as the
out-of-domain split (the head was trained elsewhere), holds out half of its null and
answerable queries, fits a threshold — and a monotone Platt rescaling — on the calibration
half, and re-measures rejection on the held-out half. If the mechanism transfers at all, a
threshold chosen on one half should discriminate on the other.

    .venv/bin/python tools/multihop_recalibrate.py

Writes `eval/benchmarks/multihop-rag/recalibration.json` and a markdown table beside it.
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
sys.path.insert(0, str(ROOT / "src"))
import evalkit  # noqa: E402

BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"


# ── small statistics helpers ─────────────────────────────────────────────

def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC-AUC; 0.5 is chance. Ties get average ranks."""
    labels = np.asarray(labels)
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # average ties
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def abstention_rates(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """`answerable = score >= threshold`; abstention is the complement."""
    predicted_answerable = scores >= threshold
    null = labels == 0
    positive = labels == 1
    null_abstain = float((~predicted_answerable)[null].mean()) if null.any() else float("nan")
    pos_abstain = float((~predicted_answerable)[positive].mean()) if positive.any() else float("nan")
    tpr = 1.0 - pos_abstain
    tnr = null_abstain
    return {
        "threshold": round(float(threshold), 4),
        "null_abstention": round(null_abstain, 4),
        "positive_abstention": round(pos_abstain, 4),
        "discrimination": round(null_abstain - pos_abstain, 4),
        "balanced_accuracy": round((tpr + tnr) / 2.0, 4),
    }


def best_threshold(scores: np.ndarray, labels: np.ndarray, grid: int = 200) -> float:
    """The threshold maximizing balanced accuracy on `scores`; ties go to the middle."""
    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        return 0.5
    candidates = np.unique(np.concatenate([scores, np.linspace(lo, hi, grid)]), axis=0)
    best, best_score = 0.5, -1.0
    for threshold in candidates:
        ba = abstention_rates(scores, labels, threshold)["balanced_accuracy"]
        # Prefer the threshold nearest 0.5 among equivalently good ones, so the choice is
        # not an artifact of the grid.
        if ba > best_score + 1e-12 or (abs(ba - best_score) <= 1e-12 and abs(threshold - 0.5) < abs(best - 0.5)):
            best, best_score = float(threshold), float(ba)
    return best


def bootstrap_auc(scores: np.ndarray, labels: np.ndarray, samples: int, seed: int = 23) -> dict:
    rng = np.random.default_rng(seed)
    n = len(scores)
    values = []
    for _ in range(samples):
        idx = rng.integers(0, n, n)
        value = auc(scores[idx], labels[idx])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return {"point": round(float(auc(scores, labels)), 4), "lo": None, "hi": None}
    return {
        "point": round(float(auc(scores, labels)), 4),
        "lo": round(float(np.percentile(values, 2.5)), 4),
        "hi": round(float(np.percentile(values, 97.5)), 4),
    }


def brier(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((scores - labels) ** 2))


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration `P(y=1) = sigmoid(a * logit(p) + b)` by gradient descent.

    A slope `a <= 0` means the head's probability points the *wrong* way once calibrated;
    `a ~ 0` means it carries no information about answerability on this corpus. Slope 1 and
    intercept 0 leave it untouched.
    """
    eps = 1e-6
    p = np.clip(scores.astype(float), eps, 1 - eps)
    x = np.log(p / (1 - p))
    a, b = 1.0, 0.0
    lr = 0.5
    for _ in range(4000):
        z = a * x + b
        pred = 1.0 / (1.0 + np.exp(-z))
        grad_a = float(np.mean((pred - labels) * x))
        grad_b = float(np.mean(pred - labels))
        a -= lr * grad_a
        b -= lr * grad_b
    return float(a), float(b)


# ── the experiment ───────────────────────────────────────────────────────

def stratified_split(labels: np.ndarray, seed: int, fraction: float = 0.5):
    rng = random.Random(seed)
    calib, test = [], []
    for cls in (0, 1):
        idx = [i for i, label in enumerate(labels) if label == cls]
        rng.shuffle(idx)
        cut = int(round(len(idx) * fraction))
        calib.extend(idx[:cut])
        test.extend(idx[cut:])
    rng.shuffle(calib)
    rng.shuffle(test)
    return np.array(calib), np.array(test)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="multihop_recalibrate.py")
    ap.add_argument("--gate", default=str(BENCH / "gate.json"))
    ap.add_argument("--out", default=str(BENCH / "recalibration"))
    ap.add_argument("--splits", type=int, default=500, help="repeated held-out splits for the transfer test")
    ap.add_argument("--bootstrap-samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=17, help="the canonical split shown in detail")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    gate = json.loads(Path(args.gate).read_text(encoding="utf-8"))
    nulls = [row for row in gate["null"]]
    positives = [row for row in gate["positive"]]
    scores = np.array([row["p"] for row in nulls + positives], dtype=float)
    labels = np.array([0] * len(nulls) + [1] * len(positives), dtype=int)

    report: dict = {
        "source": str(args.gate),
        "n": {"null": len(nulls), "positive": len(positives)},
        "shipped_threshold": 0.5,
        "shipped": abstention_rates(scores, labels, 0.5),
        "auc_all": bootstrap_auc(scores, labels, args.bootstrap_samples, args.seed + 1),
    }
    # The head only runs when evidence survives the relevance gate; p == 0.0 is the
    # short-circuit, not a head output. Report the head's signal both ways.
    reached = scores > 0.0
    report["reached_head"] = int(reached.sum())
    if reached.sum() and len(set(labels[reached].tolist())) == 2:
        report["auc_reached_head"] = bootstrap_auc(
            scores[reached], labels[reached], args.bootstrap_samples, args.seed + 2
        )
    else:
        report["auc_reached_head"] = None

    # Canonical split: fit on half, measure on the other, and show the fitted rescaling.
    calib_idx, test_idx = stratified_split(labels, args.seed)
    cal_scores, cal_labels = scores[calib_idx], labels[calib_idx]
    test_scores, test_labels = scores[test_idx], labels[test_idx]
    fitted_threshold = best_threshold(cal_scores, cal_labels)
    a, b = fit_platt(cal_scores, cal_labels)
    platt_test = 1.0 / (1.0 + np.exp(-(a * np.log(np.clip(test_scores, 1e-6, 1 - 1e-6) / (1 - np.clip(test_scores, 1e-6, 1 - 1e-6))) + b)))
    report["canonical_split"] = {
        "seed": args.seed,
        "n_calibration": int(len(calib_idx)),
        "n_test": int(len(test_idx)),
        "calibration_fit": {
            "best_threshold": round(fitted_threshold, 4),
            "balanced_accuracy": abstention_rates(cal_scores, cal_labels, fitted_threshold)["balanced_accuracy"],
            "oracle_threshold": "the fit is already the calibration-half optimum",
        },
        "test_at_shipped_0.5": abstention_rates(test_scores, test_labels, 0.5),
        "test_at_calibrated_threshold": abstention_rates(test_scores, test_labels, fitted_threshold),
        "test_oracle": {
            **abstention_rates(test_scores, test_labels, best_threshold(test_scores, test_labels)),
            "note": "threshold fitted on the test half itself — an upper bound, not a result",
        },
        "platt": {
            "slope": round(a, 4),
            "intercept": round(b, 4),
            "brier_before": round(brier(test_scores, test_labels), 4),
            "brier_after": round(brier(platt_test, test_labels), 4),
            "note": "slope ~ 0 means the head's probability carries no signal to rescale",
        },
        "auc_test_at_shipped_scores": round(auc(test_scores, test_labels), 4),
    }

    # Repeated stratified half-splits: fit on one half, score balanced accuracy on the
    # other. If the mechanism transfers, the test distribution sits above 0.5.
    transfer = []
    for seed in range(args.splits):
        cal_idx, test_idx = stratified_split(labels, 1000 + seed)
        threshold = best_threshold(scores[cal_idx], labels[cal_idx])
        result = abstention_rates(scores[test_idx], labels[test_idx], threshold)
        result["calibration_threshold"] = round(float(threshold), 4)
        transfer.append(result["balanced_accuracy"])
    transfer = np.array(transfer)
    report["transfer"] = {
        "splits": args.splits,
        "test_balanced_accuracy_mean": round(float(transfer.mean()), 4),
        "test_balanced_accuracy_lo": round(float(np.percentile(transfer, 2.5)), 4),
        "test_balanced_accuracy_hi": round(float(np.percentile(transfer, 97.5)), 4),
        "fraction_above_chance": round(float((transfer > 0.5).mean()), 4),
        "chance": 0.5,
        "note": "threshold fitted on a calibration half, balanced accuracy measured on the held-out half",
    }

    out = Path(args.out)
    evalkit.write_json(out.with_suffix(".json"), report)
    shipped = report["shipped"]
    canon = report["canonical_split"]
    transfer_row = report["transfer"]
    table = evalkit.markdown_table([
        {"reading": "shipped threshold 0.5 (all queries)", "null abstain": f"{shipped['null_abstention']:.3f}",
         "answerable abstain": f"{shipped['positive_abstention']:.3f}", "gap": f"{shipped['discrimination']:+.3f}",
         "balanced accuracy": f"{shipped['balanced_accuracy']:.3f}"},
        {"reading": "test half, calibrated threshold", "null abstain": f"{canon['test_at_calibrated_threshold']['null_abstention']:.3f}",
         "answerable abstain": f"{canon['test_at_calibrated_threshold']['positive_abstention']:.3f}",
         "gap": f"{canon['test_at_calibrated_threshold']['discrimination']:+.3f}",
         "balanced accuracy": f"{canon['test_at_calibrated_threshold']['balanced_accuracy']:.3f}"},
        {"reading": "test half, threshold fit on it (oracle)", "null abstain": f"{canon['test_oracle']['null_abstention']:.3f}",
         "answerable abstain": f"{canon['test_oracle']['positive_abstention']:.3f}",
         "gap": f"{canon['test_oracle']['discrimination']:+.3f}",
         "balanced accuracy": f"{canon['test_oracle']['balanced_accuracy']:.3f}"},
    ])
    evalkit.write_text(
        out.with_suffix(".md"),
        "# MultiHop-RAG answerability recalibration\n\n"
        + table
        + f"\n\nROC-AUC over all {len(scores)} queries: "
        + f"{report['auc_all']['point']:.3f} [{report['auc_all']['lo']:.3f}, {report['auc_all']['hi']:.3f}]. "
        + f"Platt slope fitted on the calibration half: {a:+.3f}.\n"
        + f"Repeated half-split transfer: balanced accuracy {transfer_row['test_balanced_accuracy_mean']:.3f} "
        + f"[{transfer_row['test_balanced_accuracy_lo']:.3f}, {transfer_row['test_balanced_accuracy_hi']:.3f}] "
        + f"on the held-out half across {transfer_row['splits']} splits.\n",
    )
    print(json.dumps(report, indent=2))
    print("\nwrote", out.with_suffix(".json"), "and", out.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
