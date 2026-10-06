"""The calibration row: reliability, ECE, Brier, NLL and risk-coverage for the answerability decision.

MTEB grades a model; BrowseComp-Plus grades calibration error; and no standard retrieval benchmark
grades a retriever's *relevance or answerability calibration* at all. That is the unclaimed row this
project is positioned to publish, and this tool produces the first artifact for it: a reliability
diagram and a calibration table for the engine's answerability decision on MultiHop-RAG, the one
corpus that ships both positive and null (`null_query`) labels.

The measurement is of artifacts the repository already has. `gate.json` holds Laya's calibrated
probability for 601 queries (301 null, 300 answerable); `answerability-selective.json` holds the
held-out ECE/Brier/NLL/risk-coverage for both the shipped scalar and the feature gate. This
consolidates them and renders the reliability curve the audit asked for, and is explicit about the
corpora it cannot yet cover.

    .venv/bin/python tools/calibration_report.py
    .venv/bin/python tools/calibration_report.py --chart docs/charts/reliability.png
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
import answerability_train as train  # noqa: E402
from laya_rag.answerability import FEATURE_NAMES, LogisticGate, roc_auc  # noqa: E402

MULTIHOP = ROOT / "eval" / "benchmarks" / "multihop-rag"
GATE = MULTIHOP / "gate.json"
SELECTIVE = MULTIHOP / "answerability-selective.json"
#: The 12-feature gate inputs (nine score features + three evidence-shape features) with labels.
SHAPE = MULTIHOP / "gate-shape.json"
#: Frozen seed for the held-out split, so the diagram is reproducible.
SPLIT_SEED = 17

#: Corpora the calibration row cannot yet cover, and why. Stated in the report so the
#: single-corpus result is never read as a cross-corpus claim.
UNCALIBRATED = {
    "BEIR-10": "no per-query answerability labels; the corpus has relevance qrels, not nulls",
    "MuSiQue": "retrieval run only; no answerable/null labelling",
    "FRAMES": "every query is answerable by construction, so there is no negative class",
}


def load_head(path: Path = GATE) -> tuple[list[float], list[int]]:
    """Per-query probability and label for the shipped scalar head.

    `gate.json` groups the records rather than labelling them: every `null` row is a negative
    and every `positive` row is an answerable query.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    probabilities: list[float] = []
    labels: list[int] = []
    for group, label in (("null", 0), ("positive", 1)):
        for row in payload.get(group) or []:
            probabilities.append(float(row["p"]))
            labels.append(label)
    return probabilities, labels


def calibration(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, Any]:
    """Every calibration and selective-prediction reading for one decision."""
    outcomes = [int(v) for v in labels]
    probs = [float(v) for v in probabilities]
    sweep = evalkit.ece_sweep(probs, outcomes)
    return {
        "n": len(probs),
        "positive_rate": round(sum(outcomes) / len(outcomes), 4) if outcomes else 0.0,
        "auc": round(roc_auc(outcomes, probs), 4),
        "ece_equal_width": evalkit.expected_calibration_error(probs, outcomes, n_bins=10),
        "ece_equal_mass": sweep["ece"],
        "ece_bins": sweep["bins"],
        "ece_monotone": sweep["monotone"],
        "brier": evalkit.brier_score(probs, outcomes),
        "nll": evalkit.negative_log_likelihood(probs, outcomes),
        "aurc": evalkit.aurc(probs, outcomes),
        "augrc": evalkit.augrc(probs, outcomes),
        "risk@80": evalkit.risk_at_coverage(probs, outcomes, 0.80),
        "risk@90": evalkit.risk_at_coverage(probs, outcomes, 0.90),
        "reliability_equal_width": evalkit.reliability_bins(probs, outcomes, n_bins=10),
        "reliability_equal_mass": evalkit.reliability_curve_equal_mass(probs, outcomes, n_bins=10),
    }


def feature_gate_reading(path: Path = SELECTIVE) -> dict[str, Any]:
    """The held-out 12-feature gate and scalar readings, read from the existing report."""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "source": str(path.relative_to(ROOT)),
        "n": payload.get("n"),
        "features": payload.get("features"),
        "gate": payload.get("held_out", {}),
    }


def gate_heldout(path: Path = SHAPE, seed: int = SPLIT_SEED) -> dict[str, Any]:
    """Per-query held-out predictions for the feature gate and the scalar head.

    Trains `LogisticGate` on one stratified split of the labelled gate records and returns the
    test-split labels, the gate's probabilities and the shipped scalar's probabilities on the
    *same* queries, so the two reliability curves are directly comparable. Pure NumPy — no model
    forward pass — because the feature vectors are already recorded.
    """
    if not path.is_file():
        return {}
    records = train.load_records(path)
    X, y, head = train.build_matrix(records)
    train_idx, test_idx = train.stratified_split(y, seed)
    gate = LogisticGate().fit(X[train_idx], y[train_idx])
    return {
        "seed": seed,
        "source": str(path.relative_to(ROOT)),
        "n_test": int(len(test_idx)),
        "features": list(FEATURE_NAMES),
        "labels": [int(v) for v in y[test_idx]],
        "gate_probabilities": [float(v) for v in gate.predict_proba(X[test_idx])],
        "head_probabilities": [float(v) for v in head[test_idx]],
    }


def build_report(gate_path: Path = GATE, selective_path: Path = SELECTIVE) -> dict[str, Any]:
    probabilities, labels = load_head(gate_path)
    held = gate_heldout()
    report: dict[str, Any] = {
        "corpus": "MultiHop-RAG",
        "shipped_head": calibration(labels, probabilities),
        "feature_gate": feature_gate_reading(selective_path),
        "uncalibrated_corpora": UNCALIBRATED,
        "chart": "docs/charts/reliability.png",
    }
    if held:
        gate_labels = held["labels"]
        report["gate_heldout"] = {
            "seed": held["seed"],
            "source": held["source"],
            "n_test": held["n_test"],
            "features": held["features"],
            "gate": calibration(gate_labels, held["gate_probabilities"]),
            "head": calibration(gate_labels, held["head_probabilities"]),
        }
    return report


def chart_reliability(report: Mapping[str, Any], out: Path) -> None:
    """The reliability diagram: observed frequency against predicted probability."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    head = report["shipped_head"]
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1, label="perfect")
    series = [(head["reliability_equal_width"], "scalar head (all 601)", "#1f77b4")]
    held = (report.get("gate_heldout") or {}).get("gate")
    if held:
        series.append((held["reliability_equal_width"],
                       f"feature gate (held out, n={report['gate_heldout']['n_test']})", "#2ca02c"))
    for rows, name, colour in series:
        xs = [row["confidence"] for row in rows]
        ys = [row["accuracy"] for row in rows]
        ax.plot(xs, ys, marker="o", color=colour, linewidth=1.5, label=name)
    ax.set_xlabel("predicted P(answerable)")
    ax.set_ylabel("observed frequency")
    ax.set_title(
        f"MultiHop-RAG answerability reliability\n"
        f"scalar ECE={head['ece_equal_mass']:.3f}, Brier={head['brier']:.3f}"
        + (f"; gate ECE={held['ece_equal_mass']:.3f}" if held else "")
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def render_markdown(report: Mapping[str, Any]) -> str:
    head = report["shipped_head"]
    lines = [
        "# Answerability calibration — MultiHop-RAG",
        "",
        "The engine's answerability decision on the one corpus that labels both classes: "
        f"**{head['n']}** queries, positive rate **{head['positive_rate']:.4f}**.",
        "",
        "| reading | shipped scalar head |",
        "| --- | --- |",
        f"| ROC-AUC | {head['auc']:.4f} |",
        f"| ECE (equal-width, 10) | {head['ece_equal_width']:.4f} |",
        f"| ECE (equal-mass, {head['ece_bins']} bins) | {head['ece_equal_mass']:.4f} |",
        f"| Brier | {head['brier']:.4f} |",
        f"| NLL | {head['nll']:.4f} |",
        f"| AURC | {head['aurc']:.4f} |",
        f"| AUGRC | {head['augrc']:.4f} |",
        f"| risk@80 | {head['risk@80']:.4f} |",
        f"| risk@90 | {head['risk@90']:.4f} |",
        "",
        "## Reliability (equal-width bins)",
        "",
        "| bin | n | confidence | accuracy | gap |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in head["reliability_equal_width"]:
        lines.append(
            f"| {row['range']} | {row['count']} | {row['confidence']:.4f} | "
            f"{row['accuracy']:.4f} | {row['gap']:.4f} |"
        )
    held = report.get("gate_heldout")
    if held:
        gate_cal, head_cal = held["gate"], held["head"]
        lines += [
            "",
            f"## The feature gate, held out (seed {held['seed']}, {held['n_test']} test queries)",
            "",
            f"Trained on a stratified split of `{held['source']}`; the scalar head is scored on the "
            "same test queries.",
            "",
            "| reading | feature gate | scalar head (same split) |",
            "| --- | --- | --- |",
        ]
        for key, label in (
            ("auc", "ROC-AUC"),
            ("ece_equal_width", "ECE (equal-width, 10)"),
            ("brier", "Brier"),
            ("nll", "NLL"),
            ("risk@80", "risk@80"),
            ("risk@90", "risk@90"),
            ("aurc", "AURC"),
        ):
            lines.append(f"| {label} | {gate_cal[key]:.4f} | {head_cal[key]:.4f} |")
    gate = report.get("feature_gate") or {}
    if gate:
        held = gate.get("gate", {})
        features = gate.get("features") or []
        lines += [
            "",
            f"## The feature gate, held out ({len(features)} features)",
            "",
            f"From `{gate['source']}` ({gate.get('n')} labelled queries):",
            "",
            "| metric | gate | scalar |",
            "| --- | --- | --- |",
        ]
        for metric in ("gate_auc", "gate_ece", "gate_brier", "gate_nll", "gate_risk@80", "gate_aurc"):
            if metric not in held:
                continue
            base = metric.replace("gate_", "head_")
            cell = held.get(base, {})
            gate_value = held[metric]
            lines.append(
                f"| {metric.replace('gate_', '')} | "
                f"{gate_value['mean']:.4f} [{gate_value['lo']:.4f}, {gate_value['hi']:.4f}] | "
                f"{cell['mean']:.4f} [{cell['lo']:.4f}, {cell['hi']:.4f}] |"
                if isinstance(gate_value, Mapping) and isinstance(cell, Mapping)
                else f"| {metric.replace('gate_', '')} | {gate_value} | {cell} |"
            )
    lines += [
        "",
        "## What this does not cover",
        "",
        "Calibration is only measurable where a corpus labels a negative class. Still missing:",
        "",
    ]
    for corpus, reason in report["uncalibrated_corpora"].items():
        lines.append(f"- **{corpus}** — {reason}.")
    lines += [
        "",
        "The reliability diagram is written to " + report["chart"] + ".",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="calibration_report.py")
    ap.add_argument("--gate", default=str(GATE))
    ap.add_argument("--selective", default=str(SELECTIVE))
    ap.add_argument("--out", default=str(ROOT / "eval" / "calibration"))
    ap.add_argument("--chart", default=None, help="write the reliability diagram here")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    gate = Path(args.gate)
    if not gate.is_file():
        print(f"calibration-report: SKIPPED — {gate} not found")
        return 0
    report = build_report(gate, Path(args.selective))
    out = Path(args.out)
    evalkit.write_json(out.with_suffix(".json"), report)
    evalkit.write_text(out.with_suffix(".md"), render_markdown(report))
    if args.chart:
        chart_reliability(report, Path(args.chart))
    head = report["shipped_head"]
    print(
        f"calibration-report: n={head['n']} ECE(eq-mass)={head['ece_equal_mass']:.4f} "
        f"Brier={head['brier']:.4f} NLL={head['nll']:.4f} risk@80={head['risk@80']:.4f}"
    )
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
