#!/usr/bin/env python
"""Does the answerability gate transfer across corpora? — R3's measurement protocol.

The repository's headline abstention numbers are one corpus's: the feature gate reads held-out
AUC **0.8146** on the 300-query protocol (0.8290 with the 12-feature arm), while out of domain the
shipped scalar is at chance — **0.5087** AUC, risk@80 0.4927 (`docs/findings-calibration-row.md`).
No benchmark in the 2026 landscape grades a *retrieval-only* answerability predictor with a held-out
AUC, so the transfer question is one this project has to instrument for itself.

This tool is that instrument, and only that: it takes feature rows from **two** corpora and reports

* the gate fitted on one corpus and scored on the other (the transfer arm),
* the same gate on the corpus it was fitted on (the in-domain ceiling, for scale),
* a **shuffled-label** control on the training corpus, because a gate that "transfers" without
  beating its own shuffle has learned nothing,
* risk@80 — the false-accept rate when the most confident 80% are accepted — because AUC says
  nothing about deployable risk.

Rows are produced elsewhere (`tools/answerability_train.py --features`, or any run that records
`extract_features` output); this module does no retrieval and no model calls, so it is testable
offline and cheap to re-run as rows accumulate.

    python tools/gate_transfer.py --train eval/gate/corpus-a.json --test eval/gate/corpus-b.json
    python tools/gate_transfer.py --train a.json --test b.json --features shape
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

import numpy as np  # noqa: E402

import evalkit  # noqa: E402
from laya_rag.answerability import (  # noqa: E402
    FEATURE_NAMES,
    SHAPE_FEATURE_NAMES,
    LogisticGate,
    roc_auc,
)

#: The feature subsets the transfer question is asked about. `shape` is the three signals that need
#: **no decision-model call at all** (max passage coverage, coverage spread, pairwise redundancy), so
#: negatives can be mined for it at corpus scale; `all` is the twelve the shipped gate uses.
SUBSETS = {
    "shape": tuple(SHAPE_FEATURE_NAMES),
    "all": tuple(FEATURE_NAMES),
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Read a feature file: ``{"rows": [{"query", "label", "features": {...}}, ...]}``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{path}: no rows")
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SystemExit(f"{path}: row {index} is not an object")
        features = row.get("features")
        if not isinstance(features, Mapping):
            raise SystemExit(f"{path}: row {index} has no 'features' map")
        label = row.get("label")
        if label not in (0, 1, True, False):
            raise SystemExit(f"{path}: row {index} has label {label!r}; expected 0 or 1")
        out.append({"query": str(row.get("query", "")), "label": int(label), "features": features})
    return out


def matrix(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """The design matrix and labels for ``names``, refusing a row that is missing one."""
    X = np.zeros((len(rows), len(names)), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        features = row["features"]
        for j, name in enumerate(names):
            if name not in features:
                raise SystemExit(f"row {i} ({row['query'][:40]!r}) has no feature {name!r}")
            X[i, j] = float(features[name])
        y[i] = float(row["label"])
    return X, y


def _arm(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    evaluated = evalkit.evaluate_gate(y, scores) if hasattr(evalkit, "evaluate_gate") else {}
    return {
        "auc": round(float(roc_auc(y, scores)), 4),
        "risk@80": evalkit.risk_at_coverage(list(scores), list(y.astype(int)), 0.80),
        "n": int(y.shape[0]),
        "positives": int(y.sum()),
        **({"balanced_accuracy": round(float(evaluated.get("balanced_accuracy", 0.0)), 4)}
           if evaluated else {}),
    }


def _coefficients(gate: LogisticGate, names: Sequence[str]) -> dict[str, float] | None:
    """The fitted weights per feature, or ``None`` when the gate does not expose them.

    A coefficient table is diagnostic, never an input to a decision, so a missing accessor must
    degrade to "not reported" rather than to an exception — and must not become a reason to reach
    into the object's internals from a tool.
    """
    coef = getattr(gate, "coefficients", None)
    if coef is None:
        coef = getattr(gate, "coef", None)
    if coef is None:
        return None
    return {name: round(float(value), 4) for name, value in zip(names, np.asarray(coef).ravel())}


def transfer(
    train: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    *,
    features: str = "all",
    seed: int = 17,
) -> dict[str, Any]:
    """Fit on ``train``, score ``test``, and report the shuffled control.

    The control is fitted on the *same* inputs with the labels permuted. It answers the question a
    transfer number cannot answer alone: if a gate scores 0.7 across corpora and its shuffle also
    scores 0.7, the features are carrying the label, not the corpus.
    """
    names = SUBSETS[features] if features in SUBSETS else tuple(features)
    X_train, y_train = matrix(train, names)
    X_test, y_test = matrix(test, names)
    if y_train.min() == y_train.max():
        raise SystemExit("the training rows carry a single class; a gate cannot be fitted")

    gate = LogisticGate(feature_names=names).fit(X_train, y_train)
    in_domain = _arm(y_train, gate.predict_proba(X_train))
    out_of_domain = _arm(y_test, gate.predict_proba(X_test))

    rng = np.random.default_rng(seed)
    shuffled = LogisticGate(feature_names=names).fit(X_train, rng.permutation(y_train))
    control = _arm(y_test, shuffled.predict_proba(X_test))

    return {
        "features": list(names),
        "in_domain": in_domain,
        "transfer": out_of_domain,
        "shuffled_control": control,
        # `_coef` is the gate's private state: the class exposes `predict_proba` and persistence,
        # not its parameters, so a coefficient table would be reading a private field to display a
        # diagnostic. Report the weights the model *learned from the training corpus* only where the
        # class offers them, and say so plainly when it does not.
        "coefficients": _coefficients(gate, names),
        "seed": seed,
    }


def render(result: Mapping[str, Any], train_label: str = "train", test_label: str = "test") -> str:
    lines = [
        f"features: {', '.join(result['features'])}",
        "",
        "| arm | n | positives | AUC | risk@80 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, label in (
        ("in_domain", f"{train_label} (fitted here)"),
        ("transfer", f"{test_label} (held out)"),
        ("shuffled_control", f"{test_label} (shuffled labels)"),
    ):
        arm = result[name]
        lines.append(
            f"| {label} | {arm['n']} | {arm['positives']} | {arm['auc']:.4f} | {arm['risk@80']:.4f} |"
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="gate_transfer.py", description=__doc__)
    ap.add_argument("--train", required=True, help="feature rows for the corpus the gate is fitted on")
    ap.add_argument("--test", required=True, help="feature rows for the corpus it is scored on")
    ap.add_argument("--features", default="all", choices=sorted(SUBSETS),
                    help="which feature subset to use; 'shape' needs no decision-model call")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=None, help="write the JSON result here as well")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    train = load_rows(Path(args.train))
    test = load_rows(Path(args.test))
    result = transfer(train, test, features=args.features, seed=args.seed)
    result["train"] = args.train
    result["test"] = args.test
    print(render(result, Path(args.train).stem, Path(args.test).stem))
    if args.out:
        evalkit.write_json(Path(args.out), result)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
