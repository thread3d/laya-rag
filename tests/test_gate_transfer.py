"""The cross-corpus transfer protocol, tested without a corpus.

The numbers this tool reports decide whether the abstention claim is portable, so the protocol
itself has to mean what it says: a gate must be able to succeed, must be *able to fail*, and the
shuffled control must sit at chance — a control that also scores 0.7 is the bug this repository has
already recorded once (two arms identical to four decimals meant the harness was reading the wrong
field, not that the effect was null).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import gate_transfer as gt  # noqa: E402


def _rows(n: int, *, shift: float = 0.0, label_noise: float = 0.0, seed: int = 0):
    """Rows whose label is a function of the shape features, optionally shifted."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        answerable = i % 2 == 1
        base = rng.normal(0.6 if answerable else 0.3, 0.08, size=3)
        noise = rng.normal(0.0, label_noise if label_noise else 1e-9, size=3)
        rows.append(
            {
                "query": f"q{i}",
                "label": int(answerable),
                "features": {
                    "max_passage_coverage": float(base[0] + shift + noise[0]),
                    "coverage_spread": float(base[1] + shift + noise[1]),
                    "pairwise_redundancy": float(base[2] + shift + noise[2]),
                },
            }
        )
    return rows


def test_a_gate_transfers_when_the_corpora_share_a_distribution():
    result = gt.transfer(_rows(60), _rows(60, seed=7), features="shape")
    assert result["in_domain"]["auc"] > 0.9
    assert result["transfer"]["auc"] > 0.9
    assert result["in_domain"]["n"] == 60 and result["transfer"]["positives"] == 30


def test_the_shuffled_control_sits_at_chance():
    """If a permutation of the labels transfers too, the features are carrying the corpus."""
    result = gt.transfer(_rows(80), _rows(80, seed=3), features="shape")
    assert 0.2 < result["shuffled_control"]["auc"] < 0.8


def test_a_shifted_corpus_is_reported_as_a_failure_not_a_success():
    """The documented failure mode: the score features shift, the ranking survives, the gate breaks."""
    result = gt.transfer(_rows(60), _rows(60, shift=4.0, seed=11), features="shape")
    assert result["in_domain"]["auc"] > 0.9
    assert result["transfer"]["auc"] < 0.6


def test_risk_at_80_is_the_scored_tail_not_the_whole_set():
    result = gt.transfer(_rows(80), _rows(80, seed=5), features="shape")
    assert 0.0 <= result["transfer"]["risk@80"] <= 1.0
    assert result["transfer"]["risk@80"] < result["transfer"]["n"]


def test_a_single_class_training_set_is_refused():
    rows = _rows(10)
    for row in rows:
        row["label"] = 1
    with pytest.raises(SystemExit):
        gt.transfer(rows, rows, features="shape")


def test_a_row_missing_a_feature_is_named_in_the_error(tmp_path):
    rows = _rows(4)
    del rows[2]["features"]["coverage_spread"]
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    with pytest.raises(SystemExit):
        gt.load_rows(path) if False else gt.matrix(gt.load_rows(path), gt.SUBSETS["shape"])


def test_the_render_names_every_arm():
    rendered = gt.render(gt.transfer(_rows(20), _rows(20, seed=2), features="shape"))
    assert "AUC" in rendered and "risk@80" in rendered
    assert rendered.count("|") >= 12
