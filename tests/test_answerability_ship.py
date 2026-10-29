"""Shipping the fitted gate: the artefact, the loader, and the two shapes of labelled rows.

The trainer was written against one feature-file shape (`null`/`positive` groups) and the newer
`tools/gate_dataset.py` writes another (a flat `rows` list). Reading only one made the newer dataset
unusable by the trainer, and the *baseline* silently became all zeros — reporting the shipped scalar
at AUC 0.500 for the right reason by accident. Both are pinned here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import answerability_train as at  # noqa: E402
from laya_rag.answerability import FEATURE_NAMES, LogisticGate  # noqa: E402


def _row(label: int, level: float, scalar: float) -> dict:
    return {
        "query": f"q{label}",
        "label": label,
        "answerability": scalar,
        "features": {name: level for name in FEATURE_NAMES},
    }


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_the_trainer_reads_the_flat_rows_shape(tmp_path):
    path = _write(tmp_path, {"rows": [_row(1, 0.9, 0.8), _row(0, 0.2, 0.6)]})
    records = at.load_records(path)
    assert [record["label"] for record in records] == [1, 0]


def test_the_trainer_reads_the_grouped_shape_it_was_written_for(tmp_path):
    grouped = {
        "null": [{"query": "n", "features": {n: 0.1 for n in FEATURE_NAMES}}],
        "positive": [{"query": "p", "features": {n: 0.9 for n in FEATURE_NAMES}}],
    }
    records = at.load_records(_write(tmp_path, grouped))
    assert [record["label"] for record in records] == [0, 1]


def test_the_scalar_baseline_is_mapped_from_answerability_not_left_at_zero(tmp_path):
    """A zeroed baseline reports the shipped system at AUC 0.500, which is the wrong reason."""
    path = _write(tmp_path, {"rows": [_row(1, 0.9, 0.85), _row(0, 0.2, 0.15)]})
    records = at.load_records(path)
    assert [record["p"] for record in records] == [0.85, 0.15]
    _X, _y, head = at.build_matrix(records)
    assert head.tolist() != [0.0, 0.0]


def test_a_row_without_features_is_named_and_refused(tmp_path):
    path = _write(tmp_path, {"rows": [{"query": "q", "label": 1}]})
    with pytest.raises(ValueError):
        at.load_records(path)


def test_a_row_with_a_non_binary_label_is_refused(tmp_path):
    path = _write(tmp_path, {"rows": [{"query": "q", "label": 2, "features": {}}]})
    with pytest.raises(ValueError):
        at.load_records(path)


def test_the_persisted_artefact_loads_into_the_gate_the_engine_uses(tmp_path):
    """`make gate-train` writes a `.gate` file; the engine reads it by path. Round trip both ends."""
    rows = []
    for i in range(40):
        label = i % 2
        rows.append(_row(label, 0.9 if label else 0.2, 0.5))
    gate = LogisticGate().fit(
        [[row["features"][n] for n in FEATURE_NAMES] for row in rows],
        [row["label"] for row in rows],
    )
    path = tmp_path / "gate.gate"
    gate.save(path)

    restored = LogisticGate.load(path)
    probe = [[0.95 if n != "pairwise_redundancy" else 0.1 for n in FEATURE_NAMES]]
    assert restored.predict_proba(probe)[0] == pytest.approx(gate.predict_proba(probe)[0], abs=1e-9)
    assert restored.feature_names == tuple(FEATURE_NAMES)


def test_a_damaged_artefact_degrades_to_the_scalar_rather_than_breaking_queries(tmp_path):
    """The engine's contract: a stale gate must not take every query down with it."""
    from tests.support import StubModel, make_config
    from laya_rag.engine import RagEngine

    path = tmp_path / "broken.gate"
    path.write_text("{not json", encoding="utf-8")
    engine = RagEngine(make_config(tmp_path, answerability_gate_path=str(path)), model=StubModel())
    assert engine.answerability_gate is None
    assert engine.stats()["answerability_gate"] == {"configured": True, "loaded": False}
