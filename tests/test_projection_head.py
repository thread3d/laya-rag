"""Offline unit tests for the linear projection head.

Everything here runs with no checkpoints and no network: the projection, the ranking and
the three-way evaluation are pure NumPy, and the only model-touching path (``main``) is
exercised on a missing cache so it exits before importing an encoder. The synthetic corpus
is built so the teacher separates three classes cleanly while raw Laya similarity is
swamped by nuisance dimensions -- the regime the projection exists to rescue.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import projection_head  # noqa: E402
from projection_head import (  # noqa: E402
    LinearProjection,
    distillation_loss,
    evaluate_projected,
    main,
    retrieval_rankings,
)


# ── LinearProjection ─────────────────────────────────────────────────────

def test_fit_recovers_a_known_linear_map():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(200, 5))
    weights = rng.normal(size=(5, 3))
    bias = rng.normal(size=3)
    target = source @ weights + bias

    projection = LinearProjection.fit(source, target, l2=1e-8)

    assert projection.weights.shape == (5, 3)
    assert projection.bias.shape == (3,)
    assert np.allclose(projection.transform(source), target, atol=1e-4)
    assert np.allclose(projection.weights, weights, atol=1e-4)
    assert np.allclose(projection.bias, bias, atol=1e-4)


def test_fit_rejects_mismatched_or_empty_input():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        LinearProjection.fit(rng.normal(size=(3, 4)), rng.normal(size=(2, 4)))
    with pytest.raises(ValueError):
        LinearProjection.fit(np.zeros((0, 4)), np.zeros((0, 3)))
    with pytest.raises(ValueError):
        LinearProjection.fit(rng.normal(size=(3, 4)), rng.normal(size=(3, 4)), l2=-1.0)


def test_transform_handles_one_and_two_dimensions():
    projection = LinearProjection(
        weights=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        bias=np.array([1.0, -1.0]),
    )

    row = np.array([1.0, 2.0, 3.0])
    single = projection.transform(row)
    assert single.shape == (2,)
    assert single.dtype == np.float64
    assert np.allclose(single, np.array([1.0 + 3.0 + 1.0, 2.0 + 3.0 - 1.0]))

    stacked = projection.transform(np.vstack([row, row]))
    assert stacked.shape == (2, 2)
    assert np.allclose(stacked, np.vstack([single, single]))

    with pytest.raises(ValueError):
        projection.transform(np.array([1.0, 2.0, 3.0, 4.0]))


def test_transform_normalized_gives_unit_rows_and_keeps_zero_rows_zero():
    projection = LinearProjection(weights=np.eye(4), bias=np.zeros(4))
    vectors = np.array([[3.0, 0.0, 0.0, 4.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])

    unit = projection.transform_normalized(vectors)

    norms = np.linalg.norm(unit, axis=1)
    assert np.allclose(norms[:2], [1.0, 0.0], atol=1e-12)
    assert np.allclose(norms[2], 1.0, atol=1e-12)

    single = projection.transform_normalized(np.array([3.0, 0.0, 0.0, 4.0]))
    assert np.isclose(float(np.linalg.norm(single)), 1.0)


def test_save_load_round_trip_and_version_guard(tmp_path):
    rng = np.random.default_rng(2)
    projection = LinearProjection(weights=rng.normal(size=(6, 2)), bias=rng.normal(size=2))
    path = tmp_path / "nested" / "projection.json"

    projection.save(path)
    restored = LinearProjection.load(path)

    assert np.array_equal(restored.weights, projection.weights)
    assert np.array_equal(restored.bias, projection.bias)
    assert repr(restored) == "LinearProjection(weights=(6, 2), bias=(2,))"

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert not list(path.parent.glob("*.tmp"))

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"version": 99, "weights": [[1.0]], "bias": [0.0]}), encoding="utf-8")
    with pytest.raises(ValueError):
        LinearProjection.load(bad)


def test_unfitted_projection_refuses_to_transform():
    with pytest.raises(ValueError):
        LinearProjection().transform(np.zeros(3))
    assert repr(LinearProjection()) == "LinearProjection(unfitted)"


# ── distillation loss ────────────────────────────────────────────────────

def test_distillation_loss_is_zero_for_identical_and_one_for_orthogonal():
    projection = LinearProjection(weights=np.eye(4), bias=np.zeros(4))
    rng = np.random.default_rng(3)
    source = rng.normal(size=(12, 4))

    assert distillation_loss(source, source, projection) == pytest.approx(0.0, abs=1e-12)

    first = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (12, 1))
    second = np.tile(np.array([0.0, 1.0, 0.0, 0.0]), (12, 1))
    assert distillation_loss(first, second, projection) == pytest.approx(1.0, abs=1e-12)
    assert distillation_loss(first, second, projection) > 0.0


# ── ranking ──────────────────────────────────────────────────────────────

def test_retrieval_rankings_drops_the_query_id_and_ranks_the_nearest_doc_first():
    docs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]])
    ids = ["q0", "d1", "d2"]
    qrels = {"q0": {"d2": 1.0}}

    rankings = retrieval_rankings(docs, np.array([[1.0, 0.0, 0.0]]), ids, qrels, top_k=10)

    assert list(rankings) == ["q0"]
    assert "q0" not in rankings["q0"]
    assert rankings["q0"][0] == "d2"
    assert rankings["q0"][1] == "d1"


def test_retrieval_rankings_breaks_ties_by_ascending_doc_id():
    docs = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    ids = ["b", "a", "c"]  # deliberately unsorted

    rankings = retrieval_rankings(docs, np.array([[1.0, 0.0]]), ids, {"z": {"a": 1.0}}, top_k=10)

    assert rankings["z"] == ["a", "b", "c"]


def test_retrieval_rankings_accepts_a_query_mapping():
    docs = np.array([[1.0, 0.0], [0.0, 1.0]])
    rankings = retrieval_rankings(
        docs,
        {"z": np.array([0.0, 1.0])},
        ["a", "b"],
        {"z": {"b": 1.0}},
        top_k=5,
    )
    assert rankings["z"] == ["b", "a"]


# ── three-way evaluation ─────────────────────────────────────────────────

def _synthetic_corpus(seed: int = 0):
    """Teacher separates classes; raw Laya similarity is dominated by nuisance dims.

    Class dims carry an exact one-hot signal while nuisance dims are large Gaussian noise,
    so raw cosine is essentially unordered. Ridge regression averages the nuisance away
    across documents, which is exactly why the projection then retrieves cleanly.
    """
    rng = np.random.default_rng(seed)
    classes, per_class, d_laya, nuisance, signal = 3, 60, 8, 4.0, 1.0
    total = classes * per_class
    labels = np.repeat(np.arange(classes), per_class)
    one_hot = np.eye(classes)[labels]

    laya_doc = rng.normal(scale=nuisance, size=(total, d_laya))
    laya_doc[:, :classes] = one_hot * signal
    laya_query = rng.normal(scale=nuisance, size=(classes, d_laya))
    laya_query[:, :classes] = np.eye(classes) * signal
    teacher_doc = one_hot.copy()

    ids = [f"d{index}" for index in range(total)]
    qrels = {
        f"q{class_index}": {
            f"d{index}": 1.0 for index in range(total) if labels[index] == class_index
        }
        for class_index in range(classes)
    }
    queries = {f"q{class_index}": laya_query[class_index] for class_index in range(classes)}
    return laya_doc, teacher_doc, queries, qrels, ids


def test_evaluate_projected_improves_ndcg_and_reports_the_documented_keys():
    laya_doc, teacher_doc, queries, qrels, ids = _synthetic_corpus()

    result = evaluate_projected(laya_doc, teacher_doc, queries, qrels, ids, top_k=100)

    assert set(result) >= {"raw", "projected", "teacher_docs_only", "ndcg@10"}
    for name in ("raw", "projected", "teacher_docs_only"):
        metrics = result[name]
        assert {"nDCG@10", "MRR@10", "MAP@100"} <= set(metrics)

    raw = result["raw"]["nDCG@10"]
    projected = result["projected"]["nDCG@10"]
    assert projected > raw
    assert projected >= 0.9
    assert raw <= 0.6

    comparison = result["ndcg@10"]
    assert set(comparison) == {"raw", "projected", "teacher_docs_only", "delta"}
    assert comparison["delta"] == pytest.approx(projected - raw, abs=1e-4)
    assert result["settings"]["documents"] == laya_doc.shape[0]
    assert result["settings"]["queries"] == len(qrels)


def test_evaluate_projected_is_deterministic():
    corpus = _synthetic_corpus()
    first = evaluate_projected(*corpus, top_k=100)
    second = evaluate_projected(*corpus, top_k=100)
    assert first["ndcg@10"] == second["ndcg@10"]


def test_teacher_control_uses_the_raw_query_when_dimensions_already_match():
    laya_doc, teacher_doc, queries, qrels, ids = _synthetic_corpus()
    padding = np.zeros((teacher_doc.shape[0], laya_doc.shape[1] - teacher_doc.shape[1]))
    padded_teacher = np.hstack([teacher_doc, padding])

    result = evaluate_projected(laya_doc, padded_teacher, queries, qrels, ids, top_k=100)

    assert result["settings"]["teacher_control_query"] == "raw"


# ── CLI ──────────────────────────────────────────────────────────────────

def test_main_exits_cleanly_on_a_missing_cache(tmp_path, capsys):
    code = main(["--dataset", "no-such-dataset", "--data-root", str(tmp_path)])

    assert code == 1
    assert "missing cached Laya document embeddings" in capsys.readouterr().out


def test_main_help_guard_does_not_load_a_model():
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0


def test_module_imports_without_loading_a_model():
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(REPO_ROOT / 'tools')!r}); "
        "import projection_head; "
        "print('torch' in sys.modules or 'transformers' in sys.modules)"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"
