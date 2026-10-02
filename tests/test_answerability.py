"""Tests for the feature-based answerability gate."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pytest

from laya_rag.answerability import (
    FEATURE_NAMES,
    LogisticGate,
    evaluate_gate,
    extract_features,
    roc_auc,
)

EXPECTED_FEATURES = (
    "max_relevance",
    "mean_relevance",
    "min_relevance",
    "max_recall",
    "mean_recall",
    "distinct_document_ratio",
    "passage_count",
    "log_evidence_chars",
    "query_term_coverage",
)


@dataclass
class _Passage:
    """A stand-in for ``laya_rag.rerank.ScoredPassage`` with just the gate inputs."""

    text: str
    relevance: float
    recall_score: float
    path: str


def _feature(name: str) -> int:
    return FEATURE_NAMES.index(name)


def _separable_dataset(seed: int = 7, noise: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Two classes separated by the ``max_relevance`` column, plus a little noise."""
    rng = np.random.default_rng(seed)
    n = 100
    y = np.array([0.0] * 50 + [1.0] * 50)
    X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    X[:, _feature("max_relevance")] = y + rng.normal(0.0, noise, n)
    X[:, 1:] = rng.normal(0.0, 0.05, (n, len(FEATURE_NAMES) - 1))
    return X, y


# -- feature names ---------------------------------------------------------


def test_feature_names_are_documented_and_ordered():
    assert FEATURE_NAMES == EXPECTED_FEATURES
    assert len(FEATURE_NAMES) == 9
    assert LogisticGate.FEATURE_NAMES is FEATURE_NAMES
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


# -- extract_features ------------------------------------------------------


def test_extract_features_length_and_ordering():
    passages = [
        _Passage("alpha", relevance=0.2, recall_score=1.0, path="/a.md"),
        _Passage("beta", relevance=0.8, recall_score=3.0, path="/a.md"),
        _Passage("gamma", relevance=0.5, recall_score=2.0, path="/b.md"),
    ]
    features = extract_features("alpha beta", passages)

    assert features.shape == (len(FEATURE_NAMES),)
    assert features.dtype == np.float64

    assert features[_feature("max_relevance")] == pytest.approx(0.8)
    assert features[_feature("mean_relevance")] == pytest.approx(0.5)
    assert features[_feature("min_relevance")] == pytest.approx(0.2)
    assert features[_feature("max_recall")] == pytest.approx(3.0)
    assert features[_feature("mean_recall")] == pytest.approx(2.0)
    assert features[_feature("distinct_document_ratio")] == pytest.approx(2 / 3)
    assert features[_feature("passage_count")] == pytest.approx(3.0)
    assert features[_feature("log_evidence_chars")] == pytest.approx(math.log1p(14))
    assert features[_feature("query_term_coverage")] == pytest.approx(1.0)


def test_extract_features_uses_chunk_id_when_path_missing():
    class _NoPath:
        text = "refund policy"
        relevance = 0.7
        recall_score = 2.0
        chunk_id = "chunk-1"

    features = extract_features("refund", [_NoPath(), _NoPath()])
    # Both passages share one identity, so the ratio is 1/2 rather than 2/2.
    assert features[_feature("distinct_document_ratio")] == pytest.approx(0.5)


def test_extract_features_accepts_mapping_passages():
    # The JSON/HTTP surface carries passages as plain dicts; the features must match
    # the attribute-bearing ScoredPassage form exactly.
    as_objects = [_Passage("refund policy", 0.6, 2.0, "/a.md")]
    as_dicts = [{"text": "refund policy", "relevance": 0.6, "recall_score": 2.0, "path": "/a.md"}]
    np.testing.assert_array_equal(
        extract_features("refund policy", as_dicts),
        extract_features("refund policy", as_objects),
    )


def test_query_term_coverage_full_partial_and_none():
    passages = [_Passage("refund policy details", 0.5, 1.0, "/a.md")]
    assert extract_features("refund policy", passages)[_feature("query_term_coverage")] == pytest.approx(1.0)
    assert extract_features("refund invoice", passages)[_feature("query_term_coverage")] == pytest.approx(0.5)
    assert extract_features("banana", passages)[_feature("query_term_coverage")] == pytest.approx(0.0)


def test_query_term_coverage_ignores_stopwords_and_empty_query():
    passages = [_Passage("refund policy", 0.5, 1.0, "/a.md")]
    # A pure stopword query has no content terms, so coverage is defined as 0.0.
    assert extract_features("the and of", passages)[_feature("query_term_coverage")] == pytest.approx(0.0)
    assert extract_features("", passages)[_feature("query_term_coverage")] == pytest.approx(0.0)


def test_query_term_coverage_is_case_insensitive():
    passages = [_Passage("Refund POLICY", 0.5, 1.0, "/a.md")]
    assert extract_features("refund policy", passages)[_feature("query_term_coverage")] == pytest.approx(1.0)


def test_extract_features_empty_passages_is_all_zeros():
    features = extract_features("anything at all", [])
    assert features.shape == (len(FEATURE_NAMES),)
    np.testing.assert_array_equal(features, np.zeros(len(FEATURE_NAMES)))
    assert features[_feature("passage_count")] == 0.0


def test_extract_features_is_defensive_about_missing_attributes():
    features = extract_features("refund", [object(), object()])
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(features).all()
    assert features[_feature("max_relevance")] == 0.0
    assert features[_feature("log_evidence_chars")] == 0.0
    assert features[_feature("query_term_coverage")] == 0.0


# -- fitting and prediction ------------------------------------------------


def test_fit_separates_a_synthetic_dataset():
    X, y = _separable_dataset()
    gate = LogisticGate()
    assert gate.fit(X, y) is gate

    probabilities = gate.predict_proba(X)
    assert probabilities.shape == (X.shape[0],)
    assert probabilities.dtype == np.float64
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))
    assert probabilities[y == 1].mean() > probabilities[y == 0].mean()

    predictions = gate.predict(X)
    assert predictions.shape == (X.shape[0],)
    assert predictions.dtype == bool

    assert roc_auc(y, probabilities) > 0.9
    metrics = evaluate_gate(y, probabilities)
    assert metrics["balanced_accuracy"] > 0.85


def test_fit_is_deterministic():
    X, y = _separable_dataset()
    first = LogisticGate().fit(X, y).predict_proba(X)
    second = LogisticGate().fit(X, y).predict_proba(X)
    # Zero initialisation, no shuffling and no sampling: two fits agree to the
    # limit of the platform's floating-point transcendentals.
    np.testing.assert_allclose(first, second, rtol=1e-9, atol=1e-12)


def test_fit_is_not_dominated_by_a_majority_class():
    rng = np.random.default_rng(3)
    n_neg, n_pos = 90, 10
    y = np.array([0.0] * n_neg + [1.0] * n_pos)
    X = np.zeros((n_neg + n_pos, len(FEATURE_NAMES)))
    X[:, _feature("max_relevance")] = y + rng.normal(0.0, 0.2, n_neg + n_pos)
    X[:, 1:] = rng.normal(0.0, 0.05, X[:, 1:].shape)

    gate = LogisticGate().fit(X, y)
    probabilities = gate.predict_proba(X)
    assert roc_auc(y, probabilities) > 0.9
    assert evaluate_gate(y, probabilities)["balanced_accuracy"] > 0.8

    predictions = gate.predict(X)
    # A majority-class predictor would score 0.5 balanced accuracy here; the gate
    # should instead recover the 10% minority while keeping the majority mostly right.
    assert np.mean(predictions[y == 1]) > 0.8  # minority recall
    assert np.mean(predictions[y == 0]) < 0.2  # majority false-positive rate


def test_predict_threshold_is_respected():
    X, y = _separable_dataset()
    gate = LogisticGate().fit(X, y)
    probabilities = gate.predict_proba(X)
    np.testing.assert_array_equal(gate.predict(X, threshold=0.5), probabilities >= 0.5)
    assert gate.predict(X, threshold=1.1).sum() == 0
    assert gate.predict(X, threshold=-0.1).sum() == X.shape[0]


def test_predict_before_fit_raises():
    with pytest.raises(ValueError):
        LogisticGate().predict_proba(np.zeros((1, len(FEATURE_NAMES))))


def test_repr_mentions_feature_count():
    assert str(len(FEATURE_NAMES)) in repr(LogisticGate())


# -- persistence -----------------------------------------------------------


def test_save_load_round_trip_preserves_predictions(tmp_path):
    X, y = _separable_dataset()
    gate = LogisticGate().fit(X, y)
    path = tmp_path / "answerability.json"
    gate.save(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["feature_names"] == list(FEATURE_NAMES)

    loaded = LogisticGate.load(path)
    np.testing.assert_allclose(loaded.predict_proba(X), gate.predict_proba(X), rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(loaded.predict(X), gate.predict(X))


def test_load_rejects_tampered_feature_names(tmp_path):
    X, y = _separable_dataset()
    gate = LogisticGate().fit(X, y)
    path = tmp_path / "gate.json"
    gate.save(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["feature_names"] = ["wrong"] + payload["feature_names"][1:]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        LogisticGate.load(path)


def test_load_rejects_unknown_version(tmp_path):
    X, y = _separable_dataset()
    gate = LogisticGate().fit(X, y)
    path = tmp_path / "gate.json"
    gate.save(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        LogisticGate.load(path)


# -- metrics ---------------------------------------------------------------


def test_roc_auc_hand_computed():
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert roc_auc(labels, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    assert roc_auc(np.array([1, 1, 1]), np.array([0.2, 0.5, 0.9])) == pytest.approx(0.5)
    assert roc_auc(np.array([0, 0, 0]), np.array([0.2, 0.5, 0.9])) == pytest.approx(0.5)


def test_roc_auc_partial_ties():
    # One positive beats one negative and ties the other: (1 + 0.5) / 2 = 0.75.
    assert roc_auc(np.array([0, 0, 1]), np.array([0.1, 0.5, 0.5])) == pytest.approx(0.75)


def test_evaluate_gate_returns_documented_keys():
    X, y = _separable_dataset()
    probabilities = LogisticGate().fit(X, y).predict_proba(X)
    metrics = evaluate_gate(y, probabilities)
    assert set(metrics) == {"auc", "balanced_accuracy", "n"}
    assert metrics["n"] == float(len(y))
    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0


def test_evaluate_gate_balanced_accuracy_is_mean_class_recall():
    # Both positives are missed, so balanced accuracy is the average of 0.0 and 1.0.
    metrics = evaluate_gate(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4]))
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)


# -- input validation ------------------------------------------------------


def test_fit_rejects_empty_input():
    with pytest.raises(ValueError):
        LogisticGate().fit(np.zeros((0, len(FEATURE_NAMES))), np.zeros(0))
    with pytest.raises(ValueError):
        LogisticGate().fit(np.array([]), np.array([]))


def test_fit_rejects_single_class_labels():
    X = np.zeros((4, len(FEATURE_NAMES)))
    with pytest.raises(ValueError):
        LogisticGate().fit(X, np.ones(4))
    with pytest.raises(ValueError):
        LogisticGate().fit(X, np.zeros(4))
