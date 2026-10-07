"""Tests for the feature-based answerability gate."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pytest

from laya_rag.answerability import (
    FEATURE_NAMES,
    SHAPE_FEATURE_NAMES,
    LogisticGate,
    bridge_conditioned_max,
    evaluate_gate,
    extract_features,
    passage_text,
    roc_auc,
    rotated_bridge_texts,
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
    "max_passage_coverage",
    "coverage_spread",
    "pairwise_redundancy",
    "bridge_conditioned",
)

#: The nine score features every earlier artefact names, and the shape features appended
#: after them.
SCORE_FEATURES = EXPECTED_FEATURES[:9]


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
    assert len(FEATURE_NAMES) == 13
    assert LogisticGate.FEATURE_NAMES is FEATURE_NAMES
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    # The shape features stay at 9..11; the optional bridge column follows them.
    assert SHAPE_FEATURE_NAMES == FEATURE_NAMES[9:12]


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

    # Evidence-shape features, appended after the nine score features. No single passage
    # holds both query terms, so coverage spreads across two of them.
    assert features[_feature("max_passage_coverage")] == pytest.approx(0.5)
    assert features[_feature("coverage_spread")] == pytest.approx(2.0)
    assert features[_feature("pairwise_redundancy")] == pytest.approx(0.0)


def test_shape_features_measure_a_distributed_answer():
    passages = [
        _Passage("alpha only", relevance=0.5, recall_score=1.0, path="/a.md"),
        _Passage("beta only", relevance=0.5, recall_score=1.0, path="/b.md"),
    ]
    features = extract_features("alpha beta gamma", passages)
    assert features[_feature("max_passage_coverage")] == pytest.approx(1 / 3)
    # Two passages cover two of three terms; the third is absent from the evidence. They
    # share only the word "only", so Jaccard is 1/3 rather than 0.
    assert features[_feature("coverage_spread")] == pytest.approx(2.0)
    assert features[_feature("pairwise_redundancy")] == pytest.approx(1 / 3)


def test_shape_features_report_redundant_evidence():
    passages = [
        _Passage("refund policy details", relevance=0.5, recall_score=1.0, path="/a.md"),
        _Passage("refund policy details", relevance=0.5, recall_score=1.0, path="/b.md"),
    ]
    features = extract_features("refund policy details", passages)
    assert features[_feature("pairwise_redundancy")] == pytest.approx(1.0)


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


def test_a_nine_feature_gate_still_loads_and_selects_by_name(tmp_path):
    """An artefact trained before the shape features existed must keep predicting."""
    X, y = _separable_dataset()
    X9 = X[:, :9]
    gate = LogisticGate().fit(X9, y)
    assert gate.feature_names == SCORE_FEATURES

    path = tmp_path / "gate9.json"
    gate.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["feature_names"] == list(SCORE_FEATURES)

    loaded = LogisticGate.load(path)
    assert loaded.feature_names == SCORE_FEATURES
    # The full-width vector selects the same nine columns, in the gate's order.
    np.testing.assert_allclose(
        loaded.predict_proba(X), gate.predict_proba(X9), rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        loaded.predict_proba(X9), gate.predict_proba(X9), rtol=1e-9, atol=1e-12
    )


def test_a_gate_rejects_a_vector_of_the_wrong_width(tmp_path):
    X, y = _separable_dataset()
    gate = LogisticGate().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        gate.predict_proba(X[:, :5])


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


def test_the_bridge_column_defaults_to_zero_and_carries_a_supplied_value():
    """The optional BridgeRAG column is zero unless the caller computes it."""
    passages = [_Passage("alpha", relevance=0.5, recall_score=1.0, path="/a.md")]
    index = _feature("bridge_conditioned")
    default = extract_features("alpha", passages)
    assert default[index] == 0.0
    supplied = extract_features("alpha", passages, bridge_conditioned=0.7)
    assert supplied[index] == pytest.approx(0.7)
    assert np.allclose(np.delete(default, index), np.delete(supplied, index))


def test_passage_text_reads_objects_mappings_and_retrieval_hits():
    """The accessor the bridge harness uses: a dict has no attributes, so getattr reads it empty."""
    assert passage_text(_Passage("object text", 0.5, 1.0, "/a.md")) == "object text"
    assert passage_text({"text": "mapping text", "path": "/a.md"}) == "mapping text"
    # A Hit carries its content under `chunk`, not `text`.
    assert passage_text({"chunk": {"text": "hit text"}}) == "hit text"
    assert passage_text({}) == ""
    assert passage_text(None) == ""


def test_the_bridge_harness_head_is_the_mapping_text_not_the_empty_getattr():
    """Regression: `getattr(dict, 'text', '')` is '', which made both arms identical."""
    passages = [
        {"text": "the real first hop", "path": "/a.md"},
        {"text": "candidate", "path": "/a.md"},
    ]
    scorer = _StubScorer([0.5])
    bridge_conditioned_max(scorer, "q", passages)
    assert scorer.calls[0][1] == "the real first hop"
    assert scorer.calls[0][1] == passage_text(passages[0])
    # And a rotated arm must actually receive a different head.
    rotated = rotated_bridge_texts([passage_text(p) for p in passages])
    assert rotated[0] == "candidate"


# -- the bridge-conditioned column (the 13th) ------------------------------


class _StubScorer:
    """A stand-in for ``LayaReranker`` that records the conditioned call it receives."""

    def __init__(self, values, *, error: Exception | None = None) -> None:
        self.values = list(values)
        self.error = error
        self.calls: list[tuple] = []

    def score_relevance_conditioned(self, query, bridge, texts, wording="bridge"):
        self.calls.append((query, bridge, list(texts), wording))
        if self.error is not None:
            raise self.error
        return self.values[: len(texts)], {}


def test_bridge_column_scores_the_rest_against_the_top_passage():
    passages = [
        _Passage("bridge text", relevance=0.9, recall_score=1.0, path="/a.md"),
        _Passage("candidate one", relevance=0.5, recall_score=1.0, path="/a.md"),
        _Passage("candidate two", relevance=0.4, recall_score=1.0, path="/a.md"),
    ]
    scorer = _StubScorer([0.2, 0.8])
    value = bridge_conditioned_max(scorer, "q", passages)
    assert value == pytest.approx(0.8)
    # The bridge is the top-ranked passage and never scores itself against itself.
    assert scorer.calls == [("q", "bridge text", ["candidate one", "candidate two"], "bridge")]


def test_bridge_column_accepts_a_foreign_bridge_but_still_skips_the_same_slot():
    """The shuffled arm holds the scored count fixed by excluding the same slot."""
    passages = [
        _Passage("own top", relevance=0.9, recall_score=1.0, path="/a.md"),
        _Passage("candidate one", relevance=0.5, recall_score=1.0, path="/a.md"),
    ]
    scorer = _StubScorer([0.6])
    assert bridge_conditioned_max(scorer, "q", passages, bridge="another query") == pytest.approx(0.6)
    assert scorer.calls[0][1] == "another query"
    assert scorer.calls[0][2] == ["candidate one"]


def test_bridge_column_is_unmeasured_without_a_rest():
    scorer = _StubScorer([0.5])
    assert bridge_conditioned_max(scorer, "q", []) is None
    assert bridge_conditioned_max(
        scorer, "q", [_Passage("only", relevance=0.5, recall_score=1.0, path="/a.md")]
    ) is None
    assert scorer.calls == []          # no model call for a column that cannot exist


def test_bridge_column_is_unmeasured_when_the_scorer_returns_nothing():
    passages = [
        _Passage("a", relevance=0.9, recall_score=1.0, path="/a.md"),
        _Passage("b", relevance=0.5, recall_score=1.0, path="/a.md"),
    ]
    # An empty list is "not measured", not a score of zero.
    assert bridge_conditioned_max(_StubScorer([]), "q", passages) is None


def test_bridge_column_reads_mapping_passages():
    passages = [
        {"text": "top", "relevance": 0.9, "recall_score": 1.0, "path": "/a.md"},
        {"text": "rest", "relevance": 0.5, "recall_score": 1.0, "path": "/a.md"},
    ]
    scorer = _StubScorer([0.7])
    assert bridge_conditioned_max(scorer, "q", passages) == pytest.approx(0.7)
    assert scorer.calls[0][1] == "top"


def test_bridge_column_does_not_swallow_a_scorer_failure():
    """The collection tool owns the fallback; the helper must not turn a crash into a zero."""
    passages = [
        _Passage("a", relevance=0.9, recall_score=1.0, path="/a.md"),
        _Passage("b", relevance=0.5, recall_score=1.0, path="/a.md"),
    ]
    with pytest.raises(RuntimeError):
        bridge_conditioned_max(_StubScorer([0.5], error=RuntimeError("model down")), "q", passages)


def test_the_bridge_column_reaches_the_feature_vector():
    passages = [
        _Passage("bridge text", relevance=0.9, recall_score=1.0, path="/a.md"),
        _Passage("candidate one", relevance=0.5, recall_score=1.0, path="/a.md"),
    ]
    value = bridge_conditioned_max(_StubScorer([0.8]), "q", passages)
    vector = extract_features("q", passages, bridge_conditioned=value)
    assert vector[_feature("bridge_conditioned")] == pytest.approx(0.8)


def test_rotated_bridges_never_hand_a_query_its_own_first_hop():
    texts = ["q1", "q2", "q3"]
    rotated = rotated_bridge_texts(texts)
    assert rotated == ["q2", "q3", "q1"]
    assert all(left != right for left, right in zip(texts, rotated))
    # A rotation preserves the multiset, so the control's only difference is the content.
    assert sorted(rotated) == sorted(texts)


def test_rotated_bridges_with_one_item_cannot_avoid_itself():
    assert rotated_bridge_texts(["only"]) == ["only"]


def test_rotated_bridges_rejects_an_offset_that_is_the_real_arm():
    with pytest.raises(ValueError):
        rotated_bridge_texts(["a", "b"], offset=0)
    with pytest.raises(ValueError):
        rotated_bridge_texts(["a", "b"], offset=2)   # a multiple of n rotates to itself
