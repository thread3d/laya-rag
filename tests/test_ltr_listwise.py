"""The listwise objective must prefer the gold ranking, and must not be a silent pointwise.

`fit_softmax` fits a softmax over each query's candidate pool. The two properties worth pinning are that it is
actually group-aware — a candidate's score depends on its own query's pool, not on the global label rate —
and that it refuses the degenerate call, because a listwise fit with no groups is a pointwise fit wearing a
different name.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from multihop_ltr import fit_softmax  # noqa: E402


def _toy():
    # Two queries, three candidates each. Feature 0 is "ce", feature 1 a weak noise signal.
    features = np.array([
        [0.9, 0.0], [0.1, 0.0], [0.2, 0.0],      # query A: gold at row 0
        [0.2, 0.0], [0.8, 0.0], [0.3, 0.0],      # query B: gold at row 1
    ])
    labels = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    return features, labels, [(0, 3), (3, 6)]


def test_it_learns_to_put_the_gold_first():
    features, labels, offsets = _toy()
    weights = fit_softmax(features, labels, offsets)
    scores = features @ weights
    for start, end in offsets:
        pool = scores[start:end]
        assert int(np.argmax(pool)) == int(np.argmax(labels[start:end]))


def test_a_query_without_gold_contributes_no_gradient():
    """Adding a gold-free query must leave the fitted weights bit-identical.

    A query with no positives has no target distribution, so its gradient term is skipped rather than
    silently treated as "everything is a negative" — which would push every weight the wrong way.
    """
    features, labels, offsets = _toy()
    alone = fit_softmax(features, labels, offsets)

    extra_features = np.vstack([features, features[1:3]])
    extra_labels = np.concatenate([labels, np.zeros(2)])
    with_extra = fit_softmax(extra_features, extra_labels, offsets + [(len(labels), len(labels) + 2)])

    assert np.allclose(alone, with_extra)


def test_no_groups_is_refused():
    features, labels, _ = _toy()
    with pytest.raises(ValueError):
        fit_softmax(features, labels, [])
