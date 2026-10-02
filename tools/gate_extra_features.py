"""Experimental evidence-shape features for the answerability gate.

These three signals are now part of ``laya_rag.answerability.FEATURE_NAMES`` (the last
three entries). This module re-exports them so the probe tooling and its tests keep working
against one implementation, and so the experimental record stays readable.

* ``max_passage_coverage`` — can any single passage answer the question by itself?
* ``coverage_spread`` — passages a greedy set cover needs; high means distributed answers.
* ``pairwise_redundancy`` — mean Jaccard overlap between passages.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laya_rag.answerability import (  # noqa: E402
    SHAPE_FEATURE_NAMES,
    coverage_spread,
    extract_features,
    max_passage_coverage,
    pairwise_redundancy,
)

EXTRA_NAMES: tuple[str, ...] = SHAPE_FEATURE_NAMES


def extra_features(query: str, passages) -> "object":
    """The three experimental features in :data:`EXTRA_NAMES` order."""
    import numpy as np

    return np.array(
        [
            max_passage_coverage(query, passages),
            coverage_spread(query, passages),
            pairwise_redundancy(passages),
        ],
        dtype=np.float64,
    )


__all__ = [
    "EXTRA_NAMES",
    "coverage_spread",
    "extra_features",
    "extract_features",
    "max_passage_coverage",
    "pairwise_redundancy",
]
