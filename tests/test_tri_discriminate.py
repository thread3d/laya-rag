"""Offline tests for the bridge-discrimination measurement.

The question is whether the true bridge separates gold facts better than a shuffled one, and
AUC is the instrument because it is invariant to the monotone distortions that make Laya's
head a weak ranker. If the AUC implementation were wrong, "the bridge adds nothing" could be
an artefact of the measurement.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "laya_rag_tri_discriminate", ROOT / "tools" / "multihop_tri_discriminate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


disc = _load()


def test_auc_of_a_perfect_separation_is_one():
    assert disc.auc([0.9, 0.8, 0.1, 0.0], [1, 1, 0, 0]) == 1.0


def test_auc_of_a_perfect_inversion_is_zero():
    assert disc.auc([0.0, 0.1, 0.8, 0.9], [1, 1, 0, 0]) == 0.0


def test_auc_of_an_interleaved_order_is_computed_pairwise():
    """Counted by hand: positives 0.3 and 0.1 against negatives 0.4 and 0.2.

    Concordant (positive above negative) pairs: only 0.3 > 0.2. That is 1 of 4 -> 0.25.
    """
    assert disc.auc([0.4, 0.3, 0.2, 0.1], [0, 1, 0, 1]) == 0.25


def test_auc_of_a_half_concordant_order_is_one_half():
    """Positives 0.4 and 0.1 against negatives 0.3 and 0.2: 2 of 4 pairs concordant."""
    assert disc.auc([0.4, 0.3, 0.2, 0.1], [1, 0, 0, 1]) == 0.5


def test_auc_counts_ties_as_half_credit():
    """A fully tied score carries no information and must read as chance."""
    assert disc.auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == 0.5


def test_auc_is_undefined_with_one_class():
    assert disc.auc([0.9, 0.1], [1, 1]) != disc.auc([0.9, 0.1], [1, 1])  # NaN
    assert disc.auc([0.9, 0.1], [0, 0]) != disc.auc([0.9, 0.1], [0, 0])


def test_auc_is_invariant_to_a_monotone_rescaling():
    """The property that makes AUC the right instrument for a badly-ordered head."""
    base = [0.2, 0.7, 0.35, 0.9, 0.1]
    labels = [0, 1, 0, 1, 0]
    squashed = [value ** 3 for value in base]
    assert disc.auc(base, labels) == disc.auc(squashed, labels)


def test_auc_matches_hand_computed_examples():
    """Positives 0.9 and 0.6 against negatives 0.8 and 0.7: 2 of 4 pairs concordant."""
    assert disc.auc([0.9, 0.8, 0.7, 0.6], [1, 0, 0, 1]) == 0.5
    # Both positives above both negatives: 4 of 4.
    assert disc.auc([0.9, 0.8, 0.7, 0.6], [1, 1, 0, 0]) == 1.0


def test_render_markdown_reports_every_arm_and_comparison():
    result = {
        "counts": {"queries": 400, "candidates": 40000, "positives": 5000, "defined": 398},
        "summary": {
            "real": {"mean_auc": 0.71, "queries": 398},
            "shuffled": {"mean_auc": 0.62, "queries": 398},
            "ce": {"mean_auc": 0.88, "queries": 398},
        },
        "comparisons": {
            "real vs shuffled AUC": {
                "bootstrap": {"delta": 0.09, "ci_low": 0.05, "ci_high": 0.13, "significant": True},
                "permutation": {"p_value": 0.001},
            }
        },
    }
    text = disc.render_markdown(result)
    assert "| real | 0.7100 | 398 |" in text
    assert "real vs shuffled AUC" in text
    assert "+0.0900" in text
