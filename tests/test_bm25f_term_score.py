"""The BM25F term score, as a function the on-disk path can apply to spilled postings.

The two properties a refactor can silently lose: fields are summed **before** saturation, and each
field is normalised by its own length and `b`. Both wrong versions produce plausible numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from laya_rag.lexical import bm25f_term_score  # noqa: E402


def _plain_bm25(tf: float, idf: float, k1: float = 1.2) -> float:
    return idf * tf * (k1 + 1.0) / (k1 + tf)


def test_one_field_with_b_zero_is_plain_bm25():
    """The reduction that makes this checkable against the formula everyone knows."""
    score = bm25f_term_score(
        term_frequencies={"body": 3.0}, lengths={"body": 100.0}, avg_lengths={"body": 100.0},
        weights={"body": 1.0}, b_fields={"body": 0.0}, k1=1.2, idf=2.0,
    )
    assert score == pytest.approx(_plain_bm25(3.0, 2.0))


def test_saturation_happens_once_across_fields_not_once_per_field():
    """The property that distinguishes BM25F from saturating each field and then adding."""
    shared = dict(lengths={"title": 10.0, "body": 100.0}, avg_lengths={"title": 10.0, "body": 100.0},
                  weights={"title": 1.0, "body": 1.0}, b_fields={"title": 0.0, "body": 0.0},
                  k1=1.2, idf=1.0)
    combined = bm25f_term_score(term_frequencies={"title": 1.0, "body": 1.0}, **shared)
    assert combined == pytest.approx(_plain_bm25(2.0, 1.0))
    per_field = _plain_bm25(1.0, 1.0) * 2
    assert combined != pytest.approx(per_field)          # summing after saturating is the wrong one


def test_a_field_weight_scales_that_fields_frequency():
    base = dict(lengths={"title": 10.0, "body": 100.0}, avg_lengths={"title": 10.0, "body": 100.0},
                b_fields={"title": 0.0, "body": 0.0}, k1=1.2, idf=1.0)
    weighted = bm25f_term_score(term_frequencies={"title": 1.0, "body": 0.0},
                                weights={"title": 3.0, "body": 1.0}, **base)
    assert weighted == pytest.approx(_plain_bm25(3.0, 1.0))


def test_each_field_is_normalised_by_its_own_length():
    """A long body must not dilute a short title: the proxy (repeating the title) gets this wrong."""
    common = dict(term_frequencies={"title": 2.0, "body": 2.0}, weights={"title": 1.0, "body": 1.0},
                  b_fields={"title": 0.5, "body": 0.5}, k1=1.2, idf=1.0)
    short = bm25f_term_score(lengths={"title": 5.0, "body": 50.0},
                             avg_lengths={"title": 5.0, "body": 100.0}, **common)
    long_body = bm25f_term_score(lengths={"title": 5.0, "body": 400.0},
                                 avg_lengths={"title": 5.0, "body": 100.0}, **common)
    assert long_body < short                 # the over-long body field is discounted, the title is not


def test_a_term_that_does_not_occur_scores_zero():
    assert bm25f_term_score(
        term_frequencies={"title": 0.0, "body": 0.0}, lengths={"title": 1.0, "body": 1.0},
        avg_lengths={"title": 1.0, "body": 1.0}, weights={}, b_fields={}, k1=1.2, idf=9.0,
    ) == 0.0


def test_a_missing_average_length_does_not_divide_by_zero():
    score = bm25f_term_score(
        term_frequencies={"body": 1.0}, lengths={"body": 1.0}, avg_lengths={"body": 0.0},
        weights={"body": 1.0}, b_fields={"body": 0.5}, k1=1.2, idf=1.0,
    )
    assert score > 0.0
