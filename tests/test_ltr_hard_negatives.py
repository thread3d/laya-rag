"""The hard-negative selector must never return a gold candidate.

§P11's last axis mines negatives from the CE cache's pool, which lists candidates in retrieval order. The one property that
makes a mined negative usable is that it is *not* gold — a selector that leaked a gold passage would train the ranker to
demote a correct answer, and no metric downstream would reveal it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from multihop_ltr import select_hard_negatives  # noqa: E402


CACHE = {
    "golds": {"q1": ["gold-a"], "q2": []},
    "queries": {
        "q1": {"pool": ["gold-a", "neg-1", "neg-2", "neg-3"]},
        "q2": {"pool": ["x", "y"]},
        "q3": {"pool": []},
    },
}


def test_it_never_returns_a_gold_candidate():
    picked = select_hard_negatives(CACHE, 2)
    assert "gold-a" not in picked["q1"]
    assert picked["q1"] == ["neg-1", "neg-2"]


def test_it_keeps_retrieval_order_and_respects_count():
    assert select_hard_negatives(CACHE, 3)["q1"] == ["neg-1", "neg-2", "neg-3"]
    assert select_hard_negatives(CACHE, 99)["q1"] == ["neg-1", "neg-2", "neg-3"]


def test_a_query_without_golds_keeps_its_whole_pool():
    assert select_hard_negatives(CACHE, 2)["q2"] == ["x", "y"]


def test_an_empty_pool_yields_no_negatives_rather_than_failing():
    assert select_hard_negatives(CACHE, 2)["q3"] == []


def test_mismatched_identity_spaces_are_refused_rather_than_silently_leaking_golds():
    """The real cache stores golds as passage texts and pool entries as url#chunk ids, which never intersect.

    A selector that compared them blindly would call every candidate non-gold — including true golds — and train the ranker
    to demote correct answers, with no metric downstream to reveal it. So it refuses.
    """
    import pytest

    cache = {"golds": {"q": ["a gold passage text"]}, "queries": {"q": {"pool": ["https://x#1", "https://x#2"]}}}
    with pytest.raises(ValueError, match="different identity spaces"):
        select_hard_negatives(cache, 5)


def test_a_nonsense_count_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        select_hard_negatives(CACHE, 0)
