"""The listwise adapter's one piece of real logic: mapping a returned order back to document ids.

A listwise reranker returns a ranked list of *positions*, and the positions are the model's own indexing
of the document list it was given. Mapping those back to the pool's document ids is where a silent bug
would live — an off-by-one or an order-confusion would produce a plausible ranking of the wrong documents,
and no metric would flag it because every metric would be computed from the same wrong mapping.

The model is stubbed: this tests the adapter, not the model.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import listwise_rerank as lr  # noqa: E402


class FakeListwise:
    """Returns the documents in a fixed order, the way a listwise model returns positions."""

    def __init__(self, order: list[int]) -> None:
        self.order = order
        self.calls = 0
        self.seen: list[str] = []

    def rerank(self, *, query, documents, top_n=None):
        self.calls += 1
        self.seen = list(documents)
        return [
            {"index": index, "relevance_score": 1.0 / (rank + 1), "document": documents[index]}
            for rank, index in enumerate(self.order[: (top_n or len(documents))])
        ]


def test_positions_map_back_to_the_pool_ids_in_the_returned_order():
    corpus = {"d0": "zero", "d1": "one", "d2": "two"}
    pools = {"q1": ["d0", "d1", "d2"]}
    ranked = lr.score_listwise(FakeListwise([2, 0, 1]), None, {"q1": "q"}, corpus, pools, top_k=3)
    assert ranked["q1"] == ["d2", "d0", "d1"]


def test_the_pool_is_what_the_model_sees_not_the_whole_corpus():
    corpus = {"d0": "zero", "d1": "one", "d2": "two", "d9": "never pooled"}
    model = FakeListwise([0, 1])
    lr.score_listwise(model, None, {"q1": "q"}, corpus, {"q1": ["d1", "d0"]}, top_k=2)
    assert model.seen == ["one", "zero"], "the model must see the pool, in the pool's order"


def test_top_k_truncates_but_does_not_reorder():
    corpus = {"a": "A", "b": "B", "c": "C"}
    ranked = lr.score_listwise(FakeListwise([1, 2, 0]), None, {"q": "q"}, corpus,
                               {"q": ["a", "b", "c"]}, top_k=2)
    assert ranked["q"] == ["b", "c"]


def test_every_query_is_reranked_once():
    corpus = {"a": "A", "b": "B"}
    model = FakeListwise([1, 0])
    pools = {"q1": ["a", "b"], "q2": ["b", "a"]}
    ranked = lr.score_listwise(model, None, {"q1": "x", "q2": "y"}, corpus, pools, top_k=2)
    assert model.calls == 2
    assert ranked["q1"] == ["b", "a"] and ranked["q2"] == ["a", "b"]
