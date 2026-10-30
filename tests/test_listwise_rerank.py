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


def test_the_embedding_cache_lives_beside_the_run_not_in_tmp():
    """A corpus embedding is the long part of an evaluation; `/tmp` is not durable on this host.

    The first version cached under `/tmp/listwise-dense-cache` with `use_cache=False`, which meant a re-run
    that re-derived a number would pay to embed ArguAna again — and would do it somewhere nothing ever
    cleans up or finds. Caught by reading the tool rather than by running it, which is the cheaper of the two.
    """
    import ast

    source = Path(lr.__file__).read_text(encoding="utf-8")
    # String *literals* only: the comment above the fix explains why `/tmp` was wrong, and a crude substring
    # search would fail on the explanation rather than on the behaviour.
    literals = [
        node.value for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    offenders = [value for value in literals if value.startswith("/tmp")]
    assert not offenders, f"the tool writes outside the workspace: {offenders}"
    assert any("listwise-dense-cache" in value for value in literals), "no cache path at all"
    assert "use_cache=True" in source


def test_a_long_run_reports_progress(capsys):
    """Four hours of silence is indistinguishable from a hung process.

    The first version printed nothing until the run finished. Its own cost projection (~4.5 h for 400 queries) is the
    reason that matters: `tools/engine_extractive.py` prints a line every 25 queries and this is the same rule.
    """
    corpus = {"a": "A", "b": "B"}
    pools = {"q1": ["a", "b"], "q2": ["b", "a"]}
    lr.score_listwise(FakeListwise([1, 0]), None, {"q1": "x", "q2": "y"}, corpus, pools, top_k=2,
                      progress_every=1)
    captured = capsys.readouterr().out
    assert "1/2" in captured and "2/2" in captured
    assert "s/query" in captured and "left" in captured


def test_rankings_are_streamed_to_disk_as_they_are_computed(tmp_path):
    """Four and a half hours of rankings must not live only in memory.

    The run holds its result until the report is written, so an exception anywhere in the scoring or evaluation step
    would have cost the whole pass — the failure `cached_embed`'s chunked checkpointing exists to prevent for the
    embedding half. The writer now persists each ranking as it is produced, so an interrupted run leaves a readable
    partial ranking instead of nothing.
    """
    import json as _json

    source = Path(lr.__file__).read_text(encoding="utf-8")
    assert ".rankings.jsonl" in source
    assert "sink.write(" in source and "sink.flush()" in source, "the writer must flush per query"

    # and the streaming helper yields before the next query is computed
    corpus = {"a": "A", "b": "B"}
    streamed = list(lr._iter_listwise(FakeListwise([1, 0]), None, {"q1": "x", "q2": "y"}, corpus,
                                      {"q1": ["a", "b"], "q2": ["b", "a"]}, top_k=2, progress_every=0))
    assert [qid for qid, _ in streamed] == ["q1", "q2"]
    assert isinstance(_json.dumps({"q": streamed[0]}), str)
