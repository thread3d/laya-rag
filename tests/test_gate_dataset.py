"""The gate dataset's selection and labelling, and the adapter that feeds the features.

The point of this dataset is the *negative class*: MultiHop-RAG's 301 `null_query` records, which
its own retrieval metric excludes. Getting the labels or the sampling wrong would produce a gate
that looks calibrated and refuses the wrong things, so both are pinned here without an engine.
"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import gate_dataset as gd  # noqa: E402
from laya_rag.answerability import PassageView, extract_features, passage_view  # noqa: E402


def _records(n_answerable: int, n_null: int) -> list[dict]:
    rows = [{"query": f"a{i}", "question_type": "inference_query"} for i in range(n_answerable)]
    rows += [{"query": f"n{i}", "question_type": "null_query"} for i in range(n_null)]
    return rows


def test_null_query_records_are_the_negative_class():
    assert gd.is_null_query({"question_type": "null_query"}) is True
    assert gd.is_null_query({"question_type": "inference_query"}) is False
    assert gd.is_null_query({}) is False


def test_every_null_is_kept_and_only_the_positives_are_sampled():
    """There are 301 nulls; sampling them would discard the class being measured."""
    rows = gd.select(_records(1000, 301), limit=100, sample_seed=17)
    labels = [label for _query, label in rows]
    assert labels.count(0) == 301
    assert labels.count(1) == 100
    # the nulls come first and are untouched
    assert rows[0][0] == "n0"


def test_the_sample_is_reproducible_and_actually_samples():
    first = [q for q, label in gd.select(_records(500, 5), limit=50, sample_seed=17) if label == 1]
    second = [q for q, label in gd.select(_records(500, 5), limit=50, sample_seed=17) if label == 1]
    assert first == second
    assert len(first) == 50 and len(set(first)) == 50
    assert first != [f"a{i}" for i in range(50)]


def test_a_limit_above_the_available_positives_does_not_pad():
    rows = gd.select(_records(5, 3), limit=100, sample_seed=17)
    assert sum(1 for _q, label in rows if label == 1) == 5


def test_nulls_can_be_excluded_deliberately():
    rows = gd.select(_records(10, 4), limit=5, sample_seed=17, include_nulls=False)
    assert all(label == 1 for _q, label in rows)


def test_the_adapter_reads_a_dict_and_an_object_identically():
    class Passage:
        text = "Refunds take five business days."
        relevance = 0.8
        recall_score = 0.25
        path = "/docs/refunds.md"
        chunk_id = "abc"

    from_dict = passage_view(
        {"text": "Refunds take five business days.", "relevance": 0.8, "recall_score": 0.25,
         "path": "/docs/refunds.md", "chunk_id": "abc"}
    )
    from_object = passage_view(Passage())
    for name in ("text", "relevance", "recall_score", "path", "chunk_id"):
        assert getattr(from_dict, name) == getattr(from_object, name)


def test_the_adapter_and_a_raw_dict_produce_the_same_features():
    """`extract_features` is already dual-shape; the adapter must not change the answer.

    Which is why there is no "the adapter fixes silent zeros" test here: that premise is false
    (`_field` reads a mapping by key and an object by attribute, so a dict was never zeroed). The
    adapter is a typed boundary for tools, and this pins that it is a *neutral* one.
    """
    passage = {"text": "Refunds take five business days.", "relevance": 0.8, "path": "/d.md"}
    shaped = extract_features("how long do refunds take", [passage_view(passage)])
    unshaped = extract_features("how long do refunds take", [passage])
    assert shaped.tolist() == unshaped.tolist()


def test_a_mapping_missing_optional_fields_is_not_an_error():
    view = PassageView({"text": "only text here"})
    assert view.relevance == 0.0 and view.path is None
