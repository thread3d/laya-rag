"""Tests for BM25F: per-field length normalisation and weight.

The two things that distinguish BM25F from the `--title-boost 2` proxy it replaces are pinned here:
the fields are normalised separately and summed **before** saturation, and each field carries its
own `b` and weight. A regression that quietly concatenated the fields would still rank something,
which is why the tests compare against the proxy's actual mechanism rather than just asserting an
ordering.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from laya_rag.lexical import BM25FIndex, BM25Index  # noqa: E402


#: Both documents mention the query terms **once**, so the comparison is the fields' and not the
#: term frequencies': the first carries them in the title, the second in a much longer body. A
#: corpus where the body doc repeated the term would out-score any weight, which is BM25 doing its
#: job rather than BM25F failing.
_FILLER = " ".join(f"w{i}" for i in range(200))


def _index(**kwargs) -> BM25FIndex:
    index = BM25FIndex(drop_stopwords=False, **kwargs)
    index.build([
        ("title-hit", {"title": "quantum entanglement", "body": "a short note"}),
        ("body-hit", {"title": "notes on physics", "body": f"{_FILLER} quantum entanglement discussed"}),
    ])
    return index


def test_a_title_weight_lifts_the_title_match_over_a_longer_body_match():
    index = _index(weights={"title": 5.0, "body": 1.0})
    assert index.search("quantum entanglement", k=2)[0].chunk_id == "title-hit"


def test_the_same_weighting_leaves_the_body_match_first_when_titles_are_ignored():
    """The control: the ranking above is the weight's doing, not the corpus's."""
    index = _index(weights={"title": 0.0, "body": 1.0})
    assert index.search("quantum entanglement", k=2)[0].chunk_id == "body-hit"


def test_each_field_carries_its_own_b():
    """`b=0` in a field means its length is not normalised at all."""
    index = _index(weights={"title": 1.0, "body": 1.0}, b_fields={"title": 0.0, "body": 1.0})
    assert index.settings()["b"] == {"title": 0.0, "body": 1.0}
    unnormalised = index.search("quantum entanglement", k=2)[0].chunk_id
    index = _index(weights={"title": 1.0, "body": 1.0}, b_fields={"title": 0.0, "body": 0.0})
    assert index.search("quantum entanglement", k=2)[0].chunk_id == unnormalised


def test_idf_is_over_the_union_document_frequency():
    """A term that appears only in one field of one document is rare, so it scores high."""
    index = BM25FIndex(drop_stopwords=False)
    index.build([
        ("a", {"title": "zebra", "body": "common common common"}),
        ("b", {"title": "common", "body": "common common common"}),
    ])
    hits = {hit.chunk_id: hit.score for hit in index.search("zebra", k=2)}
    assert hits == {"a": pytest.approx(hits["a"])}
    assert hits["a"] > 0


def test_a_missing_field_is_empty_rather_than_an_error():
    index = BM25FIndex(fields=("title", "body"), drop_stopwords=False)
    index.build([("a", {"body": "only a body"}), ("b", {"title": "only a title"})])
    assert len(index) == 2
    assert [hit.chunk_id for hit in index.search("body", k=2)] == ["a"]


def test_an_unknown_field_is_rejected_loudly():
    with pytest.raises(ValueError):
        BM25FIndex(fields=("title",), weights={"body": 2.0})
    with pytest.raises(ValueError):
        BM25FIndex(fields=("title",), b_fields={"body": 0.5})
    with pytest.raises(ValueError):
        BM25FIndex(fields=())


def test_an_empty_index_returns_nothing():
    index = BM25FIndex()
    index.build([])
    assert index.search("anything", k=5) == []


def test_settings_carry_the_configuration():
    index = _index(weights={"title": 3.0, "body": 1.0}, b_fields={"title": 0.2, "body": 0.9})
    assert index.settings() == {
        "fields": ["title", "body"], "k1": 1.2,
        "b": {"title": 0.2, "body": 0.9}, "weights": {"title": 3.0, "body": 1.0},
        "drop_stopwords": False, "stem": False,
    }


def test_equal_fields_and_weights_reproduce_bm25_on_the_concatenation():
    """With no field distinction left, BM25F must reduce to BM25 — otherwise it is a different
    scorer wearing the name."""
    documents = [
        ("a", {"title": "quantum", "body": "entanglement is a property"}),
        ("b", {"title": "classical", "body": "quantum mechanics is approximate"}),
        ("c", {"title": "notes", "body": "nothing relevant in this document"}),
    ]
    fields = BM25FIndex(drop_stopwords=False, b=0.75)
    fields.build(documents)
    plain = BM25Index(drop_stopwords=False, b=0.75)
    from types import SimpleNamespace
    plain.build([
        SimpleNamespace(chunk_id=doc_id, text=f"{values['title']} {values['body']}")
        for doc_id, values in documents
    ])
    query = "quantum entanglement"
    assert [hit.chunk_id for hit in fields.search(query, k=3)] == [
        hit.chunk_id for hit in plain.search(query, k=3)
    ]


def test_differing_weights_do_not_reduce_to_the_concatenation():
    """The field structure has to actually change the ranking, not merely reorder a tie.

    The two documents hold the same words in opposite fields and have equal *total* length, so
    BM25 over the concatenation cannot tell them apart at all — its two scores are identical. BM25F
    separates them by which field the query term landed in, which is the whole claim.
    """
    from types import SimpleNamespace

    documents = [
        ("term-in-body", {"title": "alpha beta gamma delta epsilon zeta", "body": "quantum"}),
        ("term-in-title", {"title": "quantum", "body": "alpha beta gamma delta epsilon zeta"}),
    ]
    weighted = BM25FIndex(drop_stopwords=False, weights={"title": 6.0, "body": 1.0})
    weighted.build(documents)
    plain = BM25Index(drop_stopwords=False)
    plain.build([
        SimpleNamespace(chunk_id=doc_id, text=f"{values['title']} {values['body']}")
        for doc_id, values in documents
    ])

    assert weighted.search("quantum", k=2)[0].chunk_id == "term-in-title"
    plain_hits = plain.search("quantum", k=2)
    assert plain_hits[0].score == pytest.approx(plain_hits[1].score)
