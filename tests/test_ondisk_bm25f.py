"""BM25F's on-disk build: one field index each, plus the union document frequency.

The composition leans on what `OnDiskBM25` already stores, so these tests are about the three
structural constraints rather than the formula — a field must not overwrite another, the union must
be accumulated rather than derived, and the whole thing must be re-iterable-safe for a corpus that is
a mapping the caller already holds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from laya_rag.ondisk import OnDiskBM25, OnDiskBM25F  # noqa: E402

DOCS = {
    "d1": {"title": "refund policy", "body": "refunds take five business days"},
    "d2": {"title": "shipping", "body": "refund requests need a receipt"},
    "d3": {"title": "refund policy", "body": "refunds are processed quickly"},
}


def _built(tmp_path: Path, **kwargs) -> OnDiskBM25F:
    index = OnDiskBM25F(tmp_path / "bm25f", fields=("title", "body"), stem=False, **kwargs)
    index.build(DOCS)
    return index


def test_nothing_exists_until_it_is_built(tmp_path):
    assert not OnDiskBM25F.exists_on_disk(tmp_path / "bm25f")
    _built(tmp_path)
    assert OnDiskBM25F.exists_on_disk(tmp_path / "bm25f")


def test_each_field_gets_its_own_directory_and_they_do_not_overwrite(tmp_path):
    """Sharing one directory makes the second field overwrite the first — a one-field index
    wearing a two-field design."""
    index = _built(tmp_path)
    assert OnDiskBM25.exists_on_disk(tmp_path / "bm25f" / "title.bm25")
    assert OnDiskBM25.exists_on_disk(tmp_path / "bm25f" / "body.bm25")
    assert index._indexes["title"].n_docs == 3
    assert index._indexes["body"].n_docs == 3
    # `shipping` is a title-only term: present in one index, absent from the other, which is the
    # evidence that the two are genuinely separate postings.
    assert "shipping" in index._indexes["title"].terms
    assert "shipping" not in index._indexes["body"].terms


def test_the_union_document_frequency_counts_each_document_once(tmp_path):
    """`refund`/`refunds` appear across two fields and three documents; no per-field DF can say so."""
    union = _built(tmp_path).union_df
    assert union["refund"] == 3          # d1 body, d2 body, d3 title+body — one document each
    assert union["policy"] == 2         # d1 title, d3 title
    assert union["shipping"] == 1       # d2 title only


def test_the_union_is_persisted_for_the_scoring_pass(tmp_path):
    index = _built(tmp_path)
    written = json.loads((tmp_path / "bm25f" / "union-df.json").read_text(encoding="utf-8"))
    assert written == index.union_df
    assert written["refund"] == 3


def test_the_build_streams_a_mapping_so_a_second_field_can_still_read_it(tmp_path):
    """A generator would be exhausted by the first field; this is why the signature is a mapping."""
    index = _built(tmp_path)
    assert len(index.union_df) > 0
    assert index._indexes["body"].n_docs == len(DOCS)


def test_bm25f_needs_at_least_one_field(tmp_path):
    with pytest.raises(ValueError):
        OnDiskBM25F(tmp_path / "bm25f", fields=())


# ── search: the loop over the field postings ──────────────────────────────

DOCS_BODY = {
    "d1": {"body": "refunds take five business days and refunds are quick"},
    "d2": {"body": "shipping is free and refunds need a receipt"},
    "d3": {"body": "support is open on weekdays"},
}


def test_search_ranks_by_bm25f_and_names_the_documents(tmp_path):
    index = _built(tmp_path)
    hits = index.search("refund", k=10)
    ids = [hit.chunk_id for hit in hits]
    assert set(ids) == {"d1", "d2", "d3"}          # refunds/refund appear in all three
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert index.search("refund", k=0) == []
    assert index.search("nonexistentterm", k=5) == []


def test_a_title_only_term_ranks_the_document_whose_title_has_it_first(tmp_path):
    assert _built(tmp_path).search("shipping", k=3)[0].chunk_id == "d2"


def test_one_field_with_b_zero_matches_the_single_field_backend(tmp_path):
    """The parity the two backends were built for, now for the composed one."""
    from laya_rag.ondisk import OnDiskBM25

    plain = OnDiskBM25(tmp_path / "plain", k1=1.2, b=0.0, drop_stopwords=True, stem=False)
    plain.build((doc_id, payload["body"]) for doc_id, payload in DOCS_BODY.items())
    composed = OnDiskBM25F(tmp_path / "composed", fields=("body",), k1=1.2,
                           b_fields={"body": 0.0}, stem=False)
    composed.build(DOCS_BODY)

    for query in ("refund", "shipping free", "weekdays support"):
        left = [(h.chunk_id, round(h.score, 9)) for h in plain.search(query, k=5)]
        right = [(h.chunk_id, round(h.score, 9)) for h in composed.search(query, k=5)]
        assert left == right, query


def test_the_idf_is_the_union_not_a_fields_own(tmp_path):
    """Using `OnDiskBM25._idf` here would make a title-only term look common."""
    index = _built(tmp_path)
    body = index._indexes["body"]
    body_idf = body._idf(body.terms["refund"]) if "refund" in body.terms else 0.0
    union_idf = index._union_idf("refund")
    assert union_idf > 0.0
    # `refund` is in the body of two documents and the union of three, so the union IDF is smaller.
    assert union_idf < body_idf


def test_reopening_gives_the_same_ranking(tmp_path):
    built = _built(tmp_path)
    before = [(h.chunk_id, round(h.score, 9)) for h in built.search("refund policy", k=3)]
    reopened = OnDiskBM25F.open(tmp_path / "bm25f", fields=("title", "body"))
    after = [(h.chunk_id, round(h.score, 9)) for h in reopened.search("refund policy", k=3)]
    assert before == after
