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
