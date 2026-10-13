"""BM25F's IDF input cannot be derived from per-field document frequencies.

The on-disk BM25F design composes one index per field, and the tempting shortcut is to derive the
union document frequency from the per-field ones. It is wrong in both available directions — `max`
under-counts a term that appears in *different* documents in each field, the sum over-counts one that
appears in the *same* document in both — so the count has to be accumulated while building.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from laya_rag.ondisk import union_document_frequency  # noqa: E402


def test_a_term_in_different_documents_counts_each_document_once():
    """The case `max` gets wrong: title and body mention the term in *different* documents."""
    documents = [
        (["refund"], []),          # doc 1: title only
        ([], ["refund"]),          # doc 2: body only
    ]
    assert union_document_frequency(documents) == {"refund": 2}


def test_a_term_in_both_fields_of_one_document_counts_once():
    """The case the sum gets wrong: both fields of the *same* document."""
    documents = [(["refund"], ["refund"])]
    assert union_document_frequency(documents) == {"refund": 1}


def test_tokens_are_deduplicated_within_and_across_fields():
    documents = [(["a", "a", "b"], ["b", "c"])]
    assert union_document_frequency(documents) == {"a": 1, "b": 1, "c": 1}


def test_the_lower_and_upper_bounds_are_both_wrong_here():
    """A direct statement of the trap, so a future shortcut has to argue with this test."""
    documents = [(["x"], []), ([], ["x"]), (["x"], ["x"])]
    union = union_document_frequency(documents)
    assert union["x"] == 3
    df_title, df_body = 2, 2                 # docs 1 and 3 mention it in the title; 2 and 3 in body
    assert max(df_title, df_body) == 2 != union["x"]      # lower bound
    assert df_title + df_body == 4 != union["x"]          # upper bound


def test_nothing_indexed_is_an_empty_union():
    assert union_document_frequency([]) == {}


def test_an_index_missing_the_union_file_is_not_reusable(tmp_path):
    """`open` requires `union-df.json`, so `exists_on_disk` must too.

    A killed build on hotpotqa left every field present and the union file absent; `exists_on_disk` said
    reusable, `open` raised, and the fallback silently recomputed the union over 5,232,329 documents — about
    thirty minutes of work that the completeness check could have named instead.
    """
    from laya_rag.ondisk import OnDiskBM25F

    directory = tmp_path / "idx"
    index = OnDiskBM25F(directory, fields=("title", "body"))
    index.build({"a": {"title": "refunds", "body": "refunds take five days"}})
    assert OnDiskBM25F.exists_on_disk(directory) is True
    (directory / "union-df.json").unlink()
    assert OnDiskBM25F.exists_on_disk(directory) is False


def test_the_merge_phase_reports_its_progress(tmp_path, capsys, monkeypatch):
    """The instrument added for a silence that cost three rounds: a line per merge group.

    Two things have to be forced, and my first attempt forced only one: **many runs** (`block_docs=1`) *and* a
    **fan-in below the run count**, because the collapse loop only runs while the level exceeds the fan-in. With
    the default fan-in and twenty runs, nothing merges and the test proves nothing about the line.
    """
    import laya_rag.ondisk as od
    from laya_rag.ondisk import OnDiskBM25

    monkeypatch.setattr(od, "_MERGE_FAN_IN", 2)
    docs = [(str(i), f"shared term{i} words here") for i in range(20)]
    index = OnDiskBM25(tmp_path / "idx", drop_stopwords=False, stem=False)
    index.build(docs, block_docs=1)
    out = capsys.readouterr().out
    assert "merge" in out and "at this level" in out
    assert len(index.search("shared words", 5)) > 0
