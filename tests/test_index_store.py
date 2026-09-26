"""Tests for the persistent vector index."""
from __future__ import annotations

import numpy as np
import pytest

from laya_rag.chunking import Chunk
from laya_rag.index_store import IndexError_, VectorIndex, wipe


def make_chunk(chunk_id: str, text: str, doc_id: str = "d1", path: str = "/doc.md", index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        path=path,
        text=text,
        index=index,
        start_char=0,
        end_char=len(text),
    )


def test_empty_index_returns_no_hits():
    index = VectorIndex()
    assert len(index) == 0
    assert index.search(np.array([1.0, 0.0], dtype=np.float32), k=5) == []
    assert index.stats()["chunks"] == 0


def test_search_ranks_by_cosine_similarity():
    index = VectorIndex()
    chunks = [make_chunk("a", "alpha"), make_chunk("b", "beta"), make_chunk("c", "gamma")]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32)
    index.add(chunks, vectors)

    hits = index.search(np.array([1.0, 0.0], dtype=np.float32), k=2)
    assert [hit.chunk.chunk_id for hit in hits] == ["a", "c"]
    assert hits[0].score > hits[1].score


def test_scores_are_normalised_so_magnitude_does_not_matter():
    index = VectorIndex()
    index.add([make_chunk("a", "alpha")], np.array([[10.0, 0.0]], dtype=np.float32))
    hits = index.search(np.array([0.001, 0.0], dtype=np.float32), k=1)
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)


def test_adding_the_same_chunk_id_replaces_it():
    index = VectorIndex()
    index.add([make_chunk("a", "old")], np.array([[1.0, 0.0]], dtype=np.float32))
    replaced = index.add([make_chunk("a", "new")], np.array([[0.0, 1.0]], dtype=np.float32))

    assert replaced == 1
    assert len(index) == 1
    hits = index.search(np.array([0.0, 1.0], dtype=np.float32), k=1)
    assert hits[0].chunk.text == "new"


def test_length_mismatch_is_rejected():
    index = VectorIndex()
    with pytest.raises(ValueError, match="chunks but"):
        index.add([make_chunk("a", "x")], np.zeros((2, 3), dtype=np.float32))


def test_dimension_change_is_rejected():
    index = VectorIndex()
    index.add([make_chunk("a", "x")], np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(IndexError_, match="dimension changed"):
        index.add([make_chunk("b", "y")], np.zeros((1, 4), dtype=np.float32))


def test_query_dimension_is_checked():
    index = VectorIndex()
    index.add([make_chunk("a", "x")], np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(IndexError_, match="dimensions"):
        index.search(np.zeros(5, dtype=np.float32), k=1)


def test_get_returns_chunk_or_none():
    index = VectorIndex()
    index.add([make_chunk("a", "alpha")], np.zeros((1, 2), dtype=np.float32))
    assert index.get("a").text == "alpha"
    assert index.get("missing") is None


def test_remove_paths_drops_that_documents_chunks():
    index = VectorIndex()
    index.add(
        [make_chunk("a", "one", "d1", "/a.md"), make_chunk("b", "two", "d2", "/b.md")],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    removed = index.remove_paths(["/a.md"])
    assert removed == 1
    assert len(index) == 1
    assert index.get("a") is None
    assert index.get("b") is not None


def test_save_and_load_round_trip(tmp_path):
    index = VectorIndex()
    chunks = [make_chunk("a", "alpha"), make_chunk("b", "beta")]
    index.add(chunks, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    index.save(tmp_path / "idx")

    assert VectorIndex.exists_on_disk(tmp_path / "idx")
    restored = VectorIndex.load(tmp_path / "idx")

    assert len(restored) == 2
    assert restored.dim == 2
    assert restored.get("a").text == "alpha"
    hits = restored.search(np.array([0.0, 1.0], dtype=np.float32), k=1)
    assert hits[0].chunk.chunk_id == "b"


def test_save_leaves_no_temporary_files(tmp_path):
    index = VectorIndex()
    index.add([make_chunk("a", "alpha")], np.zeros((1, 2), dtype=np.float32))
    index.save(tmp_path / "idx")
    leftovers = [p.name for p in (tmp_path / "idx").iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_load_rejects_missing_index(tmp_path):
    with pytest.raises(IndexError_, match="index.json is missing"):
        VectorIndex.load(tmp_path / "absent")


def test_load_rejects_unknown_format_version(tmp_path):
    target = tmp_path / "idx"
    index = VectorIndex()
    index.add([make_chunk("a", "alpha")], np.zeros((1, 2), dtype=np.float32))
    index.save(target)

    meta = target / "index.json"
    meta.write_text(meta.read_text().replace('"format_version": 1', '"format_version": 99'))
    with pytest.raises(IndexError_, match="not supported"):
        VectorIndex.load(target)


def test_stats_counts_documents_and_paths():
    index = VectorIndex()
    index.add(
        [
            make_chunk("a", "one", "d1", "/a.md"),
            make_chunk("b", "two", "d1", "/a.md"),
            make_chunk("c", "three", "d2", "/b.md"),
        ],
        np.zeros((3, 2), dtype=np.float32),
    )
    stats = index.stats()
    assert stats == {"chunks": 3, "documents": 2, "dim": 2, "paths": 2}


def test_wipe_removes_the_directory(tmp_path):
    target = tmp_path / "idx"
    target.mkdir()
    (target / "f").write_text("x")
    wipe(target)
    assert not target.exists()
