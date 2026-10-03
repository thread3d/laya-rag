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


def test_save_and_load_preserves_the_chunk_context(tmp_path):
    contextual = Chunk(
        chunk_id="a", doc_id="d1", path="/doc.md", text="alpha", index=0,
        start_char=0, end_char=5, context="doc.md | Title | first line",
    )
    index = VectorIndex()
    index.add([contextual], np.array([[1.0, 0.0]], dtype=np.float32))
    index.save(tmp_path / "idx")

    restored = VectorIndex.load(tmp_path / "idx")
    assert restored.get("a").context == "doc.md | Title | first line"
    assert restored.get("a").text == "alpha"


def test_an_index_written_before_context_exists_loads_with_no_context(tmp_path):
    import json

    from laya_rag.index_store import INDEX_META

    index = VectorIndex()
    index.add([make_chunk("a", "alpha")], np.array([[1.0, 0.0]], dtype=np.float32))
    index.save(tmp_path / "idx")
    meta_path = tmp_path / "idx" / INDEX_META
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    for record in payload["chunks"]:
        record.pop("context", None)
    meta_path.write_text(json.dumps(payload), encoding="utf-8")

    restored = VectorIndex.load(tmp_path / "idx")
    assert restored.get("a").context == ""


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


# ── the metadata predicate (trust / ACL pushdown) ───────────────────────

def _tenant_index() -> VectorIndex:
    """Six chunks, three per tenant, with tenant A ranked ahead of tenant B."""
    index = VectorIndex()
    chunks, vectors = [], []
    for i in range(3):
        chunks.append(make_chunk(f"a{i}", f"alpha {i}", path="a.md", index=i))
        vectors.append([1.0, i * 0.01])
    for i in range(3):
        chunks.append(make_chunk(f"b{i}", f"beta {i}", path="b.md", index=i))
        vectors.append([0.9, 0.1 + i * 0.01])
    index.add(chunks, np.array(vectors, dtype=np.float32))
    return index


def test_search_predicate_refills_from_deeper_candidates():
    """A predicate must change *which* chunks come back, not how many.

    Filtering the unfiltered top-2 would return one hit; filtering inside the search
    reaches past the excluded rows and still fills the requested k.
    """
    index = _tenant_index()
    query = np.array([1.0, 0.0], dtype=np.float32)
    unfiltered = index.search(query, k=2)
    assert all(hit.chunk.path == "a.md" for hit in unfiltered)

    filtered = index.search(query, k=2, allow=lambda path: path == "b.md")
    assert len(filtered) == 2
    assert all(hit.chunk.path == "b.md" for hit in filtered)


def test_search_predicate_returns_fewer_when_the_index_has_fewer():
    """Exhausting the permitted set is the only reason to return short."""
    index = _tenant_index()
    hits = index.search(np.array([1.0, 0.0], dtype=np.float32), k=10,
                        allow=lambda path: path == "b.md")
    assert len(hits) == 3
    assert all(hit.chunk.path == "b.md" for hit in hits)


def test_search_predicate_excluding_everything_returns_nothing():
    index = _tenant_index()
    hits = index.search(np.array([1.0, 0.0], dtype=np.float32), k=5,
                        allow=lambda path: False)
    assert hits == []


def test_search_predicate_matches_the_unfiltered_ranking_order():
    """Filtering must not reorder: permitted hits keep their cosine order."""
    index = _tenant_index()
    query = np.array([1.0, 0.0], dtype=np.float32)
    unfiltered = [hit.chunk.chunk_id for hit in index.search(query, k=6)]
    filtered = [hit.chunk.chunk_id for hit in index.search(query, k=6, allow=lambda path: True)]
    assert filtered == unfiltered


def test_search_predicate_falls_back_to_the_exact_scan_with_ann():
    """An ANN index cannot take a predicate, so it must still return the right answer.

    The ANN window is fetched wide, and when it cannot refill the requested k the exact
    scan decides — the result must equal the exact filtered result.
    """
    query = np.array([1.0, 0.0], dtype=np.float32)
    exact = _tenant_index().search(query, k=2, allow=lambda path: path == "b.md")
    index = _tenant_index()
    if not index.build_ann(nlist=1):
        pytest.skip("faiss is not installed, so there is no ANN path to fall back from")
    index.ann_nprobe = 1
    approx = index.search(query, k=2, allow=lambda path: path == "b.md")
    assert [hit.chunk.chunk_id for hit in approx] == [hit.chunk.chunk_id for hit in exact]
