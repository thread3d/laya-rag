"""Tests for the optional approximate (IVF) dense index.

Exact scanning is the default and stays exact. These pin the ANN path: that it agrees with
the exhaustive scan when it is asked to search enough cells, that it survives a save/load,
and that a missing faiss degrades to the exact scan rather than failing.
"""
from __future__ import annotations

import numpy as np
import pytest

from laya_rag.chunking import Chunk
from laya_rag.index_store import INDEX_ANN, VectorIndex


def make_chunks(count: int) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"c{i}",
            doc_id="d",
            path=f"/{i}.md",
            text=f"chunk {i}",
            index=i,
            start_char=0,
            end_char=8,
        )
        for i in range(count)
    ]


def clustered(count: int, dim: int = 32, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(8, dim)).astype(np.float32)
    assignment = rng.integers(0, len(centres), size=count)
    vectors = centres[assignment] + rng.normal(scale=0.05, size=(count, dim)).astype(np.float32)
    return vectors


def test_the_index_is_exact_until_an_ann_is_built():
    index = VectorIndex()
    index.add(make_chunks(10), clustered(10))
    assert index.is_approximate is False


def test_building_an_ann_marks_the_index_approximate():
    pytest.importorskip("faiss")
    index = VectorIndex()
    index.add(make_chunks(200), clustered(200))
    assert index.build_ann() is True
    assert index.is_approximate is True


def test_ann_search_agrees_with_exact_search_when_it_searches_every_cell():
    pytest.importorskip("faiss")
    vectors = clustered(300)
    index = VectorIndex()
    index.add(make_chunks(300), vectors)
    query = vectors[17]

    exact = index.search(query, k=5)
    index.ann_nprobe = 300  # nlist is sqrt(300) ~ 17, so this visits every cell
    index.build_ann()
    approximate = index.search(query, k=5)

    assert [hit.chunk.chunk_id for hit in approximate] == [hit.chunk.chunk_id for hit in exact]


def test_ann_survives_a_save_and_load(tmp_path):
    pytest.importorskip("faiss")
    index = VectorIndex()
    index.add(make_chunks(200), clustered(200))
    index.build_ann()
    index.save(tmp_path)

    assert (tmp_path / INDEX_ANN).is_file()
    restored = VectorIndex.load(tmp_path)
    assert restored.is_approximate is True
    assert restored.search(clustered(200)[3], k=3)


def test_an_index_without_an_ann_file_loads_as_exact(tmp_path):
    index = VectorIndex()
    index.add(make_chunks(20), clustered(20))
    index.save(tmp_path)

    restored = VectorIndex.load(tmp_path)
    assert restored.is_approximate is False


def test_an_ann_over_an_empty_index_is_a_no_op():
    assert VectorIndex().build_ann() is False


def test_engine_builds_an_ann_only_when_configured(tmp_path):
    from .support import StubModel, make_config, write_corpus
    from laya_rag.engine import RagEngine

    plain = RagEngine(make_config(tmp_path, ann=False), model=StubModel(dim=32), embedder=StubModel(dim=32))
    plain.index_paths([write_corpus(tmp_path / "corpus")])
    assert plain.index.is_approximate is False

    configured = RagEngine(
        make_config(tmp_path / "second", ann=True, ann_nprobe=4),
        model=StubModel(dim=32),
        embedder=StubModel(dim=32),
    )
    configured.index_paths([write_corpus(tmp_path / "second" / "corpus")])
    assert configured.index.is_approximate is True
    assert configured.index.ann_nprobe == 4
