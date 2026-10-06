"""Tests for the on-disk engine backend (``LAYA_RAG_INDEX_BACKEND=ondisk``).

The point of the backend is that query-time memory stops scaling with corpus size while the
*ranking* stays identical to the in-memory path. These tests pin both: the on-disk BM25 and
streamed vector store must reproduce the in-memory ranking, and chunk text must round-trip
through the deferred sidecar.
"""
from __future__ import annotations

import numpy as np
import pytest

from laya_rag.chunking import Chunk, chunk_documents, load_documents
from laya_rag.config import EngineConfig
from laya_rag.engine import RagEngine
from laya_rag.lexical import BM25Index
from laya_rag.ondisk import BM25_DIRNAME, DENSE_DIRNAME, OnDiskBM25, OnDiskVectorIndex

from .support import StubModel, make_config, write_corpus


def _chunks(root):
    loaded = load_documents([root])
    return chunk_documents(loaded.documents)


def _chunk(index: int, text: str, doc_id: str = "d", path: str = "d.md") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}:{index:04d}",
        doc_id=doc_id,
        path=path,
        text=text,
        index=index,
        start_char=0,
        end_char=len(text),
    )


# ── config ───────────────────────────────────────────────────────────────

def test_index_backend_defaults_to_memory():
    assert EngineConfig().resolved_index_backend == "memory"


def test_an_unknown_index_backend_is_a_configuration_error():
    with pytest.raises(ValueError):
        EngineConfig(index_backend="quantum").resolved_index_backend


# ── lexical parity ───────────────────────────────────────────────────────

def test_ondisk_bm25_matches_the_in_memory_ranking(tmp_path):
    chunks = _chunks(write_corpus(tmp_path / "corpus"))
    memory = BM25Index(stem=True)
    memory.build(chunks)
    ondisk = OnDiskBM25(tmp_path / "bm25", stem=True)
    ondisk.build((chunk.chunk_id, chunk.text) for chunk in chunks)
    reopened = OnDiskBM25.open(tmp_path / "bm25")

    for query in ("refund business days", "shipping international", "password security", "rate limit"):
        expected = [hit.chunk_id for hit in memory.search(query, k=10)]
        assert [hit.chunk_id for hit in ondisk.search(query, k=10)] == expected
        assert [hit.chunk_id for hit in reopened.search(query, k=10)] == expected


def test_ondisk_bm25_ranking_is_stable_across_build_blocks(tmp_path):
    """A token- or doc-capped block forces many runs through the heap merge; the
    ranking must stay byte-identical to the single-run build."""
    chunks = _chunks(write_corpus(tmp_path / "corpus"))
    memory = BM25Index(stem=True)
    memory.build(chunks)
    queries = ("refund business days", "shipping international", "password security")

    for block_docs, block_tokens in ((1, 1_000_000), (2, 5), (1000, 1_000_000)):
        directory = tmp_path / f"bm25-{block_docs}-{block_tokens}"
        ondisk = OnDiskBM25(directory, stem=True)
        ondisk.build(
            ((chunk.chunk_id, chunk.text) for chunk in chunks),
            block_docs=block_docs,
            block_tokens=block_tokens,
        )
        reopened = OnDiskBM25.open(directory)
        for query in queries:
            expected = [hit.chunk_id for hit in memory.search(query, k=10)]
            assert [hit.chunk_id for hit in reopened.search(query, k=10)] == expected
        # Every term's offset window must line up with its postings, not just the ones
        # the sampled queries happen to touch.
        assert int(reopened.offsets[0]) == 0
        assert int(reopened.offsets[-1]) == len(reopened._postings)


def test_ondisk_bm25_rejects_a_zero_block_budget(tmp_path):
    index = OnDiskBM25(tmp_path / "bm25", stem=True)
    with pytest.raises(ValueError):
        index.build([("d:0000", "alpha")], block_tokens=0)
    with pytest.raises(ValueError):
        index.build([("d:0000", "alpha")], block_docs=0)


# ── vector store ─────────────────────────────────────────────────────────

def test_ondisk_vector_index_defers_text_and_round_trips(tmp_path):
    index = OnDiskVectorIndex(tmp_path / "dense", dim=0)
    chunks = [_chunk(0, "alpha document text"), _chunk(1, "beta document text")]
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index.add(chunks, vectors)
    # A second append exercises the offset splice.
    index.add([_chunk(2, "gamma document text")], np.asarray([[0.7, 0.7]], dtype=np.float32))
    index.finalize()

    opened = OnDiskVectorIndex.open(tmp_path / "dense")
    assert len(opened) == 3
    assert opened.get("d:0001").text == "beta document text"
    assert opened.get("missing") is None
    assert np.allclose(opened.vector("d:0000"), [1.0, 0.0])
    hits = opened.search(np.asarray([1.0, 0.0], dtype=np.float32), k=2)
    assert hits[0].chunk.chunk_id == "d:0000"
    assert hits[0].chunk.text == "alpha document text"


def test_ondisk_vector_index_rejects_duplicate_ids(tmp_path):
    index = OnDiskVectorIndex(tmp_path / "dense", dim=2)
    chunk = _chunk(0, "alpha document text")
    index.add([chunk], np.asarray([[1.0, 0.0]], dtype=np.float32))
    with pytest.raises(Exception):
        index.add([chunk], np.asarray([[1.0, 0.0]], dtype=np.float32))


# ── engine wiring ────────────────────────────────────────────────────────

def _ondisk_engine(tmp_path):
    config = make_config(tmp_path, index_backend="ondisk", dense_weight=1.0)
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([write_corpus(tmp_path / "corpus")], rebuild=True)
    return engine


def test_ondisk_engine_answers_with_evidence(tmp_path):
    engine = _ondisk_engine(tmp_path)
    assert (tmp_path / "index" / DENSE_DIRNAME).is_dir()
    assert (tmp_path / "index" / BM25_DIRNAME).is_dir()
    result = engine.query("How long does a refund take?")
    assert result.passages
    assert result.answerability_measured is True
    assert engine.stats()["index"]["chunks"] > 0


def test_ondisk_engine_requires_rebuild_for_incremental_indexing(tmp_path):
    engine = _ondisk_engine(tmp_path)
    with pytest.raises(ValueError):
        engine.index_paths([write_corpus(tmp_path / "corpus")])


def test_memory_and_ondisk_backends_agree_on_the_top_passage(tmp_path):
    corpus = write_corpus(tmp_path / "corpus")
    memory = RagEngine(make_config(tmp_path, dense_weight=1.0), model=StubModel())
    memory.index_paths([corpus], rebuild=True)
    ondisk = RagEngine(
        make_config(tmp_path / "od", index_backend="ondisk", dense_weight=1.0), model=StubModel()
    )
    ondisk.index_paths([corpus], rebuild=True)

    for query in ("How long does a refund take?", "Is shipping free?", "How do I reset my password?"):
        expected = memory.query(query).passages
        actual = ondisk.query(query).passages
        assert expected and actual
        assert actual[0]["path"] == expected[0]["path"]


def test_ondisk_lexical_channel_reproduces_in_memory_ids(tmp_path):
    corpus = write_corpus(tmp_path / "corpus")
    memory = RagEngine(make_config(tmp_path, dense_weight=1.0), model=StubModel())
    memory.index_paths([corpus], rebuild=True)
    ondisk = RagEngine(
        make_config(tmp_path / "od", index_backend="ondisk", dense_weight=1.0), model=StubModel()
    )
    ondisk.index_paths([corpus], rebuild=True)

    for query in ("refund business days", "shipping international", "password security"):
        expected = [hit.chunk_id for hit in memory.lexical.search(query, k=10)]
        assert [hit.chunk_id for hit in ondisk.lexical.search(query, k=10)] == expected


# ── quantized ANN over the mmapped vectors ───────────────────────────────

def _clustered_index(tmp_path, count: int = 300, dim: int = 16) -> OnDiskVectorIndex:
    rng = np.random.default_rng(7)
    centres = rng.normal(size=(6, dim)).astype(np.float32)
    vectors = np.vstack([centres[i % 6] + rng.normal(scale=0.1, size=dim) for i in range(count)])
    chunks = [_chunk(i, f"chunk number {i} text", doc_id=f"d{i % 10}") for i in range(count)]
    index = OnDiskVectorIndex(tmp_path / "dense", dim=0)
    index.add(chunks, vectors.astype(np.float32))
    index.finalize()
    return OnDiskVectorIndex.open(tmp_path / "dense")


def test_ondisk_ann_is_optional_without_faiss(tmp_path, monkeypatch):
    index = _clustered_index(tmp_path)
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "faiss":
            raise ImportError("no faiss")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert index.build_ann() is False
    assert index.is_approximate is False


def test_ondisk_ivf_ann_returns_valid_hits_and_reloads(tmp_path):
    pytest.importorskip("faiss")
    index = _clustered_index(tmp_path)
    assert index.build_ann(pq=False, nlist=4) is True
    assert index.is_approximate is True

    hits = index.search(np.asarray(index.vector("d0:0000"), dtype=np.float32), k=5)
    assert hits and all(0.0 <= hit.score for hit in hits)
    assert all(hit.chunk.text for hit in hits)

    reopened = OnDiskVectorIndex.open(tmp_path / "dense")
    assert reopened.is_approximate is True
    # nprobe == nlist scans every list, so the flat IVF ranking is the exact one.
    reopened.nprobe = 4
    assert reopened.search(np.asarray(index.vector("d0:0000"), dtype=np.float32), k=1)[0].chunk.chunk_id == "d0:0000"


def test_ondisk_pq_ann_is_functional(tmp_path):
    pytest.importorskip("faiss")
    index = _clustered_index(tmp_path)
    assert index.build_ann(pq=True, nlist=4, m=4) is True
    hits = index.search(np.asarray(index.vector("d1:0001"), dtype=np.float32), k=3)
    assert len(hits) == 3
    assert len({hit.chunk.chunk_id for hit in hits}) == 3


def test_engine_builds_the_ann_when_configured(tmp_path):
    pytest.importorskip("faiss")
    config = make_config(
        tmp_path, index_backend="ondisk", dense_weight=1.0, ann=True, ann_nprobe=4
    )
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([write_corpus(tmp_path / "corpus")], rebuild=True)
    assert engine.index.is_approximate is True
    result = engine.query("How long does a refund take?")
    assert result.passages


# ── retrieval-time predicate pushdown ────────────────────────────────────
#
# `VectorIndex.search` takes an `allow` path predicate (trust tiers, tenant ACLs); the
# on-disk backend did not, so an engine on that backend could only filter after the scan.
# These pin the parity: the predicate excludes inside the scan and deepens rather than
# returning a short result.

def test_ondisk_search_allow_predicate_excludes_inside_the_scan(tmp_path):
    index = OnDiskVectorIndex(tmp_path / "dense", dim=2)
    index.add(
        [
            _chunk(0, "nearest", doc_id="a", path="allowed/a.md"),
            _chunk(1, "second", doc_id="b", path="blocked/b.md"),
            _chunk(2, "third", doc_id="c", path="allowed/c.md"),
        ],
        np.asarray([[1.0, 0.0], [0.99, 0.1], [0.9, 0.2]], dtype=np.float32),
    )
    index.finalize()
    opened = OnDiskVectorIndex.open(tmp_path / "dense")
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    assert [hit.chunk.path for hit in opened.search(query, k=3)][:2] == [
        "allowed/a.md",
        "blocked/b.md",
    ]
    kept = opened.search(query, k=2, allow=lambda path: path.startswith("allowed/"))
    assert [hit.chunk.path for hit in kept] == ["allowed/a.md", "allowed/c.md"]


def test_ondisk_search_allow_deepens_when_the_window_is_short(tmp_path):
    index = OnDiskVectorIndex(tmp_path / "dense", dim=2)
    chunks = [
        _chunk(i, f"blocked {i}", doc_id=f"b{i}", path="blocked/x.md") for i in range(6)
    ]
    vectors = [[1.0, 0.001 * i] for i in range(6)]
    chunks.append(_chunk(6, "allowed", doc_id="a", path="allowed/y.md"))
    vectors.append([0.0, 1.0])
    index.add(chunks, np.asarray(vectors, dtype=np.float32))
    index.finalize()
    opened = OnDiskVectorIndex.open(tmp_path / "dense")

    hits = opened.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        k=1,
        allow=lambda path: path.startswith("allowed/"),
    )
    assert [hit.chunk.path for hit in hits] == ["allowed/y.md"]


def test_ondisk_search_without_allow_is_unchanged(tmp_path):
    index = OnDiskVectorIndex(tmp_path / "dense", dim=2)
    index.add(
        [_chunk(0, "one", doc_id="a", path="a.md"), _chunk(1, "two", doc_id="b", path="b.md")],
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    index.finalize()
    opened = OnDiskVectorIndex.open(tmp_path / "dense")
    hits = opened.search(np.asarray([1.0, 0.0], dtype=np.float32), k=2)
    assert [hit.chunk.path for hit in hits] == ["a.md", "b.md"]


def test_ondisk_remove_paths_tombstones_and_persists(tmp_path):
    index = OnDiskVectorIndex(tmp_path / "dense", dim=2)
    index.add(
        [
            _chunk(0, "nearest", doc_id="a", path="keep/a.md"),
            _chunk(1, "second", doc_id="b", path="drop/b.md"),
            _chunk(2, "third", doc_id="c", path="keep/c.md"),
        ],
        np.asarray([[1.0, 0.0], [0.99, 0.1], [0.9, 0.2]], dtype=np.float32),
    )
    index.finalize()
    opened = OnDiskVectorIndex.open(tmp_path / "dense")
    assert len(opened) == 3
    assert opened.remove_paths(["drop/b.md"]) == 1
    assert len(opened) == 2
    assert opened.get("b:0001") is None
    hits = opened.search(np.asarray([1.0, 0.0], dtype=np.float32), k=3)
    assert [hit.chunk.path for hit in hits] == ["keep/a.md", "keep/c.md"]

    # The tombstone is on disk, so a fresh open agrees.
    reopened = OnDiskVectorIndex.open(tmp_path / "dense")
    assert len(reopened) == 2
    assert reopened.get("b:0001") is None
    assert "drop/b.md" not in reopened.paths()
