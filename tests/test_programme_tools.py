"""Offline tests for the "what would raise the score" tools.

These pin the properties that make the new measurements trustworthy: the on-disk BM25
must return the *identical* ranking to the in-memory one (otherwise the disk path is a
different retriever), the streaming dense index must round-trip through its memmap, and
the recalibration statistics must agree with their definitions.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import beir  # noqa: E402
from laya_rag.chunking import Chunk  # noqa: E402
from laya_rag.engine import RagEngine  # noqa: E402
from laya_rag.index_store import Hit, VectorIndex  # noqa: E402
from laya_rag.lexical import BM25Index  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ondisk = _load("ondisk")
recalibrate = _load("multihop_recalibrate")

CORPUS = {
    "d1": "alpha beta beta gamma",
    "d2": "beta gamma delta",
    "d3": "alpha alpha delta epsilon",
    "d4": "gamma epsilon zeta",
    "d5": "alpha beta delta epsilon zeta",
    "d6": "unrelated filler words only",
}


def test_ondisk_bm25_matches_the_in_memory_ranking(tmp_path):
    on_disk = ondisk.OnDiskBM25(tmp_path / "bm25", k1=1.2, b=0.75, drop_stopwords=True, stem=False)
    on_disk.build(CORPUS.items(), block_docs=2)

    memory = BM25Index(k1=1.2, b=0.75, drop_stopwords=True, stem=False)
    memory.build(beir.make_chunks(CORPUS))

    for query in ("alpha", "beta gamma", "delta epsilon", "filler", "zeta zeta"):
        disk_ids = [doc_id for doc_id, _ in on_disk.search(query, k=4)]
        memory_ids = [hit.chunk_id for hit in memory.search(query, k=4)]
        assert disk_ids == memory_ids


def test_ondisk_bm25_reopens_without_the_build_state(tmp_path):
    on_disk = ondisk.OnDiskBM25(tmp_path / "bm25", k1=1.2, b=0.75, drop_stopwords=True, stem=False)
    on_disk.build(CORPUS.items(), block_docs=2)
    reopened = ondisk.OnDiskBM25.open(tmp_path / "bm25")
    assert [doc_id for doc_id, _ in reopened.search("alpha beta", k=3)] == [
        doc_id for doc_id, _ in on_disk.search("alpha beta", k=3)
    ]


def test_streaming_vector_index_round_trips_and_ranks_by_cosine(tmp_path):
    index = ondisk.StreamingVectorIndex(tmp_path / "dense", dim=4)
    index.add(["a", "b"], np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32))
    index.add(["c"], np.array([[0.9, 0.1, 0.0, 0.0]], dtype=np.float32))
    index.finalize()

    reopened = ondisk.StreamingVectorIndex.open(tmp_path / "dense")
    hits = reopened.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=3)
    assert [doc_id for doc_id, _ in hits] == ["a", "c", "b"]
    assert hits[0][1] > hits[1][1] > hits[2][1]


def test_recalibration_statistics_match_their_definitions():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])

    assert recalibrate.auc(scores, labels) == 1.0
    assert recalibrate.auc(np.array([0.5, 0.5, 0.5, 0.5]), labels) == 0.5

    threshold = recalibrate.best_threshold(scores, labels)
    rates = recalibrate.abstention_rates(scores, labels, threshold)
    assert rates["balanced_accuracy"] == 1.0
    assert rates["discrimination"] == 1.0

    # Abstention is the complement of `score >= threshold`, so the shipped 0.5 threshold
    # on this toy separates perfectly too.
    shipped = recalibrate.abstention_rates(scores, labels, 0.5)
    assert shipped["null_abstention"] == 1.0
    assert shipped["positive_abstention"] == 0.0


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id=chunk_id, path=chunk_id, text=chunk_id,
        index=0, start_char=0, end_char=len(chunk_id),
    )


def test_fuse_dedup_drops_a_near_duplicate_and_spends_the_slot_on_a_new_chunk():
    index = VectorIndex()
    ids = ["a", "a-copy", "b", "c"]
    vectors = np.array(
        [[1.0, 0.0], [0.999, 0.01], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32
    )
    index.add([_chunk(cid) for cid in ids], vectors)
    hits = [Hit(chunk=index.get(cid), score=1.0, channels=("dense",)) for cid in ids]

    plain = RagEngine._fuse(index, hits, [], limit=3, dense_weight=1.0, lexical_weight=0.0)
    assert [hit.chunk.chunk_id for hit in plain] == ["a", "a-copy", "b"]

    deduped = RagEngine._fuse(
        index, hits, [], limit=3, dense_weight=1.0, lexical_weight=0.0, dedup_threshold=0.9
    )
    assert [hit.chunk.chunk_id for hit in deduped] == ["a", "b", "c"]


def test_dedup_is_off_unless_a_threshold_is_set(tmp_path):
    # The engine default must not change recall for existing deployments.
    from laya_rag.config import EngineConfig

    config = EngineConfig()
    assert config.recall_dedup_threshold == 0.0

