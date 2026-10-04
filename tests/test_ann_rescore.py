"""Offline tests for the IVF-PQ rescore arm of the ANN benchmark.

The scale claim rests on this arm: a quantized index proposes candidates and the exact vectors
decide the order. If the rescore did not actually re-order by exact similarity, the arm would
measure the *codes'* recall while claiming the vectors' — so the tests pin the reordering and the
padding, not the faiss wiring.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    name = "laya_rag_ann_bench"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "ann_bench.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ann = _load()

VECTORS = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
QUERY = np.asarray([[1.0, 0.0]], dtype=np.float32)


class StubIndex:
    """A faiss-shaped index that returns a *deliberately wrong* candidate order."""

    def __init__(self, ids):
        self.ids = np.asarray(ids, dtype=np.int64)

    def search(self, queries, n):
        return np.zeros((len(queries), self.ids.shape[1]), dtype=np.float32), self.ids


def test_the_rescore_reorders_by_exact_similarity():
    stub = StubIndex([[3, 2, 1, 0]])          # codes rank them worst-last
    out = ann.rescored_search(stub, VECTORS, QUERY, 3, 4)
    assert out[0].tolist() == [0, 1, 3]        # exact inner product: 1.0, 0.8, 0.5


def test_the_rescore_pads_when_k_exceeds_the_candidate_list():
    stub = StubIndex([[3, 2, 1, 0]])
    out = ann.rescored_search(stub, VECTORS, QUERY, 6, 4)
    assert out[0][:4].tolist() == [0, 1, 3, 2]
    assert out[0][4:].tolist() == [-1, -1]


def test_a_row_with_no_candidates_stays_empty():
    stub = StubIndex([[-1, -1]])
    out = ann.rescored_search(stub, VECTORS, QUERY, 2, 2)
    assert out[0].tolist() == [-1, -1]


def test_the_rescore_uses_only_the_proposed_candidates():
    """An index that never proposes the true best cannot have it restored by the rescore."""
    stub = StubIndex([[3, 2, 1]])              # id 0 not proposed
    out = ann.rescored_search(stub, VECTORS, QUERY, 1, 3)
    assert out[0].tolist() == [1]              # best *among the candidates*, not best overall


def test_pq_index_size_is_codes_plus_centroids():
    # 1M x 48 B of codes + 1000 centroids x 384 dims x 4 B
    assert ann.pq_index_mb(1_000_000, 48, 1000, 384) == pytest.approx(49.5, abs=0.1)


def test_pq_is_far_smaller_than_the_flat_index():
    flat_mb = 1_000_000 * 384 * 4 / 1e6
    assert ann.pq_index_mb(1_000_000, 48, 1000, 384) < flat_mb / 20
