"""Tests for the quantization probe.

The probe answers a question about the *real* encoder vectors, but its metric and codec plumbing are
pinned here on small synthetic data so the full run can be trusted.
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
sys.path.insert(0, str(ROOT / "src"))


def _load():
    name = "laya_rag_quantization_probe"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "quantization_probe.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


qp = _load()
faiss = pytest.importorskip("faiss")


def test_recall_at_k_is_the_mean_overlap():
    exact = np.asarray([[0, 1, 2]])
    assert qp.recall_at_k(np.asarray([[0, 1, 3]]), exact, 3) == pytest.approx(2 / 3)
    assert qp.recall_at_k(exact.copy(), exact, 3) == 1.0
    assert qp.recall_at_k(np.asarray([[9, 8, 7]]), exact, 3) == 0.0


def test_load_vectors_normalises_and_limits(tmp_path):
    path = tmp_path / "v.npy"
    np.save(path, np.asarray([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]], dtype=np.float32))
    vectors = qp.load_vectors(path, faiss=faiss)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
    assert qp.load_vectors(path, limit=2, faiss=faiss).shape == (2, 2)


def test_load_vectors_rejects_a_matrix_that_is_not_2d(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.asarray([1.0, 2.0], dtype=np.float32))
    with pytest.raises(ValueError):
        qp.load_vectors(path, faiss=faiss)


def _clustered(count: int = 2000, dim: int = 32, clusters: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(clusters, dim)).astype(np.float32)
    picks = rng.integers(0, clusters, size=count)
    vectors = centres[picks] + 0.1 * rng.normal(size=(count, dim)).astype(np.float32)
    faiss.normalize_L2(vectors)
    return vectors


def test_the_probe_runs_every_codec_and_flat_is_exact():
    report = qp.probe(_clustered(), queries=50, k=10, nlist=8, m=8, nbits=8, faiss=faiss)
    rows = {row["kind"]: row for row in report["rows"]}
    assert set(rows) == {"flat-exact", "ivf-flat", "ivf-pq", "ivf-sq8", "ivf-fp16"}
    assert rows["flat-exact"]["recall@10"] == 1.0
    for row in rows.values():
        assert 0.0 <= row["recall@10"] <= 1.0
        assert row["index_bytes"] > 0


def test_a_more_compressed_codec_is_smaller_than_flat():
    report = qp.probe(_clustered(), queries=50, k=10, nlist=8, m=8, nbits=8, faiss=faiss)
    rows = {row["kind"]: row for row in report["rows"]}
    assert rows["ivf-pq"]["index_bytes"] < rows["flat-exact"]["index_bytes"]
    assert rows["ivf-sq8"]["index_bytes"] < rows["flat-exact"]["index_bytes"]
