"""Offline tests for the lexical backend benchmark's helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "laya_rag_bench_lexical", ROOT / "tools" / "bench_lexical.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load()


def test_estimate_size_is_positive_and_grows_with_content():
    small = bench.estimate_size({"a": [1, 2, 3]})
    large = bench.estimate_size({"a": list(range(1000))})
    assert small > 0
    assert large > small


def test_estimate_size_counts_shared_objects_once():
    import numpy as np

    shared = np.zeros(1000, dtype=np.float64)
    once = bench.estimate_size({"a": shared})
    twice = bench.estimate_size({"a": shared, "b": shared})
    # The second reference must not double-count the array.
    assert twice - once < shared.nbytes


def test_make_corpus_is_deterministic_and_sized():
    first = bench.make_corpus(10, 50, 20, seed=1)
    second = bench.make_corpus(10, 50, 20, seed=1)
    assert [d.text for d in first] == [d.text for d in second]
    assert len(first) == 10
    assert first[0].chunk_id == "d0"
    assert len(first[0].text.split()) == 20


def test_make_queries_is_deterministic():
    assert bench.make_queries(50, 5, seed=2) == bench.make_queries(50, 5, seed=2)
    assert len(bench.make_queries(50, 5, seed=2)) == 5


def test_render_markdown_reports_parity_and_both_backends():
    payload = {
        "_settings": {"docs": 10, "length": 20, "vocab": 50, "queries": 2, "k": 10},
        "parity": {"identical_queries": 2, "queries": 2, "top_k_identical": True},
        "results": {
            "bm25": {"build_seconds": 0.1, "index_bytes": 2_000_000, "ms_per_query": 2.0},
            "eager": {"build_seconds": 0.1, "index_bytes": 1_000_000, "ms_per_query": 1.0},
        },
        "speedup": 2.0,
        "size_ratio": 2.0,
    }
    text = bench.render_markdown(payload)
    assert "| bm25 |" in text and "| eager |" in text
    assert "2/2 queries identical" in text
