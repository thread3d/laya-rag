"""Offline tests for the BRIGHT retrieval adapter's helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    spec = importlib.util.spec_from_file_location("laya_rag_bright", ROOT / "tools" / "bright.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bright = _load()


def test_parse_gold_accepts_lists_and_python_reprs():
    assert bright.parse_gold(["a", "b"]) == ["a", "b"]
    assert bright.parse_gold("['a', 'b']") == ["a", "b"]
    assert bright.parse_gold("N/A") == []
    assert bright.parse_gold(None) == []
    assert bright.parse_gold("not a list") == []


def test_build_dataset_maps_rows_and_drops_unresolvable_gold():
    examples = [
        {"id": "0", "query": "q0", "gold_ids": ["d1", "missing"]},
        {"id": "1", "query": "q1", "gold_ids": []},
    ]
    documents = [{"id": "d1", "content": "content one"}, {"id": "d2", "content": "content two"}]
    corpus, queries, qrels = bright.build_dataset(examples, documents)
    assert corpus == {"d1": "content one", "d2": "content two"}
    assert queries == {"0": "q0", "1": "q1"}
    # Only resolvable gold ids survive, and a query with no gold is not scored.
    assert qrels == {"0": {"d1": 1}}


def test_cache_round_trip(tmp_path):
    payload = {
        "corpus": {"d1": "alpha", "d2": "beta"},
        "queries": {"0": "query"},
        "qrels": {"0": {"d1": 1}},
    }
    bright.write_cache("tiny", tmp_path, payload)
    restored = bright.read_cache("tiny", tmp_path)
    assert restored == payload
    assert bright.read_cache("absent", tmp_path) is None


def test_render_markdown_lists_subsets_and_comparisons():
    payload = {
        "_settings": {},
        "subsets": {
            "pony": {
                "documents": 10,
                "queries": 2,
                "summary": {"bm25": {"nDCG@10": 0.20, "nDCG@100": 0.30},
                            "dense": {"nDCG@10": 0.15, "nDCG@100": 0.25}},
                "comparisons": {
                    "dense vs bm25": {"delta": -0.05, "ci_low": -0.09, "ci_high": -0.01,
                                      "significant": True},
                },
            }
        },
    }
    text = bright.render_markdown(payload)
    assert "| pony |" in text and "| bm25 |" in text
    assert "dense vs bm25" in text and "significant" in text
