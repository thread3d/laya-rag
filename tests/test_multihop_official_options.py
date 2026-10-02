"""Tests for the MultiHop-RAG harness options added for the rerank comparison.

Running the cross-encoder over the full 2,255-query protocol is hours on this host, so the
harness gained a deterministic subset and a target selector. These pin the option parsing
and the word-window chunking the protocol depends on — no benchmark data or model needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import multihop_official as mh  # noqa: E402


def test_defaults_rerank_both_stages_with_no_subset():
    args = mh.parse_args([])
    assert args.limit_queries is None
    assert args.rerank_targets == "dense,hybrid"
    assert args.rerank is False
    assert args.rerank_depth == 100


def test_subset_and_target_options_parse():
    args = mh.parse_args(["--limit-queries", "400", "--rerank-targets", "hybrid", "--rerank"])
    assert args.limit_queries == 400
    assert args.rerank_targets == "hybrid"
    assert args.rerank is True


def test_resolve_models_maps_short_names():
    assert mh.resolve_models("bge-small") == [("bge-small", "BAAI/bge-small-en-v1.5")]
    resolved = mh.resolve_models("BAAI/custom-model")
    assert resolved == [("BAAI/custom-model", "BAAI/custom-model")]


def test_chunk_text_uses_overlapping_word_windows():
    assert mh.chunk_text("a b c d", 2, 1) == ["a b", "b c", "c d"]
    assert mh.chunk_text("a b c d", 2, 0) == ["a b", "c d"]
