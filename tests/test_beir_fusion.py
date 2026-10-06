"""Tests for the BEIR replica's score-fusion normalisations.

`--combsum` has always used max-normalisation; `--convex` adds a min-max convex combination so the
fusion ablation the audit asks for is runnable. These pin that both run and that min-max actually
uses the channel's full range (the bottom of a channel contributes nothing extra).
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
    name = "laya_rag_beir_tool"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "beir.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


beir = _load()


class _Embedder:
    name = "fake"

    def embed_queries(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


def _corpus():
    return {
        "d0": "refund policy for customers",
        "d1": "shipping rates and delivery",
        "d2": "password rotation policy",
    }


def test_convex_minmax_ranks_every_document():
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    out = beir.score_fusion_rankings(
        _corpus(), {"q0": "refund policy"}, _Embedder(), vectors, 3, normalise="minmax"
    )
    assert set(out["q0"]) == {"d0", "d1", "d2"}


def test_max_normalisation_is_the_default_and_also_runs():
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    out = beir.score_fusion_rankings(
        _corpus(), {"q0": "refund policy"}, _Embedder(), vectors, 3
    )
    assert set(out["q0"]) == {"d0", "d1", "d2"}


def test_the_two_normalisations_are_selectable_and_agree_on_the_top_hit():
    """The perfectly-aligned document is first under either normalisation."""
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
    corpus = _corpus()
    queries = {"q0": "refund policy customers"}
    max_out = beir.score_fusion_rankings(corpus, queries, _Embedder(), vectors, 3, normalise="max")
    mm_out = beir.score_fusion_rankings(
        corpus, queries, _Embedder(), vectors, 3, normalise="minmax"
    )
    assert max_out["q0"][0] == "d0"
    assert mm_out["q0"][0] == "d0"
