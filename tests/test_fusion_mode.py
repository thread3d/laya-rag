"""Tests for the convex-combination fusion mode.

RRF keeps ranks and throws scores away; the published evidence says a tuned convex combination
can beat it, so the alternative ships behind `FUSION_MODE` for the ablation. These pin the
normalisation contract (each channel independently to [0, 1], presence counts, weight 0 drops a
channel) and that the engine actually selects the rule.
"""
from __future__ import annotations

import pytest

from laya_rag.engine import RagEngine
from laya_rag.lexical import convex_combination

from .support import StubModel, make_config


def test_each_channel_is_normalised_independently():
    dense = [("x", 10.0), ("y", 9.0)]
    lexical = [("y", 0.5), ("z", 0.4)]
    fused = dict(convex_combination([dense, lexical]))
    assert fused == {"x": 1.0, "y": 1.0, "z": 0.0}


def test_a_flat_channel_contributes_presence_not_zero():
    fused = dict(convex_combination([[("a", 3.0), ("b", 3.0)]]))
    assert fused == {"a": 1.0, "b": 1.0}


def test_a_zero_weight_drops_a_channel():
    fused = dict(convex_combination([[("a", 1.0)], [("b", 1.0)]], weights=[0.0, 1.0]))
    assert "a" not in fused and fused["b"] == 1.0


def test_weights_must_match_channels():
    with pytest.raises(ValueError):
        convex_combination([[("a", 1.0)]], weights=[1.0, 2.0])


def test_limit_truncates_and_order_is_best_first():
    fused = convex_combination([[("a", 2.0), ("b", 1.0)]], limit=1)
    assert fused == [("a", 1.0)]


def _corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir(parents=True, exist_ok=True)
    (root / "refunds.md").write_text(
        "# Refunds\n\nRefunds are processed within five business days.\n", encoding="utf-8"
    )
    (root / "shipping.md").write_text(
        "# Shipping\n\nShipping is free on orders over fifty dollars.\n", encoding="utf-8"
    )
    return root


@pytest.mark.parametrize("mode", ["rrf", "convex"])
def test_engine_queries_under_either_fusion_mode(tmp_path, mode):
    config = make_config(
        tmp_path,
        fusion_mode=mode,
        dense_weight=1.0,
        relevance_threshold=0.0,
        answerability_threshold=0.0,
    )
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([_corpus(tmp_path)], rebuild=True)
    assert engine.query("refund processing time").passages


def test_an_unknown_fusion_mode_fails_loudly(tmp_path):
    config = make_config(
        tmp_path,
        fusion_mode="magic",
        dense_weight=1.0,
        relevance_threshold=0.0,
        answerability_threshold=0.0,
    )
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([_corpus(tmp_path)], rebuild=True)
    with pytest.raises(ValueError, match="unknown fusion_mode"):
        engine.query("refund processing time")
