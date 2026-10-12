"""The engine picks the reranker class the configuration names, and the generative one is scoreable.

§P4's open question is whether a *listwise* ranker beats a pointwise cross-encoder, and the recorded
rejection was specifically "a pointwise MS MARCO cross-encoder over the fused top-100".
Qwen3-Reranker-0.6B is Apache-2.0 and does not load through `AutoModelForSequenceClassification`, so
it needs this path. These pin the parts that do not need the weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from laya_rag.config import EngineConfig  # noqa: E402
from laya_rag.model import (  # noqa: E402
    CrossEncoderModel,
    GenerativeRerankerModel,
    build_reranker,
    is_generative_reranker,
    qwen3_rerank_prompt,
    yes_no_probability,
)


# ── the prompt is the interface ───────────────────────────────────────────

def test_the_prompt_is_the_documented_chat_template():
    text = qwen3_rerank_prompt("how long do refunds take", "five business days")
    assert text.startswith("<|im_start|>system\n")
    assert '<|im_end|>\n<|im_start|>user\n' in text
    assert "<Query>: how long do refunds take" in text
    assert "<Document>: five business days" in text
    assert text.endswith("<|im_start|>assistant\n")
    assert "yes" in text and "no" in text            # the answer is constrained, as the card says


def test_a_custom_instruction_replaces_the_default():
    text = qwen3_rerank_prompt("q", "d", instruction="Find the abstract")
    assert "<Instruct>: Find the abstract" in text


# ── the score is P("yes") ──────────────────────────────────────────────────

def test_equal_logits_are_a_coin_flip_and_a_gap_is_not():
    assert yes_no_probability(0.0, 0.0) == pytest.approx(0.5)
    assert yes_no_probability(5.0, 0.0) > 0.99
    assert yes_no_probability(0.0, 5.0) < 0.01
    # antisymmetric, which is what makes the number a probability rather than an arbitrary score
    assert yes_no_probability(1.5, 0.5) == pytest.approx(1.0 - yes_no_probability(0.5, 1.5))


def test_a_huge_gap_does_not_overflow():
    """Softmax over two logits has to survive the extremes a 0.6B model can emit."""
    assert yes_no_probability(1000.0, -1000.0) == 1.0
    assert yes_no_probability(-1000.0, 1000.0) == 0.0


# ── the engine picks the class the configuration names ────────────────────

def test_the_generative_predicate_is_narrow():
    assert is_generative_reranker("Qwen/Qwen3-Reranker-0.6B")
    assert not is_generative_reranker("BAAI/bge-reranker-base")
    assert not is_generative_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert not is_generative_reranker("")


def test_build_reranker_returns_the_matching_class_without_loading_weights():
    assert build_reranker(EngineConfig()) is None
    generative = build_reranker(EngineConfig(rerank_model="Qwen/Qwen3-Reranker-0.6B"))
    cross = build_reranker(EngineConfig(rerank_model="BAAI/bge-reranker-base"))
    assert isinstance(generative, GenerativeRerankerModel)
    assert isinstance(cross, CrossEncoderModel)
    # lazy: nothing is loaded until `score_pairs`, so choosing is free
    assert generative.info()["loaded"] is False
    assert generative.info()["kind"] == "generative-reranker"
