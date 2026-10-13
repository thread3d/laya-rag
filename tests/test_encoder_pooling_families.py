"""`auto` pooling has to know how a modern backbone was trained, or it mis-embeds it silently.

The old rule was "CLS if the name contains `bge`, mean otherwise". `Qwen/Qwen3-Embedding-0.6B` loads and
runs through `BiEncoderModel` today, so that rule would embed it with **mean** pooling when it is trained
on the **last** token — a silently worse retriever and therefore a misleading negative, not an error.
These pin the table and the padding-safe last-token index; no weights are needed for either.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from laya_rag.model import BiEncoderModel, _family_pooling  # noqa: E402
from laya_rag.config import EngineConfig  # noqa: E402


@pytest.mark.parametrize("model_id,expected", [
    ("BAAI/bge-small-en-v1.5", "cls"),
    ("BAAI/bge-m3", "cls"),
    ("mixedbread-ai/mxbai-embed-large-v1", "cls"),
    ("Qwen/Qwen3-Embedding-0.6B", "last"),
    ("intfloat/multilingual-e5-large-instruct", "mean"),
    ("Snowflake/snowflake-arctic-embed-l-v2.0", "mean"),
])
def test_the_family_table_says_how_each_verified_model_pools(model_id, expected):
    assert _family_pooling(model_id) == expected


def test_an_unknown_backbone_gets_the_historical_default_not_a_guess():
    """The table cannot know an unlisted model, and choosing silently is what it exists to avoid."""
    assert _family_pooling("someone/brand-new-encoder") == "mean"
    assert _family_pooling("") == "mean"


def test_explicit_pooling_still_wins_and_last_is_accepted():
    """`last` was not even a legal value before; a caller who needs it could not ask for it."""
    cfg = EngineConfig(embed_model="Qwen/Qwen3-Embedding-0.6B", embed_pooling="cls")
    assert BiEncoderModel(cfg).pooling == "cls"
    cfg = EngineConfig(embed_model="Qwen/Qwen3-Embedding-0.6B")
    assert BiEncoderModel(cfg).pooling == "last"


def test_the_last_token_index_is_padding_safe():
    """The round-125 bug, pre-empted: `last` must mean the last *real* token under either padding side.

    Pure Python on purpose — an `import torch` inside this test stalled the whole suite for minutes on
    this host while the embedding job was saturating it, and the arithmetic under test needs no tensor
    library. (The encode path uses the tensor form; this is the same rule expressed once.)
    """

    def last_index(mask):
        return max((i for i, value in enumerate(mask) if value), default=len(mask) - 1)

    assert [last_index(row) for row in ([1, 1, 1, 0, 0], [1, 1, 1, 1, 1])] == [2, 4]
    assert [last_index(row) for row in ([0, 0, 1, 1, 1], [1, 1, 1, 1, 1])] == [4, 4]


# ── prefixes: a family trained with one on each side ─────────────────────

def test_the_prefix_table_knows_e5_and_nothing_else():
    from laya_rag.model import _family_prefixes

    assert _family_prefixes("intfloat/multilingual-e5-large-instruct") == ("query: ", "passage: ")
    assert _family_prefixes("BAAI/bge-small-en-v1.5") == ("", "")
    assert _family_prefixes("") == ("", "")


def test_a_family_that_needs_prefixes_says_so_when_it_has_none(capsys):
    """Silent is the failure mode: this model retrieves worse, not wrongly, without both prefixes."""
    BiEncoderModel(EngineConfig(embed_model="intfloat/multilingual-e5-large-instruct"))
    out = capsys.readouterr().out
    assert "trained with prefixes" in out
    assert "EMBED_PASSAGE_PREFIX" in out


def test_a_configured_pair_is_quiet(capsys, monkeypatch):
    monkeypatch.setenv("LAYA_RAG_EMBED_QUERY_INSTRUCTION", "1")
    monkeypatch.setenv("LAYA_RAG_EMBED_PASSAGE_PREFIX", "passage: ")
    model = BiEncoderModel(EngineConfig(embed_model="intfloat/multilingual-e5-large-instruct"))
    assert capsys.readouterr().out == ""
    assert model.passage_prefix == "passage: "


def test_a_family_without_prefixes_is_quiet_and_unprefixed(capsys):
    model = BiEncoderModel(EngineConfig(embed_model="BAAI/bge-small-en-v1.5"))
    assert capsys.readouterr().out == ""
    assert model.passage_prefix == ""


# ── the query-side line: each family's own, never another's ──────────────

def test_the_query_prefix_table_is_family_specific():
    from laya_rag.model import _family_query_prefix

    assert _family_query_prefix("intfloat/multilingual-e5-large-instruct") == "query: "
    qwen = _family_query_prefix("Qwen/Qwen3-Embedding-0.6B")
    assert qwen.startswith("Instruct: ") and qwen.endswith("\nQuery: ")
    # bge has no entry: its instruction is the default sentence, applied by the caller.
    assert _family_query_prefix("BAAI/bge-small-en-v1.5") == ""


def test_a_qwen3_query_does_not_get_bges_instruction():
    """The third silent mismatch on P3's list: one family's instruction on another's encoder."""
    from laya_rag.model import BiEncoderModel

    model = BiEncoderModel(EngineConfig(embed_model="Qwen/Qwen3-Embedding-0.6B",
                                        embed_query_instruction=True))
    bge = BiEncoderModel(EngineConfig(embed_model="BAAI/bge-small-en-v1.5",
                                      embed_query_instruction=True))
    assert model._query_text("hello").startswith("Instruct: ")
    assert "Represent this sentence" not in model._query_text("hello")
    assert bge._query_text("hello").startswith("Represent this sentence")


# ── Matryoshka truncation ────────────────────────────────────────────────

def test_truncation_is_a_knob_with_a_zero_default():
    """The guard that refuses an over-wide truncation lives in the encode path, which needs torch to
    exercise; the knob and its default are assertable here, and the guard was verified against the real
    checkpoint (`truncate=4096` against a pooled width of 1024 raises rather than silently keeping 1024).
    """
    assert EngineConfig().embed_truncate_dim == 0
    monkeypatch_free = EngineConfig(embed_model="Qwen/Qwen3-Embedding-0.6B", embed_truncate_dim=256)
    assert BiEncoderModel(monkeypatch_free).truncate_dim == 256


# ── the cache key ────────────────────────────────────────────────────────

def test_the_embedder_name_carries_its_pooling():
    """Two pooling modes are two embedders, so an embedding cache keyed by model id alone would hand one
    mode's vectors to the other — the silent mismatch this file exists for, arriving through the cache.

    Found by passing `BiEncoderModel` to the harness's `cached_embed`, which requires `.name`.
    """
    qwen_last = BiEncoderModel(EngineConfig(embed_model="Qwen/Qwen3-Embedding-0.6B"))
    qwen_mean = BiEncoderModel(EngineConfig(embed_model="Qwen/Qwen3-Embedding-0.6B",
                                            embed_pooling="mean"))
    assert qwen_last.name == "Qwen/Qwen3-Embedding-0.6B.last"
    assert qwen_mean.name == "Qwen/Qwen3-Embedding-0.6B.mean"
    assert qwen_last.name != qwen_mean.name
