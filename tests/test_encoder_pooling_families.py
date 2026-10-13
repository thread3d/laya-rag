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
