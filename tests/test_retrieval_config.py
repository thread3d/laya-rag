"""Tests for the retrieval configuration: fusion weights and the dense-channel embedder.

These pin the decision the BEIR ablations produced. The dense channel defaults to off
because every non-zero weight lowered the score on BEIR *and* on this project's own
planted corpus; it turns back on automatically when a retrieval-trained embedder is
configured, which is the only case where it earned a weight.
"""
from __future__ import annotations

import numpy as np
import pytest

from laya_rag.config import EngineConfig
from laya_rag.engine import RagEngine
from laya_rag.model import BiEncoderModel, HashingEmbedder

from .support import StubModel, make_config, write_corpus


# ── auto dense weight ────────────────────────────────────────────────────

def test_dense_weight_defaults_to_off_for_layas_own_encoder():
    assert EngineConfig().resolved_dense_weight == 0.0


def test_dense_weight_turns_on_for_a_retrieval_trained_embedder():
    config = EngineConfig(embed_model="BAAI/bge-small-en-v1.5")
    assert config.resolved_dense_weight == 1.0


def test_an_explicit_dense_weight_overrides_auto():
    config = EngineConfig(embed_model="BAAI/bge-small-en-v1.5", dense_weight=0.25)
    assert config.resolved_dense_weight == 0.25


def test_an_explicit_zero_beats_a_configured_embedder():
    config = EngineConfig(embed_model="BAAI/bge-small-en-v1.5", dense_weight=0.0)
    assert config.resolved_dense_weight == 0.0


def test_blank_embed_model_is_treated_as_unset():
    assert EngineConfig(embed_model="   ").resolved_dense_weight == 0.0


def test_stats_report_the_resolved_weight_not_the_auto_sentinel():
    payload = EngineConfig().as_dict()
    assert payload["dense_weight"] == -1.0
    assert payload["dense_weight_resolved"] == 0.0


# ── engine wiring ────────────────────────────────────────────────────────

def test_engine_uses_its_own_model_as_the_embedder_by_default(tmp_path):
    model = StubModel(dim=64)
    engine = RagEngine(make_config(tmp_path), model=model)
    assert engine.embedder is engine.model


def test_a_configured_embed_model_selects_the_bi_encoder_without_loading_it(tmp_path):
    config = make_config(tmp_path, embed_model="BAAI/bge-small-en-v1.5")
    engine = RagEngine(config, model=StubModel(dim=64))
    assert isinstance(engine.embedder, BiEncoderModel)
    # Constructing it must not reach the network; only ensure_loaded would.
    assert engine.embedder.info()["loaded"] is False


def test_index_dimension_follows_the_injected_embedder_not_the_decision_model(tmp_path):
    engine = RagEngine(
        make_config(tmp_path, dense_weight=1.0),
        model=StubModel(dim=64),
        embedder=HashingEmbedder(dim=32),
    )
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    assert engine.index.dim == 32
    assert engine.model.dim == 64
    assert engine.embedder.dim == 32


def test_zero_dense_weight_keeps_candidates_lexical_only(tmp_path):
    engine = RagEngine(make_config(tmp_path), model=StubModel(dim=64))
    engine.index_paths([write_corpus(tmp_path / "corpus")])

    result = engine.query("How long do refunds take?", top_k=5)

    assert result.passages
    assert all("dense" not in passage["channels"] for passage in result.passages)
    assert all("lexical" in passage["channels"] for passage in result.passages)


def test_a_positive_dense_weight_brings_the_dense_channel_back(tmp_path):
    engine = RagEngine(make_config(tmp_path, dense_weight=1.0), model=StubModel(dim=64))
    engine.index_paths([write_corpus(tmp_path / "corpus")])

    result = engine.query("How long do refunds take?", top_k=5)

    assert result.passages
    assert any("dense" in passage["channels"] for passage in result.passages)


def test_zero_dense_weight_is_reported_in_stats(tmp_path):
    engine = RagEngine(make_config(tmp_path), model=StubModel(dim=64))
    assert engine.stats()["config"]["dense_weight_resolved"] == 0.0


def test_an_injected_embedder_that_changes_width_is_still_indexed(tmp_path):
    """The embedder's output width, not the model's, decides the index dimension."""
    embedder = HashingEmbedder(dim=16)
    engine = RagEngine(
        make_config(tmp_path, dense_weight=1.0), model=StubModel(dim=8), embedder=embedder
    )
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    vector = np.asarray(embedder.embed_one("refund"), dtype=np.float32)
    assert vector.shape[0] == 16


# ── rerank order ─────────────────────────────────────────────────────────

def test_rerank_order_defaults_to_recall_without_a_cross_encoder():
    assert EngineConfig().resolved_rerank_order == "recall"


def test_rerank_order_resolves_to_relevance_when_a_cross_encoder_is_configured():
    config = EngineConfig(rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert config.resolved_rerank_order == "relevance"


def test_an_explicit_recall_order_overrides_the_cross_encoder_default():
    config = EngineConfig(
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2", rerank_order="recall"
    )
    assert config.resolved_rerank_order == "recall"


def test_rerank_order_accepts_relevance_case_insensitively():
    assert EngineConfig(rerank_order="RELEVANCE").resolved_rerank_order == "relevance"


def test_an_unknown_rerank_order_falls_back_to_auto():
    assert EngineConfig(rerank_order="sideways").resolved_rerank_order == "recall"
    configured = EngineConfig(rerank_order="sideways", rerank_model="some/cross-encoder")
    assert configured.resolved_rerank_order == "relevance"


def _two_hits():
    from laya_rag.chunking import Chunk
    from laya_rag.index_store import Hit

    def hit(chunk_id: str, text: str, score: float) -> Hit:
        return Hit(
            chunk=Chunk(
                chunk_id=chunk_id, doc_id="d", path=f"/{chunk_id}", text=text,
                index=0, start_char=0, end_char=len(text),
            ),
            score=score,
            channels=("lexical",),
        )

    # "a" is retrieved first but shares only one of the query's two terms, so Laya
    # scores it lower than "b", which the retrieval ranked second.
    return [hit("a", "refund", 0.9), hit("b", "refund policy", 0.5)]


def test_relevance_order_ranks_by_layas_probability():
    from laya_rag.rerank import LayaReranker

    result = LayaReranker(StubModel()).rerank(
        "refund policy annual plan", _two_hits(), top_k=2, relevance_threshold=0.3, order="relevance"
    )
    assert [passage.chunk_id for passage in result.passages] == ["b", "a"]


def test_recall_order_keeps_the_retrieval_ranking():
    from laya_rag.rerank import LayaReranker

    result = LayaReranker(StubModel()).rerank(
        "refund policy annual plan", _two_hits(), top_k=2, relevance_threshold=0.3, order="recall"
    )
    assert [passage.chunk_id for passage in result.passages] == ["a", "b"]


def test_recall_order_still_filters_below_the_threshold():
    from laya_rag.rerank import LayaReranker

    result = LayaReranker(StubModel()).rerank(
        "refund policy annual plan", _two_hits(), top_k=2, relevance_threshold=1.01, order="recall"
    )
    assert result.kept == 0
    assert result.dropped_below_threshold == 2


# ── bi-encoder pooling ───────────────────────────────────────────────────

def test_bge_models_default_to_cls_pooling(tmp_path):
    from laya_rag.model import BiEncoderModel

    engine = BiEncoderModel(make_config(tmp_path, embed_model="BAAI/bge-small-en-v1.5"))
    assert engine.pooling == "cls"


def test_other_models_default_to_mean_pooling(tmp_path):
    from laya_rag.model import BiEncoderModel

    engine = BiEncoderModel(
        make_config(tmp_path, embed_model="sentence-transformers/all-MiniLM-L6-v2")
    )
    assert engine.pooling == "mean"


def test_pooling_can_be_forced(tmp_path):
    from laya_rag.model import BiEncoderModel

    engine = BiEncoderModel(
        make_config(tmp_path, embed_model="BAAI/bge-small-en-v1.5", embed_pooling="mean")
    )
    assert engine.pooling == "mean"


def test_an_unknown_pooling_is_rejected(tmp_path):
    from laya_rag.model import BiEncoderModel, ModelUnavailableError

    with pytest.raises(ModelUnavailableError, match="EMBED_POOLING"):
        BiEncoderModel(
            make_config(tmp_path, embed_model="BAAI/bge-small-en-v1.5", embed_pooling="max")
        )


def test_the_query_instruction_is_off_by_default_and_can_be_turned_on(tmp_path):
    from laya_rag.model import BiEncoderModel

    config = make_config(tmp_path, embed_model="BAAI/bge-small-en-v1.5")
    assert BiEncoderModel(config).query_instruction is False
    config.embed_query_instruction = True
    assert BiEncoderModel(config).query_instruction is True


def test_the_embedder_reports_its_pooling(tmp_path):
    from laya_rag.model import BiEncoderModel

    info = BiEncoderModel(make_config(tmp_path, embed_model="BAAI/bge-small-en-v1.5")).info()
    assert info["pooling"] == "cls"
    assert info["query_instruction"] is False


# ── cross-encoder reranking ──────────────────────────────────────────────

def test_no_rerank_model_means_no_cross_encoder(tmp_path):
    engine = RagEngine(make_config(tmp_path), model=StubModel(dim=64))
    assert engine.cross_encoder is None
    assert engine.stats()["reranker"]["kind"] == "laya decisions"


def test_a_configured_rerank_model_selects_the_cross_encoder_without_loading_it(tmp_path):
    from laya_rag.model import CrossEncoderModel

    engine = RagEngine(
        make_config(tmp_path, rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2"),
        model=StubModel(dim=64),
    )
    assert isinstance(engine.cross_encoder, CrossEncoderModel)
    assert engine.cross_encoder.info()["loaded"] is False
    assert engine.stats()["reranker"]["kind"] == "cross-encoder"


def test_rerank_can_take_scores_from_another_reranker():
    from laya_rag.rerank import LayaReranker

    reranker = LayaReranker(StubModel())
    # The query shares nothing with the stub's scoring, so only `scores` can decide.
    forward = reranker.rerank("zzz", _two_hits(), top_k=2, relevance_threshold=0.3, scores=[0.9, 0.4])
    reverse = reranker.rerank("zzz", _two_hits(), top_k=2, relevance_threshold=0.3, scores=[0.4, 0.9])
    assert [p.chunk_id for p in forward.passages] == ["a", "b"]
    assert [p.chunk_id for p in reverse.passages] == ["b", "a"]


def test_rerank_scores_are_still_filtered_by_the_threshold():
    from laya_rag.rerank import LayaReranker

    result = LayaReranker(StubModel()).rerank(
        "zzz", _two_hits(), top_k=2, relevance_threshold=0.5, scores=[0.9, 0.1]
    )
    assert [p.chunk_id for p in result.passages] == ["a"]
    assert result.dropped_below_threshold == 1


def test_rerank_rejects_a_score_count_mismatch():
    from laya_rag.rerank import DecisionError, LayaReranker

    with pytest.raises(DecisionError, match="scores for"):
        LayaReranker(StubModel()).rerank("q", _two_hits(), scores=[0.5])


def test_lexical_stemming_is_on_by_default():
    assert EngineConfig().lexical_stem is True


def test_the_engine_orders_evidence_by_an_injected_cross_encoder(tmp_path):
    engine = RagEngine(make_config(tmp_path), model=StubModel(dim=64))
    engine.index_paths([write_corpus(tmp_path / "corpus")])

    class FakeCrossEncoder:
        def score_pairs(self, pairs):
            return [0.9 if "refund" in passage.lower() else 0.1 for _, passage in pairs]

        def info(self):
            return {"kind": "cross-encoder"}

    engine.cross_encoder = FakeCrossEncoder()
    result = engine.query("How long do refunds take?", top_k=3)

    assert result.passages
    assert "refund" in result.passages[0]["path"]
    # Only the cross-encoder's above-threshold passages survive.
    assert all("refund" in passage["path"] for passage in result.passages)
