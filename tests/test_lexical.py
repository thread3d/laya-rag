"""Tests for BM25 lexical recall and rank fusion."""
from __future__ import annotations

import json

import pytest

from laya_rag.chunking import Chunk
from laya_rag.lexical import (
    EAGER_LEXICAL_FILE,
    BM25Index,
    EagerBM25Index,
    reciprocal_rank_fusion,
    tokenize,
)


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="d",
        path=f"/{chunk_id}.md",
        text=text,
        index=0,
        start_char=0,
        end_char=len(text),
    )


def test_tokenize_lowercases_and_splits():
    assert tokenize("Hello, Refund-Policy!") == ["hello", "refund", "policy"]


def test_tokenize_keeps_cjk_characters():
    assert "退" in tokenize("退款政策")
    assert "policy" in tokenize("policy 退款")


def test_empty_index_returns_no_hits():
    assert BM25Index().search("anything", k=5) == []


def test_rare_terms_beat_common_ones():
    index = BM25Index()
    index.build([
        make_chunk("common", "policy policy policy policy"),
        make_chunk("rare", "policy langchain integration"),
    ])
    hits = index.search("langchain", k=2)
    assert hits
    assert hits[0].chunk_id == "rare"


def test_search_returns_at_most_k():
    index = BM25Index()
    index.build([make_chunk(f"c{i}", f"refund number {i}") for i in range(10)])
    assert len(index.search("refund", k=3)) <= 3


def test_common_query_words_score_nothing():
    """A query of only stopwords matches nothing rather than everything."""
    index = BM25Index()
    index.build([make_chunk("a", "the and of")])
    assert index.search("the and of", k=5) == []


def test_zero_score_chunks_are_not_returned():
    index = BM25Index()
    index.build([make_chunk("a", "refund policy"), make_chunk("b", "shipping rates")])
    hits = index.search("refund", k=10)
    assert [hit.chunk_id for hit in hits] == ["a"]


def test_search_rejects_non_positive_k():
    index = BM25Index()
    index.build([make_chunk("a", "text")])
    with pytest.raises(ValueError, match="k must be positive"):
        index.search("text", k=0)


def test_save_and_load_round_trip(tmp_path):
    index = BM25Index()
    index.build([make_chunk("a", "refund policy"), make_chunk("b", "shipping rates")])
    target = tmp_path / "lexical.json"
    index.save(target)

    restored = BM25Index.load(target)
    assert len(restored) == 2
    assert [hit.chunk_id for hit in restored.search("shipping", k=2)] == ["b"]


def test_load_rejects_unknown_version(tmp_path):
    target = tmp_path / "lexical.json"
    BM25Index().save(target)
    target.write_text(target.read_text().replace('"version": 1', '"version": 42'))
    with pytest.raises(ValueError, match="unsupported"):
        BM25Index.load(target)


def test_fusion_promotes_documents_found_by_both_channels():
    dense = ["x", "shared", "y"]
    lexical = ["shared", "z"]
    fused = dict(reciprocal_rank_fusion([dense, lexical]))
    assert fused["shared"] > fused["x"]
    assert fused["shared"] > fused["z"]


def test_fusion_ordering_is_deterministic_for_ties():
    first = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    second = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    assert first == second


def test_fusion_respects_limit():
    fused = reciprocal_rank_fusion([["a", "b", "c", "d"]], limit=2)
    assert [identifier for identifier, _ in fused] == ["a", "b"]


def test_fusion_of_empty_channels_is_empty():
    assert reciprocal_rank_fusion([[], []]) == []


def test_fusion_weights_let_one_channel_dominate():
    dense = ["dense_only", "shared"]
    lexical = ["shared", "lexical_only"]
    # A low dense weight must not let its exclusive document outrank the lexical
    # channel's top hit, even though the dense channel lists it first.
    fused = dict(reciprocal_rank_fusion([dense, lexical], weights=[0.1, 1.0]))
    assert fused["shared"] > fused["lexical_only"] > fused["dense_only"]


def test_fusion_weight_zero_drops_the_channel_entirely():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["c"]], weights=[0.0, 1.0]))
    assert "a" not in fused and "b" not in fused
    assert "c" in fused


def test_fusion_weight_must_match_the_channel_count():
    with pytest.raises(ValueError, match="2 channels but 1 weights"):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_fusion_default_weights_are_still_equal():
    weighted = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], weights=[1.0, 1.0])
    assert weighted == reciprocal_rank_fusion([["a", "b"], ["b", "a"]])


# ── stemming ─────────────────────────────────────────────────────────────

def test_tokenize_stems_when_asked():
    assert tokenize("Refunds are processed", stem=True) == ["refund", "are", "process"]


def test_tokenize_leaves_identifiers_and_digits_alone():
    assert tokenize("user_ids v2", stem=True) == ["user", "id", "v2"]


def test_stemming_matches_morphological_variants_that_exact_matching_misses():
    docs = [make_chunk("d1", "the refund policy")]
    plain = BM25Index()
    plain.build(docs)
    assert plain.search("refunds", k=5) == []

    stemmed = BM25Index(stem=True)
    stemmed.build(docs)
    assert [hit.chunk_id for hit in stemmed.search("refunds", k=5)] == ["d1"]


def test_stemming_survives_a_save_and_load(tmp_path):
    index = BM25Index(stem=True)
    index.build([make_chunk("d1", "refunds processed")])
    target = tmp_path / "lexical.json"
    index.save(target)

    restored = BM25Index.load(target)
    assert restored.stem is True
    assert [hit.chunk_id for hit in restored.search("refund", k=5)] == ["d1"]


def test_an_index_without_the_stem_key_loads_as_unstemmed(tmp_path):
    index = BM25Index()
    index.build([make_chunk("d1", "refunds")])
    target = tmp_path / "lexical.json"
    index.save(target)
    payload = json.loads(target.read_text())
    payload.pop("stem")
    target.write_text(json.dumps(payload))

    assert BM25Index.load(target).stem is False


def test_search_is_unchanged_by_a_save_and_load(tmp_path):
    """The inverted index is rebuilt on load, so ranking must survive the round trip."""
    docs = [make_chunk(f"d{i}", f"refund policy number {i}") for i in range(5)]
    index = BM25Index(stem=True)
    index.build(docs)
    before = [(hit.chunk_id, round(hit.score, 6)) for hit in index.search("refund policy", k=5)]

    target = tmp_path / "lexical.json"
    index.save(target)
    after = [
        (hit.chunk_id, round(hit.score, 6))
        for hit in BM25Index.load(target).search("refund policy", k=5)
    ]
    assert before == after


def test_a_term_absent_from_the_index_scores_nothing(tmp_path):
    index = BM25Index()
    index.build([make_chunk("d1", "refund policy")])
    assert index.search("zephyrous", k=5) == []


# ── eager BM25 ───────────────────────────────────────────────────────────

def _parity_corpus():
    return [
        make_chunk("d0", "refund policy refund window refund"),
        make_chunk("d1", "shipping rates and delivery windows for orders"),
        make_chunk("d2", "refund refund refund policy policy"),
        make_chunk("d3", "account security and password reset"),
        make_chunk("d4", "the quick brown fox jumps over the lazy dog"),
        make_chunk("d5", "refund policy shipping rates account security"),
    ]


def test_eager_bm25_ranks_identically_to_bm25():
    docs = _parity_corpus()
    plain = BM25Index(stem=True)
    plain.build(docs)
    eager = EagerBM25Index(stem=True)
    eager.build(docs)
    for query in ("refund policy", "shipping rates windows", "security password",
                  "quick fox", "refund", "nonexistentterm", "the and of"):
        expected = [(h.chunk_id, round(h.score, 9)) for h in plain.search(query, k=6)]
        got = [(h.chunk_id, round(h.score, 9)) for h in eager.search(query, k=6)]
        assert got == expected, f"mismatch for {query!r}: {got} != {expected}"


def test_eager_bm25_matches_at_several_depth_cuts():
    docs = _parity_corpus()
    plain = BM25Index(stem=False)
    plain.build(docs)
    eager = EagerBM25Index(stem=False)
    eager.build(docs)
    for k in (1, 3, 6):
        assert [h.chunk_id for h in eager.search("refund policy", k=k)] == \
               [h.chunk_id for h in plain.search("refund policy", k=k)]


def test_eager_bm25_survives_a_save_and_load(tmp_path):
    docs = _parity_corpus()
    eager = EagerBM25Index(stem=True)
    eager.build(docs)
    before = [(h.chunk_id, round(h.score, 9)) for h in eager.search("refund policy", k=6)]
    eager.save(tmp_path / EAGER_LEXICAL_FILE)
    loaded = EagerBM25Index.load(tmp_path / EAGER_LEXICAL_FILE)
    after = [(h.chunk_id, round(h.score, 9)) for h in loaded.search("refund policy", k=6)]
    assert after == before


def test_eager_bm25_validates_k_and_handles_an_empty_index(tmp_path):
    eager = EagerBM25Index()
    assert eager.search("anything", k=3) == []
    eager.build(_parity_corpus())
    with pytest.raises(ValueError):
        eager.search("refund", k=0)


def test_eager_load_rejects_an_unsupported_version(tmp_path):
    eager = EagerBM25Index()
    eager.build(_parity_corpus())
    eager.save(tmp_path / EAGER_LEXICAL_FILE)
    meta_path = (tmp_path / EAGER_LEXICAL_FILE).with_suffix(
        (tmp_path / EAGER_LEXICAL_FILE).suffix + ".meta.json"
    )
    payload = json.loads(meta_path.read_text())
    payload["version"] = 999
    meta_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        EagerBM25Index.load(tmp_path / EAGER_LEXICAL_FILE)


def test_resolved_lexical_backend_validates():
    from laya_rag.config import EngineConfig

    assert EngineConfig().resolved_lexical_backend == "bm25"
    assert EngineConfig(lexical_backend="eager").resolved_lexical_backend == "eager"
    with pytest.raises(ValueError):
        EngineConfig(lexical_backend="turbo").resolved_lexical_backend


def test_engine_builds_and_queries_with_the_eager_backend(tmp_path):
    from laya_rag.engine import RagEngine
    from tests.support import StubModel, make_config, write_corpus

    config = make_config(tmp_path, dense_weight=1.0, rerank_k=3, lexical_backend="eager")
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    assert (config.index_dir / EAGER_LEXICAL_FILE).is_file()
    assert engine.lexical is not None and len(engine.lexical) > 0

    reloaded = RagEngine(make_config(tmp_path, lexical_backend="eager"), model=StubModel())
    assert reloaded.lexical is not None
    result = reloaded.query("How long does a refund take?")
    assert result.passages
    assert result.considered > 0


# ── structural context prefix ────────────────────────────────────────────

def _contextual_chunk(context: str) -> Chunk:
    return Chunk(
        chunk_id="d1",
        doc_id="d",
        path="/corpus/refunds.md",
        text="refunds are processed within five business days",
        index=0,
        start_char=0,
        end_char=50,
        context=context,
    )


def test_bm25_indexes_the_context_but_not_as_evidence():
    index = BM25Index(stem=True)
    index.build([_contextual_chunk("billing/invoices.md | Invoice policy")])
    assert [hit.chunk_id for hit in index.search("invoices", k=3)] == ["d1"]
    # With no context the same term finds nothing.
    plain = BM25Index(stem=True)
    plain.build([_contextual_chunk("")])
    assert plain.search("invoices", k=3) == []


def test_eager_bm25_indexes_the_context_too():
    index = EagerBM25Index(stem=True)
    index.build([_contextual_chunk("billing/invoices.md | Invoice policy")])
    assert [hit.chunk_id for hit in index.search("invoices", k=3)] == ["d1"]


def test_engine_context_prefix_retrieves_on_the_path_but_returns_clean_text(tmp_path):
    from laya_rag.engine import RagEngine
    from tests.support import StubModel, make_config

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "zebra_handbook.md").write_text(
        "# Handbook\n\nThe procedure describes how to escalate an incident.\n",
        encoding="utf-8",
    )
    config = make_config(
        tmp_path, dense_weight=1.0, rerank_k=3, context_prefix=True,
        chunk_chars=400, chunk_overlap_chars=0, relevance_threshold=0.0,
    )
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([corpus])
    # "zebra" appears only in the file name, which is indexed as context. The relevance
    # gate reads the chunk text, so it is kept only because the threshold is 0.0 here —
    # the point is that the context made the chunk retrievable at all.
    result = engine.query("zebra")
    assert result.passages
    assert all("zebra" not in passage["text"].casefold() for passage in result.passages)
