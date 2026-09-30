"""Tests for BM25 lexical recall and rank fusion."""
from __future__ import annotations

import json

import pytest

from laya_rag.chunking import Chunk
from laya_rag.lexical import BM25Index, reciprocal_rank_fusion, tokenize


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
