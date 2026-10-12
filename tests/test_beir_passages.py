"""Adaptive passage budgets and aggregation beyond max-pool.

§P7's falsifier was met on a *fixed* 600-character cut (`docs/findings-passages-long.md`), and its
second clause named the two ways that cut was wrong: the budget ignored the document, and max-pool
threw away every other piece. These tests pin the knobs that make those measurable, and pin that the
historical `max` behaviour is unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def _load_beir():
    import importlib.util

    name = "laya_rag_beir_passages"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "beir.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


beir = _load_beir()


# ── the adaptive budget ──────────────────────────────────────────────────

def test_a_document_of_short_sentences_gets_the_floor_not_a_fragment():
    """Without a floor a one-word-sentence document would be indexed as fragments."""
    text = "Yes. No. Maybe. Fine. Ok."
    assert beir._adaptive_passage_chars(text, ceiling=1200) == 300


def test_a_document_of_long_sentences_gets_the_ceiling():
    """A piece longer than the ceiling would defeat the point of passage retrieval."""
    sentence = "word " * 300   # ~1500 characters, one sentence
    assert beir._adaptive_passage_chars(sentence.strip() + ".", ceiling=1200) == 1200


def test_the_budget_is_the_documents_own_p90():
    """Nine long sentences and one short one: the p90 is the long one, and it is not clamped away.

    The mirror case matters too and is *not* a bug: nine short sentences and one long one gives a
    short p90, so that document gets the floor — a document whose sentences are 90% short should
    have short passages, which is the whole point of sizing from the document.
    """
    long_sentence = ("a much longer sentence that is clearly the ninetieth percentile " * 7).strip() + "."
    text = " ".join(["short one."] + [long_sentence] * 9)
    size = beir._adaptive_passage_chars(text, ceiling=1200)
    assert size == len(long_sentence)
    assert 300 < size < 1200


def test_an_empty_document_falls_back_to_the_ceiling():
    assert beir._adaptive_passage_chars("   ", ceiling=1200) == 1200


# ── aggregation ──────────────────────────────────────────────────────────

def test_an_unknown_aggregation_is_rejected_loudly():
    with pytest.raises(ValueError):
        beir.passage_rankings({}, {}, 10, passage_chars=600, aggregate="mean")


def test_max_keeps_the_best_piece_and_sum_adds_them():
    """The mechanism, on a corpus built so the two must disagree.

    `wide` mentions the query in five separate pieces and matches it nowhere strongly; `narrow`
    mentions it once, strongly. Max-pool takes `narrow` by its best piece; sum gives `wide` the
    benefit of all five, which is the long-document bias this knob exists to measure.
    """
    corpus = {
        "narrow": "refund refund refund refund refund.",
        "wide": " ".join([f"a sentence mentioning refund weakly number {i}."
                          for i in range(20)]),
    }
    queries = {"q": "refund"}
    by_max = beir.passage_rankings(corpus, queries, 2, passage_chars=200, aggregate="max")
    by_sum = beir.passage_rankings(corpus, queries, 2, passage_chars=200, aggregate="sum")
    assert set(by_max["q"]) == {"narrow", "wide"}
    assert set(by_sum["q"]) == {"narrow", "wide"}
    assert by_max["q"][0] == "narrow"
    assert by_sum["q"][0] == "wide"


def test_max_matches_sorting_by_the_best_piece():
    """The historical behaviour must not move: `max` orders by each document's best piece."""
    corpus = {"a": "alpha alpha alpha.", "b": "alpha beta gamma.", "c": "beta gamma delta."}
    ranking = beir.passage_rankings(corpus, {"q": "alpha"}, 3, passage_chars=600)["q"]
    assert ranking[0] == "a"
    assert "c" not in ranking


def test_adaptive_can_change_the_ranking_against_a_fixed_cut():
    """The knob has to reach the retriever, not just be accepted."""
    corpus = {"long": " ".join(["alpha beta gamma delta epsilon zeta eta theta."] * 40)}
    queries = {"q": "alpha"}
    fixed = beir.passage_rankings(corpus, queries, 1, passage_chars=200)["q"]
    adaptive = beir.passage_rankings(corpus, queries, 1, passage_chars=1200, adaptive=True)["q"]
    assert fixed == adaptive == ["long"]           # same answer here…
    # …but the piece counts differ, which is what makes the *budget* testable elsewhere.
    assert beir._adaptive_passage_chars(corpus["long"], ceiling=1200) != 200


def test_the_sum_arm_reads_deeper_than_max():
    """`sum` has to see every piece of a document to add them, so it cannot share `max`'s depth."""
    corpus = {"d": " ".join([f"alpha filler{i}." for i in range(60)])}
    # Many pieces at 200 characters; a shallow pool would see only some of them.
    short = beir.passage_rankings(corpus, {"q": "alpha"}, 1, passage_chars=200,
                                  aggregate="sum")["q"]
    assert short == ["d"]
