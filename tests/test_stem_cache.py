"""Tests for the memoised stemmer.

`snowballstemmer` is pure Python and a corpus repeats its words heavily — 6.2M tokens in a
20,000-document sample collapsed to 74,334 distinct stems — so stemming the same token over and
over was ~97% of a 54-minute lexical build. The memo must not change a single result, which is
what these tests are for.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laya_rag import lexical  # noqa: E402


@pytest.fixture(autouse=True)
def _cold_cache():
    """Each test starts from an empty memo, so a warm one cannot mask a behaviour change."""
    lexical._stem_cache.clear()
    yield
    lexical._stem_cache.clear()


def test_the_memo_does_not_change_the_tokens():
    text = "Refunds are processed within five business days of the REFUND request"
    cold = lexical.tokenize(text, stem=True)
    lexical._stem_cache.clear()
    lexical._stemmer = None
    warm = lexical.tokenize(text, stem=True)
    assert cold == warm


def test_a_cached_stem_equals_a_fresh_one():
    token = "processed"
    first = lexical._stem_word(token)
    assert lexical._stem_cache[token] == first
    lexical._stemmer = None                      # force a fresh stemmer, memo left warm
    assert lexical._stem_word(token) == first


def test_only_stemmable_tokens_are_memoised():
    lexical._stem_cache.clear()
    lexical._stem_word("1234")
    lexical._stem_word("東京")
    lexical._stem_word("two-words")
    assert lexical._stem_cache == {}, "digits, CJK and identifiers are returned untouched"


def test_a_repeated_corpus_hits_the_memo():
    text = "refunds refunds refunds shipping shipping"
    lexical.tokenize(text, stem=True)
    size = len(lexical._stem_cache)
    assert size == 2, "one entry per distinct token, not per occurrence"
    lexical.tokenize(text, stem=True)
    assert len(lexical._stem_cache) == size, "a second pass adds no new stems"


def test_stemming_off_does_not_touch_the_memo():
    lexical._stem_cache.clear()
    assert lexical.tokenize("Refunds processed", stem=False) == ["refunds", "processed"]
    assert lexical._stem_cache == {}


def test_the_memo_is_bounded_by_vocabulary_not_length():
    """The memory it adds is a function of distinct tokens, which is the point."""
    for _ in range(500):
        lexical.tokenize("alpha beta gamma", stem=True)
    assert len(lexical._stem_cache) <= 3
