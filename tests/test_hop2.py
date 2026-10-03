"""Offline tests for the hop-2 expansion probe.

The measurement decides whether the bridge signal is worth spending a cross-encoder on, so
the mining rule has to do what it claims: propose terms the query does *not* already contain,
prefer rare ones, and never invent a bridge when the top candidate has nothing new to say.

The mining works on **stemmed** tokens (the same tokenizer the lexical channel uses), so
these tests derive their expected keys through that tokenizer rather than assuming a raw
word survives unchanged.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from laya_rag.lexical import STOPWORDS, tokenize  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location(
        "laya_rag_multihop_hop2", ROOT / "tools" / "multihop_hop2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hop2 = _load()


def stem(word: str) -> str:
    """The mining key for a word: the tool tokenizes with stemming."""
    return tokenize(word, stem=True)[0]


def terms(text: str) -> set[str]:
    return {t for t in tokenize(text, stem=True) if t not in STOPWORDS}


CORPUS = {
    "a#0": "the terminator is a 1984 film directed by james cameron",
    "b#0": "arnold schwarzenegger starred in the terminator and later governed california",
    "c#0": "maria shriver is a journalist and the wife of arnold schwarzenegger",
    "d#0": "generic filler about shipping and refunds and other unrelated policy matters",
}


def test_document_frequency_counts_chunks_not_occurrences():
    """A term repeated inside one chunk counts once; a term in two chunks counts twice."""
    df = hop2.document_frequency({
        "x#0": "alpha alpha alpha beta",
        "y#0": "beta",
    })
    assert df[stem("alpha")] == 1
    assert df[stem("beta")] == 2


def test_bridge_terms_exclude_every_term_the_query_already_has():
    df = hop2.document_frequency(CORPUS)
    query = "who played the terminator"
    mined = hop2.bridge_terms(query, ["a#0"], CORPUS, df, max_terms=5)
    assert mined, "the top candidate has new terms to offer"
    assert not (set(mined) & terms(query)), "a bridge term must be absent from the query"


def test_bridge_terms_prefer_rare_terms():
    """A word in every chunk is not an entity; a name in one is."""
    corpus = {"a#0": "alpha zzunique", "b#0": "alpha", "c#0": "alpha"}
    df = hop2.document_frequency(corpus)
    mined = hop2.bridge_terms("q", ["a#0"], corpus, df, max_terms=1)
    assert mined == [stem("zzunique")]


def test_bridge_terms_are_capped_and_deduplicated():
    df = hop2.document_frequency(CORPUS)
    mined = hop2.bridge_terms("q", ["b#0"], CORPUS, df, max_terms=3)
    assert len(mined) == 3
    assert len(set(mined)) == 3


def test_bridge_terms_can_read_several_candidates():
    df = hop2.document_frequency(CORPUS)
    one = hop2.bridge_terms("q", ["a#0", "b#0"], CORPUS, df, top_candidates=1, max_terms=5)
    two = hop2.bridge_terms("q", ["a#0", "b#0"], CORPUS, df, top_candidates=2, max_terms=5)
    assert len(two) >= len(one)
    # Reading wider can only add candidates for the same ranking, never remove one.
    assert set(one) <= set(two) or len(two) == 5


def test_bridge_terms_of_an_empty_pool_is_empty():
    df = hop2.document_frequency(CORPUS)
    assert hop2.bridge_terms("q", [], CORPUS, df) == []


def test_render_markdown_reports_both_recalls_and_the_delta():
    result = {
        "_settings": {"queries": 2255, "base_depth": 100, "expand_depth": 100,
                      "top_candidates": 1, "max_terms": 5},
        "summary": {"base_pool_fact_recall": 0.8666, "expanded_pool_fact_recall": 0.901,
                    "queries_improved": 200, "queries_worsened": 0, "gold_facts_added": 90},
        "comparison": {
            "bootstrap": {"delta": 0.0344, "ci_low": 0.02, "ci_high": 0.05, "significant": True},
            "permutation": {"p_value": 0.001},
        },
    }
    text = hop2.render_markdown(result)
    assert "Hop-2 expansion" in text
    assert "0.8666" in text and "0.9010" in text
    assert "+0.0344" in text
