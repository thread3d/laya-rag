"""Tests for the engine's extractive end-to-end row.

The row's whole value is that every component is the engine's, so the two things worth pinning are
the ones that would silently make it someone else's: an answer that is *not* a verbatim span of the
returned passages must fail the citation check rather than pass it, and the retrieval half must be
the benchmark's own metric over the same passages — not a re-derivation.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def _load():
    name = "laya_rag_engine_extractive"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "engine_extractive.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ex = _load()


def _payload(answer: str, sentences, text: str, *, abstained=False, query="who?"):
    return {
        "query": query,
        "answer": answer,
        "abstained": abstained,
        "sentences": sentences,
        "answerable": True,
        "answerability": 0.8,
        "passages": [{"path": "/docs/000000.txt", "text": text, "chunk_id": "a:0000"}],
    }


def _item(*facts: str):
    return {"evidence_list": [{"fact": fact} for fact in facts]}


def test_a_verbatim_answer_that_cites_cleanly_scores_clean():
    fact = "Refunds are processed within five business days."
    row = ex.score_one(
        _payload("[1] Refunds are processed within five business days.",
                 [{"text": fact, "citation": 1, "path": "/docs/000000.txt", "chunk_id": "a:0000"}],
                 fact),
        _item(fact),
    )
    assert row["citation_clean"] is True
    assert row["citation_uncited"] == 0
    assert row["nugget_recall"] == pytest.approx(1.0)


def test_an_uncited_answer_is_not_clean():
    """The contract the row rests on: below 1.0 on attribution is a bug, and it must show up."""
    fact = "Refunds are processed within five business days."
    row = ex.score_one(
        _payload("Refunds are processed within five business days.", [], fact),
        _item(fact),
    )
    assert row["citation_clean"] is False
    assert row["citation_uncited"] == 1


def test_an_answer_quoting_something_not_in_the_evidence_is_not_clean():
    """A span that is not in the passages cannot be verified, so the row must not call it clean."""
    fact = "Refunds are processed within five business days."
    row = ex.score_one(
        _payload('[1] "Refunds take a fortnight" is the policy.', [
            {"text": "Refunds take a fortnight is the policy.", "citation": 1},
        ], fact),
        _item(fact),
    )
    assert row["citation_clean"] is False
    assert row["citation_unfound_quotes"] >= 1


def test_the_retrieval_half_is_the_benchmark_s_metric():
    """A gold fact present in the returned passage is a hit; one that is not, is not."""
    hit = ex.score_one(
        _payload("[1] Refunds are processed within five business days.", [], 
                 "Refunds are processed within five business days."),
        _item("Refunds are processed within five business days."),
    )
    miss = ex.score_one(
        _payload("[1] Something else entirely about shipping.", [],
                 "Something else entirely about shipping."),
        _item("Refunds are processed within five business days."),
    )
    assert hit["Hits@10"] == 1.0
    assert miss["Hits@10"] == 0.0


def test_answer_nugget_recall_is_the_answer_s_not_the_passage_s():
    """The point of the metric: the passage covers the fact, a short extract of it may not."""
    fact = "Refunds arrive in 5 days; shipping is free over $50."
    row = ex.score_one(
        _payload("[1] Refunds arrive in 5 days.", [], fact + " " + fact),
        _item(fact),
    )
    assert row["Hits@10"] == 1.0                 # the passage covers it
    assert row["nugget_recall"] < 1.0           # the answer does not


def test_a_query_with_no_nuggets_reports_none_rather_than_zero():
    row = ex.score_one(_payload("[1] A.", [], "A."), _item("a"))
    assert row["nugget_recall"] is None


def test_summarize_counts_abstention_and_skips_null_recalls():
    # The answer has to be long enough to be a claim: `split_claims` drops a fragment of fewer
    # than three words, and a report with no claims is not a clean one.
    fact = "Refunds are processed within five business days."
    rows = [
        {**ex.score_one(_payload(f"[1] {fact}", [], fact), _item(fact))},
        {
            "abstained": True, "citation_clean": False, "citation_uncited": 0, "citation_claims": 0,
            "citation_invalid": 0, "citation_unfound_quotes": 0, "nugget_recall": None,
            "nuggets": 0, "Hits@10": 0.0, "Hits@4": 0.0, "MAP@10": 0.0, "MRR@10": 0.0,
        },
    ]
    summary = ex.summarize(rows)
    assert summary["queries"] == 2
    assert summary["answered"] == 1
    assert summary["abstention_rate"] == pytest.approx(0.5)
    # Attribution is a property of an answer: the rate is over the one that was emitted, while
    # `citation_clean_all` carries the abstention's uncited sentence.
    assert summary["citation_clean_rate"] == pytest.approx(1.0)
    assert summary["citation_clean_all"] == pytest.approx(0.5)
    assert summary["answer_nugget_recall"] == pytest.approx(1.0)


def test_summarize_of_nothing_is_empty_not_a_crash():
    assert ex.summarize([]) == {"queries": 0}


def test_render_states_that_a_citation_score_below_one_is_a_bug():
    text = ex.render({"summary": {
        "queries": 2, "answered": 1, "abstention_rate": 0.5, "citation_clean_rate": 1.0,
        "citation_clean_all": 1.0, "uncited_sentences": 0,
        "claims": 3, "answer_nugget_recall": 0.4,
        "Hits@10": 0.9, "Hits@4": 0.7, "MAP@10": 0.3, "MRR@10": 0.6,
    }})
    assert "1.0 by construction" in text
    assert "answer nugget recall" in text


def test_render_handles_a_missing_nugget_reading():
    text = ex.render({"summary": {
        "queries": 1, "answered": 0, "abstention_rate": 1.0, "citation_clean_rate": None,
        "citation_clean_all": 1.0, "uncited_sentences": 0,
        "claims": 0, "answer_nugget_recall": None,
        "Hits@10": 0.0, "Hits@4": 0.0, "MAP@10": 0.0, "MRR@10": 0.0,
    }})
    assert "| n/a |" in text
