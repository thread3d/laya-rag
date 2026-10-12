"""Tests for the FRAMES extractive row.

FRAMES differs from MultiHop-RAG in the way that matters here: the gold is a set of article
*titles* and the gold answer is usually a few words. Those two are the things these tests pin —
the positional path -> title mapping the retrieval half depends on, and that the nugget count is
taken against the gold answer rather than against the evidence.
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
    name = "laya_rag_frames_extractive"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "frames_extractive.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fx = _load()

TITLES = ["alpha article", "beta article"]


def _payload(answer, text, path="/docs/000001.txt", abstained=False):
    return {
        "query": "who?",
        "answer": answer,
        "abstained": abstained,
        "sentences": [{"text": answer, "citation": 1, "path": path, "chunk_id": "c"}],
        "passages": [{"path": path, "text": text, "chunk_id": "c"}],
    }


def test_a_returned_passage_maps_back_to_its_article_title():
    row = fx.score_one(
        _payload("[1] Beta article says the answer is 2005.", "Beta article says the answer is 2005."),
        {"beta article"}, "2005", TITLES,
    )
    assert row["recall@10"] == 1.0
    assert row["mrr@10"] == 1.0


def test_a_passage_from_the_wrong_article_is_not_a_hit():
    row = fx.score_one(
        _payload("[1] Alpha article says something else entirely.",
                 "Alpha article says something else entirely.",
                 path="/docs/000000.txt"),
        {"beta article"}, "2005", TITLES,
    )
    assert row["recall@10"] == 0.0


def test_the_nugget_count_is_against_the_gold_answer_not_the_evidence():
    """The regime FRAMES tests: the passage states the answer, a short extract may still miss it."""
    row = fx.score_one(
        _payload("[1] A sentence about something else.",
                 "A sentence about something else. The gold answer is 2005.",
                 path="/docs/000000.txt"),
        {"alpha article"}, "2005", TITLES,
    )
    assert row["recall@10"] == 1.0          # the passage covers the gold article
    assert row["nugget_recall"] == 0.0      # the answer does not carry the gold answer's nugget


def test_an_answer_carrying_the_gold_answer_scores_one():
    row = fx.score_one(
        _payload("[1] It was 2005.", "It was 2005.", path="/docs/000000.txt"),
        {"alpha article"}, "2005", TITLES,
    )
    assert row["nugget_recall"] == pytest.approx(1.0)


def test_a_question_with_no_gold_nuggets_reports_none():
    row = fx.score_one(_payload("[1] A sentence.", "A sentence."), {"alpha article"}, "", TITLES)
    assert row["nugget_recall"] is None


def test_an_abstention_is_counted_separately_from_attribution():
    row = fx.score_one(
        _payload("The retrieved evidence does not contain a sentence that answers the question.",
                 "nothing useful", abstained=True),
        {"alpha article"}, "2005", TITLES,
    )
    summary = fx.summarize([row])
    assert summary["abstention_rate"] == 1.0
    assert summary["citation_clean_rate"] is None
    assert summary["answered"] == 0


def test_summarize_carries_the_retrieval_metrics_through():
    row = fx.score_one(
        _payload("[1] Beta article says the answer is 2005.", "Beta article says the answer is 2005."),
        {"beta article"}, "2005", TITLES,
    )
    summary = fx.summarize([row])
    assert summary["recall@10"] == 1.0
    assert summary["ndcg@10"] == pytest.approx(1.0)


def test_summarize_of_nothing_is_empty_not_a_crash():
    assert fx.summarize([]) == {"queries": 0}


def test_the_gate_configuration_is_explicit():
    """The row's abstention number is the composer's only when the gates are zeroed.

    `configure_engine` zeroes them for the retrieval rows; putting them back is the engine as
    deployed. Either is defensible, and which one this run used has to be a flag rather than a
    default nobody can see — the first attempt at this row ran with the shipped gates and read an
    abstention of 0.163 that was the gate's refusal, not the composer's.
    """
    assert fx.parse_args([]).zero_thresholds is False
    assert fx.parse_args(["--zero-thresholds"]).zero_thresholds is True


def test_render_states_the_bug_contract_and_the_gold_answer_metric():
    text = fx.render({"summary": {
        "queries": 1, "answered": 1, "abstention_rate": 0.0, "citation_clean_rate": 1.0,
        "uncited_sentences": 0, "claims": 3, "answer_nugget_recall": 0.5,
        "recall@10": 1.0, "mrr@10": 1.0, "ndcg@10": 1.0,
    }})
    assert "1.0 by construction" in text
    assert "answer nugget recall" in text
    assert "recall@10" in text
