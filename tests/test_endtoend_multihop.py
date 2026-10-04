"""Offline tests for the end-to-end MultiHop-RAG harness.

The scoring half is pure and is what makes the row readable: EM and F1 must match SQuAD's
normalisation (articles, punctuation, case), and abstention on the answerable records must never be
averaged together with rejection on the null records — they are opposite kinds of correct.
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
    name = "laya_rag_endtoend_multihop"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "endtoend_multihop.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


e2e = _load()


def test_normalize_drops_case_punctuation_articles_and_whitespace():
    assert e2e.normalize_answer("The  Sam Bankman-Fried!") == "sam bankman fried"
    assert e2e.normalize_answer("A  B\n C") == "b c"
    assert e2e.normalize_answer(None) == ""


def test_exact_match_ignores_the_normalization_it_defines():
    assert e2e.exact_match("The Sam Bankman-Fried", "sam bankman fried") == 1.0
    assert e2e.exact_match("Sam Bankman Fried", "FTX") == 0.0


def test_f1_rewards_partial_overlap():
    assert e2e.f1("sam bankman fried", "sam bankman fried") == pytest.approx(1.0)
    # 2 of 2 predicted tokens are right, 2 of 3 gold tokens are covered -> 2 * 1 * (2/3) / (5/3)
    assert e2e.f1("sam fried", "sam bankman fried") == pytest.approx(0.8)
    assert e2e.f1("unrelated", "sam bankman fried") == 0.0


def test_f1_of_two_empty_strings_is_one_and_of_one_is_zero():
    assert e2e.f1("", "") == 1.0
    assert e2e.f1("something", "") == 0.0


def _rows():
    return {
        "0": {"is_null": False, "abstained": False, "answer": "Sam Bankman-Fried",
              "gold": "Sam Bankman-Fried"},
        "1": {"is_null": False, "abstained": True, "answer": "", "gold": "Something else"},
        "2": {"is_null": True, "abstained": True, "answer": "", "gold": "Insufficient information."},
        "3": {"is_null": True, "abstained": False, "answer": "A guess",
              "gold": "Insufficient information.", "error": "boom"},
    }


def test_summarize_separates_abstention_from_rejection():
    summary = e2e.summarize(_rows())
    assert summary["answerable"] == 2 and summary["null"] == 2
    assert summary["abstained_on_answerable"] == 1
    assert summary["rejected_on_null"] == 1
    assert summary["rejection_rate"] == pytest.approx(0.5)
    assert summary["abstention_rate"] == pytest.approx(0.5)
    assert summary["errors"] == 1


def test_summarize_scores_only_the_answers_that_were_written():
    summary = e2e.summarize(_rows())
    assert summary["answered"] == 1
    assert summary["exact_match"] == 1.0
    assert summary["f1"] == 1.0


def test_summarize_of_an_empty_row_set_does_not_divide_by_zero():
    assert e2e.summarize({}) == {
        "answerable": 0, "null": 0, "answered": 0,
        "abstained_on_answerable": 0, "rejected_on_null": 0, "errors": 0,
    }


def test_args_default_to_a_bounded_labelled_run():
    args = e2e.parse_args([])
    assert args.answerable == 30 and args.null_count == 15
    assert args.shard == "0/1" and args.merge == 0
