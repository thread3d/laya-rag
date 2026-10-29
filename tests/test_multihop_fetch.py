"""The corpus fetch asserts the dataset's shape before any measurement is derived from it.

Every abstention number in this repository rests on one property of the download: exactly 301 of the
2,556 records are `null_query`. A truncated or re-versioned corpus would produce a *smaller* number that
reads as a result, so the check is a test rather than a comment.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import multihop_fetch as mf  # noqa: E402


def _records(n: int, nulls: int) -> list[dict]:
    return [{"query": f"q{i}", "question_type": "inference_query"} for i in range(n - nulls)] + [
        {"query": f"n{i}", "question_type": "null_query"} for i in range(nulls)
    ]


def _corpus(n: int) -> list[dict]:
    return [{"title": f"t{i}", "body": f"b{i}"} for i in range(n)]


def test_a_correct_download_has_no_complaints():
    assert mf.verify(_records(mf.RECORDS, mf.NULL_QUERIES), _corpus(mf.DOCUMENTS)) == []


def test_a_truncated_corpus_is_refused():
    problems = mf.verify(_records(mf.RECORDS, mf.NULL_QUERIES), _corpus(mf.DOCUMENTS - 1))
    assert any("documents" in problem for problem in problems)


def test_a_record_count_that_moved_is_refused():
    problems = mf.verify(_records(mf.RECORDS - 1, mf.NULL_QUERIES), _corpus(mf.DOCUMENTS))
    assert any("records" in problem for problem in problems)


def test_the_null_query_count_is_asserted_separately_and_by_name():
    """This is the number the abstention axis is scored on, so its failure says so."""
    problems = mf.verify(_records(mf.RECORDS, mf.NULL_QUERIES - 2), _corpus(mf.DOCUMENTS))
    assert any("null_query" in problem and "abstention" in problem for problem in problems)


def test_a_record_without_a_query_is_refused():
    records = _records(mf.RECORDS, mf.NULL_QUERIES)
    records[3]["query"] = "   "
    assert any("no query" in problem for problem in mf.verify(records, _corpus(mf.DOCUMENTS)))


def test_a_document_without_a_body_is_refused():
    corpus = _corpus(mf.DOCUMENTS)
    corpus[5]["body"] = ""
    assert any("no body" in problem for problem in mf.verify(_records(mf.RECORDS, mf.NULL_QUERIES), corpus))


def test_check_mode_refuses_a_missing_file_without_touching_the_network(tmp_path):
    import pytest

    with pytest.raises(SystemExit):
        mf.main(["--check", "--out", str(tmp_path)])
