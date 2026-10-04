"""Offline tests for the citation-mechanics report.

This measures the half of an end-to-end row that belongs to the engine rather than the generator,
so the arithmetic has to be exact: a coverage ratio that counted answers instead of sentences, or a
quote validity that ignored unfound quotes, would report the engine as better than it is.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def _load():
    name = "laya_rag_citation_report"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "citation_report.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cr = _load()


def _row(is_null=False, claims=4, cited=3, quotes_checked=2, quotes_found=2, clean=True,
         invalid=0, injection=0):
    return {"is_null": is_null, "clean": clean, "claims": claims, "cited": cited,
            "quotes_checked": quotes_checked, "quotes_found": quotes_found,
            "invalid_citations": invalid, "injection": injection, "quote_statuses": []}


def test_ratio_of_nothing_is_zero_not_an_error():
    assert cr._ratio(0, 0) == 0.0
    assert cr._ratio(1, 4) == 0.25


def test_summarize_counts_sentences_not_answers():
    rows = {"0": _row(claims=4, cited=3), "1": _row(claims=2, cited=1)}
    summary = cr.summarize(rows)["answerable"]
    assert summary["claims"] == 6 and summary["cited_claims"] == 4
    assert summary["citation_coverage"] == pytest.approx(round(4 / 6, 4))


def test_summarize_splits_answerable_from_null():
    rows = {"0": _row(is_null=False), "1": _row(is_null=True, quotes_checked=2, quotes_found=0)}
    summary = cr.summarize(rows)
    assert summary["all"]["answers"] == 2
    assert summary["answerable"]["answers"] == 1
    assert summary["null"]["answers"] == 1
    assert summary["null"]["quote_validity"] == 0.0
    assert summary["answerable"]["quote_validity"] == 1.0


def test_quote_validity_counts_unfound_quotes_against_the_engine():
    rows = {"0": _row(quotes_checked=4, quotes_found=1)}
    assert cr.summarize(rows)["all"]["quote_validity"] == pytest.approx(0.25)


def test_summarize_totals_invalid_citations_and_injection_flags():
    rows = {"0": _row(invalid=2, injection=1), "1": _row(invalid=0, injection=0)}
    summary = cr.summarize(rows)["all"]
    assert summary["invalid_citations"] == 2
    assert summary["injection_flags"] == 1


def test_clean_rate_is_a_share_of_answers():
    rows = {"0": _row(clean=True), "1": _row(clean=False)}
    assert cr.summarize(rows)["all"]["clean_rate"] == pytest.approx(0.5)


def test_render_markdown_names_all_three_groups():
    payload = {"_settings": {"answers": 2, "report": "x.json"},
               "summary": cr.summarize({"0": _row(), "1": _row(is_null=True)})}
    text = cr.render_markdown(payload)
    for label in ("| all |", "| answerable |", "| null |"):
        assert label in text


def test_main_refuses_a_report_with_no_answers(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"_settings": {}, "per_query": {"0": {"is_null": False}}}))
    with pytest.raises(SystemExit, match="no composed answers"):
        cr.main(["--report", str(empty), "--out", str(tmp_path / "out")])
