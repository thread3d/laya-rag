"""Unit tests for the evaluation harness's metric functions.

These run offline and deterministic: the harness decides what "the engine is good"
means, so its own arithmetic has to be checked before its verdicts are believed. The
real-weights runs in ``tools/eval.py`` are only as trustworthy as this module.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import evalkit  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── ranks ────────────────────────────────────────────────────────────────

def test_first_match_rank_is_one_based_and_none_when_absent():
    paths = ["a.md", "billing/refunds.md", "c.md"]
    assert evalkit.first_match_rank(paths, ["refunds.md"]) == 2
    assert evalkit.first_match_rank(paths, ["missing.md"]) is None
    assert evalkit.first_match_rank(paths, []) is None


def test_first_match_rank_prefers_the_earliest_of_several_gold_files():
    paths = ["other.md", "docs/hooks/api.md", "docs/hooks/lifecycle.md"]
    assert evalkit.first_match_rank(paths, ["docs/hooks"]) == 2


def test_rank_summary_counts_a_miss_against_recall_and_mrr():
    summary = evalkit.rank_summary([1, 2, None], k=5)
    assert summary["n"] == 3
    assert summary["found"] == 2
    assert summary["recall"] == pytest.approx(0.6667, abs=1e-4)
    assert summary["mrr"] == pytest.approx((1 + 0.5) / 3, abs=1e-4)
    assert summary["precision@5"] == pytest.approx(0.6667, abs=1e-4)
    assert summary["median_rank"] == 1.5
    assert summary["worst_rank"] == 2


def test_rank_summary_on_nothing_is_zero_not_an_error():
    summary = evalkit.rank_summary([], k=5)
    assert summary["n"] == 0
    assert summary["recall"] == 0.0
    assert summary["median_rank"] is None


def test_rank_summary_worst_rank_is_the_largest_not_the_median():
    summary = evalkit.rank_summary([1, 9, 3], k=5)
    assert summary["worst_rank"] == 9
    assert summary["precision@5"] == pytest.approx(2 / 3, abs=1e-4)


def test_margin_summary_reports_how_many_queries_lost_to_a_negative():
    summary = evalkit.margin_summary([0.4, -0.1, 0.2, None])
    assert summary["n"] == 3
    assert summary["negative"] == 1
    assert summary["min"] == -0.1
    assert summary["mean"] == pytest.approx(1 / 6, abs=1e-4)


# ── calibration ──────────────────────────────────────────────────────────

def test_a_perfectly_calibrated_decision_has_zero_error():
    probabilities = [0.5, 0.5, 0.5, 0.5]
    labels = [1, 0, 1, 0]
    assert evalkit.expected_calibration_error(probabilities, labels) == 0.0


def test_an_overconfident_decision_is_penalised():
    probabilities = [0.9, 0.9, 0.9, 0.9]
    labels = [1, 0, 0, 0]
    # One of four is right at a stated confidence of 0.9, so the gap is 0.65.
    assert evalkit.expected_calibration_error(probabilities, labels) == pytest.approx(0.65, abs=1e-4)


def test_reliability_bins_place_a_boundary_probability_in_the_upper_bin():
    bins = evalkit.reliability_bins([0.9], [1])
    assert len(bins) == 1
    assert bins[0]["range"] == "0.9-1.0"
    assert bins[0]["count"] == 1


def test_reliability_bins_omit_empty_buckets():
    bins = evalkit.reliability_bins([0.05, 0.95], [0, 1])
    assert [entry["bin"] for entry in bins] == [0, 9]


def test_calibration_on_no_pairs_is_zero():
    assert evalkit.expected_calibration_error([], []) == 0.0
    assert evalkit.reliability_bins([], []) == []


# ── threshold sweep ──────────────────────────────────────────────────────

def test_a_threshold_above_every_probability_predicts_nothing():
    row = [r for r in evalkit.threshold_sweep([0.9, 0.1], [1, 0], [0.95])][0]
    assert row["recall"] == 0.0
    assert row["precision"] == 0.0
    assert row["fn"] == 1


def test_a_threshold_below_every_probability_predicts_everything():
    row = evalkit.threshold_sweep([0.9, 0.1], [1, 0], [0.05])[0]
    assert row["recall"] == 1.0
    assert row["precision"] == pytest.approx(0.5)
    assert row["fp"] == 1


def test_the_shipped_relevance_threshold_separates_a_clean_pair():
    rows = evalkit.threshold_sweep([0.79, 0.001], [1, 0], [0.5])
    assert rows[0]["f1"] == 1.0
    assert rows[0]["tp"] == 1 and rows[0]["tn"] == 1


def test_the_default_sweep_covers_the_unit_interval_without_the_extremes():
    sweep = evalkit.threshold_sweep([0.5], [1])
    assert sweep[0]["threshold"] == 0.05
    assert sweep[-1]["threshold"] == 0.95
    assert len(sweep) == 19


# ── query sets ───────────────────────────────────────────────────────────

def test_queries_round_trip_through_jsonl():
    original = evalkit.planted_queries()
    parsed = evalkit.parse_queries(evalkit.dump_queries(original))
    assert parsed == original


def test_query_parsing_ignores_comments_and_blank_lines():
    parsed = evalkit.parse_queries(
        '# a comment\n\n{"id":"a","query":"Q?","gold":["x.md"]}\n'
    )
    assert len(parsed) == 1
    assert parsed[0].gold == ("x.md",)


def test_a_malformed_query_line_names_its_line_number():
    with pytest.raises(ValueError, match="line 2"):
        evalkit.parse_queries('{"id":"a","query":"Q?"}\nnot json')


def test_a_query_matches_a_gold_file_by_substring():
    query = evalkit.EvalQuery(id="q", query="Q?", gold=("billing/refunds.md",))
    assert query.matches("/repo/billing/refunds.md")
    assert not query.matches("/repo/billing/invoices.md")


def test_planted_query_ids_are_unique():
    ids = [query.id for query in evalkit.planted_queries()]
    assert len(ids) == len(set(ids))


# ── fixture integrity ────────────────────────────────────────────────────

def test_every_answerable_planted_query_names_a_file_that_exists():
    corpus = evalkit.planted_corpus()
    for query in evalkit.planted_queries():
        if query.answerable:
            assert query.gold, f"{query.id} is answerable but names no gold file"
            for fragment in query.gold:
                assert any(fragment in path for path in corpus), (
                    f"{query.id} names {fragment!r}, which is not in the planted corpus"
                )


def test_unanswerable_queries_name_no_gold_file():
    for query in evalkit.planted_queries():
        if not query.answerable:
            assert query.gold == ()


def test_the_unanswerable_planted_facts_are_actually_absent_from_the_corpus():
    corpus = evalkit.planted_corpus().values()
    text = " ".join(corpus).casefold()
    # One term per unanswerable query that would have to appear for the answer to exist.
    for term in ("reykjavik", "cryptocurrency", "salary", "chief executive's home", "gift card"):
        assert term not in text, f"{term!r} leaked into the planted corpus"


def test_there_are_hard_negatives_so_the_answerability_gate_is_actually_exercised():
    """Without a negative that clears relevance, abstention is decided by relevance alone."""
    categories = {query.category for query in evalkit.planted_queries() if not query.answerable}
    assert "hard-negative" in categories


def test_docs_queries_label_files_in_the_vendored_checkout():
    docs = REPO_ROOT / "vendor" / "laya" / "docs"
    for query in evalkit.docs_queries():
        if not query.answerable:
            continue
        assert query.gold
        assert any(str(path).endswith(fragment) or fragment in str(path)
                   for path in docs.rglob("*") for fragment in query.gold)


# ── reporting ────────────────────────────────────────────────────────────

def test_markdown_table_renders_the_header_and_every_row():
    table = evalkit.markdown_table([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
    lines = table.splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| 1 | x |"
    assert len(lines) == 4


def test_markdown_table_renders_missing_values_as_a_dash():
    assert "—" in evalkit.markdown_table([{"a": None}])


def test_compare_reports_finds_a_numeric_delta():
    differences = evalkit.compare_reports({"mrr": 0.5}, {"mrr": 0.6})
    assert differences == [{"metric": "mrr", "baseline": 0.5, "current": 0.6, "delta": 0.1}]


def test_compare_reports_is_silent_when_nothing_moved():
    assert evalkit.compare_reports({"mrr": 0.5, "label": "x"}, {"mrr": 0.5, "label": "x"}) == []
