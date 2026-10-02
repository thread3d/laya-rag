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


# ── significance: permutation + FDR ──────────────────────────────────────

def test_permutation_test_is_exact_and_flags_a_consistent_win():
    baseline = {str(i): 0.0 for i in range(6)}
    candidate = {str(i): 0.1 + 0.01 * i for i in range(6)}
    result = evalkit.permutation_test(baseline, candidate)
    assert result["method"] == "exact"
    assert result["queries"] == 6
    # All six deltas positive: only the all-plus and all-minus sign assignments reach
    # the observed magnitude, so p = 2 / 2**6 (rounded to four decimals on report).
    assert result["p_value"] == pytest.approx(round(2 / 64, 4), abs=1e-9)
    assert result["significant"] is True


def test_permutation_test_is_one_for_two_identical_systems():
    scores = {str(i): 0.3 for i in range(5)}
    result = evalkit.permutation_test(scores, dict(scores))
    assert result["p_value"] == 1.0
    assert result["significant"] is False


def test_permutation_test_is_not_significant_for_a_mixed_delta():
    baseline = {str(i): 0.0 for i in range(6)}
    candidate = {"0": 0.2, "1": 0.2, "2": 0.1, "3": -0.1, "4": -0.2, "5": -0.2}
    result = evalkit.permutation_test(baseline, candidate)
    assert result["p_value"] > 0.05
    assert result["significant"] is False


def test_permutation_test_falls_back_to_sampling_above_the_exact_cut():
    baseline = {str(i): 0.0 for i in range(10)}
    candidate = {str(i): 0.05 + 0.001 * i for i in range(10)}
    result = evalkit.permutation_test(baseline, candidate, exact_max_n=4, samples=2000)
    assert result["method"] == "approximate"
    assert result["p_value"] > 0.0  # the +1 correction keeps it off zero
    assert result["draws"] == 2000


def test_permutation_test_requires_shared_queries():
    with pytest.raises(ValueError):
        evalkit.permutation_test({"a": 1.0}, {"b": 1.0})


def test_benjamini_hochberg_matches_the_step_up_definition():
    result = evalkit.benjamini_hochberg({"a": 0.001, "b": 0.04}, q=0.05)
    # m=2: adjusted(a) = min(0.04, 0.001*2/1) = 0.002; adjusted(b) = 0.04*2/2 = 0.04.
    assert result["adjusted"]["a"] == pytest.approx(0.002)
    assert result["adjusted"]["b"] == pytest.approx(0.04)
    assert set(result["rejected"]) == {"a", "b"}
    assert result["any"] is True


def test_benjamini_hochberg_adjusted_values_are_monotone_in_p():
    pvalues = {"a": 0.001, "b": 0.02, "c": 0.2, "d": 0.9}
    adjusted = evalkit.benjamini_hochberg(pvalues, q=0.05)["adjusted"]
    order = sorted(pvalues, key=lambda name: pvalues[name])
    values = [adjusted[name] for name in order]
    assert values == sorted(values)


def test_benjamini_hochberg_rejects_nothing_when_all_p_are_large():
    result = evalkit.benjamini_hochberg({str(i): 0.5 for i in range(10)}, q=0.05)
    assert result["rejected"] == []
    assert result["any"] is False


def test_benjamini_hochberg_handles_an_empty_family():
    result = evalkit.benjamini_hochberg({}, q=0.05)
    assert result == {"q": 0.05, "tests": 0, "adjusted": {}, "rejected": [], "any": False}


# ── selective prediction ─────────────────────────────────────────────────

def test_risk_coverage_curve_tracks_false_accepts():
    probabilities = [0.9, 0.8, 0.2, 0.1]
    labels = [1, 0, 0, 1]
    curve = evalkit.risk_coverage_curve(probabilities, labels)
    assert curve["coverage"] == [0.25, 0.5, 0.75, 1.0]
    # Accept the top one (correct), then a wrong one, then another wrong one, then all.
    assert curve["risk"] == [0.0, 0.5, pytest.approx(2 / 3, abs=1e-6), 0.5]


def test_risk_at_coverage_reads_the_same_curve():
    probabilities = [0.9, 0.8, 0.2, 0.1]
    labels = [1, 0, 0, 1]
    assert evalkit.risk_at_coverage(probabilities, labels, 0.25) == 0.0
    assert evalkit.risk_at_coverage(probabilities, labels, 0.5) == 0.5
    assert evalkit.risk_at_coverage(probabilities, labels, 1.0) == 0.5
    with pytest.raises(ValueError):
        evalkit.risk_at_coverage(probabilities, labels, 0.0)


def test_aurc_rewards_a_better_ranker():
    good = ([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    bad = ([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0])
    assert evalkit.aurc(*good) < evalkit.aurc(*bad)
    assert evalkit.augrc(*good) < evalkit.augrc(*bad)
    # A perfect ranker accepts only positives up to the class prior, so the risk is zero
    # there; AURC is not zero overall because coverage past the prior must accept negatives.
    assert evalkit.risk_at_coverage(good[0], good[1], 0.5) == 0.0
    assert evalkit.aurc(*good) == pytest.approx(0.1458, abs=1e-4)


def test_aurc_handles_empty_input():
    assert evalkit.aurc([], []) == 0.0
    assert evalkit.augrc([], []) == 0.0


def test_brier_and_nll_are_low_for_correct_confident_predictions():
    probabilities = [0.01, 0.99]
    labels = [0, 1]
    assert evalkit.brier_score(probabilities, labels) < 0.01
    assert evalkit.negative_log_likelihood(probabilities, labels) < 0.05
    assert evalkit.brier_score([0.99, 0.01], labels) > 0.9


def test_ece_sweep_is_zero_for_calibrated_probabilities():
    probabilities = [0.0, 0.0, 1.0, 1.0]
    labels = [0, 0, 1, 1]
    result = evalkit.ece_sweep(probabilities, labels, min_bins=2, max_bins=4)
    assert result["ece"] == 0.0
    assert result["monotone"] is True
    assert result["bins"] >= 2


def test_ece_sweep_is_large_for_confidently_wrong_probabilities():
    probabilities = [0.9, 0.9, 0.9, 0.9]
    labels = [0, 0, 0, 0]
    result = evalkit.ece_sweep(probabilities, labels, min_bins=2, max_bins=4)
    assert result["ece"] > 0.8


# ── minimum detectable effect ────────────────────────────────────────────

def test_paired_mde_matches_the_normal_approximation():
    import statistics

    deltas = [0.1, 0.2] * 50
    sd = statistics.stdev(deltas)
    z = statistics.NormalDist().inv_cdf(0.975) + statistics.NormalDist().inv_cdf(0.8)
    expected = round(z * sd / (len(deltas) ** 0.5), 4)
    assert evalkit.paired_mde(deltas) == expected


def test_paired_mde_is_undefined_below_two_queries():
    assert evalkit.paired_mde([0.1]) is None
    assert evalkit.paired_mde([]) is None


def test_paired_mde_shrinks_with_more_queries_and_grows_with_power():
    small = [0.1, 0.2] * 10
    large = [0.1, 0.2] * 40
    assert evalkit.paired_mde(large) < evalkit.paired_mde(small)
    assert evalkit.paired_mde(small, power=0.9) > evalkit.paired_mde(small, power=0.8)
