"""Unit tests for the evidence-set coverage selectors in ``tools/multihop_coverage.py``.

These exercise the pure selection arithmetic — document quotas, fact-atom dedup, atom
facility location, and the pool reorder ceiling — against hand-computed values. No
encoder, network, or benchmark data is touched: importing the module only pulls in its
pure ``beir``/``evalkit`` imports.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import multihop_coverage as mc  # noqa: E402


# ── fact atoms ──────────────────────────────────────────────────────────

def test_split_atoms_splits_sentences_and_drops_short_fragments():
    text = "The first fact is long enough to keep. Short. \nAnother sufficiently long fact here!"
    assert mc.split_atoms(text) == [
        "The first fact is long enough to keep.",
        "Another sufficiently long fact here!",
    ]


def test_split_atoms_respects_min_chars():
    assert mc.split_atoms("Alpha. Beta.", min_chars=100) == []
    assert mc.split_atoms("Alpha. Beta.", min_chars=1) == ["Alpha.", "Beta."]


def test_atom_index_deduplicates_identical_atoms_across_candidates():
    a = "The same overlapping fact appears here."
    b = "A distinct second fact lives in this chunk."
    per_candidate, weight, texts = mc.atom_index([[a, b], [a], [a.upper()]])

    assert len(texts) == 2
    assert per_candidate == [[0, 1], [0], [0]]
    assert all(w >= 0 for w in weight)
    assert weight[0] == pytest.approx(float(len(" ".join(a.lower().split()))))
    assert weight[1] == pytest.approx(float(len(" ".join(b.lower().split()))))


def test_atom_index_deduplicates_within_a_candidate():
    per_candidate, _, texts = mc.atom_index([["Repeated fact, long enough to matter.", "repeated   fact, long enough to matter."]])
    assert len(texts) == 1
    assert per_candidate == [[0]]


# ── document quota ──────────────────────────────────────────────────────

def test_doc_quota_select_represents_min_docs_and_respects_top_k():
    rel = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    docs = np.array(["a", "a", "b", "c", "a"])
    selected = mc.doc_quota_select(rel, docs, top_k=4, min_docs=3)

    assert len(selected) == 4
    assert len({docs[i] for i in selected}) == 3
    # first pass takes the best-ranked candidate for each new document
    assert selected[:3] == [0, 2, 3]
    # second pass fills from untouched candidates in descending relevance
    assert selected[3] == 1


def test_doc_quota_select_is_deterministic_and_ties_break_to_lower_index():
    docs = np.array(["a", "a", "b", "b"])
    rel = np.zeros(4)
    first = mc.doc_quota_select(rel, docs, top_k=3, min_docs=2)
    second = mc.doc_quota_select(rel, docs, top_k=3, min_docs=2)

    assert first == second
    assert first == [0, 2, 1]


def test_doc_quota_select_falls_back_when_fewer_docs_exist():
    rel = np.array([0.5, 0.4, 0.3])
    docs = np.array(["a", "a", "a"])
    selected = mc.doc_quota_select(rel, docs, top_k=3, min_docs=5)

    assert selected == [0, 1, 2]
    assert len(selected) <= 3


# ── atom facility location ──────────────────────────────────────────────

def test_facility_location_select_covers_distinct_atoms_not_duplicates():
    # A and B carry the same two atoms; C carries one new, higher-weight atom.
    rel = np.zeros(3)
    cand_atoms = [[0, 1], [0, 1], [2]]
    weights = [50.0, 50.0, 100.0]
    gram = np.zeros((3, 3))

    selected = mc.facility_location_select(rel, cand_atoms, weights, gram, top_k=3, lam=0.5)

    assert selected[0] == 0
    assert selected[1] == 2  # B repeats A's atoms and adds nothing; C adds atom 2
    covered: set[int] = set()
    growth = []
    for i in selected:
        covered.update(cand_atoms[i])
        growth.append(len(covered))
    assert growth == [2, 3, 3]
    assert covered == {0, 1, 2}


def test_facility_location_select_respects_top_k_and_valid_indices():
    rel = np.array([0.4, 0.9, 0.1, 0.7])
    cand_atoms = [[0], [1, 2], [3], [4, 5]]
    weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    gram = np.eye(4)

    selected = mc.facility_location_select(rel, cand_atoms, weights, gram, top_k=2, lam=0.5)

    assert len(selected) == 2
    assert len(set(selected)) == 2
    assert all(0 <= i < 4 for i in selected)


def test_facility_location_select_handles_constant_relevance():
    cand_atoms = [[0], [1]]
    gram = np.zeros((2, 2))
    # constant rel must normalize to all-zeros rather than divide by zero
    selected = mc.facility_location_select(np.array([3.0, 3.0]), cand_atoms, [1.0, 1.0], gram, top_k=1)
    assert selected == [0]


def test_facility_location_select_can_drop_the_redundancy_penalty():
    # Candidates 1 and 2 add distinct atoms and are otherwise identical; candidate 1 is
    # nearly parallel to candidate 0. The gram penalty should avoid it, and dropping the
    # penalty should let the tie break to the lower index instead.
    rel = np.array([1.0, 1.0, 1.0])
    cand_atoms = [[0], [1], [2]]
    weights = [1.0, 1.0, 1.0]
    gram = np.zeros((3, 3))
    gram[0, 1] = gram[1, 0] = 0.9

    with_penalty = mc.facility_location_select(rel, cand_atoms, weights, gram, top_k=2, lam=0.5)
    without = mc.facility_location_select(
        rel, cand_atoms, weights, gram, top_k=2, lam=0.5, use_redundancy=False
    )

    assert with_penalty == [0, 2]
    assert without == [0, 1]


# ── pool reorder ceiling ────────────────────────────────────────────────

def test_pool_ceiling_map_places_the_largest_new_match_first():
    # Greedy picks candidate 1 (two new facts) at rank 1 -> AP = 2/1 / min(2,2) = 1.0.
    assert mc.pool_ceiling_map(None, [{0}, {0, 1}], 2, gold_count=2) == pytest.approx(1.0)


def test_pool_ceiling_map_hand_computed_partial_coverage():
    # pool [[g1],[g2],[g3]] top_k=2: ranks find one new fact each ->
    # AP = (1/1 + 1/2) / min(3, 2) = 0.75.
    assert mc.pool_ceiling_map(None, [{0}, {1}, {2}], 2, gold_count=3) == pytest.approx(0.75)


def test_pool_ceiling_map_is_zero_without_matches():
    assert mc.pool_ceiling_map(None, [set(), set()], 2, gold_count=2) == 0.0
    assert mc.pool_ceiling_map(None, [], 10, gold_count=0) == 0.0


def test_pool_ceiling_map_divides_by_the_true_gold_count_not_the_pool_matches():
    """The bug this signature exists to prevent.

    Three gold facts, but the pool only contains one of them. The ceiling normalises by the
    true count (3), so the single reachable fact at rank 1 gives 1/1 / 3 = 0.333 — not
    1/1 / 1 = 1.0, which is what dividing by the pool's own matched facts would report. The
    wrong denominator *inflates* the ceiling on exactly the queries where recall is failing.
    """
    assert mc.pool_ceiling_map(None, [{0}, set(), set()], 10, gold_count=3) == pytest.approx(1 / 3)
    # Denominator is capped at top_k, matching the official metric.
    assert mc.pool_ceiling_map(
        None, [{i} for i in range(20)], 10, gold_count=20
    ) == pytest.approx(sum(1 / r for r in range(1, 11)) / 10)


def test_pool_ceiling_map_requires_a_gold_count():
    """No default: the only obvious default is the one that silently overstates."""
    with pytest.raises(TypeError):
        mc.pool_ceiling_map(None, [{0}], 10)


# ── metric-definition audit ─────────────────────────────────────────────

def test_official_metric_scoring_examines_exactly_the_top_k():
    """A retrieved list longer than k must not have its k+1-th item examined at all.

    The metric sliced ``retrieved[:11]`` while guarding on ``rank <= 10``: the extra item
    was discarded, so the number was right, but any edit to the guard would have let rank
    11 score. The slice is now exactly k, and this pins it.
    """
    gold = ["refunds take five business days"]
    hit_at_11 = ["filler"] * 10 + ["refunds take five business days"]
    assert mc.official_metrics_one(hit_at_11, gold)["Hits@10"] == 0.0
    hit_at_10 = ["filler"] * 9 + ["refunds take five business days"]
    assert mc.official_metrics_one(hit_at_10, gold)["Hits@10"] == 1.0


def test_official_metric_normalizes_average_precision_by_the_true_gold_count():
    """MAP divides by the query's gold facts, not the ones the list happened to match."""
    # Two gold facts, only one ever retrieved: AP 1/1 over min(2, 10).
    one = mc.official_metrics_one(["alpha beta"], ["alpha", "gamma"])
    assert one["MAP@10"] == pytest.approx(0.5)
    both = mc.official_metrics_one(
        ["alpha", "gamma"], ["alpha", "gamma"]
    )
    assert both["MAP@10"] == pytest.approx((1 + 1 / 2) / 2)


# ── atom weighting and beam search (estimator improvements) ─────────────

def test_idf_atom_weight_ranks_a_rare_atom_above_a_common_one():
    """A specific fact uses rare terms; a generic sentence uses common ones."""
    texts = [
        "refunds are processed within five business days",   # rare-ish
        "the company said this about the thing",             # generic
        "the company said that about the thing",             # generic
        "the company said more about the thing",             # generic
    ]
    weights = mc.idf_atom_weight(texts, reduction="mean")
    assert weights[0] > weights[1]
    assert weights[1] == pytest.approx(weights[2]) == pytest.approx(weights[3])


def test_idf_atom_weight_counts_distinct_terms_only():
    """Repeating a word must not raise an atom's weight; adding a rare one must."""
    repeated = ["alpha beta"] * 3 + ["alpha beta alpha beta alpha beta"]
    mean = mc.idf_atom_weight(repeated, reduction="mean")
    total = mc.idf_atom_weight(repeated, reduction="sum")
    assert mean[3] == pytest.approx(mean[0])
    assert total[3] == pytest.approx(total[0])

    # Two extra rare terms: 'sum' grows with the number of distinct terms, 'mean' need not.
    extended = ["alpha beta"] * 3 + ["alpha beta gamma delta"]
    assert mc.idf_atom_weight(extended, reduction="sum")[3] > \
        mc.idf_atom_weight(extended, reduction="sum")[0]
    assert len(mc.idf_atom_weight(extended, reduction="mean")) == 4


def test_idf_atom_weight_handles_empty_input_and_empty_atoms():
    assert mc.idf_atom_weight([]) == []
    assert mc.idf_atom_weight(["", "  "]) == [0.0, 0.0]


def test_idf_atom_weight_rejects_an_unknown_reduction():
    with pytest.raises(ValueError, match="reduction"):
        mc.idf_atom_weight(["a b"], reduction="max")


def test_beam_search_returns_exactly_top_k_distinct_candidates():
    rel = np.array([0.9, 0.8, 0.7, 0.1])
    atoms = [[0], [1], [2], [3]]
    weight = [1.0, 1.0, 1.0, 1.0]
    out = mc.beam_facility_location_select(rel, atoms, weight, top_k=3, lam=0.7, beam_width=4)
    assert len(out) == 3
    assert len(set(out)) == 3
    assert all(0 <= i < 4 for i in out)


def test_beam_search_prefers_coverage_when_relevance_is_flat():
    """With no relevance signal the objective is pure coverage, so distinct atoms win."""
    rel = np.array([0.5, 0.5, 0.5])
    atoms = [[0, 1], [0, 1], [2]]          # candidate 2 is the only new fact
    weight = [1.0, 1.0, 1.0]
    out = mc.beam_facility_location_select(rel, atoms, weight, top_k=2, lam=0.0, beam_width=4)
    assert 2 in out


def test_beam_search_honours_relevance_when_lambda_is_one():
    rel = np.array([0.1, 0.9, 0.5])
    atoms = [[0], [1], [2]]
    weight = [1.0, 1.0, 1.0]
    out = mc.beam_facility_location_select(rel, atoms, weight, top_k=1, lam=1.0, beam_width=4)
    assert out == [1]


def test_beam_search_handles_degenerate_input():
    assert mc.beam_facility_location_select(np.array([]), [], [], top_k=3) == []
    assert mc.beam_facility_location_select(np.array([0.5]), [[0]], [1.0], top_k=0) == []
    # A pool smaller than top_k still returns every candidate exactly once.
    out = mc.beam_facility_location_select(np.array([0.5, 0.4]), [[0], [1]], [1.0, 1.0], top_k=5)
    assert sorted(out) == [0, 1]


def _greedy(scores, atoms, weight, top_k, lam):
    """The greedy coverage rule this module ships, with the redundancy term off."""
    n = len(scores)
    return mc.facility_location_select(
        np.asarray(scores, dtype=float), atoms, weight, np.zeros((n, n)),
        top_k, lam, use_redundancy=False,
    )


def test_coverage_beam_with_width_one_reproduces_greedy_exactly():
    """The control that makes the lookahead attributable: beam(1) == the greedy rule."""
    scores = [0.9, 0.8, 0.7, 0.2, 0.1]
    atoms = [[0], [1], [2], [0, 1], [3]]
    weight = [1.0, 1.0, 1.0, 2.0, 1.0]
    assert mc.coverage_beam_order(
        scores, atoms, weight, top_k=3, lam=0.5, beam_width=1
    ) == _greedy(scores, atoms, weight, 3, 0.5)


def test_coverage_beam_with_lookahead_keeps_alternatives():
    scores = [0.9, 0.8, 0.1, 0.1]
    atoms = [[0], [0, 1], [2], [2]]
    weight = [1.0, 1.0, 1.0, 1.0]
    beam = mc.coverage_beam_order(scores, atoms, weight, top_k=2, lam=0.5, beam_width=4)
    assert len(beam) == 2
    assert len(set(beam)) == 2
    # With lookahead the search may leave the greedy choice, but never returns an invalid set.
    assert all(0 <= i < len(scores) for i in beam)


def test_coverage_beam_handles_degenerate_input():
    assert mc.coverage_beam_order([], [], [], top_k=3) == []
    assert mc.coverage_beam_order([0.5], [[0]], [1.0], top_k=0) == []
    out = mc.coverage_beam_order([0.5, 0.4], [[0], [1]], [1.0, 1.0], top_k=5)
    assert sorted(out) == [0, 1]
