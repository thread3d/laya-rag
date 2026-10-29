"""A retrieval row may only be placed when it measured the member, not a subset.

The tool exists to stop a familiar over-claim: a report over 150 of 2,255 answerable queries looks
exactly like a report over all of them, and the benchmark's table scores the full set. That is failure
class 11 in this repository's ledger — *a subset read as the member* — and the guard is a refusal, not a
footnote.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import multihop_row as mr  # noqa: E402


def _report(queries: int) -> dict:
    return {
        "summary": {
            "queries": queries,
            "Hits@10": 0.94, "Hits@4": 0.847, "MAP@10": 0.371, "MRR@10": 0.688,
            "citation_clean_all": 1.0, "answer_nugget_recall": 0.2348,
        }
    }


def test_a_report_without_the_four_columns_is_refused():
    with pytest.raises(SystemExit):
        mr.summarise({"summary": {"queries": 10, "Hits@10": 0.5}})


def test_a_subset_is_not_placeable_and_says_how_many_queries_it_saw():
    row = mr.summarise(_report(150))
    problems = mr.placement_problems(row)
    assert problems and "150 queries" in problems[0]
    assert "subset" in problems[0]


def test_the_full_set_is_placeable():
    assert mr.placement_problems(mr.summarise(_report(mr.FULL_SET))) == []


def test_a_subset_render_refuses_first_and_still_shows_its_numbers():
    rendered = mr.render(mr.summarise(_report(150)), label="ours: test arm")
    assert rendered.startswith("**not placeable**")
    assert "engine_extractive.py" in rendered      # the command that would make it placeable
    assert "**ours: test arm**" in rendered        # but the numbers are shown, clearly marked as a subset


def test_a_full_set_render_does_not_refuse():
    rendered = mr.render(mr.summarise(_report(mr.FULL_SET)), label="ours: test arm")
    assert "not placeable" not in rendered
    assert "2255" in rendered


def test_the_published_anchors_travel_with_their_own_caveats():
    """A sorted column would imply the protocols match; they do not, and the page says so."""
    rendered = mr.render(mr.summarise(_report(mr.FULL_SET)), label="ours")
    assert "voyage-02" in rendered
    assert "Chunking differs" in rendered
    assert "no lexical baseline" in rendered
