"""Tests for the per-corpus lexical profile.

The claim `SCORE-LIFT.md` §P2 pre-registers is falsifiable in a specific way: *the selector picks
the same settings on every corpus*. These tests pin the two properties that make the tool able to
answer it — a configuration's value is the modal one across the current reports, and a
deliberately different corpus (titles stripped) does not contribute one.
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


def _load():
    name = "laya_rag_lexical_selector"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "lexical_selector.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ls = _load()


def _report(directory: Path, stem: str, rows: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.json").write_text(
        json.dumps({"datasets": {"scifact": {"retrievers": rows}}})
    )


def _row(value: float) -> dict:
    return {"nDCG@10": value, "scored_queries": 300}


def test_collect_takes_the_modal_value_across_reports(tmp_path):
    _report(tmp_path, "a", {"bm25-engine": _row(0.6762)})
    _report(tmp_path, "b", {"bm25-engine": _row(0.6762)})
    _report(tmp_path, "c", {"bm25-engine": _row(0.6700)})
    collected = ls.collect(tmp_path)
    assert collected["scifact"]["bm25-engine"]["value"] == 0.6762
    assert collected["scifact"]["bm25-engine"]["variants"] == [0.67]


def test_a_deliberately_different_corpus_does_not_contribute(tmp_path):
    """`arguana-notitles` strips ArguAna's titles; its row is an experiment, not the config."""
    _report(tmp_path, "arguana-notitles-fixed", {"bm25-engine": _row(0.3809)})
    _report(tmp_path, "five-small-fixed", {"bm25-engine": _row(0.4169)})
    collected = ls.collect(tmp_path)
    assert collected["scifact"]["bm25-engine"]["value"] == 0.4169
    assert collected["scifact"]["bm25-engine"]["variants"] == []


def test_a_named_subset_and_a_superseded_report_are_skipped(tmp_path):
    _report(tmp_path, "hotpotqa-subset", {"bm25-engine": _row(0.6365)})
    _report(tmp_path, "report-stem", {"bm25-engine": _row(0.9999)})   # superseded
    _report(tmp_path, "five-small-fixed", {"bm25-engine": _row(0.6762)})
    assert ls.collect(tmp_path)["scifact"]["bm25-engine"]["value"] == 0.6762


def test_the_profile_reports_the_picks_and_both_macros(tmp_path):
    _report(tmp_path, "a", {"bm25-engine": _row(0.30), "bm25-title": _row(0.40)})
    result = ls.profile(ls.collect(tmp_path))
    assert result["picks"] == {"bm25-title": 1}
    assert result["shipped_macro"] == pytest.approx(0.30)
    assert result["best_macro"] == pytest.approx(0.40)
    assert result["gain"] == pytest.approx(0.10)


def test_a_tie_goes_to_the_shipped_configuration(tmp_path):
    """The shipped settings must not lose a tie to a variant that was never shown to be better."""
    _report(tmp_path, "a", {"bm25-engine": _row(0.44), "bm25-title": _row(0.44)})
    assert ls.profile(ls.collect(tmp_path))["picks"] == {"bm25-engine": 1}


def test_no_configuration_disagrees_with_itself_on_the_committed_reports():
    """A variant appearing once is an experiment; two values with equal support is drift."""
    collected = ls.collect(ls.BEIR_DIR)
    if not collected:
        pytest.skip("no eval/beir reports to profile")
    divergent = {
        member: {config: row["variants"] for config, row in configs.items() if row["variants"]}
        for member, configs in collected.items()
    }
    assert not any(divergent.values()), {k: v for k, v in divergent.items() if v}


def test_the_selector_is_not_degenerate_on_the_committed_reports():
    """§P2's falsifier: if every member picked the same settings there would be nothing to choose."""
    collected = ls.collect(ls.BEIR_DIR)
    if len(collected) < 18:
        pytest.skip("all eighteen members are not measured here")
    result = ls.profile(collected)
    assert len(result["picks"]) > 1, result["picks"]
    assert result["gain"] > 0
