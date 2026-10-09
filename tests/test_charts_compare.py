"""Tests for the cross-system comparison charts.

The comparison figures are the ones a reader is most likely to quote, and they rest on two inputs
that can be wrong silently: the pinned published table (which came from a parsed web page) and the
membership rule (which decides whether a corpus outside the benchmark is allowed into an "of 18"
average). Both are pinned directly, along with the classification of the open-source survey and the
"a missing report is a gap, not a zero" behaviour of the loaders.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def _load():
    name = "laya_rag_charts_compare"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "charts_compare.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ch = _load()


def _plt_or_skip():
    try:
        return ch.cmp._plt_mod()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"matplotlib unavailable: {exc}")


# -- the pinned published table --------------------------------------------


def test_self_check_passes():
    assert ch.self_check() == []


def test_check_mode_exits_zero(capsys):
    assert ch.main(["--check"]) == 0
    assert "invariants hold" in capsys.readouterr().out


def test_the_published_table_is_internally_consistent():
    """The paper's own average must be the mean of its own 18 rows, or the table was mis-parsed."""
    datasets = [name for name in ch.published.NDCG10 if name != "Avg."]
    assert len(datasets) == 18
    for system in ch.published.SYSTEMS:
        values = [ch.published.NDCG10[name][system] for name in datasets]
        assert all(0.0 <= value <= 1.0 for value in values), system
        assert sum(values) / len(values) == pytest.approx(ch.published.AVERAGE[system], abs=0.002)


def test_every_published_dataset_maps_to_a_canonical_member():
    mapped = set(ch.published.CANONICAL.values())
    assert mapped == set(ch.BEIR_ORDER)
    assert len(ch.BEIR_ORDER) == 18
    assert len(set(ch.BEIR_ORDER)) == 18


def test_a_convention_difference_is_not_charted_as_agreement():
    """Touché's published row is the v2 release; the repository's anchor is v1 (0.442)."""
    source = next(key for key, canon in ch.published.CANONICAL.items() if canon == "webis-touche2020")
    assert ch.published.NDCG10[source]["BM25"] != ch.cmp.PUBLISHED_BM25["webis-touche2020"]


# -- membership ------------------------------------------------------------


def test_coverage_classifies_members_by_how_far_they_got():
    states = ch.coverage_states(
        measured=["scifact", "fever"], extras=["msmarco"], dense=["scifact"]
    )
    kinds = {row["dataset"]: row["kind"] for row in states}
    assert kinds["scifact"] == "measured-dense"
    assert kinds["fever"] == "measured-lexical"
    assert kinds["robust04"] == "blocked" and kinds["bioasq"] == "blocked"
    assert kinds["cqadupstack"] == "needs-adapter"
    assert kinds["msmarco"] == "extra"
    assert len(states) == 19


def test_our_macro_never_averages_a_non_member():
    """MS MARCO is measured but is not one of the 18; it must not move the macro."""
    rows = {
        "scifact": {"best": ("x", 0.70)},
        "fever": {"best": ("y", 0.60)},
        "msmarco": {"best": ("z", 0.10)},
    }
    assert ch.our_macro(rows, ["scifact", "fever"]) == {"mean": 0.65, "n": 2}
    assert ch.our_macro(rows, ["scifact"]) == {"mean": 0.7, "n": 1}
    # And a member with no report is skipped rather than counted as zero.
    assert ch.our_macro(rows, ["scifact", "nq"]) == {"mean": 0.7, "n": 1}


def test_macro_reports_its_own_count():
    assert ch.macro([0.2, 0.4]) == {"mean": 0.3, "n": 2}
    assert ch.macro([]) == {"mean": 0.0, "n": 0}


# -- loaders ---------------------------------------------------------------


def test_our_rows_reads_reports_and_skips_superseded(tmp_path, monkeypatch):
    monkeypatch.setattr(ch, "BEIR_DIR", tmp_path)
    monkeypatch.setattr(ch.cc, "BEIR", tmp_path)
    good = {
        "datasets": {
            "scifact": {"retrievers": {
                "bm25-engine": {"nDCG@10": 0.6762},
                "biencoder": {"nDCG@10": 0.7196},
            }},
        }
    }
    (tmp_path / "five-small-fixed.json").write_text(json.dumps(good))
    # A superseded *report* keeps its name; the quarantine is a separate marker file, which has
    # no `datasets` key at all and must be ignored on that ground rather than by its filename.
    (tmp_path / "three-base.json").write_text(json.dumps(
        {"datasets": {"scifact": {"retrievers": {"bm25-engine": {"nDCG@10": 0.999}}}}}
    ))
    (tmp_path / "three-base.superseded.json").write_text(json.dumps(
        {"superseded_by": "large-splade-fixed", "reason": "pre-fix", "note": "generated"}
    ))
    rows = ch.our_rows(tmp_path)
    assert rows["scifact"]["best"] == ("biencoder", 0.7196)
    assert rows["scifact"]["lexical"] == ("bm25-engine", 0.6762)
    assert rows["scifact"]["has_dense"] is True


def test_a_seeded_subset_never_beats_a_full_run(tmp_path, monkeypatch):
    """The HotpotQA estimate reads higher than its full run: the maximum is not the result."""
    monkeypatch.setattr(ch, "BEIR_DIR", tmp_path)
    (tmp_path / "hotpotqa-subset.json").write_text(json.dumps({
        "datasets": {"hotpotqa": {"queries": 97852, "scored_queries": 600,
                                  "retrievers": {"bm25-engine": {"nDCG@10": 0.6365}}}}
    }))
    (tmp_path / "hotpotqa.json").write_text(json.dumps({
        "datasets": {"hotpotqa": {"queries": 97852, "scored_queries": 7405,
                                  "retrievers": {"bm25-engine": {"nDCG@10": 0.6352}}}}
    }))
    row = ch.our_rows(tmp_path)["hotpotqa"]
    assert row["best"] == ("bm25-engine", 0.6352)
    assert row["from_subset"] is False
    assert row["reports"] == ["hotpotqa"]      # the estimate is not a contributing report


def test_a_subset_is_used_only_when_no_full_run_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(ch, "BEIR_DIR", tmp_path)
    (tmp_path / "nq-subset.json").write_text(json.dumps({
        "datasets": {"nq": {"queries": 3452, "scored_queries": 400,
                            "retrievers": {"bm25-engine": {"nDCG@10": 0.40}}}}
    }))
    row = ch.our_rows(tmp_path)["nq"]
    assert row["best"] == ("bm25-engine", 0.40) and row["from_subset"] is True


def test_an_unnamed_lower_scoring_run_is_still_treated_as_a_subset(tmp_path, monkeypatch):
    """A run that scored fewer queries than another report for the same dataset is an estimate."""
    monkeypatch.setattr(ch, "BEIR_DIR", tmp_path)
    (tmp_path / "hotpotqa.json").write_text(json.dumps({
        "datasets": {"hotpotqa": {"scored_queries": 7405,
                                  "retrievers": {"bm25-engine": {"nDCG@10": 0.6352}}}}
    }))
    (tmp_path / "hotpotqa-seeded.json").write_text(json.dumps({
        "datasets": {"hotpotqa": {"scored_queries": 600,
                                  "retrievers": {"bm25-engine": {"nDCG@10": 0.70}}}}
    }))
    row = ch.our_rows(tmp_path)["hotpotqa"]
    assert row["best"] == ("bm25-engine", 0.6352), "a 600-query estimate won over the full run"
    assert row["reports"] == ["hotpotqa"]


def test_ce_deltas_reads_the_beir_report_not_the_multihop_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(ch, "BEIR_DIR", tmp_path)
    monkeypatch.setattr(ch, "MH_DIR", tmp_path / "absent")
    (tmp_path / "stage2-rerank.json").write_text(json.dumps({
        "datasets": {"arguana": {"retrievers": {
            "hybrid-biencoder": {"nDCG@10": 0.5679},
            "crossencoder": {"nDCG@10": 0.2497},
        }}}
    }))
    rows = ch.ce_deltas()
    assert len(rows) == 1
    assert rows[0]["corpus"] == "arguana"
    assert rows[0]["delta"] == pytest.approx(-0.3182)


def test_lexical_gap_pairs_our_control_with_the_published_row():
    rows = {"scifact": {"lexical": ("bm25-engine", 0.6762)}}
    gaps = ch.lexical_gap(rows)
    scifact = next(row for row in gaps if row["dataset"] == "scifact")
    assert scifact["published"] == ch.published.NDCG10["SciFact"]["BM25"] == 0.665
    assert scifact["gap"] == pytest.approx(0.0112, abs=1e-4)
    # A dataset with no report is absent from the comparison, not charted at zero.
    assert all(row["dataset"] != "robust04" for row in gaps)


def test_multihop_gaps_subtract_per_metric():
    official = {"hybrid (ours)@ov200": {"Hits@10": 0.80, "Hits@4": 0.60,
                                        "MAP@10": 0.25, "MRR@10": 0.50}}
    gaps = ch.multihop_gaps(official, [("published", 0.70, 0.60, 0.40, 0.45)])
    assert gaps[0]["gaps"] == {"Hits@10": 0.1, "Hits@4": 0.0, "MAP@10": -0.15, "MRR@10": 0.05}


def test_bright_rows_reads_both_reports_and_reports_the_macro(tmp_path, monkeypatch):
    def report(subset, documents, queries, bm25, dense, hybrid):
        return {subset: {"documents": documents, "queries": queries, "summary": {
            "bm25": {"nDCG@10": bm25}, "dense": {"nDCG@10": dense}, "hybrid": {"nDCG@10": hybrid},
        }}}

    first, second = tmp_path / "report.json", tmp_path / "report-small.json"
    first.write_text(json.dumps({"subsets": report("pony", 7894, 112, 0.1221, 0.0398, 0.1281)}))
    second.write_text(json.dumps({"subsets": report("biology", 57359, 103, 0.2089, 0.1076, 0.1789)}))
    monkeypatch.setattr(ch, "BRIGHT_REPORTS", (str(first), str(second)))
    # The adapted-encoder arm would otherwise be merged in from the working tree; this test is
    # about the two pinned reports only.
    monkeypatch.setattr(ch, "BRIGHT_ADAPTED_REPORT", "eval/bright/absent-for-this-test.json")
    rows = ch.bright_rows()
    assert [row["subset"] for row in rows] == ["pony", "biology"]
    assert rows[0]["arms"] == {"bm25": 0.1221, "dense": 0.0398, "hybrid": 0.1281}
    macro = sum(row["arms"]["hybrid"] for row in rows) / len(rows)
    assert macro == pytest.approx(0.1535, abs=1e-4)


def test_bright_rows_merges_the_adapted_arm_beside_the_bge_one(tmp_path, monkeypatch):
    """The adapted encoder is drawn as an extra arm, never as a replacement for the bge one."""
    base = tmp_path / "report.json"
    base.write_text(json.dumps({"subsets": {"pony": {
        "documents": 7894, "queries": 112,
        "summary": {"bm25": {"nDCG@10": 0.1221}, "dense": {"nDCG@10": 0.0398},
                    "hybrid": {"nDCG@10": 0.1281}},
    }}}))
    adapted = tmp_path / "report-laya.json"
    adapted.write_text(json.dumps({"subsets": {"pony": {
        "summary": {"bm25": {"nDCG@10": 0.1221}, "dense": {"nDCG@10": 0.0196},
                    "hybrid": {"nDCG@10": 0.0715}},
    }}}))
    monkeypatch.setattr(ch, "BRIGHT_REPORTS", (str(base),))
    # An absolute path is used verbatim by the `ROOT / name` join.
    monkeypatch.setattr(ch, "BRIGHT_ADAPTED_REPORT", str(adapted))
    rows = ch.bright_rows()
    assert rows[0]["arms"]["dense"] == 0.0398
    assert rows[0]["arms"]["dense (Laya adapted)"] == 0.0196


def test_bright_skips_a_missing_report_rather_than_inventing_a_subset(tmp_path, monkeypatch):
    monkeypatch.setattr(ch, "BRIGHT_REPORTS", (str(tmp_path / "absent.json"),))
    assert ch.bright_rows() == []


def test_the_bright_chart_writes_a_figure(tmp_path, monkeypatch):
    plt = _plt_or_skip()
    source = tmp_path / "report.json"
    source.write_text(json.dumps({"subsets": {"pony": {
        "documents": 7894, "queries": 112,
        "summary": {"bm25": {"nDCG@10": 0.12}, "dense": {"nDCG@10": 0.04},
                    "hybrid": {"nDCG@10": 0.13}},
    }}}))
    monkeypatch.setattr(ch, "BRIGHT_REPORTS", (str(source),))
    assert ch.chart_bright(tmp_path, plt) is True
    assert (tmp_path / "cmp_bright.png").is_file()


# -- the open-source survey ------------------------------------------------


def test_oss_rows_drops_incomplete_fetches_and_classifies_by_name(tmp_path):
    payload = [
        {"full_name": "None", "stars": "None", "pushed_at": "None"},
        {"full_name": "qdrant/qdrant", "stars": 34942, "pushed_at": "2026-10-06T06:50:23Z",
         "language": "Rust", "archived": False},
        {"full_name": "bizreach-inc/light-splade", "stars": 16, "pushed_at": "2025-01-01T00:00:00Z",
         "language": "Python", "archived": False},
    ]
    path = tmp_path / "repo-metadata.json"
    path.write_text(json.dumps(payload))
    rows = ch.oss_rows(path, now=datetime(2026, 10, 8, tzinfo=timezone.utc))
    assert [row["name"] for row in rows] == ["qdrant/qdrant", "bizreach-inc/light-splade"]
    assert rows[0]["category"] == "search engine / vector DB"
    # The case-insensitive lookup is what makes the metadata's spelling usable as a key.
    assert rows[1]["category"] == "learned retrieval"
    assert rows[0]["days_since_push"] == 1


def test_every_surveyed_project_has_a_category():
    """No silent `other`: a new repository in the survey must be classified deliberately."""
    if not ch.OSS_METADATA.is_file():
        pytest.skip("the open-source survey is not present")
    rows = ch.oss_rows()
    unclassified = [row["name"] for row in rows if row["category"] == "other"]
    assert unclassified == [], f"uncategorised in the survey: {unclassified}"


def test_category_totals_roll_up_counts_and_stars():
    totals = ch.category_totals([
        {"category": "graph RAG", "stars": 10},
        {"category": "graph RAG", "stars": 5},
        {"category": "lexical retrieval", "stars": 1},
    ])
    assert totals[0] == {"category": "graph RAG", "projects": 2, "stars": 15}
    assert totals[1]["category"] == "lexical retrieval"


# -- rendering -------------------------------------------------------------


def test_the_matrix_chart_writes_a_figure_with_gaps_for_unmeasured_datasets(tmp_path, monkeypatch):
    plt = _plt_or_skip()
    monkeypatch.setattr(ch, "BEIR_DIR", tmp_path)
    (tmp_path / "five-small-fixed.json").write_text(json.dumps({
        "datasets": {"scifact": {"retrievers": {"biencoder": {"nDCG@10": 0.7196}}}}
    }))
    assert ch.chart_beir_matrix(tmp_path, plt) is True
    written = tmp_path / "cmp_beir18_matrix.png"
    assert written.is_file() and written.stat().st_size > 10_000


def test_the_coverage_chart_writes_a_figure(tmp_path, monkeypatch):
    plt = _plt_or_skip()
    monkeypatch.setattr(ch.cc, "BEIR", tmp_path)
    (tmp_path / "five-small-fixed.json").write_text(json.dumps({
        "datasets": {"scifact": {"retrievers": {"biencoder": {"nDCG@10": 0.7196}}}}
    }))
    assert ch.chart_coverage(tmp_path, plt) is True
    assert (tmp_path / "cmp_coverage.png").is_file()


def test_charts_skip_rather_than_fail_when_a_source_is_absent(tmp_path, monkeypatch):
    plt = _plt_or_skip()
    monkeypatch.setattr(ch, "BEIR_DIR", tmp_path / "absent")
    monkeypatch.setattr(ch, "MH_DIR", tmp_path / "absent")
    monkeypatch.setattr(ch, "OSS_METADATA", tmp_path / "absent.json")
    assert ch.chart_multihop_delta(tmp_path, plt) is False
    assert ch.chart_reranker(tmp_path, plt) is False
    assert ch.chart_oss_landscape(tmp_path, plt) is False
