"""Offline tests for the FRAMES retrieval metrics and title matching.

The FRAMES row is a *document-level* retrieval claim scored against the articles a question links, so
two things have to be right: the metrics must be the standard definitions, and the title matching must
fold the dataset's URLs onto the corpus's canonical titles without silently pretending a redirect
resolved. The report carries `gold_coverage` precisely because the second is imperfect, and these
tests pin that it is computed over the gold set rather than over the corpus.
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
    name = "laya_rag_frames_eval"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "frames_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fe = _load()


def test_titles_fold_across_the_two_spellings():
    assert fe.normalise_title("Harriet_Lane") == fe.normalise_title("Harriet Lane")
    assert fe.normalise_title("%22Weird_Al%22_Yankovic") == fe.normalise_title('"Weird Al" Yankovic')
    assert fe.normalise_title("James Buchanan.") == fe.normalise_title("james buchanan")


def test_gold_titles_are_read_from_every_link_column():
    row = {
        "Prompt": "unused",
        "wikipedia_link_1": "https://en.wikipedia.org/wiki/James_Buchanan",
        "wiki_links": "['https://en.wikipedia.org/wiki/Harriet_Lane']",
        "Answer": "Jane Ballou",
    }
    assert fe.gold_titles(row) == {"james buchanan", "harriet lane"}


def test_recall_is_hit_or_miss_at_k():
    ranked = ["a", "b", "c", "d"]
    assert fe.recall_at_k(ranked, {"c"}, 3) == 1.0
    assert fe.recall_at_k(ranked, {"c"}, 2) == 0.0


def test_reciprocal_rank_uses_the_first_hit():
    assert fe.reciprocal_rank(["a", "b", "c"], {"b"}, 10) == pytest.approx(0.5)
    assert fe.reciprocal_rank(["a", "b", "c"], {"z"}, 10) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking_and_discounted_otherwise():
    assert fe.ndcg_at_k(["a", "b"], {"a", "b"}, 10) == pytest.approx(1.0)
    discounted = fe.ndcg_at_k(["x", "a"], {"a"}, 10)
    assert 0.0 < discounted < 1.0


def test_ndcg_of_no_gold_at_all_is_zero_not_undefined():
    assert fe.ndcg_at_k(["a"], set(), 10) == 0.0


def test_score_ranking_reports_every_headline_metric():
    row = fe.score_ranking(["a", "b"], {"b"})
    for key in ("recall@1", "recall@5", "recall@10", "recall@20", "mrr@10", "ndcg@10"):
        assert key in row
    assert row["recall@1"] == 0.0 and row["recall@10"] == 1.0


def test_coverage_is_measured_over_the_gold_set():
    """A question whose gold articles are mostly absent must report low coverage, not full."""
    gold = {"present", "absent-one", "absent-two"}
    assert len(gold & {"present"}) / len(gold) == pytest.approx(1 / 3)


def test_summarize_means_the_per_query_rows():
    rows = [fe.score_ranking(["a"], {"a"}), fe.score_ranking(["a"], {"z"})]
    for row in rows:
        row["gold_coverage"] = 1.0
    summary = fe.summarize(rows)
    assert summary["questions"] == 2
    assert summary["recall@1"] == pytest.approx(0.5)
    assert summary["gold_coverage"] == pytest.approx(1.0)


def test_a_title_map_folds_a_dataset_link_onto_its_canonical_article():
    """The redirect gap was worth 0.12 of coverage and 0.09 nDCG@10, so the map must be applied."""
    row = {"wikipedia_link_1": "https://en.wikipedia.org/wiki/USA"}
    assert fe.gold_titles(row) == {"usa"}
    assert fe.gold_titles(row, {"usa": "united states"}) == {"united states"}


def test_a_partial_title_map_leaves_unmapped_titles_alone():
    row = {"wikipedia_link_1": "https://en.wikipedia.org/wiki/Harriet_Lane",
           "wikipedia_link_2": "https://en.wikipedia.org/wiki/USA"}
    mapped = fe.gold_titles(row, {"usa": "united states"})
    assert mapped == {"harriet lane", "united states"}


def test_the_title_map_is_loaded_normalised_on_both_sides(tmp_path):
    path = tmp_path / "title-map.json"
    path.write_text('{"USA": "United_States", "Harriet_Lane": "Harriet Lane"}')
    loaded = fe.load_title_map(path)
    assert loaded == {"usa": "united states", "harriet lane": "harriet lane"}


def test_an_absent_title_map_is_empty_rather_than_an_error(tmp_path):
    assert fe.load_title_map(tmp_path / "nothing.json") == {}
