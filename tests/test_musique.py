"""Offline tests for the MuSiQue adapter.

The replication's credibility rests on the mapping: a paragraph is a corpus item, the
`is_supporting` paragraphs are the gold set, and `document_of` must group paragraphs from
one article so the document-quota rule means the same thing it means on MultiHop-RAG. These
pin that mapping on synthetic records, with no dataset and no model.
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
    spec = importlib.util.spec_from_file_location("laya_rag_musique", ROOT / "tools" / "musique.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


musique = _load()


def _record(qid: str, question: str, paragraphs, hops: int = 2) -> dict:
    return {
        "id": qid,
        "question": question,
        "answer": "x",
        "paragraphs": paragraphs,
        "question_decomposition": [{"id": i} for i in range(hops)],
    }


def _paragraph(title: str, text: str, supporting: bool = False) -> dict:
    return {"idx": 0, "title": title, "paragraph_text": text, "is_supporting": supporting}


def _write(tmp_path: Path, records) -> Path:
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_slug_is_id_and_filesystem_safe():
    assert musique.slug("Grant's First Stand") == "grant-s-first-stand"
    assert musique.slug("") == "untitled"
    assert musique.slug("  A/B  ") == "a-b"


def test_load_split_maps_supporting_paragraphs_to_gold_indices(tmp_path):
    path = _write(tmp_path, [
        _record("q1", "who?", [
            _paragraph("Alpha", "alpha text", supporting=True),
            _paragraph("Beta", "beta text", supporting=True),
            _paragraph("Gamma", "gamma text"),
        ]),
    ])
    data = musique.load_split(path)
    assert list(data["queries"]) == ["q1"]
    assert len(data["golds"]["q1"]) == 2
    order = data["order"]
    gold_ids = [doc for doc, position in order.items() if position in set(data["golds"]["q1"])]
    assert sorted(gold_ids) == ["alpha#0", "beta#0"]
    assert data["hops"]["q1"] == 2


def test_load_split_deduplicates_a_paragraph_repeated_across_records(tmp_path):
    """The same passage in two candidates is one corpus item, not two slots."""
    shared = _paragraph("Shared", "the same text", supporting=True)
    path = _write(tmp_path, [
        _record("q1", "a", [shared, _paragraph("X", "x text")]),
        _record("q2", "b", [dict(shared), _paragraph("Y", "y text")]),
    ])
    data = musique.load_split(path)
    assert len(data["corpus"]) == 3, data["corpus"]
    assert sum(1 for doc in data["corpus"] if doc.startswith("shared")) == 1


def test_load_split_gives_document_of_one_article_to_share_a_prefix(tmp_path):
    path = _write(tmp_path, [
        _record("q1", "a", [
            _paragraph("Same Article", "first paragraph", supporting=True),
            _paragraph("Same Article", "second paragraph"),
        ]),
    ])
    data = musique.load_split(path)
    docs = [doc.rsplit("#", 1)[0] for doc in data["corpus"]]
    assert docs == ["same-article", "same-article"]


def test_load_split_skips_empty_paragraph_text(tmp_path):
    path = _write(tmp_path, [
        _record("q1", "a", [_paragraph("A", "real text", supporting=True), _paragraph("B", "   ")]),
    ])
    data = musique.load_split(path)
    assert len(data["corpus"]) == 1
    assert len(data["golds"]["q1"]) == 1


def test_load_split_honours_the_limit(tmp_path):
    path = _write(tmp_path, [
        _record(f"q{i}", "a", [_paragraph("A", f"text {i}", supporting=True)]) for i in range(5)
    ])
    assert len(musique.load_split(path, limit=3)["queries"]) == 3


def test_load_split_treats_a_missing_decomposition_as_one_hop(tmp_path):
    record = _record("q1", "a", [_paragraph("A", "t", supporting=True)])
    record["question_decomposition"] = []
    path = _write(tmp_path, [record])
    assert musique.load_split(path)["hops"]["q1"] == 1


def test_gold_matches_flags_only_supporting_candidates():
    order = {"a#0": 0, "b#0": 1, "c#0": 2}
    matches = musique.gold_matches(["c#0", "a#0", "b#0"], [0, 2], order)
    assert matches == [{2}, {0}, set()]


def test_gold_matches_is_empty_for_an_unknown_document():
    assert musique.gold_matches(["nope#9"], [0], {"a#0": 0}) == [set()]


def test_render_markdown_reports_every_rule_and_comparison():
    result = {
        "_settings": {"split": "validation", "queries": 2417, "corpus": 30000, "depth": 100,
                      "top_k": 10, "model": "bge-small"},
        "summary": {
            "fused": {"Hits@10": 0.6, "Hits@4": 0.4, "MAP@10": 0.3, "MRR@10": 0.45},
            "covquota-3-0.7": {"Hits@10": 0.65, "Hits@4": 0.45, "MAP@10": 0.34, "MRR@10": 0.5},
        },
        "comparisons": {
            "covquota-3-0.7 vs fused MAP@10": {
                "bootstrap": {"delta": 0.04, "ci_low": 0.01, "ci_high": 0.07, "significant": True},
                "permutation": {"p_value": 0.001},
            }
        },
    }
    text = musique.render_markdown(result)
    assert "MuSiQue" in text
    assert "covquota-3-0.7" in text
    assert "+0.0400" in text
