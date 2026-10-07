"""Offline tests for the CQADupStack adapter.

CQADupStack is the one BEIR member that is not one corpus: the release ships twelve independent
sub-collections and the benchmark's number for the member is the **macro-average of the twelve**.
`tools/beir.py` models one corpus per dataset, so the member has its own tool, and these tests pin
the two things that would silently produce a wrong number: the averaging, and the query-id namespacing
that keeps a paired bootstrap from pairing a `tex` query with a `physics` query of the same integer
id.
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


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _cqadupstack():
    return _load("laya_rag_cqadupstack", "cqadupstack.py")


def _forum(documents: int, n_dcg: float, per_query: dict[str, float]) -> dict:
    return {
        "documents": documents,
        "queries": len(per_query),
        "scored_queries": len(per_query),
        "retrievers": {
            "bm25": {"nDCG@10": n_dcg, "MAP@100": n_dcg / 2, "seconds": 1.0,
                     "scored_queries": len(per_query)},
            "bm25-engine": {"nDCG@10": n_dcg + 0.1, "MAP@100": n_dcg / 2 + 0.05, "seconds": 2.0,
                            "scored_queries": len(per_query)},
        },
        "per_query_nDCG@10": {
            "bm25": dict(per_query),
            "bm25-engine": {q: v + 0.1 for q, v in per_query.items()},
        },
    }


def test_the_release_has_exactly_twelve_subforums():
    cqa = _cqadupstack()
    assert len(cqa.SUBFORUMS) == 12
    assert len(set(cqa.SUBFORUMS)) == 12


def test_aggregate_macro_averages_the_subforums():
    """The member's score is the mean of the twelve sub-forum scores, not a pooled run."""
    cqa = _cqadupstack()
    per_forum = {
        "android": _forum(100, 0.20, {"1": 0.2, "2": 0.2}),
        "tex": _forum(300, 0.60, {"1": 0.6, "2": 0.6}),
    }
    agg = cqa.aggregate(per_forum)

    assert agg["documents"] == 400
    assert agg["scored_queries"] == 4
    assert agg["retrievers"]["bm25"]["nDCG@10"] == pytest.approx(0.40)
    assert agg["retrievers"]["bm25-engine"]["nDCG@10"] == pytest.approx(0.50)
    # Seconds sum; they are the cost of the twelve runs, not an average of them.
    assert agg["retrievers"]["bm25"]["seconds"] == pytest.approx(2.0)


def test_query_ids_are_namespaced_by_forum():
    """Query ids are per-forum integers and collide: `19399` exists in more than one forum."""
    cqa = _cqadupstack()
    per_forum = {
        "english": _forum(10, 0.1, {"19399": 0.1}),
        "tex": _forum(10, 0.9, {"19399": 0.9}),
    }
    pooled = cqa.aggregate(per_forum)["per_query_nDCG@10"]["bm25"]

    assert set(pooled) == {"english:19399", "tex:19399"}, "an unprefixed id would collide silently"
    assert pooled["tex:19399"] == pytest.approx(0.9)


def test_aggregate_carries_the_per_forum_detail():
    cqa = _cqadupstack()
    agg = cqa.aggregate({"gis": _forum(7, 0.5, {"1": 0.5})})
    assert set(agg["subforums"]) == {"gis"}
    assert agg["subforums"]["gis"]["documents"] == 7
    assert agg["subforums"]["gis"]["retrievers"]["bm25-engine"]["nDCG@10"] == pytest.approx(0.6)


def test_missing_subforums_fail_loudly_without_download(tmp_path):
    cqa = _cqadupstack()
    with pytest.raises(FileNotFoundError, match="--no-download"):
        cqa.ensure_subforums(tmp_path, download=False)


def test_the_one_corpus_harness_refuses_cqadupstack_with_a_pointer():
    """`beir.py` would otherwise extract 5.3 GB and *then* fail its dataset-level corpus check."""
    beir = _load("laya_rag_beir_guard", "beir.py")
    with pytest.raises(ValueError, match="cqadupstack.py"):
        beir.ensure_dataset("cqadupstack", ROOT / "eval" / "beir", download=False)
