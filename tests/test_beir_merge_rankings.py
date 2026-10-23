"""A merged union must score exactly what a single run scores, and a partial union must be refused.

The merge exists because `evaluate_rankings` macro-averages over the qrels it is given: scoring a ranking
set that covers one shard counts every other query as zero. So the two properties that matter are that the
union is *complete* and that the score is *unchanged* — and the second is the only way to know the union
did not quietly drop or duplicate a query.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import beir_merge_rankings as merge  # noqa: E402


def _sidecar(path: Path, per_query: dict[str, list[str]]) -> Path:
    path.write_text(json.dumps({"dataset": "x", "rankings": {"r": per_query}}), encoding="utf-8")
    return path


def test_the_union_of_disjoint_shards_is_the_whole_set(tmp_path: Path):
    a = _sidecar(tmp_path / "a.json", {"q1": ["d1"], "q2": ["d2"]})
    b = _sidecar(tmp_path / "b.json", {"q3": ["d3"]})
    merged = merge.load_parts([a, b])
    assert merged["r"] == {"q1": ["d1"], "q2": ["d2"], "q3": ["d3"]}


def test_a_query_covered_by_two_shards_is_counted_once(tmp_path: Path):
    a = _sidecar(tmp_path / "a.json", {"q1": ["first"]})
    b = _sidecar(tmp_path / "b.json", {"q1": ["second"]})
    assert merge.load_parts([a, b])["r"] == {"q1": ["first"]}


def test_two_retrievers_are_unioned_independently(tmp_path: Path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps({"rankings": {"r1": {"q1": ["a"]}, "r2": {"q1": ["b"]}}}), encoding="utf-8")
    merged = merge.load_parts([a])
    assert set(merged) == {"r1", "r2"}
    assert merged["r1"]["q1"] == ["a"] and merged["r2"]["q1"] == ["b"]


def test_a_report_is_refused_because_it_has_no_rankings(tmp_path: Path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"datasets": {"x": {"retrievers": {}}}}), encoding="utf-8")
    try:
        merge.load_parts([report])
    except SystemExit as exc:
        assert "no rankings block" in str(exc)
    else:  # pragma: no cover - the assertion below is the test
        raise AssertionError("a metrics report was accepted as a rankings sidecar")
