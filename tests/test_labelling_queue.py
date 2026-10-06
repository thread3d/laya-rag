"""Tests for the hand-labelling harness.

The harness never writes a label; these pin that the queue starts unlabelled, that the proposal is
never promoted into the label, and that the score reports the metric the roadmap owes (no-coverage
recall) as well as agreement.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    name = "laya_rag_labelling_queue"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "labelling_queue.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lq = _load()


def test_load_records_reads_a_json_list(tmp_path):
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps([{"id": "a"}, {"id": "b"}]))
    assert [r["id"] for r in lq.load_records(path)] == ["a", "b"]


def test_load_records_reads_jsonl(tmp_path):
    path = tmp_path / "inputs.jsonl"
    path.write_text('{"id": "a"}\n\n{"id": "b"}\n')
    assert [r["id"] for r in lq.load_records(path)] == ["a", "b"]


def test_build_queue_leaves_every_item_unlabelled():
    queue = lq.build_queue([{"id": "x", "claim": "c", "proposed_label": "supported"}])
    item = queue[0]
    assert item["label"] is None
    assert item["labelled_by"] is None
    assert item["status"] == "unlabelled"
    assert item["proposed_label"] == "supported"   # kept, but never promoted
    assert item["claim"] == "c"


def test_build_queue_assigns_missing_ids():
    queue = lq.build_queue([{}, {}])
    assert [item["id"] for item in queue] == ["item-0", "item-1"]


def test_score_reports_agreement_and_no_coverage_recall():
    records = [
        {"id": "a", "label": "no_coverage", "proposed_label": "no_coverage"},
        {"id": "b", "label": "no_coverage", "proposed_label": "supported"},
        {"id": "c", "label": "supported", "proposed_label": "supported"},
    ]
    result = lq.score(records)
    assert result["labelled"] == 3
    assert result["agreement"] == 0.6667
    assert result["no_coverage_items"] == 2
    assert result["no_coverage_recall"] == 0.5
    assert result["label_counts"] == {"no_coverage": 2, "supported": 1}


def test_score_with_no_labels_does_not_invent_any():
    result = lq.score([{"id": "a", "label": None, "proposed_label": "supported"}])
    assert result["labelled"] == 0
    assert result["unlabelled"] == 1
    assert "agreement" not in result
    assert "note" in result


def test_reviewers_are_counted():
    records = [
        {"id": "a", "label": "supported", "proposed_label": "supported", "labelled_by": "alice"},
        {"id": "b", "label": "supported", "proposed_label": "supported", "labelled_by": "bob"},
        {"id": "c", "label": "supported", "proposed_label": "supported"},
    ]
    assert lq.score(records)["reviewers"] == {"alice": 1, "bob": 1, "unknown": 1}


def test_cli_build_then_score_round_trips(tmp_path):
    source = tmp_path / "pairs.json"
    source.write_text(json.dumps([{"id": "x", "proposed_label": "contradicted"}]))
    queue = tmp_path / "queue.jsonl"
    assert lq.main(["build", "--input", str(source), "--out", str(queue)]) == 0
    items = [json.loads(line) for line in queue.read_text().splitlines()]
    assert items[0]["label"] is None
    # Label it by hand, then score.
    items[0]["label"] = "contradicted"
    queue.write_text("\n".join(json.dumps(item) for item in items) + "\n")
    assert lq.main(["score", "--input", str(queue), "--json"]) == 0
