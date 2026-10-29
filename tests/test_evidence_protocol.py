"""The evidence-layer protocol's four axes, and its refusal to omit one.

The protocol's value is that a reader can see which axes were measured. A report that silently drops
an absent axis is worse than no report, because it reads as a pass — so "not measured" naming the
command that would produce it is a tested property here, not a formatting choice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import evidence_protocol as ep  # noqa: E402
from laya_rag.answerability import FEATURE_NAMES  # noqa: E402


def _answer_row(**summary):
    base = {"queries": 150, "MAP@10": 0.3128, "Hits@10": 0.9067, "Hits@4": 0.74,
            "MRR@10": 0.6152, "answer_nugget_recall": 0.261, "citation_clean_all": 1.0,
            "abstention_rate": 0.0}
    base.update(summary)
    return {"summary": base, "per_query": {}}


def _gate_rows(n: int = 40, separable: bool = True):
    """Rows where the features separate the classes and the scalar deliberately does not."""
    rows = []
    for i in range(n):
        label = i % 2
        level = (0.9 if label else 0.2) if separable else 0.5
        rows.append(
            {
                "query": f"q{i}",
                "label": label,
                "answerability": 0.5,  # the shipped scalar: a coin flip on this sample
                "features": {name: level for name in FEATURE_NAMES},
            }
        )
    return rows


def test_evidence_quality_reads_the_retrieval_rows_and_the_answers_own_numbers():
    quality = ep.evidence_quality(_answer_row())
    assert quality["retrieval"]["MAP@10"] == pytest.approx(0.3128)
    assert quality["answer_nugget_recall"] == pytest.approx(0.261)
    assert quality["citation_clean"] == pytest.approx(1.0)
    assert quality["queries"] == 150


def test_a_report_without_retrieval_metrics_is_refused():
    with pytest.raises(SystemExit):
        ep.evidence_quality({"summary": {"queries": 3}})


def test_the_gate_is_scored_against_the_shipped_scalar_on_the_same_half():
    axis = ep.abstention_axis(_gate_rows())
    assert axis["held_out"]["n"] == axis["scalar"]["n"] == axis["rows"] // 2
    assert axis["held_out"]["positives"] == axis["scalar"]["positives"]
    assert axis["held_out"]["auc"] > 0.9
    assert axis["scalar"]["auc"] == pytest.approx(0.5, abs=0.1)


def test_a_single_class_input_is_refused_rather_than_scored():
    rows = _gate_rows()
    for row in rows:
        row["label"] = 1
    with pytest.raises(SystemExit):
        ep.abstention_axis(rows)


def test_the_joint_score_penalises_false_access_and_says_what_it_is_not():
    joint = ep.joint_score(0.3, {"held_out": {"risk@80": 0.25}})
    assert joint["joint"] == pytest.approx(0.225)
    assert "different samples" in joint["caveat"]


def test_cost_is_absent_rather_than_zero_when_no_report_exists(tmp_path):
    assert ep.cost_axis(None) is None
    assert ep.cost_axis(tmp_path / "missing.json") is None


def test_cost_reads_the_device_and_thread_count_alongside_the_seconds(tmp_path):
    import json

    path = tmp_path / "cost.json"
    path.write_text(json.dumps({
        "device": "cpu", "torch_threads": 16, "index_bytes": 1024,
        "latency_seconds": {"mean": 6.71}, "cpu_seconds": {"mean": 96.3},
    }), encoding="utf-8")
    cost = ep.cost_axis(path)
    assert cost["device"] == "cpu" and cost["torch_threads"] == 16
    assert cost["wall_mean_seconds"] == pytest.approx(6.71)


def test_render_names_an_absent_axis_and_the_command_that_would_produce_it():
    rendered = ep.render({"evidence_quality": None, "abstention": None, "joint": None, "cost": None})
    assert rendered.count("not measured") == 4  # axes 1, 2, 3 and 4
    assert "engine_extractive.py" in rendered
    assert "gate_dataset.py" in rendered


def test_render_carries_every_measured_axis():
    report = {
        "evidence_quality": ep.evidence_quality(_answer_row()),
        "abstention": ep.abstention_axis(_gate_rows()),
        "joint": ep.joint_score(0.3128, ep.abstention_axis(_gate_rows())),
        "cost": {"device": "cpu", "torch_threads": 16, "index_bytes": 1024,
                 "wall_mean_seconds": 6.71, "cpu_mean_seconds": 96.3},
    }
    rendered = ep.render(report)
    assert "not measured" not in rendered
    assert "0.3128" in rendered and "cpu" in rendered and "96.3" in rendered


# ── which configuration produced the numbers ─────────────────────────────

def test_the_report_says_when_two_axes_share_a_query_sample():
    answer = {"_settings": {"limit": 150, "sample_seed": 17, "rerank_model": None}}
    gate = {"_settings": {"limit": 150, "sample_seed": 17}}
    config = ep.configuration(answer, gate)
    assert config["same_query_sample"] is True
    assert "share a query sample" in config["note"]


def test_the_report_says_so_when_they_do_not():
    """The joint score multiplies two samples; the report should compute that, not repeat a caveat."""
    answer = {"_settings": {"limit": 150, "sample_seed": 17}}
    gate = {"_settings": {"limit": 300, "sample_seed": 17}}
    config = ep.configuration(answer, gate)
    assert config["same_query_sample"] is False
    assert "different samples" in config["note"]


def test_a_gate_dataset_without_settings_block_is_still_described():
    """`gate_dataset.py` records its settings at the top level, not under `_settings`."""
    config = ep.configuration({"limit": 150, "sample_seed": 17}, {"limit": 300, "sample_seed": 17, "records": "x.json"})
    assert config["gate_rows"]["limit"] == 300
    assert config["same_query_sample"] is False


def test_the_rendered_page_carries_the_configuration_and_its_warning():
    report = {
        "evidence_quality": ep.evidence_quality(_answer_row()),
        "abstention": ep.abstention_axis(_gate_rows()),
        "joint": ep.joint_score(0.3128, ep.abstention_axis(_gate_rows())),
        "cost": None,
        "configuration": ep.configuration({"limit": 150, "sample_seed": 17}, {"limit": 300, "sample_seed": 17}),
    }
    rendered = ep.render(report)
    assert "Which configuration produced these" in rendered
    assert "different samples" in rendered
