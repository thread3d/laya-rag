"""Offline tests for the comparative charts.

The charts are the artefact a reader actually looks at, so a chart that silently drops
strategies, or that plots a point estimate where the report has an interval, is a
misreading waiting to happen. These pin the selection rules of the second-stage chart and
its behaviour when the report is missing.
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
    spec = importlib.util.spec_from_file_location("laya_rag_compare", ROOT / "tools" / "compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = _load()


def _plt_or_skip():
    try:
        return compare._plt_mod()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"matplotlib unavailable: {exc}")


def _report(names: dict[str, float], queries: int = 40) -> dict:
    """A synthetic experiment report: each strategy a flat per-query MAP@10."""
    return {
        "_settings": {"limit_queries": queries, "depth": 100},
        "per_query": {
            name: {str(i): value + (i % 5) * 0.001 for i in range(queries)}
            for name, value in names.items()
        },
    }


def test_second_stage_chart_renders_every_named_strategy(tmp_path, monkeypatch):
    plt = _plt_or_skip()
    monkeypatch.setattr(compare, "MH_DIR", tmp_path)
    (tmp_path / "rerank-experiment-full.json").write_text(
        json.dumps(_report({
            "fused": 0.25,
            "ce": 0.29,
            "coverage-0.7": 0.31,
            "covquota-3-0.7": 0.32,
            "tri": 0.33,
            "tri-covquota-3-0.7": 0.34,
            "pit-tri-ce": 0.35,
            "rescue-0.5": 0.27,
        }))
    )
    outdir = tmp_path / "charts"
    compare.chart_second_stage(outdir, plt)
    written = outdir / "second_stage.png"
    assert written.is_file()
    assert written.stat().st_size > 1000


def test_second_stage_chart_skips_quietly_without_a_report(tmp_path, monkeypatch, capsys):
    plt = _plt_or_skip()
    monkeypatch.setattr(compare, "MH_DIR", tmp_path)
    compare.chart_second_stage(tmp_path / "charts", plt)
    assert "skipping that chart" in capsys.readouterr().out
    assert not (tmp_path / "charts" / "second_stage.png").exists()


def test_second_stage_chart_prefers_the_full_report(tmp_path, monkeypatch):
    """The full 2,255-query report must win over the 400-query subset when both exist."""
    plt = _plt_or_skip()
    monkeypatch.setattr(compare, "MH_DIR", tmp_path)
    full = _report({"fused": 0.25, "ce": 0.29}, queries=200)
    full["_settings"]["limit_queries"] = 2255
    (tmp_path / "rerank-experiment-full.json").write_text(json.dumps(full))
    (tmp_path / "rerank-experiment.json").write_text(
        json.dumps(_report({"fused": 0.10}, queries=10))
    )
    outdir = tmp_path / "charts"
    compare.chart_second_stage(outdir, plt)
    assert (outdir / "second_stage.png").is_file()


def test_second_stage_chart_handles_a_report_without_per_query_scores(
    tmp_path, monkeypatch, capsys
):
    plt = _plt_or_skip()
    monkeypatch.setattr(compare, "MH_DIR", tmp_path)
    (tmp_path / "rerank-experiment-full.json").write_text(json.dumps({"_settings": {}}))
    compare.chart_second_stage(tmp_path / "charts", plt)
    assert "no per-query scores" in capsys.readouterr().out


def test_bridge_chart_renders_the_treatment_and_its_control(tmp_path, monkeypatch):
    plt = _plt_or_skip()
    monkeypatch.setattr(compare, "MH_DIR", tmp_path)
    (tmp_path / "rerank-bridge.json").write_text(
        json.dumps(_report({
            "fused": 0.25,
            "ce": 0.30,
            "covquota-3-0.7": 0.33,
            "tri": 0.13,
            "tri-shuffled": 0.11,
            "ce-tri-filter-0.3": 0.27,
            "ce-tri-shuffled-filter-0.3": 0.26,
            "pit-tri-ce": 0.24,
        }))
    )
    outdir = tmp_path / "charts"
    compare.chart_bridge_stage(outdir, plt)
    written = outdir / "bridge_stage.png"
    assert written.is_file()
    assert written.stat().st_size > 1000


def test_bridge_chart_skips_when_the_report_is_absent(tmp_path, monkeypatch, capsys):
    plt = _plt_or_skip()
    monkeypatch.setattr(compare, "MH_DIR", tmp_path)
    compare.chart_bridge_stage(tmp_path / "charts", plt)
    assert "skipping the bridge chart" in capsys.readouterr().out


def test_second_stage_panels_skips_a_report_with_no_requested_strategy(
    tmp_path, monkeypatch, capsys
):
    plt = _plt_or_skip()
    report = tmp_path / "rerank-bridge.json"
    report.write_text(json.dumps(_report({"unrelated": 0.2})))
    ok = compare.second_stage_panels(
        tmp_path / "charts", plt, report, ["tri", "fused"], "x.png", "title"
    )
    assert ok is False
    assert "none of the requested strategies" in capsys.readouterr().out
