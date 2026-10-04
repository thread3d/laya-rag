"""Offline tests for the cost-axis report.

The numbers here carry the project's winnable claim — quality per unit of compute — so the
statistics have to mean what they say: the percentile must be the interpolated one (a p95 that is
really "the largest value" would flatter the tail), and the memory figure must be normalised, since
`ru_maxrss` is bytes on macOS and kilobytes on Linux.
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
    name = "laya_rag_cost_report"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "cost_report.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cost = _load()


def test_percentile_interpolates_between_ranks():
    values = [1.0, 2.0, 3.0, 4.0]
    assert cost.percentile(values, 50) == pytest.approx(2.5)
    assert cost.percentile(values, 0) == pytest.approx(1.0)
    assert cost.percentile(values, 100) == pytest.approx(4.0)


def test_percentile_is_not_just_the_maximum():
    """A p95 reported as the largest value would flatter the tail."""
    values = [1.0] * 99 + [900.0]
    assert cost.percentile(values, 95) == pytest.approx(1.0)
    assert cost.percentile(values, 99) > 1.0


def test_percentile_handles_degenerate_inputs():
    assert cost.percentile([], 95) == 0.0
    assert cost.percentile([7.0], 95) == pytest.approx(7.0)


def test_summarize_latency_reports_the_rates_and_the_tail():
    summary = cost.summarize_latency([1.0, 2.0, 3.0, 4.0])
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["median"] == pytest.approx(2.5)
    assert summary["p95"] == pytest.approx(3.85)
    assert summary["queries_per_second"] == pytest.approx(0.4)  # 4 queries over 10 seconds


def test_summarize_latency_of_nothing_is_empty_not_zeroed():
    assert cost.summarize_latency([]) == {}


def test_index_bytes_sums_every_artifact(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.bin").write_bytes(b"y" * 5)
    assert cost.index_bytes(tmp_path) == 15


def test_index_bytes_of_a_missing_directory_is_zero(tmp_path):
    assert cost.index_bytes(tmp_path / "absent") == 0


def test_peak_rss_is_normalised_to_bytes():
    """macOS reports bytes and Linux kilobytes; the report must mean bytes either way."""
    value = cost.ru_maxrss_bytes()
    assert value > 0
    assert value >= 2**20, f"implausibly small for a Python process: {value}"


def test_args_default_to_the_shipped_default_index():
    args = cost.parse_args([])
    assert args.index_dir.endswith("engine-default")
    assert args.queries == 60 and args.warmup == 3
