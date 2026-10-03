"""Offline tests for the per-type routing analysis.

The measured result is a null, and a null is only worth recording if the instrument could
have found a real effect. The central test therefore builds a world where two question types
genuinely prefer different strategies and checks that the nested protocol detects it; the rest
pins the protocol's discipline (every query scored by a rule chosen without it) and its
handling of degenerate input.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "laya_rag_multihop_routing", ROOT / "tools" / "multihop_routing.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


routing = _load()


def test_mean_of_nothing_is_zero():
    assert routing.mean([]) == 0.0
    assert routing.mean([0.2, 0.4]) == pytest.approx(0.3)


def test_route_by_type_detects_a_world_where_types_really_differ():
    """The instrument check: if A prefers X and B prefers Y, routing must win.

    Strategy X is excellent on type A and useless on B; Y is the reverse. No single
    strategy is good at both, so routing has to beat the globally-selected rule.
    """
    per_query = {"X": {}, "Y": {}}
    types = {}
    for i in range(200):
        q = str(i)
        if i % 2 == 0:
            types[q] = "A"
            per_query["X"][q] = 0.9
            per_query["Y"][q] = 0.1
        else:
            types[q] = "B"
            per_query["X"][q] = 0.1
            per_query["Y"][q] = 0.9
    result = routing.route_by_type(per_query, types, seed=3)
    routed = routing.mean(list(result["routed"].values()))
    global_selected = routing.mean(list(result["global"].values()))
    assert routed == pytest.approx(0.9)
    # The seeded halves are only approximately balanced, so the globally-selected rule lands
    # near 0.5 rather than exactly on it; the point is that it cannot approach 0.9.
    assert global_selected < 0.6
    assert routed > global_selected
    assert result["picks"]["A"] == ["X", "X"]
    assert result["picks"]["B"] == ["Y", "Y"]


def test_route_by_type_scores_every_query_out_of_sample():
    """Every query must be scored by a rule picked on the other half, never on its own."""
    per_query = {"X": {str(i): 0.5 for i in range(40)}}
    types = {str(i): "A" for i in range(40)}
    result = routing.route_by_type(per_query, types, seed=5)
    assert len(result["routed"]) == 40
    assert len(result["global"]) == 40
    assert set(result["routed"]) == set(result["global"])


def test_route_by_type_records_a_pick_per_type_per_direction():
    per_query = {"X": {str(i): 0.5 for i in range(20)}, "Y": {str(i): 0.6 for i in range(20)}}
    types = {str(i): ("A" if i < 10 else "B") for i in range(20)}
    result = routing.route_by_type(per_query, types, seed=1)
    for qtype in ("A", "B"):
        assert len(result["picks"][qtype]) == 2
    assert len(result["global_picks"]) == 2


def test_route_by_type_handles_an_empty_report():
    result = routing.route_by_type({}, {}, seed=1)
    assert result["routed"] == {}
    assert result["global"] == {}
    assert result["strategies"] == []


def test_route_by_type_uses_only_queries_every_strategy_scored():
    """A strategy missing a query would otherwise silently shrink the comparison set."""
    per_query = {
        "X": {str(i): 0.5 for i in range(40)},
        "Y": {str(i): 0.6 for i in range(30)},   # missing the last ten
    }
    types = {str(i): "A" for i in range(40)}
    result = routing.route_by_type(per_query, types, seed=2)
    assert len(result["routed"]) == 30
    assert all(int(q) < 30 for q in result["routed"])


def test_render_markdown_reports_picks_and_comparisons():
    payload = {
        "_settings": {"queries": 2255, "report": "r.json", "seed": 17},
        "summary": {"routed_MAP@10": 0.3169, "globally_selected_MAP@10": 0.3192},
        "type_counts": {"comparison_query": 856, "inference_query": 816},
        "picks_by_type": {
            "comparison_query": ["covquota-3-0.7", "covquota-3-0.7"],
            "inference_query": ["coverage-0.5", "covquota-3-0.5"],
        },
        "global_picks": ["covquota-3-0.7", "covquota-3-0.7"],
        "comparisons": {
            "routed vs global MAP@10": {
                "bootstrap": {"delta": -0.0023, "ci_low": -0.0056, "ci_high": 0.0008,
                              "significant": False},
                "permutation": {"p_value": 0.1738},
            }
        },
    }
    text = routing.render_markdown(payload)
    assert "routed_MAP@10" in text
    assert "inference_query" in text
    assert "-0.0023" in text
