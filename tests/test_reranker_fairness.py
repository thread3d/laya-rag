"""Tests for the reranker-fairness pack.

The whole point of the pack is that the pool is *reranker-excluded* and that R@k cannot move under
reordering, so those two properties are pinned directly, along with the refusal of a pool that was
built with a reranker in the loop and the R@1000 bound/measurement switch.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def _load():
    name = "laya_rag_reranker_fairness"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "reranker_fairness.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rf = _load()


def _payload(*, reranker_key: str | None = None) -> dict:
    settings = {"rrf_k": 5, "model": "tiny", "overlap": 200}
    if reranker_key:
        settings[reranker_key] = "bge-reranker-base"
    return {
        "_settings": settings,
        "queries": {
            "0": {"pool": ["a#0", "b#0", "c#0"], "ce": [0.1, 0.9, 0.5]},
            "1": {"pool": ["d#0", "e#0"], "ce": [0.4, 0.6]},
        },
    }


# -- the offline invariants -------------------------------------------------


def test_self_check_passes():
    assert rf.self_check() == []


def test_check_mode_exits_zero(capsys):
    assert rf.main(["--check"]) == 0
    assert "invariants hold" in capsys.readouterr().out


# -- the pool is reranker-excluded, and says so -----------------------------


def test_pool_provenance_accepts_a_plain_fused_pool():
    provenance = rf.pool_provenance(_payload())
    assert provenance["excluded"] is True
    assert provenance["reranker_keys_in_pool_settings"] == []
    assert provenance["settings"]["rrf_k"] == 5


def test_pool_provenance_refuses_a_pool_built_with_a_reranker():
    provenance = rf.pool_provenance(_payload(reranker_key="pool_reranker"))
    assert provenance["excluded"] is False
    assert provenance["reranker_keys_in_pool_settings"] == ["pool_reranker"]


def test_main_refuses_a_reranker_built_pool(tmp_path, capsys):
    """A fairness claim from a reranker-built pool is the failure Tier 0.5 exists to prevent."""
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps(_payload(reranker_key="reranker_model")))
    code = rf.main(["--scores", str(scores), "--out", str(tmp_path / "out")])
    assert code == 1
    assert "NOT reranker-excluded" in capsys.readouterr().out


# -- reordering cannot change membership ------------------------------------


def test_arm_rankings_keep_the_pool_order_and_sort_by_score():
    fused, ce = rf.arm_rankings(_payload())
    assert fused["0"] == ["a#0", "b#0", "c#0"]
    assert ce["0"] == ["b#0", "c#0", "a#0"]
    assert fused["1"] == ["d#0", "e#0"]
    assert ce["1"] == ["e#0", "d#0"]


def test_the_candidate_set_is_identical_at_the_pool_depth():
    fused, ce = rf.arm_rankings(_payload())
    row = rf.membership_invariance(fused, ce)
    assert row["checked"] == 2
    assert row["identical"] == 2
    assert row["all_identical"] is True


def test_the_reordering_still_moves_the_top_k_set():
    """Below the pool depth the arms *must* differ — that difference is the ordering gain."""
    fused, ce = rf.arm_rankings(_payload())
    assert rf.divergent_top_k(fused, ce, 1)["divergent"] == 2
    assert rf.divergent_top_k(fused, ce, 3)["divergent"] == 0


def test_an_empty_comparison_is_not_all_identical():
    """A vacuous pass would let an empty cache claim the invariant."""
    assert rf.membership_invariance({}, {})["all_identical"] is False


def test_membership_fails_when_the_second_arm_adds_a_candidate():
    fused = {"0": ["a#0", "b#0"]}
    ce = {"0": ["z#0", "b#0"]}          # a retriever, not a reranker: membership moves
    row = rf.membership_invariance(fused, ce)
    assert row["identical"] == 0 and row["all_identical"] is False


# -- the depth curve and the R@1000 bound -----------------------------------


def test_pool_depth_curve_reads_a_diagnostics_shape(tmp_path):
    report = {
        "_settings": {"max_depth": 500, "depths": [10, 100, 500]},
        "per_query": [
            {"pool_fact_recall@10": 0.5, "pool_fact_recall@100": 1.0, "pool_fact_recall@500": 1.0},
            {"pool_fact_recall@10": 0.0, "pool_fact_recall@100": 0.5, "pool_fact_recall@500": 1.0},
        ],
    }
    path = tmp_path / "diagnostics.json"
    path.write_text(json.dumps(report))
    curve = rf.pool_depth_curve(path)
    assert curve["max_depth"] == 500
    assert curve["curve"]["10"]["mean"] == 0.25
    assert curve["curve"]["100"]["mean"] == 0.75
    assert curve["curve"]["10"]["queries"] == 2


def test_r1000_is_bounded_without_a_deep_run_and_measured_with_one():
    scores = _payload()
    golds = {"0": ["b facts"], "1": ["e facts"]}
    evidence = rf.nr.Evidence({"a#0": "nothing", "b#0": "b facts here",
                               "c#0": "c", "d#0": "d", "e#0": "e facts here"})
    diagnostics = {"source": "diagnostics.json", "max_depth": 500,
                   "curve": {"10": {"mean": 0.4, "queries": 2},
                             "500": {"mean": 0.98, "queries": 2}}}
    rerank = {"source": "rerank.json", "arms": {"fused": {"MAP@10": 0.25}, "ce": {"MAP@10": 0.29}},
              "ce_vs_fused": {}}
    bounded = rf.build_report(scores=scores, golds=golds, evidence=evidence,
                              diagnostics=diagnostics, rerank=rerank)
    assert bounded["r1000"]["measured"] is False
    assert "depth 500" in bounded["r1000"]["bound"]
    assert "needs a deeper pool" in bounded["r1000"]["bound"]
    deep = {"source": "diagnostics-1000.json", "max_depth": 1000,
            "curve": {"1000": {"mean": 0.99, "queries": 2}}}
    measured = rf.build_report(scores=scores, golds=golds, evidence=evidence,
                               diagnostics=diagnostics, rerank=rerank, deep=deep)
    assert measured["r1000"]["measured"] is True
    assert measured["r1000"]["curve"]["curve"]["1000"]["mean"] == 0.99


def test_recall_is_identical_at_the_pool_depth_and_better_at_ten_for_the_ce():
    """The pack's substantive reading: ordering only, on a pool neither arm built."""
    scores = _payload()
    # Query 0's whole fact sits in "b#0": fused ranks it second, the CE first.
    golds = {"0": ["b fact"]}
    evidence = rf.nr.Evidence({"a#0": "a", "b#0": "b fact", "c#0": "c",
                               "d#0": "d", "e#0": "e"})
    fused, ce = rf.arm_rankings(scores)
    table = rf.recall_table({"fused": fused, "ce": ce}, golds, evidence, (1, 3))
    assert table["3"]["fused"]["mean"] == 1.0
    assert table["3"]["ce"]["mean"] == 1.0          # membership: identical at the pool depth
    assert table["1"]["fused"]["mean"] == 0.0       # ordering: the CE lifts it into rank 1
    assert table["1"]["ce"]["mean"] == 1.0


def test_render_markdown_states_the_bound_and_the_invariance():
    scores = _payload()
    golds = {"0": ["b facts"], "1": ["e facts"]}
    evidence = rf.nr.Evidence({"a#0": "x", "b#0": "b facts", "c#0": "x",
                               "d#0": "x", "e#0": "e facts"})
    report = rf.build_report(
        scores=scores, golds=golds, evidence=evidence,
        diagnostics={"source": "diagnostics.json", "max_depth": 500,
                     "curve": {"500": {"mean": 0.98, "queries": 2}}},
        rerank={"source": "rerank.json", "arms": {}, "ce_vs_fused": {}},
    )
    text = rf.render_markdown(report)
    assert "reranker-excluded" in text
    assert "same candidate set" in text
    assert "Not measurable from cache" in text
