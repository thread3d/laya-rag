"""Tests for the LLM-free nugget / claim-recall protocol.

The extraction and coverage rules are the protocol's whole content, so they are pinned directly;
the integration test checks the two arms the cached pool and CE scores support.
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
    name = "laya_rag_nugget_recall"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "nugget_recall.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


nr = _load()


def test_chunk_text_uses_the_benchmark_window():
    assert nr.chunk_text("a b c d e", size=3, overlap=1) == ["a b c", "c d e"]
    assert nr.chunk_text("a b c", size=3, overlap=1) == ["a b c"]


def test_nuggets_include_numbers_dates_entities_and_terms():
    found = nr.nuggets("In 1999 the WHO reported 42 cases about vaccines.")
    assert ("number", "1999") in found
    assert ("number", "42") in found
    assert ("entity", "WHO") in found
    assert ("term", "vaccines") in found
    assert all(kind != "term" for kind, value in found if value in nr._STOPWORDS)


def test_claim_recall_is_one_when_the_evidence_covers_every_nugget():
    evidence = nr.Evidence({"c1": "The Fed raised rates in March 2020 by two points."})
    golds = {"q1": ["The Fed raised rates in March 2020"]}
    out = nr.claim_recall({"a": {"q1": ["c1"]}}, golds, evidence, k=1)
    assert out["a"]["mean"] == 1.0


def test_claim_recall_is_zero_when_the_ranking_is_empty():
    evidence = nr.Evidence({"c1": "anything"})
    golds = {"q1": ["The Fed raised rates in March 2020"]}
    out = nr.claim_recall({"a": {"q1": []}}, golds, evidence, k=1)
    assert out["a"]["mean"] == 0.0


def test_claim_recall_respects_the_depth_cut():
    evidence = nr.Evidence({"gold": "March 2020", "other": "nothing here"})
    golds = {"q1": ["March 2020"]}
    out = nr.claim_recall({"a": {"q1": ["other", "gold"]}}, golds, evidence, k=1)
    assert out["a"]["mean"] < 1.0
    out2 = nr.claim_recall({"a": {"q1": ["other", "gold"]}}, golds, evidence, k=2)
    assert out2["a"]["mean"] == 1.0


def test_ranking_arms_are_built_from_the_cached_pool_and_scores():
    if not nr.SCORES.is_file():
        pytest.skip("ce-scores-full.json not generated")
    fused, ce = nr.build_arm_rankings()
    assert fused and ce
    query_id = next(iter(fused))
    assert set(fused[query_id]) == set(ce[query_id])
    assert len(fused[query_id]) > 10


def test_the_report_is_written_when_the_inputs_exist(tmp_path):
    if not nr.SCORES.is_file() or not nr.CORPUS.is_file():
        pytest.skip("corpus reports not generated")
    out = tmp_path / "nugget"
    assert nr.main(["--out", str(out), "--depths", "10"]) == 0
    assert out.with_suffix(".json").is_file()
    assert out.with_suffix(".md").is_file()
