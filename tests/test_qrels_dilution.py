"""Tests for the qrels-dilution trap, which has now been found four times in this repository.

`evaluate_rankings` macro-averages over **every query in the qrels it is given**. So evaluating a
partial ranking set against a complete qrels scores every unranked query as zero and divides by the
wrong denominator — a silent, large under-report that looks like a retrieval failure.

It has been fixed in `arguana_stance`, `order_eval`, the BEIR query subset, `ondisk.py` and
`engine_beir.py`. The behavioural test below is the important one: it demonstrates the trap directly,
so it fails if `evaluate_rankings` ever changes in a way that makes a restricted qrels unnecessary —
and it documents why every limited path needs the guard.
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

import evalkit  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


beir = _load("laya_rag_beir_dilution", ROOT / "tools" / "beir.py")


def test_partial_rankings_against_a_full_qrels_are_diluted():
    """The trap itself: a perfect ranking on 3 of 10 queries scores far below 1.0."""
    qrels = {f"q{i}": {f"d{i}": 1.0} for i in range(10)}
    rankings = {f"q{i}": [f"d{i}"] for i in range(3)}      # perfect, but only for three queries

    diluted = evalkit.evaluate_rankings(rankings, qrels, k_values=(10,))["summary"]["nDCG@10"]
    restricted = evalkit.evaluate_rankings(
        rankings, beir.restrict_qrels(qrels, rankings), k_values=(10,))["summary"]["nDCG@10"]

    assert restricted == pytest.approx(1.0), "restricted qrels measure the ranked queries only"
    assert diluted < 0.5, "the full qrels divide a perfect 3-query result by ten"
    assert restricted > diluted


def test_the_dilution_ratio_matches_the_shortfall_in_ranked_queries():
    """The arithmetic that identified the HotpotQA bug: 0.0516 x 7405/600 = 0.637."""
    qrels = {f"q{i}": {f"d{i}": 1.0} for i in range(100)}
    rankings = {f"q{i}": [f"d{i}"] for i in range(20)}
    diluted = evalkit.evaluate_rankings(rankings, qrels, k_values=(10,))["summary"]["nDCG@10"]
    assert diluted == pytest.approx(20 / 100)


@pytest.mark.parametrize("tool", ["ondisk.py", "engine_beir.py"])
def test_every_limited_path_restricts_its_qrels(tool):
    """A source guard: each tool that can cap its query set must pair it with capped qrels."""
    source = (ROOT / "tools" / tool).read_text()
    assert "restrict_qrels" in source, f"{tool} can cap queries without restricting qrels"
