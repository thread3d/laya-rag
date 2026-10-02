"""Offline tests for the atomic-unit retrieval experiment's helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "laya_rag_atom_index", ROOT / "tools" / "multihop_atom_index.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


atom = _load()


def test_build_atoms_ids_and_parent_map():
    corpus = {
        "a#0": "Refunds are processed within five business days. Shipping is free over fifty dollars.",
        "a#1": "Contact billing support if the refund has not arrived after ten days.",
    }
    atoms, parent = atom.build_atoms(corpus, min_chars=20)
    assert len(atoms) == 3
    assert [a.chunk_id for a in atoms] == ["a#0::0", "a#0::1", "a#1::0"]
    assert parent["a#0::1"] == "a#0"
    assert parent["a#1::0"] == "a#1"


def test_build_atoms_drops_units_below_the_minimum():
    corpus = {"a#0": "Short. This sentence is comfortably long enough to be an atom."}
    atoms, _ = atom.build_atoms(corpus, min_chars=40)
    assert len(atoms) == 1
    assert "comfortably long enough" in atoms[0].text


def test_parent_rankings_keeps_the_best_atom_per_parent_in_order():
    parent = {"a#0::0": "a#0", "a#0::1": "a#0", "a#1::0": "a#1", "a#2::0": "a#2"}
    hits = {
        "q": [
            ("a#0::0", 9.0),
            ("a#0::1", 8.0),   # same parent -> skipped
            ("a#1::0", 7.0),
            ("a#2::0", 6.0),
        ]
    }
    rankings = atom.parent_rankings(hits, parent, top_k=2)
    assert rankings["q"] == ["a#0", "a#1"]


def test_parent_rankings_ignores_unknown_atom_ids():
    rankings = atom.parent_rankings({"q": [("missing", 1.0), ("a#0::0", 0.5)]},
                                    {"a#0::0": "a#0"}, top_k=5)
    assert rankings["q"] == ["a#0"]


def test_parent_rankings_handles_an_empty_query():
    assert atom.parent_rankings({"q": []}, {}, top_k=5) == {"q": []}


def test_score_strategies_wraps_each_ranked_list_for_official_metrics():
    """The regression this pins: official_metrics takes a list of per-query ranked
    lists; a flat list of strings is sliced into characters and scores zero."""
    corpus = {"a#0": "Refunds are processed within five business days."}
    ranking = {"0": ["a#0"], "1": []}
    items = [(0, {"q": "how long"}), (1, {"q": "unanswerable"})]
    golds = [["five business days"], ["shipping rates"]]
    rows = atom.score_strategies(corpus, ranking, items, golds)
    assert rows["Hits@10"]["0"] == 1.0
    assert rows["MAP@10"]["0"] == 1.0
    assert rows["Hits@10"]["1"] == 0.0
