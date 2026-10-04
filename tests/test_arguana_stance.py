"""Offline tests for the ArguAna relational (stance) reranker.

Two things decide whether the measurement means anything, and both are pure: the label columns
must be read from the checkpoint rather than assumed (a swapped contradiction/entailment pair
inverts every score), and the pool must be reordered, not re-selected, so any delta is ordering.
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
    name = "laya_rag_arguana_stance"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "arguana_stance.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


st = _load()


def test_rank_positions_descends_and_breaks_ties_by_position():
    assert st.rank_positions([0.1, 0.9, 0.5]) == [1, 2, 0]
    assert st.rank_positions([0.5, 0.5]) == [0, 1]


def test_order_doc_ids_reorders_the_given_pool_only():
    pool = ["a", "b", "c"]
    assert st.order_doc_ids(pool, [0.1, 0.9, 0.5]) == ["b", "c", "a"]
    assert sorted(st.order_doc_ids(pool, [0.1, 0.9, 0.5])) == sorted(pool)


def test_stance_labels_are_read_from_the_checkpoint():
    real = {0: "contradiction", 1: "entailment", 2: "neutral"}
    assert st.stance_label_indices(real) == (0, 1)
    # A permuted checkpoint must be followed, not overridden.
    permuted = {0: "ENTAILMENT", 1: "neutral", 2: "Contradiction"}
    assert st.stance_label_indices(permuted) == (2, 0)


def test_stance_labels_reject_a_model_without_the_pair():
    with pytest.raises(ValueError, match="no contradiction/entailment"):
        st.stance_label_indices({0: "positive", 1: "negative"})


def test_args_default_to_the_documented_pool_and_official_k():
    args = st.parse_args([])
    assert args.pool == 20 and args.top_k == 10
    assert args.shard == "0/1" and args.merge == 0 and args.compare is False


def test_fingerprint_separates_models_and_pools():
    base = st.parse_args([])
    assert st._fingerprint(base) != st._fingerprint(st.parse_args(["--pool", "50"]))
    assert st._fingerprint(base) != st._fingerprint(st.parse_args(["--model", "other"]))
    assert st._fingerprint(base) == st._fingerprint(st.parse_args(["--shard", "1/3"]))
