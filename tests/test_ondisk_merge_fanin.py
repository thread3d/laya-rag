"""Tests for the bounded-fan-in run merge.

The final merge opens every run at once, so a small block cap — which multiplies runs — could push a
build past the file-descriptor limit and fail outright. Collapsing runs in groups first fixes that,
but only if the collapse is *ranking-neutral*: it reorders nothing, it just changes how many files
are open at a time. These tests force the collapse path by shrinking the fan-in, which the real
configuration only reaches on a corpus large enough to be impractical to test on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laya_rag import ondisk  # noqa: E402

DOCS = [
    (f"d{index}",
     f"alpha beta gamma{index % 7} delta{index % 3} epsilon{index % 5} zeta{index % 11}")
    for index in range(30)
]
QUERIES = ("alpha beta", "gamma3 delta0", "epsilon2 zeta4", "delta1", "zeta9 alpha")


def _ranking(tmp_path: Path, monkeypatch, fan_in: int, name: str):
    monkeypatch.setattr(ondisk, "_MERGE_FAN_IN", fan_in)
    index = ondisk.OnDiskBM25(tmp_path / name)
    # One document per block: 30 runs, so a low fan-in must collapse them in several passes.
    index.build(DOCS, block_docs=1, block_tokens=1)
    return index, {query: [hit.chunk_id for hit in index.search(query, k=10)] for query in QUERIES}


def test_a_single_level_merge_is_the_ranking(tmp_path, monkeypatch):
    """The ordinary path (fan-in above the run count) is the reference."""
    _index, ranking = _ranking(tmp_path, monkeypatch, 64, "plain")
    assert any(ranking.values()), "the reference ranking must not be empty"


def test_collapsing_runs_does_not_change_the_ranking(tmp_path, monkeypatch):
    _plain, reference = _ranking(tmp_path, monkeypatch, 64, "plain")
    _folded, collapsed = _ranking(tmp_path, monkeypatch, 2, "folded")
    assert collapsed == reference


def _hits(tmp_path: Path, monkeypatch, fan_in: int, name: str):
    monkeypatch.setattr(ondisk, "_MERGE_FAN_IN", fan_in)
    index = ondisk.OnDiskBM25(tmp_path / name)
    index.build(DOCS, block_docs=1, block_tokens=1)
    return index, {query: [(hit.chunk_id, round(hit.score, 9))
                            for hit in index.search(query, k=10)] for query in QUERIES}


def test_collapsing_runs_does_not_change_a_single_score(tmp_path, monkeypatch):
    """Order *within* a term may legitimately differ after a collapse, so compare scores."""
    plain, reference = _hits(tmp_path, monkeypatch, 64, "plain")
    folded, collapsed = _hits(tmp_path, monkeypatch, 2, "folded")
    assert collapsed == reference
    assert plain.terms == folded.terms
    assert plain.n_docs == folded.n_docs
    assert int(plain.offsets[-1]) == int(folded.offsets[-1]), "same number of postings"


def test_a_deeper_collapse_agrees_too(tmp_path, monkeypatch):
    """Fan-in 1 cannot collapse; fan-in 3 forces several levels. Both must match the plain run."""
    _plain, reference = _ranking(tmp_path, monkeypatch, 64, "plain")
    _solo, single = _ranking(tmp_path, monkeypatch, 1, "solo")
    _deep, deeper = _ranking(tmp_path, monkeypatch, 3, "deep")
    assert single == reference and deeper == reference


def test_the_scratch_runs_are_cleaned_up_after_a_collapse(tmp_path, monkeypatch):
    """Intermediates live in the scratch directory, so a completed build must leave none behind.

    This is where the first version failed: it deleted only the runs it knew about, so the
    `merge-*.npy` intermediates kept `runs/` non-empty and the build died at `rmdir` — after the
    index itself had already been written.
    """
    index, _ = _ranking(tmp_path, monkeypatch, 2, "folded")
    assert not (tmp_path / "folded" / "runs").exists(), "the scratch directory must be gone"
    assert (index.directory / "postings.bin").is_file(), "and the index must be complete"
