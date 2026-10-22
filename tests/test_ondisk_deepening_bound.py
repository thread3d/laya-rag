"""The query loop must not deepen without a bound.

A filter that rejects most candidates made the loop re-rank 4x wider on every retry — k, 4k, 16k, up to `live_count`.
On FEVER's 5.4M documents that is ~11 passes with the last over millions of positions; it was measured as a stall
(read offsets frozen for 620 s, no output for 7 h) in a field-weighted BM25F run.

This is a **source-level guard with its limit stated**: it proves the bound exists and is consulted, not that a given
call terminates. The behavioural case (a filter rejecting everything) is covered on a small index by
`tests/test_index_store.py::test_search_allow_rejects_everything`, where the window reaches `live_count` first.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "laya_rag" / "ondisk.py"


def test_the_deepening_loop_consults_a_bound():
    text = SRC.read_text(encoding="utf-8")
    assert "_MAX_DEEPENINGS" in text, "the deepening bound constant is gone"
    assert re.search(r"if deepenings >= _MAX_DEEPENINGS:", text), "the loop no longer consults the bound"
    assert re.search(r"^_MAX_DEEPENINGS = \d+$", text, re.M), "the bound is not a plain integer literal"


def test_the_bound_returns_rather_than_raising():
    """A selective filter must degrade to best effort, not fail the query."""
    text = SRC.read_text(encoding="utf-8")
    tail = text.split("if deepenings >= _MAX_DEEPENINGS:")[1][:120]
    assert "return kept[:k]" in tail, "the bound should return the hits found so far"
