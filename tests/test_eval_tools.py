"""Offline tests for the evaluation tools' own helpers.

The harness is only as trustworthy as the guards around it. ``index_matches`` exists
because a cached index built with the 64-dimension stub was silently reused by a real
256/768-dimension run and failed deep inside ``VectorIndex.search``; that regression is
pinned here so the failure comes back as a rebuild, not a dimension error.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from laya_rag.engine import RagEngine  # noqa: E402
from tests.support import StubModel, make_config, write_corpus  # noqa: E402


def _load_eval_tool():
    """Import ``tools/eval.py`` by path; it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("laya_rag_eval_tool", ROOT / "tools" / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_tool = _load_eval_tool()


def test_index_matches_rejects_an_index_built_with_a_different_embedder(tmp_path):
    engine = RagEngine(make_config(tmp_path), model=StubModel(dim=64))
    engine.index_paths([write_corpus(tmp_path / "corpus")])

    assert eval_tool.index_matches(engine, StubModel(dim=64)) is True
    assert eval_tool.index_matches(engine, StubModel(dim=256)) is False


def test_index_matches_is_false_when_nothing_is_indexed(tmp_path):
    engine = RagEngine(make_config(tmp_path), model=StubModel(dim=64))
    assert eval_tool.index_matches(engine, StubModel(dim=64)) is False


def test_index_matches_rejects_an_unreadable_cache(tmp_path):
    RagEngine(make_config(tmp_path), model=StubModel(dim=64)).index_paths(
        [write_corpus(tmp_path / "corpus")]
    )
    (tmp_path / "index" / "index.json").write_text("{ not json", encoding="utf-8")

    # A fresh engine has no in-memory copy, so this reads the damaged file.
    fresh = RagEngine(make_config(tmp_path), model=StubModel(dim=64))
    assert eval_tool.index_matches(fresh, StubModel(dim=64)) is False
