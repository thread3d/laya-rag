"""Offline tests for the engine-level BEIR evaluation's pure helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "laya_rag_engine_beir", ROOT / "tools" / "engine_beir.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load()


def test_write_corpus_names_files_by_position_and_returns_id_order(tmp_path):
    doc_ids = tool.write_corpus({"b": "beta", "a": "alpha"}, tmp_path / "docs")
    assert doc_ids == ["b", "a"]
    assert (tmp_path / "docs" / "000000.txt").read_text(encoding="utf-8") == "beta"
    assert (tmp_path / "docs" / "000001.txt").read_text(encoding="utf-8") == "alpha"


def test_ranking_maps_paths_back_and_deduplicates():
    doc_ids = ["doc-a", "doc-b", "doc-c"]
    passages = [
        {"path": "/x/000001.txt"},  # doc-b
        {"path": "/x/000000.txt"},  # doc-a
        {"path": "/x/000001.txt"},  # duplicate chunk of doc-b
        {"path": "/x/notes.txt"},   # not a corpus file
        {"path": "/x/000099.txt"},  # out of range
        {"path": ""},               # missing
    ]
    assert tool.ranking_from_passages(passages, doc_ids) == ["doc-b", "doc-a"]


def test_parse_args_defaults():
    args = tool.parse_args([])
    assert args.dataset == "scifact"
    assert args.sparse is False
    assert args.top_k == 10
