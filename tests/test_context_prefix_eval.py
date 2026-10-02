"""Offline tests for the structural context prefix evaluation."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "laya_rag_context_prefix_eval", ROOT / "tools" / "context_prefix_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load()


def _documents():
    from laya_rag.chunking import Document

    return [
        Document(doc_id="a", path="/corpus/alpha.md",
                 text="# Alpha\n\nNothing relevant is written in this file at all.\n"),
        Document(doc_id="z", path="/corpus/zebra_handbook.md",
                 text="# Handbook\n\nGeneral procedures for the team are described here.\n"),
    ]


def _queries():
    import evalkit

    return evalkit.parse_queries(
        '{"id":"q-zebra","query":"zebra","gold":["zebra_handbook.md"],"category":"topical"}'
    )


def test_prefix_makes_a_path_only_query_retrievable():
    documents, queries = _documents(), _queries()
    off = tool.evaluate(documents, queries, context_prefix=False)
    on = tool.evaluate(documents, queries, context_prefix=True)
    # Without the prefix the term exists nowhere in the indexed text.
    assert off["recall"] == 0.0
    # The file name is part of the breadcrumb, so the chunk becomes retrievable.
    assert on["recall"] == 1.0
    assert on["mrr"] == 1.0


def test_evaluate_reports_per_query_ranks():
    result = tool.evaluate(_documents(), _queries(), context_prefix=True)
    assert result["per_query"][0]["id"] == "q-zebra"
    assert result["per_query"][0]["rank"] == 1


def test_render_markdown_includes_the_per_query_table():
    documents, queries = _documents(), _queries()
    payload = {
        "_settings": {"corpus": "tiny", "documents": 2, "k": 10},
        "without_prefix": tool.evaluate(documents, queries, context_prefix=False),
        "with_prefix": tool.evaluate(documents, queries, context_prefix=True),
        "delta": {"recall": 1.0, "mrr": 1.0, "precision@10": 1.0},
    }
    text = tool.render_markdown(payload)
    assert "| mrr |" in text
    assert "q-zebra" in text
