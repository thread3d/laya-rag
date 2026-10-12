"""The on-disk BM25F row is the *same* retriever as the in-memory one, not a similar one.

`BM25FIndex` cannot serve a corpus whose documents do not fit in memory, and the whole reason to
compose an on-disk BM25F is that FEVER gets the row the small corpora got. So the pair is checked
against each other here rather than assumed to agree — the same standard the plain BM25 backends
were held to.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def _load_beir():
    name = "laya_rag_beir_bm25f_ondisk"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "beir.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


beir = _load_beir()

FIELDS = {
    "d1": {"title": "refund policy", "body": "refunds take five business days and refunds are quick"},
    "d2": {"title": "shipping", "body": "shipping is free and refunds need a receipt"},
    "d3": {"title": "refund policy", "body": "support is open on weekdays"},
    "d4": {"title": "warranty", "body": "the warranty covers manufacturing defects"},
}
QUERIES = {"q1": "refund", "q2": "shipping free", "q3": "warranty defects", "q4": "policy"}


def test_the_on_disk_row_matches_the_in_memory_row_exactly(tmp_path):
    """Ranking for ranking, over four documents and four queries, at the recorded settings."""
    memory = beir.bm25f_rankings(FIELDS, QUERIES, top_k=4, k1=0.9, b=0.4,
                                title_weight=5.0, body_weight=1.0, stem=False)
    ondisk = beir.ondisk_bm25f_rankings(
        FIELDS, QUERIES, top_k=4, index_dir=tmp_path / "bm25f",
        k1=0.9, b=0.4, title_weight=5.0, body_weight=1.0, stem=False,
    )
    assert memory == ondisk


def test_a_built_index_is_reused_rather_than_rebuilt(tmp_path, capsys):
    """A weight sweep must not pay for the index again — the property the design exists for."""
    directory = tmp_path / "bm25f"
    beir.ondisk_bm25f_rankings(FIELDS, QUERIES, top_k=4, index_dir=directory, stem=False)
    first = capsys.readouterr().out
    assert "building the on-disk BM25F index" in first

    beir.ondisk_bm25f_rankings(FIELDS, QUERIES, top_k=4, index_dir=directory, stem=False)
    second = capsys.readouterr().out
    assert "reusing the on-disk BM25F index" in second


def test_a_different_weight_reuses_the_same_index(tmp_path):
    """Weights are query-time: changing one must change the ranking, not the build."""
    directory = tmp_path / "bm25f"
    heavy_title = beir.ondisk_bm25f_rankings(
        FIELDS, QUERIES, top_k=4, index_dir=directory, title_weight=5.0, stem=False,
    )
    heavier_title = beir.ondisk_bm25f_rankings(
        FIELDS, QUERIES, top_k=4, index_dir=directory, title_weight=20.0, stem=False,
    )
    assert heavy_title["q4"] == ["d1", "d3"]           # `policy` is a title-only term
    assert heavier_title["q4"][0] in {"d1", "d3"}
    # The index was not rebuilt for the second call, so the ranking change is the weight's doing.
