"""`--bm25f --lexical-backend ondisk` produces the BM25F row over the on-disk backend.

The adapter is parity-tested in `test_beir_bm25f_ondisk.py`; this is the wiring — one flag switching
the backend, with the row labelled so a reader can tell which one produced it. FEVER is the corpus
that needs it, and FEVER is too large to run in a test, so the switch is exercised on a four-document
corpus where both paths fit.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def _load_beir():
    name = "laya_rag_beir_bm25f_cli"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "beir.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


beir = _load_beir()


def _dataset(root: Path, name: str) -> None:
    directory = root / name
    (directory / "qrels").mkdir(parents=True, exist_ok=True)
    (directory / "corpus.jsonl").write_text(
        '{"_id": "d1", "title": "Refunds", "text": "Refunds take five business days."}\n'
        '{"_id": "d2", "title": "Shipping", "text": "Shipping is free over fifty dollars."}\n'
        '{"_id": "d3", "title": "Refunds", "text": "Refund requests need a receipt."}\n',
        encoding="utf-8",
    )
    (directory / "queries.jsonl").write_text(
        '{"_id": "q1", "text": "how long do refunds take"}\n'
        '{"_id": "q2", "text": "is shipping free"}\n',
        encoding="utf-8",
    )
    (directory / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n", encoding="utf-8"
    )


def _run(tmp_path: Path, backend: str) -> dict:
    root = tmp_path / "beir"
    root.mkdir(exist_ok=True)
    _dataset(root, "tiny")
    out = tmp_path / f"report-{backend}"
    assert beir.main([
        "--datasets", "tiny",
        "--data-root", str(root),
        "--retrievers", "bm25-engine,bm25-title",
        "--title-boost", "2",
        "--bm25f",
        "--lexical-backend", backend,
        "--out", str(out),
    ]) == 0
    return json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _no_heavy_lock(monkeypatch):
    monkeypatch.setenv("LAYA_RAG_ALLOW_CONCURRENT", "1")


def test_the_flag_switches_the_backend_and_says_so(tmp_path):
    ondisk = _run(tmp_path, "ondisk")
    row = ondisk["datasets"]["tiny"]["retrievers"]["bm25f"]
    assert "on-disk backend" in row["note"]

    memory = _run(tmp_path, "memory")
    assert "on-disk backend" not in memory["datasets"]["tiny"]["retrievers"]["bm25f"]["note"]


def test_the_two_backends_produce_the_same_row(tmp_path):
    """The switch changes where the index lives, not what the retriever is."""
    ondisk = _run(tmp_path, "ondisk")["datasets"]["tiny"]
    memory = _run(tmp_path, "memory")["datasets"]["tiny"]
    assert ondisk["retrievers"]["bm25f"]["nDCG@10"] == pytest.approx(
        memory["retrievers"]["bm25f"]["nDCG@10"]
    )
    assert ondisk["per_query_nDCG@10"]["bm25f"] == memory["per_query_nDCG@10"]["bm25f"]
