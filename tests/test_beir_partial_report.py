"""A long multi-dataset run must not lose the datasets it has already finished.

`tools/beir.py` writes its report once, at the end. That is invisible in a five-minute run and
destructive in an hours-long one: a sweep that completed Robust04 and was stopped during TREC-NEWS
would have discarded Robust04's rows with the kill. The report is now written after every dataset,
and this pins the property by making a later dataset fail.
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
    name = "laya_rag_beir_partial"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "beir.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


beir = _load_beir()


def _dataset(root: Path, name: str) -> None:
    directory = root / name
    (directory / "qrels").mkdir(parents=True)
    (directory / "corpus.jsonl").write_text(
        '{"_id": "d1", "title": "Refunds", "text": "Refunds take five business days."}\n'
        '{"_id": "d2", "title": "Shipping", "text": "Shipping is free over fifty dollars."}\n',
        encoding="utf-8",
    )
    (directory / "queries.jsonl").write_text(
        '{"_id": "q1", "text": "how long do refunds take"}\n', encoding="utf-8"
    )
    (directory / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8"
    )


def test_a_later_dataset_failing_keeps_the_earlier_one(tmp_path, capsys, monkeypatch):
    # The tool takes the heavy-job lock; the test is not a heavy job, so it opts out rather than
    # failing when a real run happens to be in flight.
    monkeypatch.setenv("LAYA_RAG_ALLOW_CONCURRENT", "1")
    root = tmp_path / "beir"
    root.mkdir()
    _dataset(root, "tiny")
    out = tmp_path / "report"

    with pytest.raises(Exception):
        beir.main([
            "--datasets", "tiny,not-a-dataset",
            "--data-root", str(root),
            "--retrievers", "bm25-engine",
            "--out", str(out),
        ])

    written = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert "tiny" in written["datasets"]
    assert "bm25-engine" in written["datasets"]["tiny"]["retrievers"]
    assert "written:" in capsys.readouterr().out


def test_the_rows_of_a_finished_dataset_are_scored_and_persisted(tmp_path, monkeypatch):
    """The control: with every dataset present, the report is complete and the row is real."""
    monkeypatch.setenv("LAYA_RAG_ALLOW_CONCURRENT", "1")
    root = tmp_path / "beir"
    root.mkdir()
    _dataset(root, "tiny")
    out = tmp_path / "report"
    assert beir.main([
        "--datasets", "tiny",
        "--data-root", str(root),
        "--retrievers", "bm25-engine",
        "--out", str(out),
    ]) == 0
    written = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert written["datasets"]["tiny"]["retrievers"]["bm25-engine"]["nDCG@10"] == pytest.approx(1.0)
