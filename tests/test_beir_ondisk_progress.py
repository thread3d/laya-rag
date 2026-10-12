"""The on-disk BM25 build reports while it works.

`OnDiskBM25.build` prints nothing between "building the on-disk lexical index" and the first
query. At BioASQ's 14.9M documents that is the longest single wait in the harness, and the first
2.7M-document run sat silent long enough to look hung — the same failure the MTEB side of this
repository already paid four rounds for: a phase that reports nothing is indistinguishable from a
phase that is stuck.
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
    name = "laya_rag_beir_ondisk_progress"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "beir.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


beir = _load_beir()


def test_the_wrapper_reports_at_the_interval_and_not_before(capsys):
    documents = [(str(i), f"text {i}") for i in range(250_002)]
    consumed = list(beir._with_progress(documents, len(documents), every=250_000))
    out = capsys.readouterr().out
    assert len(consumed) == len(documents)          # nothing is dropped or duplicated
    assert consumed == documents
    assert "250000/250002" in out
    assert out.count("on-disk index:") == 1         # 250,002 documents is one interval


def test_an_unknown_total_still_reports(capsys):
    list(beir._with_progress(iter([("a", "x")]), 0, every=1))
    assert "1/0 documents (?)" in capsys.readouterr().out


def test_progress_can_be_switched_off(capsys):
    list(beir._with_progress(iter([("a", "x")] * 3), 3, every=0))
    assert capsys.readouterr().out == ""
