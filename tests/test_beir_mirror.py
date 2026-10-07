"""Offline tests for the mirror fallback in the BEIR harness.

Three corpora — `robust04`, `trec-news` and `signal1m` — have no UKP zip any more: the files are
gone and the Wayback Machine never captured them. `ensure_dataset` falls back to a public,
non-gated mirror that publishes them in BEIR layout. The mirror's qrels are **four space-separated
TREC columns** while `load_qrels` reads **three tab-separated** ones, and getting that wrong fails
*silently*: a non-numeric doc id trips the `float()` guard so the row is dropped (robust04 would
read as an empty qrels), and a numeric one is read as the score, collapsing every judgement onto the
doc id `"0"` (signal1m). These tests pin the conversion without touching the network.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def _load():
    name = "laya_rag_beir_mirror"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "beir.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_the_three_recovered_datasets_are_the_mirror_set():
    beir = _load()
    assert set(beir.MIRROR_DATASETS) == {"robust04", "trec-news", "signal1m"}


def test_fetch_converts_the_four_column_space_separated_qrels(tmp_path, monkeypatch):
    """The mirror's `qid iter docid rel`, space-separated, becomes the loader's `qid docid rel`."""
    beir = _load()

    # A doc id with no numeric reading, exactly the shape that makes the raw file fail silently.
    raw = (
        "301 0 FBIS3-10082 1\n"
        "301 0 FBIS3-10169 0\n"
        "302 0 LA010190-0001 2\n"
    )

    calls: list[tuple[str, Path]] = []

    def fake_retrieve(url: str, destination):
        calls.append((url, Path(destination)))
        destination = Path(destination)
        if destination.name.endswith(".test.txt"):
            destination.write_text(raw, encoding="utf-8")
        elif destination.name == "queries.jsonl":
            destination.write_text('{"_id": "301", "text": "q"}\n', encoding="utf-8")
        else:
            destination.write_text('{"_id": "FBIS3-10082", "title": "", "text": "d"}\n',
                                   encoding="utf-8")

    monkeypatch.setattr(beir.urllib.request, "urlretrieve", fake_retrieve)
    beir._fetch_mirror_dataset("robust04", tmp_path / "robust04")

    target = tmp_path / "robust04"
    assert (target / "corpus.jsonl").is_file() and (target / "queries.jsonl").is_file()
    assert not (target / "qrels.beir-v1.0.0-robust04.test.txt").exists(), "the raw file is cleaned up"

    qrels = beir.load_qrels(target)
    assert qrels == {
        "301": {"FBIS3-10082": 1.0, "FBIS3-10169": 0.0},
        "302": {"LA010190-0001": 2.0},
    }
    # The failure this guards: reading the raw four-column file would leave the doc id as "0".
    assert "0" not in qrels.get("301", {})


def test_a_numeric_doc_id_is_not_read_as_the_score(tmp_path, monkeypatch):
    """`signal1m`'s doc ids are numeric, so the raw file parses — onto the wrong column."""
    beir = _load()

    def fake_retrieve(url: str, destination):
        destination = Path(destination)
        if destination.name.endswith(".test.txt"):
            destination.write_text("00ecb565 0 638727826393444352 1\n", encoding="utf-8")
        else:
            destination.write_text('{"_id": "x", "text": "y"}\n', encoding="utf-8")

    monkeypatch.setattr(beir.urllib.request, "urlretrieve", fake_retrieve)
    beir._fetch_mirror_dataset("signal1m", tmp_path / "signal1m")

    qrels = beir.load_qrels(tmp_path / "signal1m")
    assert qrels == {"00ecb565": {"638727826393444352": 1.0}}
    assert "0" not in qrels["00ecb565"]
