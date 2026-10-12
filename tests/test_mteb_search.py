"""Tests for the MTEB v2 ``SearchProtocol`` adapter.

Two things need pinning. The plumbing — corpus rows to positional files, and returned passage
paths back to corpus ids — is what makes a score belong to the right document, and a silent
off-by-one there would corrupt every MTEB number without a metric flagging it. The conformance
itself is pinned too: the adapter must satisfy MTEB's runtime-checkable ``SearchProtocol`` and
``ModelMeta.load_model`` must assign the metadata onto the object it builds.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def _load():
    name = "laya_rag_mteb_search"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "mteb_search.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


m = _load()


class _Dataset:
    """The parts of a HuggingFace dataset the adapter uses: columns and length."""

    def __init__(self, **columns):
        self._columns = columns
        self.column_names = list(columns)

    def __getitem__(self, key):
        return self._columns[key]


class _FakeEngine:
    def __init__(self, passages):
        self._passages = passages
        self.indexed: list = []
        self.queries: list = []

    def index_paths(self, paths, rebuild=False):
        self.indexed = list(paths)
        return SimpleNamespace(documents=1, chunks=1)

    def query(self, text, top_k=None, retrieve_k=None, relevance_threshold=None):
        self.queries.append((text, top_k, retrieve_k, relevance_threshold))
        return SimpleNamespace(passages=self._passages)


# ── corpus plumbing ──────────────────────────────────────────────────────

def test_corpus_rows_prepends_a_title_when_there_is_one():
    corpus = _Dataset(id=["a"], text=["body"], title=["Head"])
    assert list(m.corpus_rows(corpus)) == [("a", "Head\n\nbody")]


def test_corpus_rows_without_a_title_is_just_the_text():
    corpus = _Dataset(id=[1], text=["body"])
    assert list(m.corpus_rows(corpus)) == [("1", "body")]


def test_write_corpus_names_by_position_and_returns_the_id_order(tmp_path):
    doc_ids = m.write_corpus([("doc-a", "one"), ("doc-b", "two")], tmp_path)
    assert doc_ids == ["doc-a", "doc-b"]
    assert (tmp_path / "000000.txt").read_text() == "one"
    assert (tmp_path / "000001.txt").read_text() == "two"


# ── index and search ─────────────────────────────────────────────────────

def test_index_writes_the_corpus_and_builds_the_engine(tmp_path):
    engine = _FakeEngine([])
    adapter = m.LayaRagSearch(engine=engine, workdir=tmp_path)
    adapter.index(_Dataset(id=["d0", "d1"], text=["zero", "one"]))
    assert engine.indexed == [tmp_path]
    assert adapter._doc_ids == ["d0", "d1"]
    assert (tmp_path / "000001.txt").read_text() == "one"


def test_search_maps_passage_paths_back_to_corpus_ids(tmp_path):
    engine = _FakeEngine([
        {"path": str(tmp_path / "000001.txt"), "relevance": 0.9},
        {"path": str(tmp_path / "000000.txt"), "relevance": 0.4},
    ])
    adapter = m.LayaRagSearch(engine=engine, workdir=tmp_path)
    adapter.index(_Dataset(id=["d0", "d1"], text=["zero", "one"]))
    results = adapter.search(_Dataset(id=["q1"], text=["question"]), top_k=10)
    assert results == {"q1": {"d1": 0.9, "d0": 0.4}}


def test_search_deduplicates_a_document_to_its_best_score(tmp_path):
    engine = _FakeEngine([
        {"path": str(tmp_path / "000000.txt"), "relevance": 0.2},
        {"path": str(tmp_path / "000000.txt"), "relevance": 0.8},
    ])
    adapter = m.LayaRagSearch(engine=engine, workdir=tmp_path)
    adapter.index(_Dataset(id=["d0"], text=["zero"]))
    assert adapter.search(_Dataset(id=["q"], text=["q"]), top_k=5) == {"q": {"d0": 0.8}}


def test_search_widens_the_pool_and_does_not_gate(tmp_path):
    """A retrieval benchmark scores order; the gate is measured separately."""
    engine = _FakeEngine([])
    adapter = m.LayaRagSearch(engine=engine, workdir=tmp_path)
    adapter.index(_Dataset(id=["d0"], text=["x"]))
    adapter.search(_Dataset(id=["q"], text=["q"]), top_k=100)
    text, top_k, retrieve_k, threshold = engine.queries[0]
    assert top_k == 100 and retrieve_k >= 100 and threshold == 0.0


def test_search_filters_to_top_ranked_candidates_when_given(tmp_path):
    engine = _FakeEngine([
        {"path": str(tmp_path / "000000.txt"), "relevance": 0.9},
        {"path": str(tmp_path / "000001.txt"), "relevance": 0.5},
    ])
    adapter = m.LayaRagSearch(engine=engine, workdir=tmp_path)
    adapter.index(_Dataset(id=["d0", "d1"], text=["a", "b"]))
    results = adapter.search(
        _Dataset(id=["q"], text=["q"]), top_k=5, top_ranked={"q": ["d1"]}
    )
    assert results == {"q": {"d1": 0.5}}


def test_a_passage_that_is_not_a_positional_file_is_ignored(tmp_path):
    engine = _FakeEngine([
        {"path": "not-a-number.txt", "relevance": 0.9},
        {"path": str(tmp_path / "999999.txt"), "relevance": 0.9},
    ])
    adapter = m.LayaRagSearch(engine=engine, workdir=tmp_path)
    adapter.index(_Dataset(id=["d0"], text=["a"]))
    assert adapter.search(_Dataset(id=["q"], text=["q"]), top_k=5) == {"q": {}}


def test_close_removes_the_temporary_directory():
    adapter = m.LayaRagSearch(engine=_FakeEngine([]))
    adapter.index(_Dataset(id=["d0"], text=["a"]))
    directory = adapter._directory()
    assert directory.is_dir()
    adapter.close()
    assert not directory.exists()


def test_retrieve_k_floor_is_configurable(tmp_path):
    engine = _FakeEngine([])
    adapter = m.LayaRagSearch(engine=engine, workdir=tmp_path, retrieve_k=500)
    adapter.index(_Dataset(id=["d0"], text=["a"]))
    adapter.search(_Dataset(id=["q"], text=["q"]), top_k=10)
    assert engine.queries[0][2] == 500


def test_the_loader_ignores_arguments_the_adapter_does_not_take():
    adapter = m.load_laya_rag_search("laya-rag/engine-multilingual", revision=None,
                                     device=None, experiment_kwargs={"x": 1}, nonsense=True)
    assert isinstance(adapter, m.LayaRagSearch)


# ── MTEB conformance ─────────────────────────────────────────────────────

def _mteb(tmp_path):
    os.environ["MTEB_CACHE"] = str(tmp_path / "mteb-cache")
    pytest.importorskip("mteb")
    from mteb.models.models_protocols import SearchProtocol

    return SearchProtocol


def test_adapter_satisfies_the_search_protocol(tmp_path):
    SearchProtocol = _mteb(tmp_path)
    adapter = m.LayaRagSearch(engine=_FakeEngine([]))
    assert isinstance(adapter, SearchProtocol)


def test_model_meta_loads_the_adapter_and_sets_the_metadata(tmp_path):
    _mteb(tmp_path)
    meta = m.build_model_meta()
    assert meta.name == m.MODEL_NAME
    assert "hybrid" in meta.model_type
    loaded = meta.load_model()
    assert isinstance(loaded, m.LayaRagSearch)
    # `load_model` assigns a copy, so identity is not the contract; equality is.
    assert loaded.mteb_model_meta == meta


# ── progress, and the smoke cap that must never be read as a score ───────

def test_progress_line_reports_a_rate_and_an_estimate():
    line = m.progress_line(50, 200, 100.0)
    assert "50/200" in line
    assert "2.00s/query" in line
    assert "5 min left" in line


def test_progress_line_survives_having_done_nothing():
    assert "0/10" in m.progress_line(0, 10, 0.0)


def test_progress_line_of_a_finished_pass_has_no_time_left():
    assert "0 min left" in m.progress_line(10, 10, 20.0)


def test_the_progress_interval_and_the_smoke_cap_are_parameters():
    """Both exist so a long search can be watched and a rate measured before committing to it."""
    from laya_rag.config import EngineConfig

    adapter = m.LayaRagSearch(config=EngineConfig(), progress_every=5, limit_queries=3)
    assert adapter.progress_every == 5
    assert adapter.limit_queries == 3
    assert m.LayaRagSearch(config=EngineConfig(), progress_every=0).progress_every == 0
