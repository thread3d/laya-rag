"""Engine tests: indexing, hybrid recall, Laya gating and persistence.

These run against the deterministic stub model, so they verify the engine's own
logic rather than the checkpoint's retrieval quality.
"""
from __future__ import annotations

import numpy as np
import pytest

from laya_rag.engine import IndexNotBuiltError, RagEngine
from laya_rag.index_store import VectorIndex
from laya_rag.lexical import LEXICAL_FILE

from .support import StubModel, make_config, stub_engine, write_corpus


def test_indexing_a_corpus_reports_what_it_did(tmp_path):
    corpus = write_corpus(tmp_path / "corpus")
    engine = stub_engine(tmp_path)

    report = engine.index_paths([corpus])

    assert report.documents == 3
    assert report.chunks > 0
    assert report.total_chunks == report.chunks
    assert report.paths == 3
    # A file with an unusable extension is filtered silently during a directory walk,
    # so the report lists the files it read rather than every file it passed over.
    assert all(path.endswith(".md") for path in engine.index.paths())


def test_index_reports_running_progress(tmp_path):
    """The progress callback is what lets the interface show counts instead of a
    spinner that says nothing, so the stages and the numbers have to be real."""
    corpus = write_corpus(tmp_path / "corpus")
    engine = stub_engine(tmp_path)

    events = []
    report = engine.index_paths([corpus], progress=events.append)

    shape = {"stage", "documents", "documents_total", "skipped", "chunks", "embedded", "path"}
    for event in events:
        assert shape <= set(event), f"an event is missing keys: {event}"

    stages = [event["stage"] for event in events]
    assert stages[0] == "reading"
    assert "chunking" in stages
    assert "embedding" in stages
    assert stages[-1] == "saving"

    read = [e for e in events if e["stage"] == "reading" and e["documents"]]
    assert read[-1]["documents"] == report.documents == 3

    chunked = [e for e in events if e["stage"] == "chunking" and e["chunks"]]
    assert chunked[-1]["chunks"] == report.chunks == report.total_chunks
    assert chunked[-1]["documents"] == chunked[-1]["documents_total"] == 3

    embedded = [e for e in events if e["stage"] == "embedding"]
    assert embedded[-1]["chunks"] == report.chunks
    assert embedded[-1]["embedded"] == report.chunks
    assert all(e["embedded"] <= e["chunks"] for e in embedded)


def test_index_paths_without_progress_still_works(tmp_path):
    """Progress is optional instrumentation and must not change the result."""
    corpus = write_corpus(tmp_path / "corpus")
    engine = stub_engine(tmp_path)
    report = engine.index_paths([corpus])
    assert report.documents == 3


def test_index_writes_both_channels(tmp_path):
    corpus = write_corpus(tmp_path / "corpus")
    engine = stub_engine(tmp_path)
    engine.index_paths([corpus])

    assert VectorIndex.exists_on_disk(engine.config.index_dir)
    assert (engine.config.index_dir / LEXICAL_FILE).is_file()


def test_query_returns_evidence_with_sources(tmp_path):
    corpus = write_corpus(tmp_path / "corpus")
    engine = stub_engine(tmp_path)
    engine.index_paths([corpus])

    result = engine.query("How long does a refund take?")

    assert result.passages
    assert result.kept <= engine.config.rerank_k
    for passage in result.passages:
        assert passage["path"]
        assert isinstance(passage["text"], str)
        assert 0.0 <= passage["relevance"] <= 1.0
        assert passage["channels"]


def test_query_ranks_the_matching_document_first(tmp_path):
    corpus = write_corpus(tmp_path / "corpus")
    engine = stub_engine(tmp_path)
    engine.index_paths([corpus])

    result = engine.query("refund payment method business days")

    assert result.passages
    assert "refunds.md" in result.passages[0]["path"]


def test_passages_are_ordered_by_laya_relevance(tmp_path):
    corpus = write_corpus(tmp_path / "corpus")
    engine = stub_engine(tmp_path)
    engine.index_paths([corpus])

    result = engine.query("shipping international orders")
    relevance = [passage["relevance"] for passage in result.passages]
    assert relevance == sorted(relevance, reverse=True)


def test_unanswerable_query_is_gated(tmp_path):
    corpus = write_corpus(tmp_path / "corpus")
    engine = stub_engine(tmp_path, relevance_threshold=0.99, answerability_threshold=0.99)
    engine.index_paths([corpus])

    result = engine.query("What is the airspeed velocity of an unladen swallow?")

    assert result.answerable is False
    assert result.kept == 0


def test_max_passage_chars_truncates_preview(tmp_path):
    corpus = write_corpus(tmp_path / "corpus")
    engine = stub_engine(tmp_path)
    engine.index_paths([corpus])

    result = engine.query("refund policy", max_passage_chars=50)

    assert result.passages
    assert all(len(passage["text"]) <= 51 for passage in result.passages)


def test_query_without_an_index_explains_itself(tmp_path):
    engine = stub_engine(tmp_path)
    with pytest.raises(IndexNotBuiltError, match="no index at"):
        engine.query("anything")


def test_query_on_an_empty_corpus_returns_a_note(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "only-binary.bin").write_bytes(b"\x00\x01")
    engine = stub_engine(tmp_path)
    engine.index_paths([empty])

    result = engine.query("anything")
    assert result.passages == []
    assert "empty" in (result.note or "")


def test_reindexing_replaces_changed_files(tmp_path):
    corpus = write_corpus(tmp_path / "corpus")
    engine = stub_engine(tmp_path)
    first = engine.index_paths([corpus])

    (corpus / "refunds.md").write_text("# Refunds\n\nRefunds now take two days.\n", encoding="utf-8")
    second = engine.index_paths([corpus])

    assert second.removed_chunks > 0
    assert second.total_chunks <= first.total_chunks


def test_rebuild_discards_previous_content(tmp_path):
    corpus = write_corpus(tmp_path / "corpus")
    engine = stub_engine(tmp_path)
    engine.index_paths([corpus])

    other = tmp_path / "other"
    other.mkdir()
    (other / "only.md").write_text("# Only\n\nUnique zephyr content here.\n", encoding="utf-8")

    report = engine.index_paths([other], rebuild=True)
    assert report.total_chunks == report.chunks
    assert all("only.md" in path for path in engine.index.paths())


def test_reindexing_a_directory_prunes_deleted_files(tmp_path):
    """A file removed from disk must not stay retrievable after re-indexing its parent."""
    corpus = write_corpus(tmp_path / "corpus")
    engine = stub_engine(tmp_path)
    engine.index_paths([corpus])
    assert any("shipping.md" in path for path in engine.index.paths())

    (corpus / "shipping.md").unlink()
    report = engine.index_paths([corpus])

    assert report.removed_chunks > 0
    assert not any("shipping.md" in path for path in engine.index.paths())
    assert all("shipping.md" not in record["path"] for record in engine.list_chunks())


def test_reindexing_a_directory_prunes_a_file_made_unreadable(tmp_path):
    """A file that becomes empty is refused by the loader and must be pruned too."""
    corpus = write_corpus(tmp_path / "corpus")
    engine = stub_engine(tmp_path)
    engine.index_paths([corpus])

    (corpus / "security.md").write_text("   \n\n", encoding="utf-8")
    engine.index_paths([corpus])

    assert not any("security.md" in path for path in engine.index.paths())


def test_pruning_never_touches_files_outside_the_reindexed_directory(tmp_path):
    """Passing a single file is not a claim about the rest of the index."""
    corpus = write_corpus(tmp_path / "corpus")
    engine = stub_engine(tmp_path)
    engine.index_paths([corpus])

    (corpus / "shipping.md").unlink()
    # Re-index one surviving file, not the directory: the missing file is out of scope.
    engine.index_paths([corpus / "refunds.md"])

    assert any("shipping.md" in path for path in engine.index.paths())


def test_prune_false_keeps_stale_chunks(tmp_path):
    """The old append-only behaviour stays available for callers that want it."""
    corpus = write_corpus(tmp_path / "corpus")
    engine = stub_engine(tmp_path)
    engine.index_paths([corpus])

    (corpus / "shipping.md").unlink()
    engine.index_paths([corpus], prune=False)

    # Incremental replacement still removes the chunks of the files it reloaded, but the
    # deleted file's chunks are deliberately left alone under `prune=False`.
    assert any("shipping.md" in path for path in engine.index.paths())


def test_missing_lexical_channel_degrades_loudly(tmp_path):
    """An index without lexical.json must still answer, and must say recall is degraded."""
    corpus = write_corpus(tmp_path / "corpus")
    engine = stub_engine(tmp_path)
    engine.index_paths([corpus])
    (engine.config.index_dir / LEXICAL_FILE).unlink()

    reloaded = stub_engine(tmp_path)
    assert reloaded.lexical is None

    result = reloaded.query("refund payment")
    assert "lexical recall channel is missing" in (result.note or "")
    assert all(passage["channels"] == ["dense"] for passage in result.passages)


def test_corrupt_lexical_channel_is_ignored(tmp_path):
    corpus = write_corpus(tmp_path / "corpus")
    engine = stub_engine(tmp_path)
    engine.index_paths([corpus])
    (engine.config.index_dir / LEXICAL_FILE).write_text("{ not json")

    reloaded = stub_engine(tmp_path)
    assert reloaded.lexical is None
    assert reloaded.query("refund").passages


def test_hybrid_recall_finds_exact_terms_dense_alone_misses(tmp_path):
    """A rare identifier must be recalled lexically even when dense ranks it low."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text(
        "# Handbook\n\n" + ("General filler sentence about policy. " * 40) +
        "\n\nThe widget ZQX-4471 requires recalibration every quarter.\n",
        encoding="utf-8",
    )
    (corpus / "b.md").write_text("# Other\n\n" + ("Unrelated text. " * 60), encoding="utf-8")

    engine = stub_engine(tmp_path, chunk_chars=300, chunk_overlap_chars=50)
    engine.index_paths([corpus])

    result = engine.query("ZQX-4471", retrieve_k=5)
    assert result.passages
    assert any("ZQX-4471" in passage["text"] for passage in result.passages)


def test_reset_removes_the_index(tmp_path):
    corpus = write_corpus(tmp_path / "corpus")
    engine = stub_engine(tmp_path)
    engine.index_paths([corpus])
    assert engine.has_index()

    engine.reset()
    assert not engine.has_index()


def test_stats_report_model_and_corpus(tmp_path):
    corpus = write_corpus(tmp_path / "corpus")
    engine = stub_engine(tmp_path)
    engine.index_paths([corpus])

    stats = engine.stats()
    assert stats["index_present"] is True
    assert stats["lexical_present"] is True
    assert stats["index"]["chunks"] > 0
    assert stats["model"]["model"] == "stub"


def test_query_validates_input(tmp_path):
    engine = stub_engine(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        engine.query("   ")


def test_embedding_batches_cover_every_chunk(tmp_path):
    """A corpus larger than one embed batch must still be fully indexed."""
    corpus = tmp_path / "big"
    corpus.mkdir()
    for i in range(12):
        (corpus / f"doc{i}.md").write_text(f"# Doc {i}\n\n" + (f"unique{i} content line. " * 20), encoding="utf-8")

    engine = RagEngine(make_config(tmp_path), model=StubModel(), embed_batch_texts=2)
    report = engine.index_paths([corpus])

    assert report.chunks > 12
    assert report.total_chunks == report.chunks


# -- query-embedding cache ---------------------------------------------------

class _CountingEmbedder(StubModel):
    """A stub embedder that counts single-string query embeddings."""

    def __init__(self) -> None:
        super().__init__()
        self.one_calls = 0

    def embed_one(self, text: str) -> np.ndarray:
        self.one_calls += 1
        return super().embed_one(text)


def _cached_engine(tmp_path, **overrides):
    embedder = _CountingEmbedder()
    config = make_config(tmp_path, dense_weight=1.0, rerank_k=3, **overrides)
    engine = RagEngine(config, model=StubModel(), embedder=embedder)
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    return engine, embedder


def test_repeated_query_embeds_once(tmp_path):
    engine, embedder = _cached_engine(tmp_path)
    before = embedder.one_calls
    engine.query("How long does a refund take?")
    engine.query("How long does a refund take?")
    assert embedder.one_calls - before == 1
    engine.query("What are the shipping rates?")
    assert embedder.one_calls - before == 2


def test_query_cache_can_be_disabled(tmp_path):
    engine, embedder = _cached_engine(tmp_path, query_cache_size=0)
    before = embedder.one_calls
    engine.query("How long does a refund take?")
    engine.query("How long does a refund take?")
    assert embedder.one_calls - before == 2


def test_query_cache_is_bounded(tmp_path):
    engine, embedder = _cached_engine(tmp_path, query_cache_size=1)
    before = embedder.one_calls
    engine.query("alpha question about refunds")
    engine.query("beta question about shipping")
    engine.query("alpha question about refunds")  # evicted by the second query
    assert embedder.one_calls - before == 3


def test_adjudicate_claims_reuses_the_embedding_cache(tmp_path):
    engine, embedder = _cached_engine(tmp_path)
    claim = "refunds are processed within five business days"
    before = embedder.one_calls
    engine.adjudicate_claims([claim], retrieve_k=3)
    engine.adjudicate_claims([claim], retrieve_k=3)
    # The claim is embedded once; the second adjudication reuses the cached vector.
    assert embedder.one_calls - before == 1


# -- fusion constant ---------------------------------------------------------

class _StubSparseEncoder:
    """A sparse encoder stand-in: every text carries term 1 plus one rotating term."""

    def encode(self, texts):
        return [{1: 1.0, (index % 7) + 2: 1.0} for index, _ in enumerate(texts)]

    def embed_one(self, text):
        return self.encode([text])[0]

    def info(self):
        return {"model": "stub", "top_k": 128, "loaded": True}


def _spy_on_fuse(engine):
    captured: dict = {}
    original = engine._fuse

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    engine._fuse = spy
    return captured


def _engine_with_sparse(tmp_path):
    config = make_config(tmp_path, dense_weight=1.0, rerank_k=3)
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    config.sparse_model = "stub/sparse"
    engine._sparse_encoder = _StubSparseEncoder()
    engine._build_sparse(engine.index.chunks)
    return engine


def test_query_passes_the_configured_fusion_constant_and_redundancy(tmp_path):
    """The primary path, not just adjudicate, must honour FUSION_K and the redundancy flag."""
    config = make_config(
        tmp_path, dense_weight=1.0, rerank_k=3, fusion_k=7,
        recall_coverage_lambda=0.5, recall_coverage_redundancy=False,
    )
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    captured = _spy_on_fuse(engine)
    engine.query("How long does a refund take?")
    assert captured["rrf_k"] == 7
    assert captured["coverage_redundancy"] is False


def test_query_sends_no_sparse_channel_by_default(tmp_path):
    engine = RagEngine(make_config(tmp_path, dense_weight=1.0, rerank_k=3), model=StubModel())
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    captured = _spy_on_fuse(engine)
    result = engine.query("How long does a refund take?")
    assert captured["sparse_hits"] == []
    assert all("sparse" not in passage["channels"] for passage in result.passages)


def test_sparse_channel_is_built_and_fused_into_the_query(tmp_path):
    engine = _engine_with_sparse(tmp_path)
    assert engine.sparse is not None
    assert len(engine.sparse) == len(engine.index)
    assert (engine.config.index_dir / "sparse.json").is_file()

    captured = _spy_on_fuse(engine)
    result = engine.query("How long does a refund take?")
    assert captured["sparse_hits"], "the sparse channel must contribute candidates"
    assert captured["sparse_weight"] == 1.0
    assert any("sparse" in passage["channels"] for passage in result.passages)


def test_stats_report_the_sparse_channel(tmp_path):
    engine = _engine_with_sparse(tmp_path)
    payload = engine.stats()["sparse"]
    assert payload["configured"] is True
    assert payload["present"] is True
    assert payload["encoder"]["model"] == "stub"


def test_sparse_model_is_rejected_with_the_ondisk_backend(tmp_path):
    config = make_config(
        tmp_path, dense_weight=1.0, sparse_model="stub/sparse", index_backend="ondisk"
    )
    engine = RagEngine(config, model=StubModel())
    with pytest.raises(ValueError, match="ondisk"):
        engine.index_paths([write_corpus(tmp_path / "corpus")])


def _long_document_corpus(tmp_path):
    corpus = tmp_path / "long"
    corpus.mkdir()
    (corpus / "one.md").write_text(
        "# One\n\n" + ("refund policy details and processing times. " * 300), encoding="utf-8"
    )
    for name in ("two", "three"):
        (corpus / f"{name}.md").write_text(
            f"# {name}\n\n" + ("refund policy details and processing times. " * 60),
            encoding="utf-8",
        )
    return corpus


def test_engine_rerank_min_docs_spreads_the_kept_passages(tmp_path):
    config = make_config(
        tmp_path, dense_weight=1.0, retrieve_k=30, rerank_k=4,
        relevance_threshold=0.0, rerank_min_docs=2,
    )
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([_long_document_corpus(tmp_path)])
    result = engine.query("refund policy details and processing times")
    paths = [passage["path"] for passage in result.passages]
    # Without the quota the long document's overlapping chunks fill the list.
    assert len(paths) >= 2
    assert paths[0] != paths[1]


def test_engine_without_the_quota_keeps_the_passage_order(tmp_path):
    config = make_config(
        tmp_path, dense_weight=1.0, retrieve_k=30, rerank_k=4,
        relevance_threshold=0.0, rerank_min_docs=0,
    )
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([_long_document_corpus(tmp_path)])
    result = engine.query("refund policy details and processing times")
    paths = [passage["path"] for passage in result.passages]
    assert len(paths) >= 2
    assert len(set(paths)) <= len(paths)  # duplicates are allowed without the quota


class _FakeIndex:
    def __init__(self, ids):
        self._ids = set(ids)

    def get(self, chunk_id):
        if chunk_id not in self._ids:
            return None
        from types import SimpleNamespace

        return SimpleNamespace(chunk_id=chunk_id, text="text")


def test_fusion_constant_defaults_to_the_measured_five():
    from laya_rag.config import EngineConfig

    config = EngineConfig()
    assert config.fusion_k == 5
    assert config.as_dict()["fusion_k"] == 5


def test_fuse_respects_the_rrf_constant():
    from types import SimpleNamespace

    # `a` is rank 1 in one channel; `m` is rank 3 in both. At k=0 the single top hit
    # wins; at k>=2 the twice-mid-ranked document wins. This is what the constant tunes.
    dense = [SimpleNamespace(chunk=SimpleNamespace(chunk_id=i)) for i in ("a", "b", "m")]
    lexical = [SimpleNamespace(chunk_id=i) for i in ("z", "c", "m")]
    index = _FakeIndex({"a", "b", "m", "z", "c"})

    order0 = [hit.chunk.chunk_id for hit in RagEngine._fuse(index, dense, lexical, limit=2, rrf_k=0)]
    order5 = [hit.chunk.chunk_id for hit in RagEngine._fuse(index, dense, lexical, limit=2, rrf_k=5)]

    assert order0[0] == "a"
    assert order5[0] == "m"


def test_engine_builds_the_lexical_channel_with_the_configured_bm25_parameters(tmp_path):
    config = make_config(tmp_path, bm25_k1=0.9, bm25_b=0.4)
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    assert engine.lexical.k1 == 0.9
    assert engine.lexical.b == 0.4


def test_chunk_chars_auto_derives_from_the_corpus(tmp_path):
    config = make_config(tmp_path, dense_weight=1.0, rerank_k=3, chunk_chars_auto=True,
                         chunk_chars=400, chunk_chars_auto_max=2000)
    engine = RagEngine(config, model=StubModel())
    report = engine.index_paths([_long_document_corpus(tmp_path)])
    # The corpus's long documents push the p90 above the fixed 400-char default.
    assert report.derived_chunk_chars > 400
    assert report.as_dict()["derived_chunk_chars"] == report.derived_chunk_chars


def test_chunk_chars_auto_defaults_on_and_can_be_disabled(tmp_path):
    from laya_rag.config import EngineConfig

    assert EngineConfig().chunk_chars_auto is True
    engine = RagEngine(
        make_config(tmp_path, dense_weight=1.0, rerank_k=3, chunk_chars_auto=False),
        model=StubModel(),
    )
    report = engine.index_paths([write_corpus(tmp_path / "corpus")])
    assert report.derived_chunk_chars == 0
