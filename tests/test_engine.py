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
