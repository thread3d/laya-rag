"""Chunking tests.

The chunker previously emitted cascades of tiny overlapping chunks on documents with
irregular paragraph spacing (1606 chunks for 24 small files). These tests pin the
properties that regression violated: bounded size, forward progress, and no
duplicated work.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from laya_rag.chunking import (
    SKIP_DIRS,
    chunk_documents,
    chunk_text,
    load_documents,
    read_document,
)

from .support import document, write_corpus


def test_chunk_text_rejects_invalid_budgets():
    with pytest.raises(ValueError, match="chunk_chars must be positive"):
        chunk_text("hello", chunk_chars=0)
    with pytest.raises(ValueError, match="overlap_chars must be non-negative"):
        chunk_text("hello", chunk_chars=100, overlap_chars=-1)
    with pytest.raises(ValueError, match="must be smaller than chunk_chars"):
        chunk_text("hello", chunk_chars=100, overlap_chars=100)


def test_empty_text_yields_no_chunks():
    assert chunk_text("   \n\n  ") == []


def test_chunks_respect_the_budget():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(30))
    pieces = chunk_text(text, chunk_chars=400, overlap_chars=80)
    assert pieces
    assert max(len(piece) for piece, _, _ in pieces) <= 400


def test_chunk_starts_always_advance():
    """An overlap wider than the chunk must not let the cursor stall or reverse."""
    text = "\n\n".join(f"Para {i} " + "content " * 20 for i in range(40))
    for chunk_chars, overlap in ((200, 190), (400, 80), (120, 0)):
        pieces = chunk_text(text, chunk_chars=chunk_chars, overlap_chars=overlap)
        starts = [start for _, start, _ in pieces]
        assert all(b > a for a, b in zip(starts, starts[1:])), (chunk_chars, overlap)


def test_irregular_spacing_does_not_multiply_chunks():
    """The regression case: mixed short paragraphs must not explode the chunk count."""
    text = "\n\n\n".join(["short line"] * 200)
    pieces = chunk_text(text, chunk_chars=400, overlap_chars=80, min_chunk_chars=40)
    assert len(pieces) < len(text) / 100, f"{len(pieces)} chunks for {len(text)} chars is a cascade"


def test_long_single_paragraph_is_split_on_whitespace():
    text = "word " * 500
    pieces = chunk_text(text, chunk_chars=300, overlap_chars=50)
    assert len(pieces) > 1
    assert all(len(piece) <= 300 for piece, _, _ in pieces)


def test_offsets_index_the_original_text():
    text = "  " + "\n\n".join(f"Block {i} " + "text " * 30 for i in range(12)) + "  "
    body = text.strip()
    for piece, start, end in chunk_text(text, chunk_chars=300, overlap_chars=60):
        assert body[start:end].strip() == piece


def test_min_chunk_merges_short_tail():
    text = "\n\n".join(["alpha " * 30, "beta " * 30, "tail"])
    pieces = chunk_text(text, chunk_chars=250, overlap_chars=40, min_chunk_chars=50)
    assert len(pieces) > 1
    # The tiny trailing fragment is folded into the previous chunk, not emitted alone.
    assert len(pieces[-1][0]) >= 50


def test_chunk_documents_preserves_order_and_ids():
    docs = [document("/a.md", "alpha " * 60, "d1"), document("/b.md", "beta " * 60, "d2")]
    chunks = chunk_documents(docs, chunk_chars=200, overlap_chars=40, min_chunk_chars=20)
    assert {c.doc_id for c in chunks} == {"d1", "d2"}
    per_doc = [c for c in chunks if c.doc_id == "d1"]
    assert [c.index for c in per_doc] == list(range(len(per_doc)))
    assert per_doc[0].chunk_id == "d1:0000"


def test_load_documents_reads_text_and_skips_noise(tmp_path):
    root = write_corpus(tmp_path / "corpus")
    report = load_documents([root])

    names = {path.split("/")[-1] for path in (d.path for d in report.documents)}
    assert names == {"refunds.md", "shipping.md", "security.md"}

    # A directory walk filters by extension silently, so notes.bin is simply absent.
    # A file whose extension is allowed but whose content is unusable is reported.
    reasons = {path.split("/")[-1]: reason for path, reason in report.skipped}
    assert "empty.txt" in reasons
    assert "notes.bin" not in names


def test_extension_allowlist_is_enforced(tmp_path):
    root = write_corpus(tmp_path / "corpus")
    report = load_documents([root], extensions=[".md"])
    assert len(report.documents) == 3


def test_missing_path_is_reported_not_raised(tmp_path):
    report = load_documents([tmp_path / "nope"])
    assert report.documents == []
    assert "does not exist" in report.skipped[0][1]


def test_read_document_rejects_binary(tmp_path):
    target = tmp_path / "blob.dat"
    target.write_bytes(b"\x00\x01\x02")
    assert read_document(target) is None


def test_skip_dirs_exclude_dependency_noise():
    for name in ("node_modules", ".git", "__pycache__", ".venv"):
        assert name in SKIP_DIRS


# -- running counts ----------------------------------------------------------

def test_load_documents_reports_each_file_as_it_is_read(tmp_path):
    """The callback is what lets a long load show progress instead of a bare spinner,
    so the counts have to be the real running ones."""
    corpus = write_corpus(tmp_path / "corpus")
    seen = []
    report = load_documents(
        [corpus],
        on_document=lambda doc, documents, skipped: seen.append(
            (Path(doc.path).name, documents, skipped)
        ),
    )
    assert [documents for _, documents, _ in seen] == [1, 2, 3]
    assert len(seen) == len(report.documents) == 3
    # A file refused along the way is counted too, not hidden from the running total.
    assert seen[-1][2] == len(report.skipped)


def test_chunk_documents_reports_progress_per_document(tmp_path):
    corpus = write_corpus(tmp_path / "corpus")
    documents = load_documents([corpus]).documents
    seen = []
    chunks = chunk_documents(
        documents,
        on_document=lambda done, total, produced: seen.append((done, total, produced)),
    )
    assert [done for done, _, _ in seen] == [1, 2, 3]
    assert all(total == 3 for _, total, _ in seen)
    assert all(produced <= len(chunks) for _, _, produced in seen)
    assert seen[-1][2] == len(chunks)
