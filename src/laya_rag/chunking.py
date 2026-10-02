"""Document loading and chunking.

Laya ships no ingestion path of its own, so this module owns both halves: reading
text files off disk and splitting them into retrieval-sized pieces. Chunking is
paragraph-aware with a character overlap, which keeps a sentence from being cut in
half at a chunk boundary while staying simple enough to reason about.
"""
from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

DEFAULT_EXTENSIONS: tuple[str, ...] = (
    ".md", ".markdown", ".mdx", ".txt", ".rst", ".org",
    ".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".sh", ".bash", ".zsh", ".sql", ".graphql",
    ".html", ".htm", ".css", ".scss",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".rs", ".go", ".java", ".kt",
    ".rb", ".php", ".cs", ".swift", ".scala", ".lua", ".pl", ".r", ".jl",
    ".vue", ".svelte", ".astro", ".ex", ".exs", ".erl", ".hs", ".ml",
)

# Directories that never hold user knowledge and are expensive or pointless to walk.
SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "env", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    "dist", "build", ".next", ".nuxt", "target", ".gradle", ".idea",
    ".hf-cache", ".pip-cache", "models", ".cache", "site-packages",
})

MAX_FILE_BYTES = 2 * 1024 * 1024
_BINARY_SNIFF_BYTES = 8192


@dataclass(frozen=True)
class Document:
    """A text file read into memory."""

    doc_id: str
    path: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """A retrievable slice of a :class:`Document`."""

    chunk_id: str
    doc_id: str
    path: str
    text: str
    index: int
    start_char: int
    end_char: int


@dataclass
class LoadReport:
    """What a load attempt did, including what it refused and why."""

    documents: list[Document] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    #: Paths whose text carried invisible payloads, with the flag kinds found. Advisory:
    #: the text is still indexed after sanitisation, but the caller can surface it.
    sanitized: list[tuple[str, list[str]]] = field(default_factory=list)

    def skip(self, path: str, reason: str) -> None:
        self.skipped.append((path, reason))

    def note_sanitized(self, path: str, flags: Sequence[str]) -> None:
        if flags:
            self.sanitized.append((str(path), list(flags)))


def _looks_binary(blob: bytes) -> bool:
    if b"\x00" in blob[:_BINARY_SNIFF_BYTES]:
        return True
    # A high proportion of non-text control bytes means it is not prose or source.
    sample = blob[:_BINARY_SNIFF_BYTES]
    if not sample:
        return False
    control = sum(1 for byte in sample if byte < 9 or (13 < byte < 32))
    return control / len(sample) > 0.10


#: Characters used to smuggle instructions past a human reviewer: zero-width joiners and
#: spaces, bidi overrides, word-joiners, and the byte-order mark. They are removed rather
#: than escaped, because no indexed prose or source file needs them.
_INVISIBLE = re.compile(
    "[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]"
)

#: A long run of base64/hex-looking characters. Legitimate in a lockfile or a hash, so it
#: is *flagged*, never removed.
_LONG_ENCODED_BLOB = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")
_LONG_HEX_BLOB = re.compile(r"\b[0-9a-fA-F]{200,}\b")


def sanitize_text(text: str, *, normalize_unicode: bool = True) -> str:
    """Fold text to a canonical form and strip invisible control characters.

    NFKC folds compatibility forms (full-width digits, ligatures) so a term cannot be
    hidden by spelling it in a different width, and the invisible-character strip removes
    the zero-width/bidi payload class outright. Both change bytes, so a caller that must
    preserve source exactly can pass ``normalize_unicode=False`` and lose only the
    invisible-character removal.
    """
    folded = unicodedata.normalize("NFKC", str(text)) if normalize_unicode else str(text)
    return _INVISIBLE.sub("", folded)


def invisible_payload_flags(text: str) -> list[str]:
    """Advisory flags for the invisible/encoded payload class.

    Deterministic and near-zero false positive on the invisible class; the encoded-blob
    class is advisory because hashes and lockfiles legitimately contain long blobs.
    """
    flags: list[str] = []
    if _INVISIBLE.search(str(text)):
        flags.append("invisible_control_characters")
    if _LONG_ENCODED_BLOB.search(str(text)):
        flags.append("long_encoded_blob")
    if _LONG_HEX_BLOB.search(str(text)):
        flags.append("long_hex_blob")
    return flags


def _doc_id(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]


def iter_files(
    roots: Sequence[Path],
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
    max_file_bytes: int = MAX_FILE_BYTES,
    report: LoadReport | None = None,
) -> Iterator[Path]:
    """Yield candidate text files under ``roots``, skipping noise directories."""
    allowed = {ext.lower() for ext in extensions}
    for root in roots:
        root = Path(root).expanduser()
        if root.is_file():
            if root.suffix.lower() in allowed:
                yield root
            elif report is not None:
                report.skip(str(root), f"extension {root.suffix!r} not in the allowed set")
            continue
        if not root.is_dir():
            if report is not None:
                report.skip(str(root), "path does not exist")
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            for filename in sorted(filenames):
                candidate = Path(dirpath) / filename
                if candidate.suffix.lower() not in allowed:
                    continue
                try:
                    size = candidate.stat().st_size
                except OSError as exc:
                    if report is not None:
                        report.skip(str(candidate), f"stat failed: {exc}")
                    continue
                if size == 0:
                    if report is not None:
                        report.skip(str(candidate), "empty file")
                    continue
                if size > max_file_bytes:
                    if report is not None:
                        report.skip(str(candidate), f"{size} bytes exceeds the {max_file_bytes} byte limit")
                    continue
                yield candidate


def read_document(path: Path, report: LoadReport | None = None) -> Document | None:
    """Read one file as UTF-8 text, refusing binaries and unreadable files."""
    try:
        blob = Path(path).read_bytes()
    except OSError as exc:
        if report is not None:
            report.skip(str(path), f"read failed: {exc}")
        return None
    if _looks_binary(blob):
        if report is not None:
            report.skip(str(path), "binary content")
        return None
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = blob.decode("latin-1")
        except UnicodeDecodeError as exc:
            if report is not None:
                report.skip(str(path), f"decode failed: {exc}")
            return None
    if not text.strip():
        if report is not None:
            report.skip(str(path), "no non-whitespace content")
        return None
    # Detect payloads before sanitising: the strip is what removes the invisible ones.
    flags = invisible_payload_flags(text)
    if flags and report is not None:
        report.note_sanitized(str(path), flags)
    text = sanitize_text(text)
    return Document(doc_id=_doc_id(Path(path)), path=str(Path(path)), text=text)


def load_documents(
    roots: Sequence[Path],
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
    max_file_bytes: int = MAX_FILE_BYTES,
    on_document: Callable[[Document, int, int], None] | None = None,
) -> LoadReport:
    """Load every readable text document under ``roots``.

    ``on_document`` is called after each file is read with the document and the running
    ``(documents, skipped)`` counts, so a caller can report progress while a large load
    is still running rather than only when it finishes.
    """
    report = LoadReport()
    for path in iter_files(roots, extensions=extensions, max_file_bytes=max_file_bytes, report=report):
        document = read_document(path, report=report)
        if document is not None:
            report.documents.append(document)
            if on_document is not None:
                on_document(document, len(report.documents), len(report.skipped))
    return report


_SENTENCE_MARKERS: tuple[str, ...] = ("\n\n", ". ", "! ", "? ")


def _sentence_start(body: str, start: int, end: int) -> int:
    """Index just after the last sentence break in ``body[start:end]``, or -1."""
    best = -1
    for marker in _SENTENCE_MARKERS:
        index = body.rfind(marker, start, end)
        if index != -1:
            best = max(best, index + len(marker))
    return best


def _sentence_end(body: str, end: int, limit: int) -> int | None:
    """Index just after the next sentence break in ``body[end:limit]``, or ``None``.

    The search is bounded by ``limit`` so a document with no sentence punctuation costs
    one bounded scan per chunk rather than scanning to the end of the file each time.
    """
    best: int | None = None
    for marker in _SENTENCE_MARKERS:
        index = body.find(marker, end, limit)
        if index != -1:
            candidate = index + len(marker)
            best = candidate if best is None else min(best, candidate)
    return best


def chunk_text(
    text: str,
    chunk_chars: int = 1200,
    overlap_chars: int = 400,
    min_chunk_chars: int = 80,
) -> list[tuple[str, int, int]]:
    """Split ``text`` into ``(chunk_text, start_char, end_char)`` triples.

    Paragraph breaks are preferred split points. A paragraph longer than
    ``chunk_chars`` is split on whitespace, and the trailing ``overlap_chars`` of
    each chunk seed the next one so a fact spanning a boundary stays retrievable.

    A cut that lands *inside* a sentence backs the next chunk up to that sentence's
    start, so the sentence is wholly contained in one chunk even when it is longer
    than ``overlap_chars`` (as long as it still fits in one chunk). The guarantee is
    therefore "a fact up to ``overlap_chars`` survives any split, and a whole
    sentence up to ``chunk_chars`` survives too" rather than the overlap alone.
    """
    if chunk_chars < 1:
        raise ValueError(f"chunk_chars must be positive, got {chunk_chars}")
    if overlap_chars < 0:
        raise ValueError(f"overlap_chars must be non-negative, got {overlap_chars}")
    if overlap_chars >= chunk_chars:
        raise ValueError(
            f"overlap_chars ({overlap_chars}) must be smaller than chunk_chars ({chunk_chars})"
        )
    if not text.strip():
        return []

    # Work on the stripped text so reported offsets index the same string the
    # chunk text came from.
    body = text.strip()
    total = len(body)
    out: list[tuple[str, int, int]] = []

    start = 0
    while start < total:
        end = min(start + chunk_chars, total)
        clean = False
        if end < total:
            # Prefer a paragraph break, then a sentence end, then any space, but
            # never cut before the half-way point or the chunk would be a stub.
            floor = start + chunk_chars // 2
            cut = body.rfind("\n\n", floor, end)
            if cut != -1:
                clean = True
            else:
                sentence = body.rfind(". ", floor, end)
                if sentence != -1:
                    cut = sentence + 1
                    clean = True
                else:
                    cut = body.rfind(" ", floor, end)
            if cut != -1 and cut > start:
                end = cut

        piece = body[start:end].strip()
        if piece:
            out.append((piece, start, end))
        if end >= total:
            break

        # Overlap the tail so a fact spanning a boundary stays retrievable, and
        # always move forward: an overlap wider than the chunk itself would
        # otherwise re-emit the same region forever.
        advanced = end - overlap_chars
        if not clean and overlap_chars > 0:
            # The cut fell inside a sentence. Extend the overlap to that sentence's
            # start when it fits in one chunk, so the sentence is not split.
            sentence_start = _sentence_start(body, start, end)
            sentence_end = _sentence_end(body, end, start + chunk_chars + 2)
            if (
                sentence_start > start
                and sentence_end is not None
                and sentence_end - sentence_start <= chunk_chars
            ):
                advanced = min(advanced, sentence_start)
        start = advanced if advanced > start else start + 1

    # Fold a short trailing fragment into the previous chunk rather than dropping it.
    if len(out) > 1 and len(out[-1][0]) < min_chunk_chars:
        last_text, last_start, last_end = out.pop()
        prev_text, prev_start, _prev_end = out[-1]
        merged = body[prev_start:last_end].strip()
        if merged:
            out[-1] = (merged, prev_start, last_end)
        else:  # pragma: no cover - only reachable if the tail was pure whitespace
            out.append((last_text, last_start, last_end))
    return out


def chunk_documents(
    documents: Sequence[Document],
    chunk_chars: int = 1200,
    overlap_chars: int = 400,
    min_chunk_chars: int = 80,
    on_document: Callable[[int, int, int], None] | None = None,
) -> list[Chunk]:
    """Chunk every document, preserving reading order per document.

    ``on_document`` is called after each document with ``(done, total, chunks)``: how
    many documents have been chunked, how many there are, and the running chunk count.
    """
    chunks: list[Chunk] = []
    total = len(documents)
    for done, document in enumerate(documents, start=1):
        pieces = chunk_text(
            document.text,
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
            min_chunk_chars=min_chunk_chars,
        )
        for index, (snippet, start, end) in enumerate(pieces):
            chunks.append(Chunk(
                chunk_id=f"{document.doc_id}:{index:04d}",
                doc_id=document.doc_id,
                path=document.path,
                text=snippet,
                index=index,
                start_char=start,
                end_char=end,
            ))
        if on_document is not None:
            on_document(done, total, len(chunks))
    return chunks
