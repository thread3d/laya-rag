"""Group retrieved passages into one evidence entry per source document.

Retrieval works in chunks, and a query that matches one file in several places comes
back as several chunks of that file. Every consumer of that evidence — the composer's
numbered context, the numbered passages in the interface, and the citation check —
has to agree on what ``[n]`` means, or a citation stops pointing at what it names.
Feeding each chunk as its own numbered block also repeats the document in the
composer's context and spends citation numbers on the same file.

So the evidence is grouped here, once, and all three number the same list. A document
appears at the position of its **first** passage, keeping the ranking that retrieval
produced; its remaining chunks are joined into that one entry in rank order.

Passages with no path are independent evidence and are never merged together: there is
no document they are known to share, and inventing one would put unrelated text under a
single citation number.
"""
from __future__ import annotations

from typing import Iterable

#: Two chunks of one document are joined only if the tail of what is already joined
#: repeats at least this many characters of the next chunk's head. Adjacent chunks
#: overlap by the chunker's ``overlap_chars`` (200 by default), so a real overlap is
#: far longer than this, while a coincidental match this long between unrelated chunks
#: does not occur in practice. Short repeats are therefore joined, not trimmed.
MIN_OVERLAP_CHARS = 40


def _join_without_overlap(earlier: str, later: str) -> str:
    """Append ``later`` to ``earlier``, dropping a repeated chunk overlap.

    The largest suffix of ``earlier`` that is also a prefix of ``later`` is removed from
    ``later``, so a chunk that repeats text already present is not fed twice. When all
    of ``later`` is already present it is dropped entirely — the literal reading of
    "add this document once".
    """
    limit = min(len(earlier), len(later))
    for size in range(limit, MIN_OVERLAP_CHARS - 1, -1):
        if earlier.endswith(later[:size]):
            return earlier + later[size:]
    return f"{earlier}\n\n{later}"


def group_passages_by_document(
    passages: Iterable[tuple[str | None, str]],
) -> list[tuple[str, str]]:
    """Merge passages sharing a path into one ``(path, text)`` entry each.

    The result is in first-appearance order, so entry *n* is the document whose best
    passage ranked *n*-th among the distinct documents. Chunk text is joined in rank
    order with repeated overlap trimmed. A passage with no path stays its own entry.
    """
    order: list[str] = []
    groups: dict[str, dict[str, str]] = {}
    pathless = 0

    for path, text in passages:
        key = (path or "").strip()
        if not key:
            # Not a document anyone can share, so it gets its own entry under a key no
            # real path can collide with.
            pathless += 1
            key = f"\x00pathless\x00{pathless}"
            groups[key] = {"path": "", "text": text}
            order.append(key)
            continue
        if key in groups:
            groups[key]["text"] = _join_without_overlap(groups[key]["text"], text)
        else:
            groups[key] = {"path": path, "text": text}
            order.append(key)

    return [(groups[key]["path"], groups[key]["text"]) for key in order]
