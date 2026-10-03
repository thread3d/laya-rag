"""Index drift: the three lifecycle classes that silently degrade retrieval.

An index is not correct because it was correct when it was built. Three distinct things go
wrong afterwards, and they need three distinct detectors (Oracle, *How to Detect RAG Index
Drift*, 2026):

* **deleted documents** — a path that no longer exists on disk still has retrievable
  chunks. Re-indexing a directory prunes these (``RagEngine.index_paths``), but between
  re-indexes nothing notices;
* **duplicate content** — the same passage indexed twice under two paths, or a file copied
  rather than moved. Both copies compete for the same slots, so the effective pool is
  smaller than it looks and a quota-based selector spends slots on one fact;
* **near-duplicate chunks** — the same passage at a different offset, or a body that
  differs only in a heading. These are not identical, so exact hashing misses them.

Everything here is deterministic, model-free and offline: it reads chunk metadata and text
and reports what it found. It never repairs anything — pruning is a policy decision that
belongs to the caller, and ``index_paths`` already implements the one safe automatic rule.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .chunking import Chunk

_WHITESPACE = re.compile(r"\s+")

#: Default MinHash signature width and band count. The band schedule sets the recall of
#: candidate generation, and the reported similarity is the *exact* Jaccard on the shingle
#: sets, so the sketch only has to be good enough to not miss pairs. With 8 bands of 2 rows
#: the collision probability is ``1 - (1 - s^2)^8``: ~0.90 at s=0.5 and ~1.0 at s=0.85, which
#: is the steep S-curve an LSH needs. A schedule is validated against the requested
#: threshold rather than trusted, because a badly tuned one fails *silently* by finding
#: nothing.
DEFAULT_SIGNATURE = 16
DEFAULT_BANDS = 8
DEFAULT_SHINGLE = 5
DEFAULT_THRESHOLD = 0.85
DEFAULT_MAX_CHUNKS = 20000

_MASK64 = (1 << 64) - 1
_MIX_STRIDE = 0x9E3779B97F4A7C15


def _mix64(value: int) -> int:
    """splitmix64 finalizer — a cheap bijection that decorrelates near-identical inputs."""
    value &= _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def band_recall(bands: int, rows: int, similarity: float) -> float:
    """Probability that a pair at ``similarity`` agrees on at least one band.

    The LSH S-curve, ``1 - (1 - s**rows)**bands``. Exposed because it is the number that
    says whether a band schedule can possibly find the pairs a caller asked for.
    """
    if bands < 1 or rows < 1:
        return 0.0
    return 1.0 - (1.0 - float(similarity) ** rows) ** bands


def fingerprint(text: str) -> str:
    """A content hash invariant to whitespace, so a reformatted file is not "changed".

    Case is *not* folded: "Refunds" and "refunds" are different indexed text, and the
    duplicate detector's job is to find chunks whose stored text is redundant.
    """
    return hashlib.sha1(_WHITESPACE.sub(" ", str(text)).strip().encode("utf-8")).hexdigest()


def shingles(text: str, size: int = DEFAULT_SHINGLE) -> frozenset[str]:
    """Word-shingle set of a chunk's text."""
    if size < 1:
        raise ValueError(f"shingle size must be >= 1, got {size!r}")
    words = _WHITESPACE.sub(" ", str(text)).strip().lower().split(" ")
    words = [word for word in words if word]
    if len(words) < size:
        return frozenset({" ".join(words)}) if words else frozenset()
    return frozenset(
        " ".join(words[i : i + size]) for i in range(len(words) - size + 1)
    )


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Exact Jaccard similarity; used to verify an LSH candidate pair."""
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _minhash_signature(items: frozenset[str], width: int) -> tuple[int, ...]:
    """A MinHash sketch: ``min`` over the shingles of ``width`` independent permutations.

    Each permutation is ``_mix64(hash(shingle) + i * stride)`` — a bijection of a 64-bit
    content hash, so permuting once per shingle is enough and no per-permutation hashing
    pass is needed. Position ``i`` agreeing between two sketches has probability equal to
    the shingle Jaccard, which is what makes the banded comparison below a valid LSH.

    (Taking the ``width`` *smallest hashes of one hash* instead — a KMV sketch — estimates
    set cardinality, not Jaccard, and compares positionally almost never. It is a tempting
    shortcut that silently finds nothing.)
    """
    if not items:
        return tuple([_MASK64] * width)
    sketch = [_MASK64] * width
    for item in items:
        base = int.from_bytes(
            hashlib.blake2b(item.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for position in range(width):
            value = _mix64(base + position * _MIX_STRIDE)
            if value < sketch[position]:
                sketch[position] = value
    return tuple(sketch)


@dataclass
class DuplicateCluster:
    """Chunks whose stored text is byte-identical after whitespace normalisation."""

    fingerprint: str
    chunk_ids: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)

    @property
    def redundant(self) -> int:
        """How many retrievable chunks this cluster makes redundant."""
        return max(0, len(self.chunk_ids) - 1)


def duplicate_clusters(chunks: Sequence[Chunk]) -> list[DuplicateCluster]:
    """Exact-duplicate clusters, largest first, then by first chunk id for determinism."""
    by_fingerprint: dict[str, DuplicateCluster] = {}
    for chunk in chunks:
        key = fingerprint(chunk.text)
        cluster = by_fingerprint.get(key)
        if cluster is None:
            cluster = by_fingerprint[key] = DuplicateCluster(fingerprint=key)
        cluster.chunk_ids.append(chunk.chunk_id)
        if chunk.path not in cluster.paths:
            cluster.paths.append(chunk.path)
    duplicates = [c for c in by_fingerprint.values() if len(c.chunk_ids) > 1]
    duplicates.sort(key=lambda c: (-len(c.chunk_ids), c.chunk_ids[0]))
    return duplicates


def near_duplicate_pairs(
    chunks: Sequence[Chunk],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    shingle_size: int = DEFAULT_SHINGLE,
    signature: int = DEFAULT_SIGNATURE,
    bands: int = DEFAULT_BANDS,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> list[tuple[str, str, float]]:
    """Near-duplicate ``(chunk_id, chunk_id, jaccard)`` pairs, strongest first.

    MinHash LSH: candidates must agree on at least one band of the signature, then the
    exact Jaccard on the shingle sets decides. That is what makes this affordable on a
    large index — a naive all-pairs comparison is quadratic and would never finish on the
    17k-chunk MultiHop corpus, let alone a million.

    ``max_chunks`` bounds the work; callers should report that the scan was truncated
    rather than present it as complete. Pairs are unique and ordered by their chunk ids.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")
    if bands < 1 or signature < 1 or bands > signature:
        raise ValueError(
            f"need 1 <= bands <= signature, got bands={bands!r} signature={signature!r}"
        )
    rows = max(1, signature // bands)
    recall = band_recall(bands, rows, threshold)
    if recall < 0.5:
        raise ValueError(
            f"a {bands}-band x {rows}-row schedule finds only {recall:.0%} of pairs at "
            f"jaccard {threshold}; raise --bands, lower --threshold, or widen --signature "
            "(a schedule that cannot see the pairs would report a clean index)"
        )
    selected = list(chunks)[: max(0, int(max_chunks))]
    shingle_sets: list[frozenset[str]] = [shingles(c.text, shingle_size) for c in selected]
    signatures = [_minhash_signature(s, signature) for s in shingle_sets]

    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for index, sketch in enumerate(signatures):
        for band in range(bands):
            key = (band, sketch[band * rows : (band + 1) * rows])
            buckets.setdefault(key, []).append(index)

    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[str, str, float]] = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        for position, left in enumerate(members):
            for right in members[position + 1 :]:
                key = (left, right) if left < right else (right, left)
                if key in seen:
                    continue
                seen.add(key)
                score = jaccard(shingle_sets[left], shingle_sets[right])
                if score >= threshold:
                    pairs.append(
                        (selected[key[0]].chunk_id, selected[key[1]].chunk_id, round(score, 4))
                    )
    pairs.sort(key=lambda item: (-item[2], item[0], item[1]))
    return pairs


def stale_paths(paths: Iterable[str], *, exists: Callable[[str], bool] = os.path.exists) -> list[str]:
    """Indexed paths that are gone from disk, in order.

    ``exists`` is injectable so the check is testable without touching a real filesystem.
    """
    return [path for path in paths if not exists(path)]


def audit(
    chunks: Sequence[Chunk],
    *,
    check_paths: bool = True,
    exists: Callable[[str], bool] = os.path.exists,
    near_duplicates: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    shingle_size: int = DEFAULT_SHINGLE,
    signature: int = DEFAULT_SIGNATURE,
    bands: int = DEFAULT_BANDS,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    examples: int = 5,
) -> dict[str, Any]:
    """The drift report: what is stale, what is duplicated, what the scan skipped.

    Deliberately reporting-only. Nothing here changes the index, because the safe automatic
    rule (prune paths under a directory being re-indexed) already lives in
    ``RagEngine.index_paths`` and a detector that also deleted would be a surprise.
    """
    chunk_list = list(chunks)
    paths = sorted({chunk.path for chunk in chunk_list})
    report: dict[str, Any] = {
        "chunks": len(chunk_list),
        "paths": len(paths),
        "scanned_chunks": min(len(chunk_list), int(max_chunks)),
        "truncated": len(chunk_list) > int(max_chunks),
    }

    if check_paths:
        gone = stale_paths(paths, exists=exists)
        report["stale_paths"] = {
            "count": len(gone),
            "examples": gone[:examples],
            "stale_chunks": sum(1 for chunk in chunk_list if chunk.path in set(gone)),
        }
    else:
        report["stale_paths"] = {"count": 0, "examples": [], "stale_chunks": 0}

    clusters = duplicate_clusters(chunk_list)
    report["duplicate_content"] = {
        "clusters": len(clusters),
        "redundant_chunks": sum(cluster.redundant for cluster in clusters),
        "examples": [
            {
                "chunk_ids": cluster.chunk_ids[:examples],
                "paths": cluster.paths[:examples],
                "size": len(cluster.chunk_ids),
            }
            for cluster in clusters[:examples]
        ],
    }

    if near_duplicates:
        pairs = near_duplicate_pairs(
            chunk_list,
            threshold=threshold,
            shingle_size=shingle_size,
            signature=signature,
            bands=bands,
            max_chunks=max_chunks,
        )
        report["near_duplicate_content"] = {
            "threshold": threshold,
            "shingle_size": shingle_size,
            "pairs": len(pairs),
            "examples": [
                {"chunk_ids": [left, right], "jaccard": score}
                for left, right, score in pairs[:examples]
            ],
        }
    else:
        report["near_duplicate_content"] = None

    report["clean"] = (
        report["stale_paths"]["count"] == 0
        and report["duplicate_content"]["clusters"] == 0
        and (report["near_duplicate_content"] is None
             or report["near_duplicate_content"]["pairs"] == 0)
    )
    return report


def summarize(report: dict[str, Any]) -> str:
    """One human-readable line per drift class, for a CLI or a log."""
    lines = [
        f"chunks: {report['chunks']} across {report['paths']} paths"
        + (" (scan truncated)" if report.get("truncated") else "")
    ]
    stale = report["stale_paths"]
    lines.append(
        f"deleted documents: {stale['count']} path(s) gone, {stale['stale_chunks']} chunk(s) still indexed"
    )
    duplicates = report["duplicate_content"]
    lines.append(
        f"duplicate content: {duplicates['clusters']} cluster(s), "
        f"{duplicates['redundant_chunks']} redundant chunk(s)"
    )
    near = report.get("near_duplicate_content")
    if near is None:
        lines.append("near-duplicate content: not scanned (pass --near-duplicates)")
    else:
        lines.append(
            f"near-duplicate content: {near['pairs']} pair(s) at jaccard >= {near['threshold']}"
            + (" (scan truncated)" if report.get("truncated") else "")
        )
    lines.append("clean" if report["clean"] else "drift detected")
    return "\n".join(lines)
