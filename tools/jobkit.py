"""Resumable, checkpointed long jobs.

Every expensive measurement in this repository is a loop over queries or documents, and until
now none of them survived interruption: `beir.cached_embed` writes one all-or-nothing `.npy`
keyed by the whole doc-id list, `evalkit.write_json` is not atomic, and a tool re-run after a
kill redoes every unit it had already finished. That is tolerable for a two-minute run and not
tolerable for the hours-long ones the roadmap schedules (engine-level MultiHop, the 13 large
BEIR corpora, multi-corpus distillation).

This module fixes the three failure modes with one small contract:

* **A unit of work is a `(key, value)` pair.** The key is a stable, human-readable identity
  (`"nfcorpus:test:42"`), not a loop index, so a resumed run cannot silently pair a result with
  the wrong query.
* **`record` appends one JSON line and fsyncs it.** A kill loses at most the unit in flight;
  everything already recorded is on disk.
* **`finish` writes the report atomically**, through a sibling temporary file and `os.replace`,
  so a reader never sees a half-written report.

A run is bound to a `fingerprint` — a hash of the settings that would change the numbers (model
ids, chunking, depth, split, plus `CODE_VERSION`). Reopening a checkpoint under a different
fingerprint raises rather than silently mixing two configurations, which is the failure that
would be hardest to notice and most damaging to a claim.

Sharding is contiguous over a deterministically ordered sequence, so concatenating the shards
restores the original order and a merge is a fold rather than a sort.

Layout for a run whose final report is ``eval/benchmarks/multihop-rag/official.json``::

    official.jsonl           one {"key", "value"} per completed unit (durable)
    official.manifest.json   status, fingerprint, shard, counts (atomic, rewritten)
    official.json            the report, written once by finish() (atomic)

For a sharded job the caller gives each shard its own stem (``--out official.shard0of4``); then
:func:`merge_shards` folds them into ``official.json``, refusing a fingerprint mismatch.

    from jobkit import ResumableRun, config_fingerprint, shard_items

    fp = config_fingerprint(dataset="nfcorpus", model="BAAI/bge-small-en-v1.5")
    run = ResumableRun(out, fingerprint=fp, shard=(shard, nshards))
    for key in run.pending(keys):
        run.record(key, expensive(key))
    run.finish({"dataset": "nfcorpus", "metrics": ...})
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

# Bump when the checkpoint *format* or the meaning of a recorded value changes. It is folded
# into every fingerprint, so an old checkpoint is invalidated loudly instead of being misread.
CODE_VERSION = "1"

# The reserved key for shared, non-per-unit state (settings, gold facts, model ids). A tool
# records it once under this key and `merge_shards` treats an identical repeat across shards as
# one row rather than a duplicate — which is what lets a sharded job carry its own metadata
# without a separate sidecar.
META_KEY = "__meta__"


class FingerprintMismatch(RuntimeError):
    """A checkpoint exists for a different configuration than the one being run."""


def config_fingerprint(**settings: Any) -> str:
    """A stable hash of everything that would change a recorded number.

    Callers should pass every setting that affects the score: dataset, split, model id,
    chunking, overlap, depth, seed. Sorting the keys makes the hash independent of call order.
    """
    payload = {"code_version": CODE_VERSION, **settings}
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path | str, payload: Any) -> None:
    """Write JSON through a sibling temp file and `os.replace`, never in place.

    A reader (or a crashed run's next attempt) sees either the previous file or the complete
    new one, never a partial write. `evalkit.write_json` writes in place and is fine for a
    report a human reads after the run; this is for the one a resumed run depends on.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_json(path: Path | str, default: Any = None) -> Any:
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def shard_items(items: Sequence[Any], index: int, count: int) -> list[Any]:
    """A contiguous shard of a deterministically ordered sequence.

    Contiguous rather than round-robin so that concatenating shard 0..count-1 restores the
    input order exactly, and so a shard is a bounded *range* of the input — which is what makes
    `--limit-queries`-style smoke runs and incremental progress meaningful.

    The split is by integer arithmetic on ``len(items)``, so every index from 0 to count-1
    yields a valid (possibly empty) shard and the union is the whole input.
    """
    if count < 1:
        raise ValueError("shard count must be >= 1")
    if not 0 <= index < count:
        raise ValueError(f"shard index {index} out of range for {count} shards")
    total = len(items)
    start = (total * index) // count
    end = (total * (index + 1)) // count
    return list(items[start:end])


def shard_stem(stem: Path | str, index: int, count: int) -> Path:
    """The per-shard output stem. An unsharded run keeps the caller's exact stem."""
    stem = Path(stem)
    if count <= 1:
        return stem
    return stem.with_name(f"{stem.name}.shard{index}of{count}")


def parse_shard(text: str) -> tuple[int, int]:
    """Parse a CLI `--shard i/n` value into `(index, count)`."""
    try:
        index_text, count_text = str(text).split("/")
        index, count = int(index_text), int(count_text)
    except ValueError as exc:
        raise ValueError(f"shard must be written 'i/n', got {text!r}") from exc
    if count < 1:
        raise ValueError("shard count must be >= 1")
    if not 0 <= index < count:
        raise ValueError(f"shard index {index} out of range for {count} shards")
    return index, count


def _read_jsonl(path: Path) -> dict[str, Any]:
    """Load a checkpoint file into `key -> value`.

    A **trailing** partial line is dropped rather than raised on: that is exactly the state a
    kill during `record` leaves, and it loses only the unit that was in flight. Any other
    malformed line means the file is not this tool's checkpoint, and that is raised.
    """
    out: dict[str, Any] = {}
    if not path.is_file():
        return out
    text = path.read_text(encoding="utf-8")
    if not text:
        return out
    lines = text.split("\n")
    if not text.endswith("\n"):
        tail = lines.pop()
        if tail.strip():
            try:
                record = json.loads(tail)
                out[record["key"]] = record.get("value")
            except (ValueError, KeyError, TypeError):
                pass
    for number, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            key = record["key"]
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"{path}:{number} is not a valid checkpoint record; the file may be from "
                "another tool or a different format version"
            ) from exc
        out[key] = record.get("value")
    return out


@dataclass(frozen=True)
class WorkItem:
    """One unit of work: a stable key and whatever the caller needs to run it."""

    key: str
    payload: Any = None


def _report_stem(path: Path) -> Path:
    """`official.json` -> `official`, so sidecars read `official.jsonl`, not `official.json.jsonl`."""
    return path.with_name(path.name[: -len(".json")]) if path.name.endswith(".json") else path


class ResumableRun:
    """An append-only checkpoint over a sequence of `(key, value)` units.

    Open it at the start of the job; it loads whatever a previous attempt recorded. Iterate
    `pending(...)` rather than the raw item list so completed units are skipped, and call
    `record` as each unit finishes — not in a batch at the end, which is the whole point.

    The checkpoint is self-describing from the first `record`: a manifest carrying the
    fingerprint is written if one does not exist, so reopening under a different configuration
    is always detected, even for a run that never reached `finish`.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        fingerprint: str,
        shard: tuple[int, int] = (0, 1),
    ) -> None:
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.shard = shard
        self.stem = _report_stem(self.path)
        self.jsonl_path = Path(str(self.stem) + ".jsonl")
        self.manifest_path = Path(str(self.stem) + ".manifest.json")
        self._completed: dict[str, Any] = {}
        self._load()
        if not self.manifest_path.is_file():
            self.touch()

    # ── loading and status ──────────────────────────────────────────────

    def _load(self) -> None:
        manifest = read_json(self.manifest_path)
        if isinstance(manifest, Mapping):
            recorded = manifest.get("fingerprint")
            if recorded is not None and recorded != self.fingerprint:
                raise FingerprintMismatch(
                    f"{self.manifest_path} was written for fingerprint {recorded}, not "
                    f"{self.fingerprint}; rerun with --fresh to discard it"
                )
        if not self.jsonl_path.is_file():
            return
        self._completed = _read_jsonl(self.jsonl_path)

    @property
    def completed_keys(self) -> set[str]:
        return set(self._completed)

    def __len__(self) -> int:
        return len(self._completed)

    def __contains__(self, key: str) -> bool:
        return key in self._completed

    def get(self, key: str, default: Any = None) -> Any:
        return self._completed.get(key, default)

    def values(self) -> dict[str, Any]:
        return dict(self._completed)

    def pending(self, items: Iterable[Any], key: Any = None) -> list[Any]:
        """The items whose key is not yet recorded, in input order.

        `key` is a callable item -> str, defaulting to an item's own `.key` (so `WorkItem`
        works directly) or, failing that, the item itself when it is already a string.
        """
        if key is None:
            def key_of(item: Any) -> str:
                return item.key if isinstance(item, WorkItem) else str(item)
        else:
            key_of = key
        return [item for item in items if key_of(item) not in self._completed]

    # ── writing ─────────────────────────────────────────────────────────

    def record(self, key: str, value: Any) -> None:
        """Append one unit's result and fsync it before returning.

        The fsync is deliberate: it is what makes "killed at unit k" a bounded loss rather than
        an unbounded one. It costs a few milliseconds and buys the whole guarantee.
        """
        record = {"key": key, "value": value}
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._completed[key] = value

    def manifest(self, *, status: str, total: int | None = None, extra: Mapping | None = None) -> dict:
        index, count = self.shard
        payload: dict[str, Any] = {
            "status": status,
            "fingerprint": self.fingerprint,
            "shard_index": index,
            "shard_count": count,
            "completed": len(self._completed),
            "total": total,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if extra:
            payload.update(extra)
        return payload

    def finish(self, payload: Mapping[str, Any], *, total: int | None = None,
               extra: Mapping | None = None) -> dict:
        """Write the merged report atomically and mark the manifest complete.

        `payload` is the caller's report (metrics, settings); jobkit adds a `_job` block so a
        reader can tell which fingerprint and shard produced it. The per-unit values are *not*
        folded in automatically — a caller that wants them should put them in `payload`, so the
        report keeps whatever shape its consumers already expect.
        """
        report = dict(payload)
        report["_job"] = self.manifest(status="complete", total=total, extra=extra)
        write_json_atomic(self.path, report)
        write_json_atomic(self.manifest_path, report["_job"])
        return report

    def touch(self, *, total: int | None = None, extra: Mapping | None = None) -> None:
        """Rewrite the manifest with current progress, without producing a final report."""
        write_json_atomic(self.manifest_path, self.manifest(status="running", total=total, extra=extra))


def merge_shards(
    stem: Path | str,
    count: int,
    *,
    fingerprint: str | None = None,
    build: Any = None,
) -> dict:
    """Fold `<stem>.shard{i}of{count}.jsonl` into `<stem>.json` and a single manifest.

    Refuses to merge checkpoints written under different fingerprints, which is the case that
    would otherwise produce a report that is a blend of two configurations. The shard files are
    read directly rather than through :class:`ResumableRun`, whose whole job is to refuse a
    fingerprint it was not opened with.

    `build(rows) -> payload` lets a tool reconstruct its own report shape from the merged
    `key -> value` rows; the default is a generic `{"rows": rows}`. Shared, non-per-unit state
    (settings, gold facts, model ids) travels as an ordinary reserved row — a tool records it
    once under a key such as ``"__meta__"`` and `build` pops it — so a shard never needs a
    separate sidecar and the merge works from any process.
    """
    if count < 1:
        raise ValueError("shard count must be >= 1")
    merged: dict[str, Any] = {}
    fingerprints: set[str] = set()
    for index in range(count):
        part = shard_stem(stem, index, count)
        manifest = read_json(Path(str(part) + ".manifest.json"))
        if isinstance(manifest, Mapping) and manifest.get("fingerprint"):
            fingerprints.add(str(manifest["fingerprint"]))
        for key, value in _read_jsonl(Path(str(part) + ".jsonl")).items():
            if key in merged:
                if key == META_KEY and merged[key] == value:
                    continue  # shared state recorded identically by every shard, not a clash
                raise RuntimeError(f"key {key!r} appears in more than one shard of {stem}")
            merged[key] = value
    if fingerprint is not None:
        fingerprints.add(fingerprint)
    if len(fingerprints) > 1:
        raise FingerprintMismatch(
            f"shards of {stem} carry {len(fingerprints)} different fingerprints: {sorted(fingerprints)}"
        )
    job = {
        "status": "complete",
        "fingerprint": next(iter(fingerprints), ""),
        "shard_count": count,
        "completed": len(merged),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    report = dict(build(merged) if build is not None else {"rows": merged})
    report["_job"] = job
    write_json_atomic(Path(str(stem) + ".json"), report)
    write_json_atomic(Path(str(stem) + ".manifest.json"), job)
    return report


def iter_manifests(root: Path | str) -> Iterator[tuple[Path, dict]]:
    """Every jobkit manifest under `root`, newest first — the data behind `make job-status`."""
    root = Path(root)
    if not root.exists():
        return
    found: list[tuple[float, Path, dict]] = []
    for path in root.rglob("*.manifest.json"):
        payload = read_json(path)
        if isinstance(payload, dict) and "fingerprint" in payload:
            found.append((path.stat().st_mtime, path, payload))
    for _, path, payload in sorted(found, reverse=True, key=lambda row: row[0]):
        yield path, payload
