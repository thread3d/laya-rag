"""A tamper-evident audit log for index mutations.

The security brief asks for an audit log and the repository has none: index builds, deletes and
resets leave no durable record. This is a hash-chained append-only JSONL log — each record carries
a SHA-256 over the *previous* record's hash and the canonical form of this record — so editing,
reordering, removing or inserting a line breaks the chain at that point and :meth:`AuditLog.verify`
names the first bad record.

It is deliberately not a security boundary on its own. An attacker who can rewrite the whole file
can recompute every hash; what the chain catches is edit-after-the-fact by anyone who did not
recompute it, which is the property a log is for. The head hash is the thing to anchor elsewhere
(a release artefact, a timestamp) if that matters.

    from laya_rag.audit import AuditLog
    AuditLog("eval/audit.jsonl").append("index", detail={"chunks": 4592})
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

#: The hash a fresh chain starts from.
GENESIS = "0" * 64
VERSION = 1


def _canonical(record: Mapping[str, Any]) -> str:
    """A stable serialisation: sorted keys, no incidental whitespace."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _digest(previous: str, record: Mapping[str, Any]) -> str:
    """The link hash: `sha256(previous_hash + canonical(record))`."""
    return hashlib.sha256((previous + _canonical(record)).encode("utf-8")).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class AuditLog:
    """An append-only, hash-chained log at ``path``."""

    path: Path | str
    _path: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._path = Path(self.path)

    # -- writing ---------------------------------------------------------
    def entries(self) -> list[dict[str, Any]]:
        """Every record, in file order. A malformed line raises rather than being skipped."""
        if not self._path.is_file():
            return []
        records: list[dict[str, Any]] = []
        for number, line in enumerate(self._path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"audit log line {number} is not JSON: {exc}") from exc
        return records

    def head(self) -> str:
        """The hash of the last record, or :data:`GENESIS` for an empty log."""
        records = self.entries()
        return str(records[-1]["hash"]) if records else GENESIS

    def append(
        self,
        action: str,
        *,
        actor: str | None = None,
        detail: Mapping[str, Any] | None = None,
        at: str | None = None,
    ) -> dict[str, Any]:
        """Append one record and return it. ``action`` names the mutation (e.g. ``index``)."""
        if not action or not str(action).strip():
            raise ValueError("an audit record needs a non-empty action")
        records = self.entries()
        previous = str(records[-1]["hash"]) if records else GENESIS
        body: dict[str, Any] = {
            "version": VERSION,
            "seq": len(records) + 1,
            "at": at or _now(),
            "action": str(action),
            "actor": actor,
            "detail": dict(detail or {}),
            "prev": previous,
        }
        record = {**body, "hash": _digest(previous, body)}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(_canonical(record) + "\n")
        return record

    # -- verifying -------------------------------------------------------
    def verify(self) -> dict[str, Any]:
        """Walk the chain. Returns ``{ok, entries, head}`` or the first bad record's index."""
        previous = GENESIS
        count = 0
        for number, record in enumerate(self.entries(), start=1):
            stored = record.get("hash")
            body = {key: value for key, value in record.items() if key != "hash"}
            if record.get("prev") != previous:
                return {
                    "ok": False, "entries": count, "first_bad": number,
                    "reason": "previous-hash mismatch",
                    "expected_prev": previous, "found_prev": record.get("prev"),
                }
            expected = _digest(previous, body)
            if stored != expected:
                return {
                    "ok": False, "entries": count, "first_bad": number,
                    "reason": "record hash mismatch",
                    "expected_hash": expected, "found_hash": stored,
                }
            previous = str(stored)
            count += 1
        return {"ok": True, "entries": count, "head": previous}


def verify_file(path: Path | str) -> dict[str, Any]:
    """Convenience wrapper: :meth:`AuditLog.verify` for a path."""
    return AuditLog(path).verify()


def iter_actions(path: Path | str) -> Iterator[dict[str, Any]]:
    """Every record, for a caller that wants to read without verifying."""
    yield from AuditLog(path).entries()
