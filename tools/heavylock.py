"""An advisory lock so only one heavyweight job runs on this host at a time.

This exists because running several corpus-scale jobs at once **crashed the machine**: three
`tools/beir.py` lexical builds plus a retrieval harness were embedding, tokenising and scanning
multi-million-document corpora simultaneously, and the host went down. Every one of those jobs was
resumable from its checkpoint, so no work was lost — but the crash was avoidable, and the project's
own rule ("one long job at a time") was being enforced only by memory.

This is the enforcement. A heavy job takes an exclusive `flock` on `eval/.heavy-job.lock` before it
does any work; a second job fails fast with the holder's name, pid and start time instead of piling
on. `flock` is released by the kernel when the process dies, so a crashed or killed job leaves no
stale lock to clean up — the exact failure mode this is for.

    from heavylock import heavy_job_guard
    heavy_job_guard("beir.py")        # exits 2 with a message if another heavy job holds it

    .venv/bin/python tools/heavylock.py --status
    LAYA_RAG_ALLOW_CONCURRENT=1 ...   # deliberate override, e.g. for a tiny read-only probe
"""
from __future__ import annotations

import argparse
import atexit
import fcntl
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "eval" / ".heavy-job.lock"


class HeavyJobInProgress(RuntimeError):
    """Raised when another heavyweight job already holds the lock."""


def _allow_concurrent() -> bool:
    return os.environ.get("LAYA_RAG_ALLOW_CONCURRENT", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def read_holder(path: Path | None = None) -> dict | None:
    """The JSON the current holder wrote, or None when the lock file is absent/empty."""
    lock_path = Path(path or DEFAULT_LOCK)
    if not lock_path.is_file():
        return None
    try:
        text = lock_path.read_text(encoding="utf-8").strip()
        return json.loads(text) if text else None
    except (OSError, json.JSONDecodeError):
        return None


def is_busy(path: Path | None = None) -> bool:
    """Whether a live process holds the lock (checked with a non-blocking probe)."""
    lock_path = Path(path or DEFAULT_LOCK)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


class HeavyJobLock:
    """An exclusive advisory lock; released explicitly or by the OS on process exit."""

    def __init__(
        self,
        name: str,
        *,
        path: Path | None = None,
        allow_concurrent: bool | None = None,
    ) -> None:
        self.name = name
        self.path = Path(path or DEFAULT_LOCK)
        self._allow = _allow_concurrent() if allow_concurrent is None else allow_concurrent
        self._fd: int | None = None

    def acquire(self) -> None:
        if self._allow:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            holder = read_holder(self.path) or {}
            who = holder.get("name", "unknown")
            pid = holder.get("pid")
            started = holder.get("started")
            since = f" since {time.strftime('%H:%M:%S', time.localtime(started))}" if started else ""
            raise HeavyJobInProgress(
                f"another heavyweight job is already running: {who} (pid {pid}){since}. "
                "Wait for it, or set LAYA_RAG_ALLOW_CONCURRENT=1 to run both deliberately."
            ) from None
        os.ftruncate(fd, 0)
        os.write(
            fd,
            json.dumps({"name": self.name, "pid": os.getpid(), "started": time.time()}).encode(),
        )
        os.fsync(fd)
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "HeavyJobLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def heavy_job_guard(name: str, *, path: Path | None = None) -> HeavyJobLock:
    """Acquire the heavy-job lock or exit the process with a clear message.

    Intended as the first statement of a long-running tool's ``main``: it needs no matching
    ``finally`` because the kernel drops the lock when the process ends, and a normal exit also
    unregisters through :mod:`atexit`.
    """
    lock = HeavyJobLock(name, path=path)
    try:
        lock.acquire()
    except HeavyJobInProgress as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    atexit.register(lock.release)
    return lock


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="heavylock.py")
    ap.add_argument("--status", action="store_true", help="report whether a heavy job is running")
    ap.add_argument("--path", default=str(DEFAULT_LOCK))
    args = ap.parse_args(argv)
    if not args.status:
        ap.print_help()
        return 0
    if is_busy(args.path):
        holder = read_holder(args.path) or {}
        started = holder.get("started")
        when = f" (started {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started))})" if started else ""
        print(f"heavy job running: {holder.get('name', 'unknown')} pid {holder.get('pid')}{when}")
        return 1
    print("no heavy job running; the lock is free")
    return 0


if __name__ == "__main__":
    sys.exit(main())
