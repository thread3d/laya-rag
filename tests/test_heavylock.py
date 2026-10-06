"""Tests for the heavy-job lock.

The lock exists because running several corpus-scale jobs at once crashed the host. These pin the
two properties that make it useful: a second job fails fast while one holds it, and the OS drops the
lock when the holder goes away (no stale file to clean up).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    name = "laya_rag_heavylock"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "heavylock.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hl = _load()


def test_a_second_job_fails_fast_with_the_holder(tmp_path):
    lock_path = tmp_path / "heavy.lock"
    first = hl.HeavyJobLock("first.py", path=lock_path)
    first.acquire()
    try:
        second = hl.HeavyJobLock("second.py", path=lock_path)
        with pytest.raises(hl.HeavyJobInProgress) as excinfo:
            second.acquire()
        assert "first.py" in str(excinfo.value)
        assert str(__import__("os").getpid()) in str(excinfo.value)
    finally:
        first.release()


def test_the_lock_is_reusable_after_release(tmp_path):
    lock_path = tmp_path / "heavy.lock"
    first = hl.HeavyJobLock("first.py", path=lock_path)
    first.acquire()
    first.release()
    second = hl.HeavyJobLock("second.py", path=lock_path)
    second.acquire()
    assert hl.read_holder(lock_path)["name"] == "second.py"
    second.release()


def test_is_busy_tracks_the_holder(tmp_path):
    lock_path = tmp_path / "heavy.lock"
    assert hl.is_busy(lock_path) is False
    lock = hl.HeavyJobLock("job.py", path=lock_path)
    lock.acquire()
    try:
        assert hl.is_busy(lock_path) is True
    finally:
        lock.release()
    assert hl.is_busy(lock_path) is False


def test_allow_concurrent_bypasses_the_lock(tmp_path):
    lock_path = tmp_path / "heavy.lock"
    first = hl.HeavyJobLock("first.py", path=lock_path)
    first.acquire()
    try:
        second = hl.HeavyJobLock("probe.py", path=lock_path, allow_concurrent=True)
        second.acquire()   # no exception
        second.release()
    finally:
        first.release()


def test_the_guard_exits_rather_than_running(tmp_path):
    lock_path = tmp_path / "heavy.lock"
    first = hl.HeavyJobLock("first.py", path=lock_path)
    first.acquire()
    try:
        with pytest.raises(SystemExit) as excinfo:
            hl.heavy_job_guard("second.py", path=lock_path)
        assert excinfo.value.code == 2
    finally:
        first.release()


def test_status_command_reports_busy_and_free(tmp_path, capsys):
    lock_path = tmp_path / "heavy.lock"
    assert hl.main(["--status", "--path", str(lock_path)]) == 0
    lock = hl.HeavyJobLock("job.py", path=lock_path)
    lock.acquire()
    try:
        assert hl.main(["--status", "--path", str(lock_path)]) == 1
        assert "job.py" in capsys.readouterr().out
    finally:
        lock.release()
