"""Offline tests for the resumable-job contract.

The guarantee being tested is narrow and load-bearing: a job killed at unit *k* and restarted
must produce exactly the result of an uninterrupted run, and a checkpoint written under one
configuration must never be silently reused under another. Everything else here (sharding,
atomic writes, torn-line tolerance) exists to keep that guarantee true.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    name = "laya_rag_jobkit"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "jobkit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolves __module__ through sys.modules
    spec.loader.exec_module(module)
    return module


jobkit = _load()


# ── fingerprinting ───────────────────────────────────────────────────────

def test_fingerprint_ignores_call_order():
    assert jobkit.config_fingerprint(a=1, b=2) == jobkit.config_fingerprint(b=2, a=1)


def test_fingerprint_changes_with_any_setting():
    base = jobkit.config_fingerprint(dataset="scifact", depth=100)
    assert jobkit.config_fingerprint(dataset="scifact", depth=500) != base
    assert jobkit.config_fingerprint(dataset="nfcorpus", depth=100) != base


# ── sharding ─────────────────────────────────────────────────────────────

def test_shard_items_partition_the_input_in_order():
    items = list(range(10))
    shards = [jobkit.shard_items(items, i, 3) for i in range(3)]
    assert [len(s) for s in shards] == [3, 3, 4]
    assert [x for s in shards for x in s] == items


def test_shard_items_handles_empty_and_single():
    assert jobkit.shard_items([], 0, 3) == []
    assert jobkit.shard_items([1, 2], 0, 1) == [1, 2]


@pytest.mark.parametrize("index,count", [(3, 3), (-1, 3), (0, 0)])
def test_shard_items_rejects_a_bad_index(index, count):
    with pytest.raises(ValueError):
        jobkit.shard_items([1, 2, 3], index, count)


def test_shard_stem_keeps_the_plain_stem_when_unsharded():
    assert jobkit.shard_stem(Path("eval/official"), 0, 1) == Path("eval/official")
    assert jobkit.shard_stem(Path("eval/official"), 2, 4) == Path("eval/official.shard2of4")


def test_parse_shard_accepts_i_over_n():
    assert jobkit.parse_shard("0/1") == (0, 1)
    assert jobkit.parse_shard("2/4") == (2, 4)


@pytest.mark.parametrize("text", ["0", "4/4", "-1/3", "0/0", "a/b"])
def test_parse_shard_rejects_bad_values(text):
    with pytest.raises(ValueError):
        jobkit.parse_shard(text)


# ── atomic writes ────────────────────────────────────────────────────────

def test_write_json_atomic_replaces_and_leaves_no_temp(tmp_path):
    target = tmp_path / "nested" / "report.json"
    jobkit.write_json_atomic(target, {"a": 1})
    jobkit.write_json_atomic(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 2}
    assert list(target.parent.glob("*.tmp")) == []


# ── the checkpoint ───────────────────────────────────────────────────────

def test_records_are_durable_across_a_restart(tmp_path):
    out = tmp_path / "run.json"
    fp = jobkit.config_fingerprint(x=1)
    run = jobkit.ResumableRun(out, fingerprint=fp)
    run.record("a", 1)
    run.record("b", 2)

    reopened = jobkit.ResumableRun(out, fingerprint=fp)
    assert reopened.completed_keys == {"a", "b"}
    assert reopened.get("a") == 1
    assert len(reopened) == 2


def test_pending_skips_completed_units_in_input_order(tmp_path):
    out = tmp_path / "run.json"
    fp = jobkit.config_fingerprint(x=1)
    run = jobkit.ResumableRun(out, fingerprint=fp)
    run.record("b", 2)
    items = [jobkit.WorkItem("a"), jobkit.WorkItem("b"), jobkit.WorkItem("c")]
    assert [i.key for i in run.pending(items)] == ["a", "c"]
    assert run.pending(["a", "b", "d"]) == ["a", "d"]


def test_fingerprint_mismatch_raises_rather_than_mixing(tmp_path):
    out = tmp_path / "run.json"
    jobkit.ResumableRun(out, fingerprint="aaa").record("a", 1)
    with pytest.raises(jobkit.FingerprintMismatch):
        jobkit.ResumableRun(out, fingerprint="bbb")


def test_finish_writes_the_report_and_a_complete_manifest(tmp_path):
    out = tmp_path / "run.json"
    fp = jobkit.config_fingerprint(x=1)
    run = jobkit.ResumableRun(out, fingerprint=fp, shard=(1, 4))
    run.record("a", 1)
    report = run.finish({"metric": 0.5}, total=3)
    assert report["metric"] == 0.5
    assert report["_job"]["status"] == "complete"
    assert report["_job"]["shard_index"] == 1 and report["_job"]["shard_count"] == 4
    assert json.loads((tmp_path / "run.manifest.json").read_text())["completed"] == 1


def test_manifest_counts_work_units_not_the_reserved_row(tmp_path):
    fp = jobkit.config_fingerprint(x=1)
    run = jobkit.ResumableRun(tmp_path / "run.json", fingerprint=fp)
    run.record(jobkit.META_KEY, {"settings": {"depth": 100}})
    run.record("a", 1)
    assert run.manifest(status="running", total=5)["completed"] == 1


def test_touch_records_progress_without_a_final_report(tmp_path):
    out = tmp_path / "run.json"
    run = jobkit.ResumableRun(out, fingerprint=jobkit.config_fingerprint(x=1))
    run.record("a", 1)
    run.touch(total=10)
    manifest = json.loads((tmp_path / "run.manifest.json").read_text())
    assert manifest["status"] == "running" and manifest["completed"] == 1 and manifest["total"] == 10
    assert not out.is_file()


def test_a_torn_trailing_line_costs_only_the_unit_in_flight(tmp_path):
    out = tmp_path / "run.json"
    fp = jobkit.config_fingerprint(x=1)
    run = jobkit.ResumableRun(out, fingerprint=fp)
    run.record("a", 1)
    with open(run.jsonl_path, "a", encoding="utf-8") as fh:
        fh.write('{"key": "b", "value": ')  # a kill mid-write, no newline

    reopened = jobkit.ResumableRun(out, fingerprint=fp)
    assert reopened.completed_keys == {"a"}


def test_a_corrupt_interior_line_is_rejected(tmp_path):
    out = tmp_path / "run.json"
    fp = jobkit.config_fingerprint(x=1)
    run = jobkit.ResumableRun(out, fingerprint=fp)
    run.record("a", 1)
    with open(run.jsonl_path, "a", encoding="utf-8") as fh:
        fh.write("not json\n")
    run.record("c", 3)
    with pytest.raises(RuntimeError):
        jobkit.ResumableRun(out, fingerprint=fp)


def test_kill_and_resume_equals_an_uninterrupted_run(tmp_path):
    """The whole point: an interruption changes how long it takes, not what it produces."""
    keys = [f"q{i}" for i in range(20)]
    expected = {k: k.upper() for k in keys}

    def work(key):
        return key.upper()

    # Uninterrupted.
    out_clean = tmp_path / "clean.json"
    fp = jobkit.config_fingerprint(job="toy")
    clean = jobkit.ResumableRun(out_clean, fingerprint=fp)
    for key in clean.pending(keys):
        clean.record(key, work(key))
    clean.finish({"rows": clean.values()})

    # Killed after eight units, then resumed twice more with different batch sizes.
    out_killed = tmp_path / "killed.json"
    first = jobkit.ResumableRun(out_killed, fingerprint=fp)
    for key in first.pending(keys)[:8]:
        first.record(key, work(key))
    second = jobkit.ResumableRun(out_killed, fingerprint=fp)
    for key in second.pending(keys)[:5]:
        second.record(key, work(key))
    third = jobkit.ResumableRun(out_killed, fingerprint=fp)
    for key in third.pending(keys):
        third.record(key, work(key))
    resumed = third.finish({"rows": third.values()})

    assert resumed["rows"] == expected
    assert json.loads(out_clean.read_text())["rows"] == expected


# ── shard merge ──────────────────────────────────────────────────────────

def test_merge_shards_reconciles_in_order(tmp_path):
    stem = tmp_path / "official"
    fp = jobkit.config_fingerprint(job="toy")
    for index in range(3):
        run = jobkit.ResumableRun(jobkit.shard_stem(stem, index, 3), fingerprint=fp,
                                  shard=(index, 3))
        for key in jobkit.shard_items(list("abcdef"), index, 3):
            run.record(key, key.upper())
    merged = jobkit.merge_shards(stem, 3)
    assert merged["_job"]["completed"] == 6
    assert list(merged["rows"]) == list("abcdef")


def test_merge_shards_can_rebuild_a_tool_specific_payload(tmp_path):
    """A tool's shared state travels as a reserved row, so the merge needs no sidecar."""
    stem = tmp_path / "official"
    fp = jobkit.config_fingerprint(job="toy")
    for index in range(2):
        run = jobkit.ResumableRun(jobkit.shard_stem(stem, index, 2), fingerprint=fp, shard=(index, 2))
        if jobkit.META_KEY not in run:
            run.record(jobkit.META_KEY, {"settings": {"depth": 100}})
        for key in jobkit.shard_items(["a", "b"], index, 2):
            run.record(key, key.upper())

    def build(rows):
        meta = rows.pop(jobkit.META_KEY)
        return {"_settings": meta["settings"], "queries": rows}

    merged = jobkit.merge_shards(stem, 2, build=build)
    assert merged["_settings"] == {"depth": 100}
    assert merged["queries"] == {"a": "A", "b": "B"}
    assert merged["_job"]["completed"] == 2  # the reserved shared-state row is not a work unit


def test_merge_shards_rejects_conflicting_shared_state(tmp_path):
    stem = tmp_path / "official"
    fp = jobkit.config_fingerprint(job="toy")
    for index in range(2):
        run = jobkit.ResumableRun(jobkit.shard_stem(stem, index, 2), fingerprint=fp, shard=(index, 2))
        run.record(jobkit.META_KEY, {"depth": 100 if index == 0 else 500})
    with pytest.raises(RuntimeError):
        jobkit.merge_shards(stem, 2)


def test_merge_shards_refuses_mixed_fingerprints(tmp_path):
    stem = tmp_path / "official"
    jobkit.ResumableRun(jobkit.shard_stem(stem, 0, 2), fingerprint="aaa", shard=(0, 2)).record("a", 1)
    jobkit.ResumableRun(jobkit.shard_stem(stem, 1, 2), fingerprint="bbb", shard=(1, 2)).record("b", 2)
    with pytest.raises(jobkit.FingerprintMismatch):
        jobkit.merge_shards(stem, 2)


def test_merge_shards_refuses_a_duplicated_key(tmp_path):
    stem = tmp_path / "official"
    fp = jobkit.config_fingerprint(job="toy")
    jobkit.ResumableRun(jobkit.shard_stem(stem, 0, 2), fingerprint=fp, shard=(0, 2)).record("a", 1)
    jobkit.ResumableRun(jobkit.shard_stem(stem, 1, 2), fingerprint=fp, shard=(1, 2)).record("a", 2)
    with pytest.raises(RuntimeError):
        jobkit.merge_shards(stem, 2)


# ── status ───────────────────────────────────────────────────────────────

def test_iter_manifests_finds_jobs_newest_first(tmp_path):
    fp = jobkit.config_fingerprint(job="toy")
    jobkit.ResumableRun(tmp_path / "a.json", fingerprint=fp).touch(total=2)
    jobkit.ResumableRun(tmp_path / "b.json", fingerprint=fp).touch(total=3)
    (tmp_path / "notes.manifest.json").write_text('{"unrelated": true}')
    found = list(jobkit.iter_manifests(tmp_path))
    assert len(found) == 2
    assert {p.name for p, _ in found} == {"a.manifest.json", "b.manifest.json"}


def test_iter_manifests_of_a_missing_root_is_empty(tmp_path):
    assert list(jobkit.iter_manifests(tmp_path / "nope")) == []


def _load_job_status():
    name = "laya_rag_job_status"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "job_status.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_job_status_reports_an_empty_root(tmp_path, capsys):
    status = _load_job_status()
    assert status.main(["--root", str(tmp_path / "none")]) == 0
    assert "no jobkit manifests" in capsys.readouterr().out


def test_job_status_reports_progress_and_shard(tmp_path, capsys):
    status = _load_job_status()
    fp = jobkit.config_fingerprint(job="toy")
    run = jobkit.ResumableRun(tmp_path / "official.json", fingerprint=fp, shard=(0, 2))
    run.record("a", 1)
    run.touch(total=4)
    assert status.main(["--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "running" in out and "1/4" in out and "shard 0/2" in out

    assert status.main(["--root", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["fingerprint"] == fp and payload[0]["completed"] == 1
