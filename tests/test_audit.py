"""Tests for the hash-chained audit log.

The chain is the whole point: an intact log verifies, and any edit, deletion or reorder of a
record names the first bad one. The engine integration checks that the three mutation paths
(index, delete, reset) actually write.
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
sys.path.insert(0, str(ROOT / "src"))

from laya_rag.audit import GENESIS, AuditLog  # noqa: E402
from laya_rag.engine import RagEngine  # noqa: E402

from .support import StubModel, make_config  # noqa: E402


def test_an_empty_log_verifies_from_genesis(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    result = log.verify()
    assert result["ok"] is True and result["entries"] == 0 and result["head"] == GENESIS


def test_append_and_verify_an_intact_chain(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    first = log.append("index", detail={"chunks": 3})
    second = log.append("delete", detail={"paths": ["a.md"]})
    assert first["seq"] == 1 and second["seq"] == 2
    assert second["prev"] == first["hash"]
    result = log.verify()
    assert result["ok"] is True and result["entries"] == 2 and result["head"] == second["hash"]


def test_editing_a_record_breaks_the_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("index", detail={"chunks": 3})
    log.append("delete", detail={"paths": ["a.md"]})
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["detail"]["chunks"] = 99
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = log.verify()
    assert result["ok"] is False and result["first_bad"] == 1
    assert result["reason"] == "record hash mismatch"


def test_removing_a_record_breaks_the_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("index")
    log.append("delete")
    log.append("reset")
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
    result = log.verify()
    assert result["ok"] is False and result["first_bad"] == 2
    assert "previous-hash" in result["reason"]


def test_reordering_records_breaks_the_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("index")
    log.append("delete")
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    assert log.verify()["ok"] is False


def test_a_record_needs_an_action(tmp_path):
    with pytest.raises(ValueError):
        AuditLog(tmp_path / "audit.jsonl").append("  ")


# ── engine integration ───────────────────────────────────────────────────

def _corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir(parents=True, exist_ok=True)
    (root / "refunds.md").write_text("Refunds are processed within five business days.\n")
    (root / "shipping.md").write_text("Shipping is free over fifty dollars.\n")
    return root


def test_the_engine_logs_index_delete_and_reset(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    config = make_config(
        tmp_path, audit_log=log_path, relevance_threshold=0.0, answerability_threshold=0.0
    )
    engine = RagEngine(config, model=StubModel())
    root = _corpus(tmp_path)
    engine.index_paths([root], rebuild=True)
    engine.remove_documents([root / "shipping.md"])
    engine.reset()

    log = AuditLog(log_path)
    assert [record["action"] for record in log.entries()] == ["index", "delete", "reset"]
    assert log.entries()[0]["detail"]["chunks"] > 0
    assert log.verify()["ok"] is True


def test_no_audit_log_means_no_file(tmp_path):
    engine = RagEngine(make_config(tmp_path), model=StubModel())
    engine.index_paths([_corpus(tmp_path)], rebuild=True)
    assert not (tmp_path / "audit.jsonl").exists()


# ── the verify tool ──────────────────────────────────────────────────────

def _tool():
    name = "laya_rag_audit_verify"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "audit_verify.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_verify_tool_exit_codes(tmp_path):
    tool = _tool()
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("index")
    assert tool.main(["--log", str(path)]) == 0
    path.write_text('{"seq": 1}\n', encoding="utf-8")
    assert tool.main(["--log", str(path)]) == 1


def test_verify_tool_skips_a_missing_log(tmp_path):
    assert _tool().main(["--log", str(tmp_path / "absent.jsonl")]) == 0
