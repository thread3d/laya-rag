"""Tests for the command-line surface the engine exposes.

Only the argument plumbing is tested here (no index and no model): the parser must accept
the documented flags and `_cmd_query` must forward them to the engine, because a capability
the service has but the CLI drops is a capability half-shipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laya_rag import cli  # noqa: E402
from laya_rag.config import EngineConfig  # noqa: E402


class _FakeResult:
    def as_dict(self) -> dict:
        return {"query": "x", "passages": []}


def _capturing_engine(captured: dict):
    class _FakeEngine:
        def __init__(self, config):
            captured["config"] = config

        def query(self, query, **kwargs):
            captured["query"] = query
            captured.update(kwargs)
            return _FakeResult()

    return _FakeEngine


def test_query_parser_defaults_tenant_to_none():
    args = cli.build_parser().parse_args(["query", "how long"])
    assert args.tenant is None


def test_cli_query_forwards_the_tenant(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(cli, "RagEngine", _capturing_engine(captured))
    args = cli.build_parser().parse_args(["query", "how long", "--tenant", "acme", "--json"])
    assert args.tenant == "acme"
    code = cli._cmd_query(args, EngineConfig())
    assert code == 0
    assert captured["query"] == "how long"
    assert captured["tenant"] == "acme"


def test_cli_query_without_a_tenant_passes_none(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(cli, "RagEngine", _capturing_engine(captured))
    args = cli.build_parser().parse_args(["query", "how long", "--json"])
    cli._cmd_query(args, EngineConfig())
    assert captured["tenant"] is None
