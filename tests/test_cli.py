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


def test_cli_verify_reports_clean_for_a_supported_citation(tmp_path, capsys):
    evidence = tmp_path / "evidence.txt"
    evidence.write_text(
        "Refunds are processed within five business days.", encoding="utf-8"
    )
    args = cli.build_parser().parse_args([
        "verify", "Refunds take five business days [1].", "--evidence", str(evidence),
    ])
    code = cli._cmd_verify(args, EngineConfig())
    out = capsys.readouterr().out
    assert code == 0
    assert "clean: yes" in out
    assert "claims 1 (cited 1, uncited 0, invalid citations 0)" in out


def test_cli_verify_without_an_answer_errors(capsys):
    args = cli.build_parser().parse_args(["verify"])
    assert cli._cmd_verify(args, EngineConfig()) == 2
    assert "provide the answer" in capsys.readouterr().err


def test_cli_verify_reads_the_answer_from_a_file(tmp_path, capsys):
    answer = tmp_path / "answer.txt"
    answer.write_text("No citations here.", encoding="utf-8")
    args = cli.build_parser().parse_args(["verify", "--answer-file", str(answer)])
    assert cli._cmd_verify(args, EngineConfig()) == 0
    assert "clean:" in capsys.readouterr().out


def test_cli_adjudicate_summarises_the_report(monkeypatch, capsys):
    class _FakeEngine:
        def __init__(self, config):
            pass

        def adjudicate_claims(self, claims, tenant=None):
            assert claims == [
                "Alpha is a true statement about refunds.",
                "Beta is entirely unknown to this corpus.",
            ]
            assert tenant == "acme"
            return {
                "supported": 1, "unsupported": 0, "not_measured": 1, "total": 2,
                "relevance_threshold": 0.5, "answerability_threshold": 0.5,
                "claims": [
                    {"verdict": "supported", "text": "Alpha is a true statement about refunds."},
                    {"verdict": "not_measured", "text": "Beta is entirely unknown to this corpus."},
                ],
            }

    monkeypatch.setattr(cli, "RagEngine", _FakeEngine)
    args = cli.build_parser().parse_args([
        "adjudicate",
        "Alpha is a true statement about refunds. Beta is entirely unknown to this corpus.",
        "--tenant", "acme",
    ])
    code = cli._cmd_adjudicate(args, EngineConfig())
    out = capsys.readouterr().out
    assert code == 0
    assert "supported 1  unsupported 0  not measured 1  of 2" in out
    assert "[supported] Alpha is a true statement about refunds." in out


# ── integrity ─────────────────────────────────────────────────────────────

def test_integrity_parser_defaults():
    args = cli.build_parser().parse_args(["integrity"])
    assert args.near_duplicates is False
    assert args.threshold == 0.85
    assert args.max_chunks == 20000
    assert args.func is cli._cmd_integrity


def test_integrity_reports_drift_without_repairing_anything(tmp_path, monkeypatch, capsys):
    """The command reads the index and prints a report; it must not mutate it."""
    from .support import stub_engine, write_corpus

    corpus = write_corpus(tmp_path / "corpus")
    engine = stub_engine(tmp_path)
    engine.index_paths([str(corpus)])
    before = len(engine.index.chunks)

    class _Engine:
        def __init__(self, config):
            pass

        has_index = staticmethod(engine.has_index)
        index = engine.index

    monkeypatch.setattr(cli, "RagEngine", _Engine)
    args = cli.build_parser().parse_args(["integrity"])
    code = cli._cmd_integrity(args, EngineConfig())
    out = capsys.readouterr().out
    assert code == 0
    assert "deleted documents:" in out
    assert "duplicate content:" in out
    assert len(engine.index.chunks) == before


def test_integrity_without_an_index_is_an_error(monkeypatch, capsys):
    class _Engine:
        def __init__(self, config):
            pass

        @staticmethod
        def has_index():
            return False

    monkeypatch.setattr(cli, "RagEngine", _Engine)
    args = cli.build_parser().parse_args(["integrity"])
    assert cli._cmd_integrity(args, EngineConfig()) == 3
    assert "no index" in capsys.readouterr().err


def test_delete_parser_collects_paths_and_doc_ids():
    args = cli.build_parser().parse_args(["delete", "a.md", "b.md", "--doc-id", "d1"])
    assert args.paths == ["a.md", "b.md"]
    assert args.doc_ids == ["d1"]


def test_cli_delete_forwards_to_the_engine(monkeypatch):
    captured: dict = {}

    class _FakeEngine:
        def __init__(self, config):
            pass

        def remove_documents(self, paths, doc_ids):
            captured["paths"] = list(paths)
            captured["doc_ids"] = list(doc_ids)
            return 3

    monkeypatch.setattr(cli, "RagEngine", _FakeEngine)
    assert cli.main(["delete", "a.md", "--doc-id", "d1", "--json"]) == 0
    assert captured == {"paths": ["a.md"], "doc_ids": ["d1"]}


def test_cli_delete_reports_an_append_only_backend(monkeypatch, capsys):
    class _RefusingEngine:
        def __init__(self, config):
            pass

        def remove_documents(self, paths, doc_ids):
            raise ValueError("the on-disk index backend is append-only")

    monkeypatch.setattr(cli, "RagEngine", _RefusingEngine)
    assert cli.main(["delete", "a.md"]) == 2
    assert "append-only" in capsys.readouterr().err


def test_compact_parser_and_forwarding(monkeypatch):
    captured: dict = {}

    class _FakeEngine:
        def __init__(self, config):
            pass

        def compact(self):
            captured["called"] = True
            return 5

    monkeypatch.setattr(cli, "RagEngine", _FakeEngine)
    assert cli.main(["compact", "--json"]) == 0
    assert captured["called"] is True
