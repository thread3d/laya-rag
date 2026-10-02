"""Offline tests for the generation/placement scaffold.

The scaffold exists precisely so that the *preparation* for external generation
benchmarks can be checked without any of them: no network, no model, no download. These
tests pin the two things that would make it dishonest if they broke — a generation-side
number losing its model attribution, and a citation report being read as a truth claim —

plus the shape of the target table and the safety of the two CLI modes.
"""
from __future__ import annotations

import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from laya_rag.verify import verify_citations  # noqa: E402


def _load_tool():
    """Import ``tools/generation_eval.py`` by path; it is a script, not a package module."""
    spec = importlib.util.spec_from_file_location(
        "laya_rag_generation_eval", ROOT / "tools" / "generation_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    # ``@dataclass`` resolves annotations through ``sys.modules[cls.__module__]``, so a
    # by-path module must be registered before it is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gen = _load_tool()

REFUND = "Refunds are processed within five business days to the original payment method."
SHIPPING = "Shipping is free on orders over fifty dollars in the continental United States."
PASSAGES = [("policy.md", REFUND), ("shipping.md", SHIPPING)]
WELL_CITED = "Refunds are processed within five business days [1]."
MIS_CITED = "Refunds are processed within five business days [7]."


# -- target table ------------------------------------------------------------

def test_targets_cover_the_four_named_targets_with_unique_names():
    names = [target.name for target in gen.TARGETS]
    assert len(names) == len(set(names))
    assert set(names) == {"MTEB", "KILT", "TREC-RAG", "HHEM"}


def test_every_target_has_requires_and_blocked_by():
    """None of the four is runnable as a submission, so every blocker must be recorded.

    A future runnable target would legitimately carry ``blocked_by=None``; this test
    allows that shape and only rejects an empty or whitespace-only string.
    """
    for target in gen.TARGETS:
        assert target.requires, f"{target.name} has no requires"
        assert target.scores.strip(), f"{target.name} has no scores"
        assert target.blocked_by is None or target.blocked_by.strip(), (
            f"{target.name} has an empty blocked_by"
        )
        assert target.blocked_by is not None, (
            f"{target.name} is marked runnable; none of the four named targets is"
        )


# -- the attribution rule ----------------------------------------------------

def test_generation_side_metric_always_carries_the_generator():
    record = gen.generation_side_metric(
        generator="ornith-1.5:35b", metric="faithfulness", value=0.5
    )
    assert record["generator"] == "ornith-1.5:35b"
    assert record["metric"] == "faithfulness"
    assert record["value"] == 0.5
    assert record["side"] == "generation"
    assert "ornith-1.5:35b" in record["label"]


@pytest.mark.parametrize("bad", ["", "   ", "\t"])
def test_generation_side_metric_rejects_an_empty_generator(bad):
    with pytest.raises(ValueError):
        gen.generation_side_metric(generator=bad, metric="faithfulness", value=1.0)


# -- citation report is mechanics, not truth ---------------------------------

def test_citation_report_well_cited_returns_verify_keys():
    report = gen.citation_report(WELL_CITED, PASSAGES)
    expected_keys = verify_citations(WELL_CITED, [REFUND, SHIPPING]).keys()
    assert set(expected_keys) <= set(report)
    assert report["clean"] is True
    assert report["counts"]["uncited"] == 0
    assert report["counts"]["invalid_citations"] == 0


def test_citation_report_mis_cited_is_not_clean():
    report = gen.citation_report(MIS_CITED, PASSAGES)
    assert report["invalid_citations"] == [7]
    assert report["counts"]["invalid_citations"] == 1
    assert report["clean"] is False


def test_citation_report_carries_no_truth_claim():
    for answer in (WELL_CITED, MIS_CITED):
        report = gen.citation_report(answer, PASSAGES)
        assert report["generator"] is None
        assert report["truth_claim"] is False
        assert report["mechanics_only"] is True
        assert report["scored_by"] == "engine"
        assert "correct" not in report


# -- introspection and CLI ---------------------------------------------------

def test_describe_targets_is_json_serialisable():
    described = gen.describe_targets()
    assert len(described) == len(gen.TARGETS)
    # A round-trip is the real test; ``requires`` is a tuple in Python and a list in JSON,
    # which is the only shape difference the encoder should introduce.
    payload = json.loads(json.dumps(described))
    assert len(payload) == len(described)
    for source, encoded in zip(described, payload):
        assert set(source) == {"name", "scores", "requires", "adapter", "blocked_by", "notes"}
        assert isinstance(source["requires"], tuple)
        assert source["requires"]
        assert source["blocked_by"]
        assert encoded["requires"] == list(source["requires"])
        assert encoded["name"] == source["name"]


def _block_network(monkeypatch):
    def _blocked(*args, **kwargs):  # noqa: ANN002, ANN003 - test guard
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", _blocked)


def test_main_check_returns_zero_without_network(monkeypatch, capsys):
    _block_network(monkeypatch)
    assert gen.main(["--check"]) == 0
    assert "ok" in capsys.readouterr().out


def test_main_list_returns_zero_without_network(monkeypatch, capsys):
    _block_network(monkeypatch)
    assert gen.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "MTEB" in out and "HHEM" in out


def test_main_defaults_to_list(monkeypatch, capsys):
    _block_network(monkeypatch)
    assert gen.main([]) == 0
    assert "TREC-RAG" in capsys.readouterr().out
