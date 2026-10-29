"""The submission gate: every check can fail, and a failure is a verdict rather than a warning.

A gate that only ever passes is a formality. These drive each check with a stub embedder — no
download, no network — so the pass and fail paths are both exercised, including the two that matter
most in practice: non-unit vectors (a cosine index built on them retrieves garbage while looking
plausible) and non-determinism (a score that cannot be reproduced).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import submission_check as sc  # noqa: E402


class Stub:
    """An embedder whose behaviour each test chooses."""

    def __init__(self, *, scale: float = 1.0, drift: float = 0.0, dim: int = 8) -> None:
        self.scale, self.drift, self.dim = scale, drift, dim
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        base = np.ones((len(texts), self.dim), dtype=np.float64) / np.sqrt(self.dim)
        jitter = self.drift * self.calls
        return base * self.scale + jitter


def test_a_clean_embedder_passes_the_checks_that_do_not_need_a_model():
    stub = Stub()
    first = stub.embed(list(sc.PROBES))
    assert sc.check_shape(first) == []
    assert sc.check_unit_norm(first) == []
    assert sc.check_determinism(first, stub.embed(list(sc.PROBES))) == []


def test_unnormalised_vectors_are_refused_by_row():
    problems = sc.check_unit_norm(Stub(scale=2.0).embed(list(sc.PROBES)))
    assert len(problems) == len(sc.PROBES)
    assert all("norm" in problem for problem in problems)


def test_a_drifting_embedder_is_refused_as_non_deterministic():
    stub = Stub(drift=1e-6)
    problems = sc.check_determinism(stub.embed(list(sc.PROBES)), stub.embed(list(sc.PROBES)))
    assert problems and "differ by max" in problems[0]


def test_a_wrong_row_count_or_shape_is_refused():
    assert sc.check_shape(np.ones((2, 8))) != []
    assert sc.check_shape(np.ones(8)) != []


def test_a_hub_model_without_a_revision_is_refused(tmp_path):
    assert sc.check_revision("BAAI/bge-small-en-v1.5", None) == []
    assert sc.check_revision("BAAI/bge-small-en-v1.5", {"revision": "abc123"}) == []
    assert sc.check_revision("BAAI/bge-small-en-v1.5", {"revision": ""}) != []


def test_the_audit_returns_a_verdict_and_its_reasons():
    result = sc.audit(Stub(), "BAAI/bge-small-en-v1.5", resolved={"revision": "abc123"})
    assert "submittable" in result and "reasons" in result
    assert result["dim"] == 8
    assert result["pooling"] == "cls"          # the engine's verified family table, not a guess
    assert {check["check"] for check in result["checks"]} >= {
        "outputs are unit vectors", "deterministic across calls", "SearchProtocol adapter present",
    }


def test_a_failing_check_makes_the_verdict_not_submittable():
    result = sc.audit(Stub(scale=3.0), "BAAI/bge-small-en-v1.5", resolved={"revision": "abc123"})
    assert result["submittable"] is False
    assert any("norm" in reason for reason in result["reasons"])


def test_the_rendered_page_says_which_check_failed_and_why():
    rendered = sc.render(sc.audit(Stub(scale=3.0), "x/y", resolved={"revision": "r"}))
    assert "NOT SUBMITTABLE" in rendered
    assert "**FAIL**" in rendered
    assert "## Why not" in rendered
