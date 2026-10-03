"""Offline tests for the late-interaction feasibility probe.

The probe's expensive half needs the real checkpoint and the cached pool; these pin the
scoring rule and the report, because a MaxSim that is subtly wrong would be read as
"late interaction does not help here" when it only measured a bug.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "laya_rag_maxsim_probe", ROOT / "tools" / "laya_maxsim_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load()


def test_maxsim_of_identical_token_sets_is_the_query_length():
    tokens = np.eye(3, dtype=np.float32)
    # Every query token finds itself, so each max is 1.0.
    assert probe.maxsim(tokens, tokens) == pytest.approx(3.0)


def test_maxsim_of_orthogonal_tokens_is_zero():
    query = np.eye(3, dtype=np.float32)
    document = np.eye(3, 6, dtype=np.float32)[:, 3:]
    assert probe.maxsim(query, document) == pytest.approx(0.0)


def test_maxsim_takes_the_best_document_token_per_query_token():
    """Different query tokens may match different document tokens — the point of MaxSim."""
    query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    document = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    # Token 0 matches document token 1, token 1 matches document token 0: 1.0 + 1.0.
    assert probe.maxsim(query, document) == pytest.approx(2.0)


def test_maxsim_of_an_empty_side_is_zero():
    empty = np.zeros((0, 4), dtype=np.float32)
    assert probe.maxsim(empty, np.ones((2, 4), dtype=np.float32)) == 0.0
    assert probe.maxsim(np.ones((2, 4), dtype=np.float32), empty) == 0.0


def test_render_markdown_reports_the_delta_and_the_index_cost():
    result = {
        "_settings": {
            "limit_queries": 200, "depth": 100, "ce_cache": "ce-scores.json",
            "max_length": 512, "hidden_size": 1024,
        },
        "summary": {
            "MAP@10": {"maxsim": 0.31, "fused": 0.25},
            "Hits@4": {"maxsim": 0.70, "fused": 0.66},
        },
        "comparisons": {
            "maxsim vs fused MAP@10": {
                "bootstrap": {"delta": 0.06, "ci_low": 0.01, "ci_high": 0.11, "significant": True},
                "permutation": {"p_value": 0.001},
            }
        },
        "cost": {
            "mean_tokens_per_chunk": 180.0, "tokens_per_chunk_p95": 320,
            "resident_bytes_per_chunk_fp32": 737280,
            "projected_gib_at_1m_chunks": 0.69, "projected_gib_at_1m_chunks_int8": 0.17,
        },
    }
    text = probe.render_markdown(result)
    assert "MaxSim" in text
    assert "+0.0600" in text
    assert "0.69 GiB" in text
    assert "0.17 GiB" in text
