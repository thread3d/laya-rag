"""Offline tests for the BEIR coverage experiment's reporting."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "laya_rag_beir_coverage", ROOT / "tools" / "beir_coverage.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load()


def _payload():
    row = {
        "documents": 100,
        "summary": {
            "fused": {"nDCG@10": 0.70},
            "coverage-0.5": {"nDCG@10": 0.69},
            "coverage-nored-0.5": {"nDCG@10": 0.68},
        },
        "comparisons": {
            "coverage-0.5": {"bootstrap": {"delta": -0.01, "ci_low": -0.03, "ci_high": 0.01,
                                           "significant": False}},
        },
    }
    return {"_settings": {}, "datasets": {"scifact": row}}


def test_render_markdown_lists_datasets_and_comparisons():
    text = tool.render_markdown(_payload())
    assert "| scifact |" in text
    assert "coverage-0.5" in text
    assert "not significant" in text


def test_parse_args_defaults():
    args = tool.parse_args([])
    assert args.k == 10
    assert "scifact" in args.datasets
