"""Offline tests for the BEIR atom-channel experiment's reporting."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "laya_rag_beir_atom_channel", ROOT / "tools" / "beir_atom_channel.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load()


def _payload():
    row = {
        "summary": {
            "dense": {"nDCG@10": 0.70},
            "lexical": {"nDCG@10": 0.68},
            "atom": {"nDCG@10": 0.60},
            "hybrid": {"nDCG@10": 0.74},
            "hybrid+atom": {"nDCG@10": 0.76},
        },
        "queries": 300,
        "hybrid+atom vs hybrid": {
            "bootstrap": {"delta": 0.02, "ci_low": 0.005, "ci_high": 0.035, "significant": True},
            "permutation": {"p_value": 0.001},
        },
    }
    return {"_settings": {}, "datasets": {"scifact": row}}


def test_render_markdown_lists_every_channel_and_the_comparison():
    text = tool.render_markdown(_payload())
    for channel in ("dense", "lexical", "atom", "hybrid", "hybrid+atom"):
        assert channel in text
    assert "scifact: +0.0200" in text
    assert "significant" in text


def test_parse_args_defaults():
    args = tool.parse_args([])
    assert args.top_k == 100
    assert "scifact" in args.datasets
