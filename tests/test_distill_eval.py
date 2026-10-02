"""Offline tests for the distillation evaluation report."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    spec = importlib.util.spec_from_file_location("laya_rag_distill_eval", ROOT / "tools" / "distill_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load()


def test_render_markdown_lists_models_and_comparisons():
    payload = {
        "_settings": {"dataset": "nfcorpus"},
        "queries": 323,
        "summary": {"zero_shot": {"nDCG@10": 0.3392}, "adapted": {"nDCG@10": 0.3505}},
        "comparisons": {"adapted": {"bootstrap": {"delta": 0.0113, "ci_low": 0.002,
                                                  "ci_high": 0.021, "significant": True}}},
    }
    text = tool.render_markdown(payload)
    assert "| adapted | 0.3505 |" in text
    assert "adapted: +0.0113" in text
    assert "significant" in text


def test_parse_args_defaults_to_nfcorpus_and_teachers():
    args = tool.parse_args([])
    assert args.dataset == "nfcorpus"
    assert "bge-large" in args.teachers
