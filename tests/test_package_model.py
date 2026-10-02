"""Offline tests for the distilled-model packaging tool."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    spec = importlib.util.spec_from_file_location("laya_rag_package_model", ROOT / "tools" / "package_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load()

_REPORT = {
    "_settings": {"dataset": "nfcorpus", "teachers": "BAAI/bge-small-en-v1.5,BAAI/bge-base-en-v1.5",
                  "student": "BAAI/bge-small-en-v1.5"},
    "queries": 323,
    "summary": {
        "zero_shot": {"nDCG@10": 0.3385},
        "adapted": {"nDCG@10": 0.3505},
        "teacher_ensemble": {"nDCG@10": 0.3798},
    },
    "comparisons": {
        "adapted": {"bootstrap": {"delta": 0.0120, "ci_low": 0.0022, "ci_high": 0.0216,
                                  "significant": True}},
    },
}


def test_render_card_states_it_is_adapted_and_carries_the_numbers():
    card = tool.render_card(_REPORT, model_id="distill-nfcorpus", pooling="cls")
    assert "domain-adapted" in card
    assert "not" in card and "zero-shot" in card
    assert "declared pooling: `cls`" in card
    assert "| zero-shot student | 0.3385 |" in card
    assert "| **this model (adapted)** | **0.3505** |" in card
    assert "| teacher ensemble | 0.3798 |" in card
    assert "significant" in card


def test_main_copies_the_model_and_writes_the_card(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_REPORT), encoding="utf-8")
    out = tmp_path / "package"

    code = tool.main(["--model", str(model), "--report", str(report), "--out", str(out)])

    assert code == 0
    assert (out / "config.json").is_file()
    card = (out / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "nfcorpus" in card
