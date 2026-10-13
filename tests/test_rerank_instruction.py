"""A generative reranker is only as general as the instruction it was scored with.

Measured on `Qwen/Qwen3-Reranker-0.6B`, three instructions, same pair: the separation between a relevant
and an irrelevant passage moves **7x** — gap 0.0100 for an empty instruction, 0.0505 for the model's own
web-search default, 0.0697 for a question-answering instruction. So the knob has to exist, and a listwise
result has to name the line it used.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from laya_rag.config import EngineConfig  # noqa: E402


def test_the_instruction_defaults_to_the_models_own_line():
    assert EngineConfig().rerank_instruction == ""


def test_the_environment_can_supply_one(monkeypatch):
    monkeypatch.setenv(
        "LAYA_RAG_RERANK_INSTRUCTION",
        "Given a question, retrieve passages that answer the question",
    )
    assert EngineConfig().rerank_instruction.startswith("Given a question")


def test_the_prompt_module_passes_it_through(monkeypatch):
    """The field is only useful if the prompt reads it — the round-1 failure mode, checked."""
    from laya_rag.model import qwen3_rerank_prompt

    assert "<Instruct>: Find the abstract" in qwen3_rerank_prompt("q", "d", instruction="Find the abstract")
