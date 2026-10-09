"""An adapted checkpoint can be served through the Laya-native path.

`LayaModel._resolve_spec` historically accepted only the four shipped checkpoint names.
Fine-tuning (`tools/laya_encoder_finetune.py`) produces a checkpoint elsewhere on disk, so
a directory path must resolve too — otherwise the engine's `EMBED_MODEL=""` path can never
use the model the engine trained. These tests pin both halves: a real checkpoint directory
loads, and a name that is neither shipped nor a checkpoint still fails loudly.
"""
from __future__ import annotations

import pytest

from laya_rag.config import EngineConfig
from laya_rag.model import CHECKPOINT_SUBFOLDERS, LayaModel, ModelUnavailableError


def test_checkpoint_directory_path_resolves(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"")
    config = EngineConfig()
    config.checkpoint = str(tmp_path)
    spec, subfolder = LayaModel(config)._resolve_spec()
    assert spec == str(tmp_path)
    assert subfolder is None


def test_relative_checkpoint_path_resolves_against_repo_root():
    config = EngineConfig()
    # `models` exists in the working checkout; it holds no model.safetensors itself, so this
    # only proves the relative branch reaches the filesystem check and then reports the path
    # it looked for rather than a bare "unknown checkpoint".
    config.checkpoint = "models"
    with pytest.raises(ModelUnavailableError, match="model.safetensors"):
        LayaModel(config)._resolve_spec()


def test_unknown_checkpoint_name_still_raises():
    config = EngineConfig()
    config.checkpoint = "definitely-not-a-checkpoint"
    with pytest.raises(ModelUnavailableError, match="not one of"):
        LayaModel(config)._resolve_spec()


def test_shipped_checkpoint_name_resolves_when_present():
    config = EngineConfig()
    for name, subfolder in CHECKPOINT_SUBFOLDERS.items():
        config.checkpoint = name
        if not (config.models_dir / "laya").is_dir():
            continue
        spec, resolved = LayaModel(config)._resolve_spec()
        assert resolved == subfolder
        assert spec.endswith("laya")
