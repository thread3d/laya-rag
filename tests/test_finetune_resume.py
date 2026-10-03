"""Offline tests for step-checkpointed fine-tuning.

A long distillation run that is killed should resume where it stopped, not re-start the epoch.
The distinction that matters is between `--resume` (load weights, restart training) and
`--resume-checkpoint` (restore optimiser and scheduler state from the latest step checkpoint);
these tests pin the arguments that implement the second, without running a training job.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _load():
    name = "laya_rag_finetune"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "finetune.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


finetune = _load()


def _args(**overrides):
    base = dict(checkpoint_steps=0, checkpoint_dir=None, checkpoint_keep=0,
                resume_checkpoint=False, dataset="nfcorpus")
    base.update(overrides)
    import argparse
    return argparse.Namespace(**base)


def test_checkpointing_is_off_by_default():
    assert finetune.checkpoint_kwargs(_args()) == {}


def test_resuming_without_checkpointing_is_an_error():
    with pytest.raises(ValueError, match="needs --checkpoint-steps"):
        finetune.checkpoint_kwargs(_args(resume_checkpoint=True))


def test_step_checkpointing_sets_the_path_and_cadence():
    kwargs = finetune.checkpoint_kwargs(_args(checkpoint_steps=200, checkpoint_dir="/tmp/ckpt"))
    assert kwargs == {"checkpoint_path": "/tmp/ckpt", "checkpoint_save_steps": 200}


def test_keep_limit_is_only_set_when_positive():
    assert "checkpoint_save_total_limit" not in finetune.checkpoint_kwargs(
        _args(checkpoint_steps=10, checkpoint_dir="/tmp/c"))
    assert finetune.checkpoint_kwargs(
        _args(checkpoint_steps=10, checkpoint_dir="/tmp/c", checkpoint_keep=3)
    )["checkpoint_save_total_limit"] == 3


def test_resume_restores_optimiser_state_not_just_weights():
    kwargs = finetune.checkpoint_kwargs(
        _args(checkpoint_steps=10, checkpoint_dir="/tmp/c", resume_checkpoint=True))
    assert kwargs["resume_from_checkpoint"] is True


def test_checkpoint_dir_defaults_per_dataset_and_is_overridable():
    default = finetune.checkpoint_dir(_args(dataset="scifact"))
    assert default.endswith("finetune-checkpoints/scifact")
    assert finetune.checkpoint_dir(_args(dataset="scifact", checkpoint_dir="/tmp/x")) == "/tmp/x"


def test_existing_checkpoints_are_ordered_by_step(tmp_path):
    for step in (500, 100, 1000):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "checkpoint-notanumber").mkdir()
    (tmp_path / "not-a-checkpoint.txt").write_text("x")
    assert [p.name for p in finetune.existing_checkpoints(tmp_path)] == [
        "checkpoint-100", "checkpoint-500", "checkpoint-1000"
    ]


def test_existing_checkpoints_of_a_missing_dir_is_empty(tmp_path):
    assert finetune.existing_checkpoints(tmp_path / "absent") == []


def test_resume_checkpoint_requires_the_flag_to_be_paired():
    with pytest.raises(SystemExit, match="needs --checkpoint-steps"):
        finetune.main(["--resume-checkpoint"])


def test_args_default_to_no_step_checkpointing():
    args = finetune.parse_args([])
    assert args.checkpoint_steps == 0 and args.checkpoint_keep == 0
    assert args.resume_checkpoint is False and args.checkpoint_dir is None
