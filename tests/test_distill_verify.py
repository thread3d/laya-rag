"""Offline tests for the distilled-encoder export check.

The export was produced so MTEB could load it, and an export that loads but silently differs from
its checkpoint would corrupt every downstream number without any metric flagging it. So the
comparison must be exact (an export is a copy, not a re-derivation) and a missing artifact must
report *absent* rather than pass — the same rule `make claim-check` follows.
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
sys.path.insert(0, str(ROOT / "src"))


def _load():
    name = "laya_rag_distill_verify"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "distill_verify.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dv = _load()


def test_identical_embeddings_compare_as_exact():
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = dv.compare_embeddings(vectors, vectors.copy())
    assert result["comparable"] is True
    assert result["max_abs_diff"] == 0.0
    assert result["dimension"] == 2


def test_a_single_changed_value_is_reported_not_averaged_away():
    reference = np.zeros((3, 4), dtype=np.float32)
    candidate = reference.copy()
    candidate[1, 2] = 0.5
    result = dv.compare_embeddings(reference, candidate)
    assert result["max_abs_diff"] == pytest.approx(0.5)
    assert result["mean_abs_diff"] < result["max_abs_diff"]


def test_shape_mismatch_is_not_comparable_rather_than_a_number():
    result = dv.compare_embeddings(np.zeros((2, 4), np.float32), np.zeros((2, 8), np.float32))
    assert result["comparable"] is False
    assert "max_abs_diff" not in result


def test_a_missing_artifact_is_absent_not_a_pass(tmp_path, capsys):
    """The check must refuse to pass when the package was never built."""
    code = dv.main(["--package", str(tmp_path / "nope"), "--source", str(tmp_path / "nope2"),
                    "--check", "--out", str(tmp_path / "out")])
    assert code == 2
    assert "ABSENT" in capsys.readouterr().err


def test_the_default_paths_point_at_the_exported_package():
    args = dv.parse_args([])
    assert args.package.endswith("distill-nfcorpus-package")
    assert args.source.endswith("distill-nfcorpus-model")
    assert args.tolerance <= 1e-6, "an export is a copy; the allowance is float noise only"
