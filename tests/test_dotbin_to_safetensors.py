"""The converter's safety property: it must refuse to unpickle anything that is not a tensor rebuild.

The whole point of not calling `torch.load` is that arbitrary pickles can execute code. The allowlist is
what makes that true, so it is tested directly rather than only through a successful conversion — a
converter that works on good input and executes on bad input is worse than no converter.
"""
from __future__ import annotations

import pickle
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import dotbin_to_safetensors as conv  # noqa: E402


def test_the_storage_dtypes_cover_what_the_archives_use():
    # These are the names that appeared in the doc2query archive's BINPERSID tuples.
    for name in ("FloatStorage", "HalfStorage", "LongStorage", "IntStorage"):
        assert name in conv._DTYPES


def test_a_forbidden_global_is_refused_rather_than_executed(tmp_path: Path):
    """A pickle naming `os.system` must raise at unpickle time, not run."""
    payload = pickle.dumps({"x": __import__("os").system})
    archive = tmp_path / "evil.bin"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("archive/data.pkl", payload)
    with pytest.raises((pickle.UnpicklingError, SystemExit)):
        conv.convert(archive, tmp_path / "out.safetensors")


def test_a_non_zip_is_rejected(tmp_path: Path):
    bogus = tmp_path / "notzip.bin"
    bogus.write_bytes(b"\x80\x02}q\x00.")
    with pytest.raises(SystemExit):
        conv.convert(bogus, tmp_path / "out.safetensors")


def test_check_mode_reads_no_tensors(tmp_path: Path, capsys):
    archive = tmp_path / "empty.bin"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("archive/data.pkl", b"")
    assert conv.main(["--bin", str(archive), "--out", str(tmp_path / "o.safetensors"), "--check"]) == 0
    assert "data.pkl present: True" in capsys.readouterr().out
