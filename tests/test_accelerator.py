"""Tests for device resolution.

These run without torch: the whole point of :mod:`laya_rag.accelerator` is that a caller hands
it the module it already imported, so a stand-in with the same three attributes is enough to
exercise every branch. That also means the suite can cover the CUDA and MPS paths on a machine
that has neither.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from laya_rag.accelerator import describe_device, prepare_rocm_libraries, resolve_device


def _torch(*, cuda: bool = False, count: int = 0, mps: bool = False, hip: str | None = None,
           name: str = "Fake Card", raises: Exception | None = None,
           file: str | None = None) -> SimpleNamespace:
    """A stand-in for torch exposing only what the resolver reads."""

    def _is_available() -> bool:
        if raises is not None:
            raise raises
        return cuda

    return SimpleNamespace(
        __file__=file or "/nonexistent/torch/__init__.py",
        cuda=SimpleNamespace(
            is_available=_is_available,
            device_count=lambda: count,
            current_device=lambda: 0,
            get_device_name=lambda index: name,
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        version=SimpleNamespace(hip=hip),
    )


@pytest.mark.parametrize("requested", ["cpu", "mps", "cuda", "cuda:1", "xpu"])
def test_an_explicit_device_is_returned_verbatim(requested: str) -> None:
    # Even one this host cannot serve: the caller asked for it, and the load error that
    # follows is the answer they need. Silently substituting the CPU is the bug this module
    # exists to remove.
    assert resolve_device(requested, _torch(cuda=True, count=1)) == requested


@pytest.mark.parametrize("requested", [None, "", "auto", "  auto  "])
@pytest.mark.parametrize(
    ("available", "expected"),
    [
        # CUDA wins over Metal when both are present: a host that has both is a host whose
        # Metal device is the integrated one.
        (_torch(cuda=True, count=1, mps=True), "cuda"),
        (_torch(cuda=True, count=1), "cuda"),
        (_torch(mps=True), "mps"),
        (_torch(), "cpu"),
    ],
)
def test_auto_resolves_to_the_fastest_available_device(
    requested: str | None, available: SimpleNamespace, expected: str
) -> None:
    assert resolve_device(requested, available) == expected


def test_a_rocm_build_resolves_to_cuda() -> None:
    # HIP is reached through torch's CUDA API, so a ROCm host reports cuda. The name is
    # torch's, not the vendor's -- `describe_device` is what says AMD out loud.
    rocm = _torch(cuda=True, count=3, hip="6.1.40091-a8dbc0c19", name="AMD Radeon Graphics")
    assert resolve_device("auto", rocm) == "cuda"
    assert describe_device("cuda", rocm) == (
        "cuda:0 - AMD Radeon Graphics (ROCm 6.1.40091-a8dbc0c19)"
    )


def test_devices_that_report_available_but_enumerate_none_do_not_select_cuda() -> None:
    # A ROCm build whose shared libraries cannot be loaded answers is_available() from its
    # device nodes and then fails on the first initialisation. Counting devices catches it.
    assert resolve_device("auto", _torch(cuda=True, count=0, mps=True)) == "mps"
    assert resolve_device("auto", _torch(cuda=True, count=0)) == "cpu"


def test_a_broken_accelerator_falls_back_instead_of_raising() -> None:
    broken = _torch(cuda=True, count=1, raises=RuntimeError("libcaffe2_nvrtc.so"))
    assert resolve_device("auto", broken) == "cpu"


def test_mps_absence_is_tolerated() -> None:
    # A torch built without MPS has no `backends.mps` attribute at all.
    no_mps = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
        backends=SimpleNamespace(),
        version=SimpleNamespace(hip=None),
    )
    assert resolve_device("auto", no_mps) == "cpu"


@pytest.mark.parametrize("device", ["cpu", "mps", "unknown", ""])
def test_describe_device_leaves_non_cuda_names_alone(device: str) -> None:
    assert describe_device(device, _torch(cuda=True, count=1)) == (device or "unknown")


def test_describe_device_is_a_label_not_a_promise() -> None:
    # An unavailable cuda device is described by its name rather than raising: a report that
    # cannot read the card still has to print something.
    assert describe_device("cuda", _torch(cuda=False, count=0)) == "cuda"
    assert describe_device("cuda", _torch(cuda=True, count=0)) == "cuda"


def test_describe_device_omits_the_rocm_suffix_on_a_cuda_build() -> None:
    assert describe_device("cuda", _torch(cuda=True, count=1, name="NVIDIA H100")) == (
        "cuda:0 - NVIDIA H100"
    )


def test_describe_device_does_not_re_index_an_already_indexed_device() -> None:
    # `LayaModel.device` reports what the parameters are on, which is "cuda:0", not "cuda";
    # indexing that again produced the nonsense label "cuda:0:0".
    assert describe_device("cuda:0", _torch(cuda=True, count=1, name="NVIDIA H100")) == (
        "cuda:0 - NVIDIA H100"
    )
    assert describe_device("cuda:1", _torch(cuda=True, count=2, name="NVIDIA H100")) == (
        "cuda:1 - NVIDIA H100"
    )


def test_the_rocm_repair_is_a_no_op_off_rocm() -> None:
    # A CUDA, CPU or MPS build has no bundled ROCm runtime to open, and must not be touched.
    assert prepare_rocm_libraries(_torch(cuda=True, count=1)) is False


def test_the_rocm_repair_is_silent_when_the_wheel_keeps_its_runtime_elsewhere(
    tmp_path: object,
) -> None:
    # A system-wide /opt/rocm install keeps nothing under `torch/lib`, so there is no file to
    # preload. The repair reports "not performed" rather than failing the load.
    package = tmp_path / "torch"  # type: ignore[operator]
    package.mkdir()
    assert prepare_rocm_libraries(_torch(hip="6.1", file=str(package / "__init__.py"))) is False


def test_the_rocm_repair_never_raises_on_an_unloadable_file(tmp_path: object) -> None:
    # The preload is best-effort: a file that exists but is not a loadable object -- a truncated
    # download, a wrong architecture -- must not turn device resolution into an exception.
    package = tmp_path / "torch"  # type: ignore[operator]
    (package / "lib").mkdir(parents=True)
    (package / "lib" / "libcaffe2_nvrtc.so").write_bytes(b"not an ELF file")
    assert prepare_rocm_libraries(_torch(hip="6.1", file=str(package / "__init__.py"))) is False
