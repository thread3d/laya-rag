"""Which device the torch-backed models actually run on.

Five models are loaded in-process by the engine -- the bi-encoder, the cross-encoder
reranker, the generative reranker, the NLI answerability head and the learned-sparse encoder
-- and each of them used to resolve ``auto`` for itself. Each resolved it the same way:
Apple's Metal backend if it was present, else CPU. That was right for the Mac the engine was
developed on and wrong in a *silent* way everywhere else. A CUDA host and a ROCm host both
fell through to ``cpu``, produced CPU-only numbers, and left nothing in the result that said
so -- which is the failure mode this project is otherwise careful about.

``auto`` here means the fastest device the *installed torch* can actually reach, in
descending order: ``cuda``, then ``mps``, then ``cpu``. ROCm builds answer ``cuda`` for the
first step because HIP is exposed through torch's CUDA API; the name is torch's, not the
vendor's, which is why :func:`describe_device` exists to say so out loud. An explicit request
is honoured verbatim -- including one this host cannot serve -- because a caller that asked
for ``cuda`` needs the load error, not a silent downgrade to the CPU it was trying to avoid.

Nothing in this package imports torch at import time, and neither does this module: a caller
passes the module it has already imported, so the engine's import path stays free of torch's
several-hundred-millisecond import for callers that never load a model.
"""
from __future__ import annotations

import os
from typing import Any

#: Values of ``DEVICE`` that mean "you choose" rather than naming a device.
AUTO_ALIASES = frozenset({"", "auto"})

#: The ROCm library a ROCm wheel opens by bare name; see :func:`prepare_rocm_libraries`.
_NVRTC_LIBRARY = "libcaffe2_nvrtc.so"


def resolve_device(requested: str | None, torch: Any | None = None) -> str:
    """Resolve a configured device name to the device a model should be moved to.

    ``requested`` is the configured value -- ``DEVICE``, ``SparseEncoder.device``, a tool's
    ``--device``. Any name outside :data:`AUTO_ALIASES` is returned unchanged, stripped of
    surrounding space. For an auto request, ``torch`` is used when given and imported when
    not; a caller that already has it should pass it, since the import is the expensive part.

    The one side effect to know about is :func:`prepare_rocm_libraries`, which runs on every
    ROCm host -- including a call that names a device explicitly, because the compile worker
    torch starts probes the accelerator wherever the model itself was put. It is a no-op
    everywhere else.
    """
    if requested is not None:
        requested = str(requested).strip()
    if torch is None:  # pragma: no cover - reached only by callers that pass no module
        import torch as torch_module

        torch = torch_module
    # The ROCm loader repair runs before the early return, not inside the CUDA branch: a
    # `torch.compile`d run starts an inductor compile worker that asks for the current CUDA
    # device wherever the model itself was put, so an explicit `DEVICE=cpu` on a ROCm build
    # fails exactly the way a GPU run does. Off ROCm this is a no-op.
    prepare_rocm_libraries(torch)
    if requested and requested not in AUTO_ALIASES:
        return requested
    if _cuda_usable(torch):
        return "cuda"
    if _mps_usable(torch):
        return "mps"
    return "cpu"


def prepare_rocm_libraries(torch: Any) -> bool:
    """Pre-open the ROCm runtime that a ROCm wheel otherwise finds only through the loader.

    A ROCm torch ships its runtime beside ``torch/lib`` and opens ``libcaffe2_nvrtc.so`` by
    *bare name*, which the dynamic loader resolves through ``LD_LIBRARY_PATH`` rather than
    through the wheel's own ``$ORIGIN`` rpath. Without that variable -- in any process the
    user did not wrap by hand, including every test runner and every ``make`` target -- device
    enumeration still succeeds and the first initialisation raises
    ``Error in dlopen: libcaffe2_nvrtc.so: cannot open shared object file``, which reads like a
    missing file and is really a missing search path.

    Two repairs, because there are two kinds of process involved:

    * opening the library by absolute path puts its soname in *this* process's loader table, so
      the later bare-name lookup finds the object already loaded;
    * writing the directory into ``LD_LIBRARY_PATH`` is what fixes the *child* processes torch
      starts itself. ``torch.compile`` forks an inductor compile worker, and a new process gets
      a fresh loader with none of the above -- it fails, and reports an uncaught exception in a
      subprocess that the caller never asked for. A child inherits this environment, so setting
      it before any compile is attempted is the whole fix.

    Best-effort and silent: it never raises, does nothing on a CPU, CUDA or MPS build, and a
    host whose environment already works is unaffected. A host with a different ROCm layout (a
    system-wide ``/opt/rocm`` rather than the wheel's bundled copy) finds no file to open and
    gets its environment left alone.
    """
    if not getattr(getattr(torch, "version", None), "hip", None):
        return False
    try:
        import ctypes

        lib_dir = os.path.join(os.path.dirname(str(torch.__file__)), "lib")
        target = os.path.join(lib_dir, _NVRTC_LIBRARY)
        if not os.path.exists(target):
            return False
        ctypes.CDLL(target, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
        _inherit_library_path(lib_dir)
        return True
    except Exception:  # noqa: BLE001 - a repair that cannot run is not an error
        return False


def _inherit_library_path(lib_dir: str) -> None:
    """Add ``lib_dir`` to ``LD_LIBRARY_PATH`` for the processes that will be spawned from here.

    This cannot affect the running process -- the loader read the variable at start-up -- and
    that is fine: the parent was repaired directly. It exists for ``torch.compile``'s compile
    worker, which is a new process with a new loader.
    """
    current = os.environ.get("LD_LIBRARY_PATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if lib_dir in entries:
        return
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([lib_dir, *entries])


def _cuda_usable(torch: Any) -> bool:
    """Whether torch can run kernels on a CUDA *or ROCm* device right now.

    The two questions are not the same one, which is why both are asked: a ROCm build whose
    runtime cannot be reached answers ``is_available()`` from its device nodes and then raises
    on the first initialisation. Counting devices catches the empty case;
    :func:`prepare_rocm_libraries` has already repaired the common cause of the *non*-empty one
    by the time this runs, and is called from :func:`resolve_device` rather than here so that it
    also covers runs that name a device explicitly.

    This deliberately stops at "a device is there and can be enumerated" rather than running a
    probe kernel: resolving a device is not the place to allocate a GPU context, and a build
    that enumerates its devices and then fails is exactly the case where the load error naming
    ``cuda`` is more useful than a silent fall back to the CPU this module exists to avoid.

    Any failure means "no accelerator" rather than propagating: this function's whole job is
    to choose a device, and a broken accelerator is the case where ``cpu`` is the useful answer.
    """
    try:
        return bool(torch.cuda.is_available()) and int(torch.cuda.device_count()) > 0
    except Exception:  # noqa: BLE001 - an unreadable accelerator means "use the CPU"
        return False


def _mps_usable(torch: Any) -> bool:
    """Whether Apple's Metal backend is present. Guarded, since the namespace is macOS-only."""
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001 - a torch built without MPS has no such backend
        return False


def describe_device(device: str, torch: Any | None = None) -> str:
    """A one-line label naming the real accelerator stack behind ``device``.

    ``cuda`` is torch's API name, not a vendor: on a ROCm build the same string denotes an AMD
    GPU, so a report printing only the device name misattributes the measurement. Returns the
    bare device name whenever the detail cannot be read -- on a device this host does not
    have, or with no torch to ask -- so it is safe to call on any output path.
    """
    name = str(device or "unknown")
    if not name.startswith("cuda"):
        return name
    try:
        if torch is None:
            import torch as torch_module

            torch = torch_module
        prepare_rocm_libraries(torch)
        if not (torch.cuda.is_available() and int(torch.cuda.device_count()) > 0):
            return name
        index = torch.cuda.current_device()
        card = str(torch.cuda.get_device_name(index))
        # The engine reports the device a model's parameters are on, which is already indexed
        # ("cuda:1"); a bare "cuda" is not. Appending unconditionally produced "cuda:0:0".
        base = name if ":" in name else f"{name}:{index}"
        label = f"{base} - {card}"
        hip = getattr(getattr(torch, "version", None), "hip", None)
        return f"{label} (ROCm {hip})" if hip else label
    except Exception:  # noqa: BLE001 - a label is never worth raising over
        return name
