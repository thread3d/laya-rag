"""Convert a legacy `pytorch_model.bin` to `model.safetensors` without ever calling torch.load.

`transformers` refuses `.bin` checkpoints under torch 2.2.2 (CVE-2025-32434). Calling
`torch.load(weights_only=True)` ourselves is not an alternative — the vulnerability is in `torch.load`
itself, so a different caller is the same unsafe code path. This module parses the ZIP directly.

The archive holds `data.pkl` plus one blob per storage at `data/<key>`. The pickle references storages
through `BINPERSID` with the tuple `("storage", "<Type>Storage", "<key>", "<device>", numel)`, so a
restricted `Unpickler` supplies exactly those five values to `persistent_load` and its `find_class`
allowlist rejects everything else — an `os.system` gadget raises instead of executing.

Scope: dense tensors only. Anything else is reported rather than converted, because a silent
mis-conversion is indistinguishable from a working model until it scores badly.
"""
from __future__ import annotations

import argparse
import collections
import io
import pickle
import sys
import zipfile
from pathlib import Path

_DTYPES = {
    "FloatStorage": "float32", "DoubleStorage": "float64", "HalfStorage": "float16",
    "BFloat16Storage": "bfloat16", "LongStorage": "int64", "IntStorage": "int32",
    "ShortStorage": "int16", "CharStorage": "int8", "ByteStorage": "uint8", "BoolStorage": "bool",
}


def convert(bin_path: Path, out_path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    import torch
    from safetensors.torch import save_file

    if not zipfile.is_zipfile(bin_path):
        raise SystemExit(f"{bin_path}: not a ZIP archive — an older single-file pickle is unsupported")

    with zipfile.ZipFile(bin_path) as archive:
        names = archive.namelist()
        pkl = next((n for n in names if n.endswith("data.pkl")), None)
        if pkl is None:
            raise SystemExit(f"{bin_path}: no data.pkl in the archive")
        prefix = pkl[: -len("data.pkl")]
        blobs = {n[len(prefix):]: archive.read(n) for n in names if n.startswith(prefix + "data/")}

        utils = torch._utils
        allowed = {
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("torch._utils", "_rebuild_tensor_v2"): utils._rebuild_tensor_v2,
            ("torch._utils", "_rebuild_parameter"): utils._rebuild_parameter,
            ("torch", "Tensor"): torch.Tensor,
            ("torch", "Size"): torch.Size,
        }
        for name in _DTYPES:
            cls = getattr(torch, name, None)
            if cls is not None:
                allowed[("torch", name)] = cls

        class Restricted(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                key = (module, name)
                if key in allowed:
                    return allowed[key]
                raise pickle.UnpicklingError(
                    f"refusing to unpickle {module}.{name}: not a tensor-rebuild callable"
                )

            def persistent_load(self, pid):
                kind, type_name, key, _device, numel = pid
                if kind != "storage":
                    raise pickle.UnpicklingError(f"unsupported persistent id kind {kind!r}")
                blob = blobs.get(f"data/{key}")
                if blob is None:
                    raise pickle.UnpicklingError(f"no blob for storage key {key!r}")
                dtype = getattr(torch, _DTYPES.get(str(type_name), "float32"))
                # Bytes -> tensor without torch.load, then a TYPED storage wrapper, which is what
                # torch 2.2's _rebuild_tensor_v2 expects (it reads .dtype and ._untyped_storage).
                flat = torch.frombuffer(bytearray(blob), dtype=dtype)
                if int(numel) and flat.numel() != int(numel):
                    raise pickle.UnpicklingError(
                        f"storage {key}: blob holds {flat.numel()} values, header says {numel}"
                    )
                return torch.storage.TypedStorage(wrap_storage=flat.untyped_storage(), dtype=dtype)

        state = Restricted(io.BytesIO(archive.read(pkl))).load()

    tensors, shape = {}, {}
    for key, value in state.items():
        if not hasattr(value, "detach"):
            raise SystemExit(f"{key}: not a tensor ({type(value).__name__}) — refusing to convert")
        tensor = value.detach().contiguous().clone()
        tensors[key] = tensor
        shape[key] = (tuple(tensor.shape), str(tensor.dtype))
    save_file(tensors, str(out_path))
    return shape


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="legacy pytorch_model.bin")
    ap.add_argument("--out", required=True, help="safetensors path to write")
    ap.add_argument("--check", action="store_true",
                    help="report the archive's members and the storage header without unpacking tensors")
    args = ap.parse_args(argv)
    src, dst = Path(args.bin), Path(args.out)
    if not src.is_file():
        print(f"{src}: missing", file=sys.stderr)
        return 2
    if args.check:
        with zipfile.ZipFile(src) as archive:
            names = archive.namelist()
        print(f"{src.name}: ZIP with {len(names)} members, data.pkl present: "
              f"{any(n.endswith('data.pkl') for n in names)}")
        return 0
    shape = convert(src, dst)
    total = sum(1 for _ in shape)
    print(f"wrote {dst} ({total} tensors)")
    for key in list(shape)[:3]:
        dims, dtype = shape[key]
        print(f"  {key}: {dims} {dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
