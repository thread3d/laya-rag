"""The package must import and run without torch, because that is how CI installs it.

The CI job installs the engine's dependencies with `--no-deps` plus a short explicit list — no torch, no
transformers. Everything therefore has to import lazily: a module-level `import torch` anywhere in
`src/laya_rag` would break every CI run at collection time, and the failure would look like a broken test suite
rather than a misplaced import.

This has bitten twice. Two tests added in the October round built `torch.tensor`s directly and constructed the
bi-encoder, so they passed on the development host and failed in the CI-shaped venv — caught only by running the
suite in a venv installed the way CI installs it. The check below is the mechanical version: **no heavy import at
module scope**, so the property survives without anyone re-running that experiment.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "laya_rag"

#: Imported only inside functions, where a torch-free caller never reaches them.
HEAVY = {"torch", "transformers", "sentence_transformers", "faiss", "onnxruntime"}


def module_level_heavy_imports(directory: Path | None = None) -> list[str]:
    """Heavy imports at module scope, as `file: statement` strings."""
    found: list[str] = []
    for path in sorted((directory or PACKAGE).glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module scope only — nested imports are lazy by construction
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in HEAVY:
                        found.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in HEAVY:
                    found.append(f"{path.name}: from {node.module}")
    return found


def test_no_heavy_import_runs_at_module_scope():
    offenders = module_level_heavy_imports()
    assert not offenders, (
        "these would break a torch-free install at import time:\n  " + "\n  ".join(offenders)
    )


def test_the_engine_imports_without_torch_installed():
    """The behavioural half: importing the package must not require the heavy stack."""
    import subprocess
    import sys

    code = (
        "import sys; "
        "sys.modules['torch'] = None; "          # a poisoned entry: any import of torch now raises
        "import laya_rag, laya_rag.engine, laya_rag.extractive, laya_rag.verify; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr[-400:]


def test_the_scan_sees_a_real_tree():
    assert len(list(PACKAGE.glob("*.py"))) > 15
    planted = PACKAGE / "_planted_heavy_import_probe.py"
    try:
        planted.write_text("import torch\n", encoding="utf-8")
        assert module_level_heavy_imports(), "a module-level torch import was not detected"
    finally:
        planted.unlink(missing_ok=True)
