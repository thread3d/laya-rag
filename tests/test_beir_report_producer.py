"""Every BEIR report must name the tool that produced it.

The §P4 re-take inventory could not be turned into commands because reports record their *parameters* but not their
**producer** — one parameter set can map to several scripts, so the command is not recoverable from the artefact. The
field was added to `tools/beir.py`; this guards it at the source, because the failure it prevents is invisible: the
reports still look complete without it.

A source-level check rather than a behavioural one, deliberately and with its limit stated: it cannot prove the
field is *populated* at runtime — only a written report can — so the runtime check is the next BEIR report's
`settings.producer`, which §10.8 names.
"""
from __future__ import annotations

import re
from pathlib import Path

BEIR = Path(__file__).resolve().parents[1] / "tools" / "beir.py"


def test_the_report_settings_record_the_producer_and_argv():
    text = BEIR.read_text(encoding="utf-8")
    assert '"producer":' in text, "beir.py no longer records which tool wrote its reports"
    assert '"argv":' in text, "beir.py no longer records the arguments it was run with"


def test_the_names_it_uses_are_imported():
    """The mistake this caught in review: `pathlib.Path(...)` in a module that imports `Path` directly.

    `ast.parse` accepts that, because it is valid syntax — so the guard has to look at the imports.
    """
    text = BEIR.read_text(encoding="utf-8")
    if re.search(r"\bpathlib\.Path\(", text):
        assert re.search(r"^import pathlib\b", text, re.M), "beir.py calls pathlib.Path without importing pathlib"
    if re.search(r"\bPath\(", text):
        assert re.search(r"^from pathlib import Path\b", text, re.M), "beir.py calls Path without importing it"
    if re.search(r"\bsys\.argv\b", text):
        assert re.search(r"^import sys\b", text, re.M), "beir.py reads sys.argv without importing sys"
