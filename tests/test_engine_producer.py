"""Every engine report must name the tool that produced it, as BEIR reports do.

`tools/engine_multihop.py` wrote `producer: None` for ten re-takes in a row, which is why the §P4 mapping needed
four static screens and still ended in a tie. The field is now written there too; this guards it at the source.

A source-level check with its limit stated, as in `test_beir_report_producer.py`: it cannot prove the field is
*populated* at runtime — only a written report can — so the runtime check is the next engine report's
`settings.producer`, which §10.8 names.
"""
from __future__ import annotations

import re
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "engine_multihop.py"


def test_the_engine_settings_record_the_producer_and_argv():
    text = TOOL.read_text(encoding="utf-8")
    assert '"producer":' in text, "engine_multihop.py no longer records which tool wrote its reports"
    assert '"argv":' in text, "engine_multihop.py no longer records the arguments it was run with"


def test_the_names_it_uses_are_imported():
    """`ast.parse` accepts `sys.argv` without `import sys`; the guard has to look at the imports."""
    text = TOOL.read_text(encoding="utf-8")
    if re.search(r"\bsys\.argv\b", text):
        assert re.search(r"^import sys\b", text, re.M), "engine_multihop.py reads sys.argv without importing sys"
    if re.search(r"\bPath\(__file__\)", text):
        assert re.search(r"^from pathlib import Path\b", text, re.M), "engine_multihop.py calls Path without importing it"
