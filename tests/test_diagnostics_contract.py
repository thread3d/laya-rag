"""The diagnostics artefact and the claim ledger must agree on field names.

`tools/claim_check.py` resolves **43 of its 54 registered claims** against `eval/benchmarks/multihop-rag/diagnostics.json`,
which `tools/multihop_diagnostics.py` writes. Neither tool imports the other, so nothing but this test stops the key
names drifting: a rename in the writer would make every one of those claims skip — and a skipped claim is indistinguishable
from a passing one in the ledger's output until you read the count.

That is not hypothetical. The listwise arm's first smoke test scored all 1,406 queries while five had been ranked, because
a *consumer* expected a filtered qrels and the producer had not filtered it. The fix was one line; finding it was the
expensive part. This test is the cheap version, applied before the run rather than after.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRITER = ROOT / "tools" / "multihop_diagnostics.py"
READER = ROOT / "tools" / "claim_check.py"


def test_the_writer_emits_the_keys_the_ledger_reads():
    writer = WRITER.read_text(encoding="utf-8")
    reader = READER.read_text(encoding="utf-8")

    # per-query pool recall at each depth — the ledger's completeness statistics
    assert 'pool_fact_recall@{depth}"]' in writer, "the writer no longer stores per-query pool recall"
    assert "pool_fact_recall@{depth}" in reader, "the ledger no longer reads per-query pool recall"

    # the summary block the ceiling claim resolves against
    assert 'summary["depth"] = per_depth' in writer, "the writer no longer stores the per-depth summary"
    assert '["depth"]["100"]["pool_ceiling_MAP@10"]' in reader, "the ledger no longer reads the depth-100 ceiling"

    # and the row container each side uses
    assert '"per_query"' in writer and 'report["per_query"]' in reader


def test_the_check_sees_both_files():
    assert WRITER.is_file() and READER.is_file()
    assert len(READER.read_text(encoding="utf-8")) > 5_000
