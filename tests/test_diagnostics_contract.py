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


def test_the_claim_counts_are_stated_as_they_are():
    """28 claims resolve against the diagnostics report; 26 name their own; 11 of those have data here.

    The numbers were stated wrong three times in a row — "43 of the 54 resolve against it" — when 43 is the count of
    claims *skipped* on a checkout without the artefact (28 default-report plus 15 whose own reports are absent). The
    mistake is small and the pattern is familiar: a number that measures one thing quoted as if it measured another,
    which is class 15 of the failures ledger. This pins both halves so the next edit cannot reintroduce it.
    """
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    import claim_check as cc

    default = [c for c in cc.CHECKS if c.report is None]
    explicit = [c for c in cc.CHECKS if c.report is not None]
    assert len(cc.CHECKS) == 54
    assert len(default) == 28, f"{len(default)} claims resolve against the default report"
    assert len(explicit) == 26, f"{len(explicit)} claims name their own report"

    status = (ROOT / "docs" / "PROGRAMME-STATUS.md").read_text(encoding="utf-8")
    assert "**28** claims resolve against one diagnostics artefact" in status


def test_the_embedding_pass_is_checkpointed():
    """17,259 chunks is ~30 min; without a chunk size an interruption loses the whole pass.

    `cached_embed(..., chunk=N)` writes a `.done` marker per chunk so a stopped run resumes rather than restarts. The
    call omitted it, which is the one place in this tool where a long pass was not recoverable — the same gap the
    listwise rankings had, and the reason the rule is stated in `REPRODUCING.md`: a long job checkpoints.
    """
    writer = WRITER.read_text(encoding="utf-8")
    assert "chunk=500" in writer, "the embedding call no longer passes a chunk size"
    assert "cached_embed(" in writer
