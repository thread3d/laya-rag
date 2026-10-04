"""Tests for three-state claim adjudication (ROADMAP Phase 5.2).

The documented limit this closes: a fabricated claim about a topic the corpus never mentions was
reported `not_measured`, because the relevance gate never reached the answerability question. That
conflates "the corpus says nothing" with "the corpus says the opposite". With a local NLI scorer
configured, the claims the gate rejected are asked about anyway, and the verdict becomes
`contradicted` or `no_coverage` instead.

The scorer is injected through the `verdict_scorer` seam, so none of this needs real weights.
"""
from __future__ import annotations

import pytest

from laya_rag.answerability import NLIAnswerability, _resolve_contradiction_index

from .support import stub_engine, write_corpus


@pytest.fixture
def engine(tmp_path):
    eng = stub_engine(tmp_path)
    eng.index_paths([write_corpus(tmp_path / "corpus")])
    return eng


def _columns(entailment: float, contradiction: float):
    return lambda premise, hypothesis: {"entailment": entailment, "contradiction": contradiction}


def _with_nli(engine, scorer):
    """Inject the verdict seam, bypassing the lazy model load."""
    engine._answerability_nli = NLIAnswerability("stub", verdict_scorer=scorer)
    engine._nli_loaded = True
    return engine


# ── the label resolver ───────────────────────────────────────────────────

def test_contradiction_index_is_read_from_the_checkpoint():
    assert _resolve_contradiction_index({0: "contradiction", 1: "entailment", 2: "neutral"}) == 0
    assert _resolve_contradiction_index({0: "ENTAILMENT", 1: "neutral", 2: "Contradiction"}) == 2


def test_a_single_label_head_has_no_contradiction_column():
    """Inventing one would turn "silent" into "refutes", which is the opposite claim."""
    assert _resolve_contradiction_index({0: "entailment"}) is None


# ── the states ───────────────────────────────────────────────────────────

def test_without_an_nli_scorer_a_rejected_claim_stays_not_measured(engine):
    report = engine.adjudicate_claims(["Refunds take ninety days."], relevance_threshold=1.01)
    assert report["claims"][0]["state"] == "not_measured"
    assert report["not_measured"] == 1
    assert report["contradicted"] == 0 and report["no_coverage"] == 0
    assert report["claims"][0]["nli"] is None


def test_a_claim_the_corpus_refutes_is_contradicted(engine):
    _with_nli(engine, _columns(0.05, 0.9))
    report = engine.adjudicate_claims(["Refunds take ninety days."], relevance_threshold=1.01)
    claim = report["claims"][0]
    assert claim["state"] == "contradicted"
    assert claim["supported"] is False
    assert report["contradicted"] == 1 and report["not_measured"] == 0
    assert claim["nli"] == {"entailment": 0.05, "contradiction": 0.9}


def test_a_claim_the_corpus_entails_is_upgraded_to_supported(engine):
    """The relevance gate missed it; the NLI pass is allowed to say the corpus supports it."""
    _with_nli(engine, _columns(0.9, 0.05))
    report = engine.adjudicate_claims(["Refunds are processed quickly."], relevance_threshold=1.01)
    claim = report["claims"][0]
    assert claim["state"] == "supported" and claim["supported"] is True
    assert report["supported"] == 1


def test_a_topic_the_corpus_never_mentions_is_no_coverage_not_not_measured(engine):
    _with_nli(engine, _columns(0.1, 0.1))
    report = engine.adjudicate_claims(["The Lima office ships for free."], relevance_threshold=1.01)
    assert report["claims"][0]["state"] == "no_coverage"
    assert report["no_coverage"] == 1 and report["not_measured"] == 0


def test_the_threshold_decides_between_contradicted_and_silent(engine):
    _with_nli(engine, _columns(0.1, 0.4))
    report = engine.adjudicate_claims(["Refunds take ninety days."], relevance_threshold=1.01)
    assert report["claims"][0]["state"] == "no_coverage"   # 0.4 is below the 0.5 default


def test_a_measured_claim_is_unaffected_by_the_nli_pass(engine):
    """The NLI pass only speaks for claims the relevance gate never reached."""
    engine._answerability_nli = NLIAnswerability("stub", verdict_scorer=_columns(0.9, 0.9))
    engine._nli_loaded = True
    report = engine.adjudicate_claims(["refunds"], relevance_threshold=0.0)
    claim = report["claims"][0]
    assert claim["measured"] is True
    assert claim["nli"] is None
    assert claim["state"] in ("supported", "unsupported")


def test_the_summary_always_carries_the_new_counts(engine):
    report = engine.adjudicate_claims(["anything at all"], relevance_threshold=1.01)
    for key in ("supported", "unsupported", "not_measured", "contradicted", "no_coverage"):
        assert key in report
    assert report["total"] == 1
