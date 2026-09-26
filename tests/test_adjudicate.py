"""Tests for corpus-wide claim adjudication (layer 3 + variant 2a).

The citation check decides whether an answer *cites* evidence; this decides whether
the corpus *supports* what it says. The tests care about three things: that claims are
matched to the right candidates (a slicing bug would silently attribute one claim's
evidence to another), that a claim Laya was never asked about is reported as
unmeasured rather than as a confident zero, and that the report says which question it
reused.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from laya_rag import api
from laya_rag.engine import IndexNotBuiltError, RagEngine
from laya_rag.rerank import ANSWERABILITY_QUESTION, DecisionError, LayaReranker

from .support import StubModel, make_config, stub_engine, write_corpus


@pytest.fixture
def engine(tmp_path):
    """A stub-backed engine with the standard corpus already indexed."""
    eng = stub_engine(tmp_path)
    eng.index_paths([write_corpus(tmp_path / "corpus")])
    return eng


# -- batched scoring ---------------------------------------------------------

def test_score_relevance_pairs_scores_each_query_against_its_own_passage():
    reranker = LayaReranker(StubModel())
    scores, _usage = reranker.score_relevance_pairs([
        ("refunds", "Refunds are processed within five business days."),
        ("peru", "Shipping is free on orders over fifty dollars."),
    ])
    assert scores[0] > 0.5
    assert scores[1] < 0.5


def test_score_relevance_pairs_on_no_pairs_returns_nothing():
    assert LayaReranker(StubModel()).score_relevance_pairs([]) == ([], {})


def test_check_answerability_pairs_scores_each_pair():
    reranker = LayaReranker(StubModel())
    scores, _usage = reranker.check_answerability_pairs([
        ("refunds", "[1] Refunds are processed within five business days."),
        ("peru", "[1] Shipping is free on orders over fifty dollars."),
    ])
    assert scores[0] > 0.5
    assert scores[1] < 0.5


def test_pair_scoring_is_length_checked():
    class ShortModel(StubModel):
        def predict_many(self, states, questions, batch_size=8):
            return super().predict_many(states, questions, batch_size)[:1]

    with pytest.raises(DecisionError):
        LayaReranker(ShortModel()).score_relevance_pairs([("a", "b"), ("c", "d")])


# -- adjudication ------------------------------------------------------------

def test_a_claim_the_corpus_supports_is_marked_supported(engine):
    report = engine.adjudicate_claims(["Refunds are processed within five business days."])
    assert report["total"] == 1
    claim = report["claims"][0]
    assert claim["supported"] is True
    assert claim["measured"] is True
    assert claim["support_probability"] >= report["answerability_threshold"]
    assert claim["sources"], "a supported claim must name what supported it"


def test_evidence_is_attributed_to_the_right_claim(engine):
    """A slicing bug would give one claim another claim's source."""
    report = engine.adjudicate_claims([
        "Refunds are processed within five business days.",
        "Shipping is free on orders over fifty dollars.",
    ])
    by_text = {claim["text"]: claim for claim in report["claims"]}
    refund = by_text["Refunds are processed within five business days."]
    shipping = by_text["Shipping is free on orders over fifty dollars."]
    assert any("refunds.md" in source["path"] for source in refund["sources"])
    assert any("shipping.md" in source["path"] for source in shipping["sources"])
    assert not any("shipping.md" in source["path"] for source in refund["sources"])


def test_a_claim_with_no_candidate_is_not_measured_not_zero(engine):
    """The engine short-circuits; the 0.0 there is a default, not a Laya decision.

    The double scores by token overlap, so a claim has to share no vocabulary at all
    with the corpus before nothing clears the relevance bar.
    """
    report = engine.adjudicate_claims(["Zephyrous quintessential vocabulary."])
    claim = report["claims"][0]
    assert claim["supported"] is False
    assert claim["measured"] is False
    assert report["not_measured"] == 1
    assert report["unsupported"] == 0
    assert "not measured" not in str(claim["support_probability"])  # it stays a number


def test_a_measured_claim_below_the_gate_is_unsupported(engine):
    """Admitting every candidate and then rejecting it is a different outcome."""
    report = engine.adjudicate_claims(
        ["Zephyrous quintessential vocabulary."],
        relevance_threshold=0.0,
        answerability_threshold=0.5,
    )
    claim = report["claims"][0]
    assert claim["measured"] is True
    assert claim["supported"] is False
    assert report["unsupported"] == 1
    assert report["not_measured"] == 0


def test_supported_and_unmeasured_are_counted_separately(engine):
    report = engine.adjudicate_claims([
        "Refunds are processed within five business days.",
        "Zephyrous quintessential vocabulary.",
    ])
    assert report["supported"] == 1
    assert report["not_measured"] == 1
    assert report["total"] == 2


def test_the_report_names_the_question_it_reused(engine):
    """A caller must not mistake this for a measured claim-support question."""
    report = engine.adjudicate_claims(["Refunds are processed within five business days."])
    assert report["question"] == ANSWERABILITY_QUESTION["instructions"]
    assert "answered" in report["question"].lower()
    assert "evidence" in report["question"].lower()
    assert report["checkpoint"]


def test_thresholds_are_echoed_so_a_reader_can_interpret_the_scores(engine):
    report = engine.adjudicate_claims(
        ["Refunds are processed within five business days."],
        relevance_threshold=0.9,
        answerability_threshold=0.99,
    )
    assert report["relevance_threshold"] == 0.9
    assert report["answerability_threshold"] == 0.99


def test_no_claims_is_an_empty_report_not_an_error(engine):
    report = engine.adjudicate_claims([])
    assert report["total"] == 0
    assert report["claims"] == []
    assert report["note"] == "no claims to adjudicate"


def test_no_index_raises_so_the_api_can_report_a_conflict(tmp_path):
    """Same contract as query(): an unbuilt index is an error, not an empty result."""
    with pytest.raises(IndexNotBuiltError):
        stub_engine(tmp_path).adjudicate_claims(["anything at all here"])


def test_an_index_with_no_readable_documents_is_reported_with_a_reason(tmp_path):
    engine = stub_engine(tmp_path)
    empty = tmp_path / "nothing"
    empty.mkdir()
    (empty / "blob.bin").write_bytes(b"\x00\x01\x02binary\x00")
    engine.index_paths([empty])
    report = engine.adjudicate_claims(["Refunds are processed within five business days."])
    assert report["total"] == 0
    assert report["note"] == "the index is empty"


def test_claim_count_is_capped(tmp_path):
    """Cost is linear in claims, so the cap is a real guard, not decoration."""
    engine = stub_engine(tmp_path, claim_max=2)
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    report = engine.adjudicate_claims([
        "Refunds are processed within five business days.",
        "Shipping is free on orders over fifty dollars.",
        "Reset your password from Settings, then Security.",
    ])
    assert report["total"] == 2


def test_timings_and_usage_are_reported(engine):
    report = engine.adjudicate_claims(["Refunds are processed within five business days."])
    assert report["timings"]["total_seconds"] >= 0
    assert "input_tokens" in report["usage"]


def test_a_missing_lexical_channel_is_disclosed(tmp_path):
    """Dense-only recall is worse, and the report has to say so."""
    engine = stub_engine(tmp_path)
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    engine._lexical = None
    (engine.config.index_dir / "lexical.json").unlink(missing_ok=True)
    report = engine.adjudicate_claims(["Refunds are processed within five business days."])
    assert report["note"] and "lexical" in report["note"]


# -- the HTTP surface --------------------------------------------------------

@pytest.fixture
def ctx(tmp_path, monkeypatch):
    engine = stub_engine(tmp_path)
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    monkeypatch.setattr(api, "RagEngine", lambda _config: engine)
    return TestClient(api.create_app(make_config(tmp_path, warmup=False)))


def test_adjudicate_endpoint_returns_a_report(ctx):
    response = ctx.post("/v1/adjudicate", json={
        "answer": "Refunds are processed within five business days [1].",
    })
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["claims"][0]["supported"] is True
    assert "answered" in body["question"].lower()


def test_adjudicate_endpoint_splits_claims_the_same_way_verify_does(ctx):
    """Both endpoints must describe the same sentences or the reports contradict."""
    answer = (
        "Refunds are processed within five business days [1]. "
        "Shipping is free on orders over fifty dollars [2]. "
        "The capital of Peru is Lima and the Andes are tall [3]."
    )
    verified = ctx.post("/v1/verify", json={
        "answer": answer, "passages": [{"text": "Refunds are processed within five business days."}],
    }).json()
    adjudicated = ctx.post("/v1/adjudicate", json={"answer": answer}).json()
    assert [c["text"] for c in verified["claims"]] == [c["text"] for c in adjudicated["claims"]]


def test_adjudicate_before_indexing_is_a_conflict(tmp_path, monkeypatch):
    engine = stub_engine(tmp_path)
    monkeypatch.setattr(api, "RagEngine", lambda _config: engine)
    app = api.create_app(make_config(tmp_path, warmup=False))
    response = TestClient(app).post("/v1/adjudicate", json={"answer": "A claim about refunds [1]."})
    assert response.status_code == 409


def test_adjudicate_rejects_an_empty_answer(ctx):
    assert ctx.post("/v1/adjudicate", json={"answer": ""}).status_code == 422


def test_adjudicate_needs_the_token_when_configured(tmp_path, monkeypatch):
    engine = stub_engine(tmp_path)
    engine.index_paths([write_corpus(tmp_path / "corpus")])
    monkeypatch.setattr(api, "RagEngine", lambda _config: engine)
    app = api.create_app(make_config(tmp_path, warmup=False, api_key="secret"))
    client = TestClient(app)
    payload = {"answer": "Refunds are processed within five business days [1]."}
    assert client.post("/v1/adjudicate", json=payload).status_code == 401
    assert client.post(
        "/v1/adjudicate", json=payload, headers={"authorization": "Bearer secret"}
    ).status_code == 200


def test_a_query_never_adjudicates(ctx, monkeypatch):
    """Adjudication is opt-in: a plain query must not pay for it."""
    original = RagEngine.adjudicate_claims

    def forbidden(self, *_a, **_k):
        raise AssertionError("a plain query triggered adjudication")

    monkeypatch.setattr(RagEngine, "adjudicate_claims", forbidden)
    body = ctx.post("/v1/query", json={"query": "How long does a refund take?"}).json()
    assert body["passages"]
    monkeypatch.setattr(RagEngine, "adjudicate_claims", original)
