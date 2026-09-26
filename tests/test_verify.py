"""Tests for the deterministic citation and quote checks.

These checks are the only part of the answer pipeline that involves no model and no
threshold, so the tests are about precision: a false "quote not found" would train a
reader to ignore the check, and a missed invalid citation would let it pass as clean.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from laya_rag import api
from laya_rag.verify import (
    extract_citations,
    extract_quotes,
    normalize,
    split_claims,
    verify_citations,
)

from .support import make_config, stub_engine

REFUND = "Refunds are processed within five business days to the original payment method."
SHIPPING = "Shipping is free on orders over fifty dollars in the continental United States."
EVIDENCE = [REFUND, SHIPPING]


# -- normalisation -----------------------------------------------------------

def test_normalize_folds_case_whitespace_and_quote_style():
    assert normalize("  Refunds\u00a0ARE  processed\n within \u201cfive\u201d days. ") == (
        'refunds are processed within "five" days.'
    )


def test_normalize_does_not_rewrite_words():
    """Anything cleverer than folding would turn 'found' into a lie."""
    assert normalize("five business days") != normalize("5 business days")


# -- citations ---------------------------------------------------------------

def test_extract_citations_is_distinct_and_ordered():
    assert extract_citations("First [2], then [1], again [2].") == [2, 1]


def test_extract_citations_handles_adjacent_markers():
    assert extract_citations("Supported by two [1][2].") == [1, 2]


def test_extract_citations_ignores_markdown_links():
    """A link's text is not a citation, but a citation inside one still is."""
    assert extract_citations("See [the docs](https://example.com) for more.") == []
    assert extract_citations("See [the docs](https://x/1) [3].") == [3]


def test_extract_citations_on_empty_text():
    assert extract_citations("") == []


# -- quotes ------------------------------------------------------------------

def test_extract_quotes_finds_backticks_and_double_quotes():
    found = extract_quotes('Use `docker compose run` and "free on orders" here.')
    assert found == ["docker compose run", "free on orders"]


def test_extract_quotes_ignores_short_spans():
    assert extract_quotes('a `x` and "it" here') == []


def test_extract_quotes_handles_curly_quotes():
    assert extract_quotes("It says \u201cfive business days\u201d exactly.") == ["five business days"]


def test_extract_quotes_skips_fenced_code_blocks():
    """A fenced block is content, handled separately from inline quoting."""
    text = "Intro.\n\n```sh\ndocker compose run --build --rm laya\n```\n"
    assert extract_quotes(text) == []


def test_extract_quotes_deduplicates():
    assert extract_quotes('`alpha` and again `alpha`') == ["alpha"]


# -- claim splitting ---------------------------------------------------------

def test_split_claims_drops_headings_and_code():
    answer = (
        "## Shipping\n\n"
        "Shipping is free on orders over fifty dollars [1].\n\n"
        "```sh\ndocker compose run --rm laya\n```\n"
    )
    claims = split_claims(answer)
    assert claims == ["Shipping is free on orders over fifty dollars [1]."]


def test_split_claims_strips_bullet_markers():
    claims = split_claims("- Refunds take five business days to process [1].")
    assert claims == ["Refunds take five business days to process [1]."]


def test_split_claims_skips_lead_ins_that_end_in_a_colon():
    """'From the repository root:' introduces a block; it asserts nothing."""
    answer = "From the repository root:\n\n```sh\nmake setup\n```\n"
    assert split_claims(answer) == []


def test_split_claims_skips_table_rules_and_short_fragments():
    assert split_claims("|---|---|") == []
    assert split_claims("See [1].") == []


def test_split_claims_separates_multiple_sentences_on_one_line():
    answer = "Refunds take five business days to arrive [1]. Shipping is free over fifty dollars [2]."
    assert len(split_claims(answer)) == 2


# -- the report --------------------------------------------------------------

def test_a_fully_cited_answer_is_clean():
    report = verify_citations("Refunds arrive within five business days [1].", EVIDENCE)
    assert report["clean"] is True
    assert report["counts"]["uncited"] == 0
    assert "every claim cites evidence" in report["summary"]


def test_an_uncited_claim_is_reported():
    report = verify_citations(
        "Refunds arrive within five business days [1]. Shipping is probably free as well.", EVIDENCE
    )
    assert report["clean"] is False
    assert report["counts"]["uncited"] == 1
    assert [c["text"] for c in report["claims"] if c["uncited"]]


def test_an_out_of_range_citation_is_caught_even_in_a_short_fragment():
    """The fragment is not a checkable claim, but the bad citation must not vanish."""
    report = verify_citations(
        "Refunds arrive within five business days [1]. See [7].", EVIDENCE
    )
    assert report["invalid_citations"] == [7]
    assert report["clean"] is False
    assert "points outside" in report["summary"]


def test_a_quote_present_in_the_evidence_is_found():
    report = verify_citations(
        'The policy says "processed within five business days" exactly [1].', EVIDENCE
    )
    assert report["counts"]["quotes_not_found"] == 0
    assert report["quotes"][0]["found_in"] == [1]


def test_a_fabricated_quote_is_flagged():
    report = verify_citations(
        'The policy says "processed within two hours" exactly [1].', EVIDENCE
    )
    assert report["counts"]["quotes_not_found"] == 1
    assert report["clean"] is False
    assert "not in the evidence" in report["summary"]


def test_a_quote_is_found_across_a_line_break_in_the_passage():
    wrapped = "Refunds are processed within five\nbusiness days to the original payment method."
    report = verify_citations('It says "processed within five business days" [1].', [wrapped])
    assert report["counts"]["quotes_not_found"] == 0


def test_a_quote_is_found_even_when_its_sentence_cites_nothing():
    """Faithful but uncited is a different problem from fabricated, and not the same fix."""
    report = verify_citations('It says "processed within five business days" exactly.', EVIDENCE)
    assert report["counts"]["quotes_found"] == 1
    assert report["counts"]["uncited"] == 1


def test_an_answer_with_no_citations_at_all_is_not_clean():
    report = verify_citations(
        "Refunds arrive within five business days. Shipping is free over fifty dollars.", EVIDENCE
    )
    assert report["clean"] is False
    assert report["counts"]["uncited"] == report["counts"]["claims"]
    assert "no claim cites evidence" in report["summary"]


def test_an_answer_with_nothing_checkable_says_so():
    report = verify_citations("## Heading\n", EVIDENCE)
    assert report["clean"] is False
    assert report["summary"] == "Nothing in the answer was checkable as a claim."


def test_the_summary_never_claims_the_answer_is_true():
    """The wording carries the limitation, because a badge cannot."""
    for answer in (
        "Refunds arrive within five business days [1].",
        "Uncited sentence about refunds and shipping policy.",
    ):
        summary = verify_citations(answer, EVIDENCE)["summary"].lower()
        assert "true" not in summary
        assert "verified" not in summary
        assert "supported" not in summary


def test_single_passage_wording_reads_naturally():
    report = verify_citations("Refunds arrive within five business days [9].", [REFUND])
    assert "outside the evidence" in report["summary"]
    assert "1 passages" not in report["summary"]


def test_counts_stay_consistent():
    report = verify_citations(
        'Refunds arrive within five business days [1]. Shipping costs money now. '
        'It says "free on orders over fifty dollars" verbatim [2]. See [9].',
        EVIDENCE,
    )
    counts = report["counts"]
    assert counts["claims"] == counts["cited"] + counts["uncited"]
    assert counts["quotes_checked"] == counts["quotes_found"] + counts["quotes_not_found"]
    assert counts["claims"] == len(report["claims"])


# -- the HTTP surface --------------------------------------------------------

@pytest.fixture
def ctx(tmp_path, monkeypatch):
    engine = stub_engine(tmp_path)
    monkeypatch.setattr(api, "RagEngine", lambda _config: engine)
    return TestClient(api.create_app(make_config(tmp_path, warmup=False)))


def test_verify_endpoint_returns_a_report(ctx):
    response = ctx.post("/v1/verify", json={
        "answer": "Refunds arrive within five business days [1].",
        "passages": [{"text": REFUND, "path": "refunds.md"}],
    })
    assert response.status_code == 200
    body = response.json()
    assert body["clean"] is True
    assert body["counts"]["claims"] == 1
    assert body["passages"] == 1


def test_verify_endpoint_rejects_an_empty_answer(ctx):
    assert ctx.post("/v1/verify", json={"answer": "", "passages": [{"text": REFUND}]}).status_code == 422


def test_verify_endpoint_requires_at_least_one_passage(ctx):
    assert ctx.post("/v1/verify", json={"answer": "A claim about refunds [1].", "passages": []}).status_code == 422


def test_verify_endpoint_needs_the_token_when_configured(tmp_path, monkeypatch):
    engine = stub_engine(tmp_path)
    monkeypatch.setattr(api, "RagEngine", lambda _config: engine)
    app = api.create_app(make_config(tmp_path, warmup=False, api_key="secret"))
    client = TestClient(app)
    payload = {"answer": "Refunds arrive within five business days [1].", "passages": [{"text": REFUND}]}
    assert client.post("/v1/verify", json=payload).status_code == 401
    assert client.post(
        "/v1/verify", json=payload, headers={"authorization": "Bearer secret"}
    ).status_code == 200


def test_verify_endpoint_does_not_need_a_model(tmp_path, monkeypatch):
    """Layer 1 must work even when the checkpoint cannot load."""
    engine = stub_engine(tmp_path)
    monkeypatch.setattr(api, "RagEngine", lambda _config: engine)
    client = TestClient(api.create_app(make_config(tmp_path, warmup=False)))

    def explode(*_a, **_k):
        raise AssertionError("the citation check contacted a model")

    monkeypatch.setattr("laya_rag.compose.urllib.request.urlopen", explode)
    response = client.post("/v1/verify", json={
        "answer": "Refunds arrive within five business days [1].",
        "passages": [{"text": REFUND}],
    })
    assert response.status_code == 200

# -- quotes of the question are not fabrications -----------------------------

def test_a_quote_of_the_question_is_not_flagged():
    """An answer that quotes the question back is not quoting the evidence, and
    flagging it as "not found" would train a reader to ignore the check."""
    report = verify_citations(
        'Your question is "Laya in Docker", which is a topic rather than a question.',
        EVIDENCE,
        query="Laya in Docker",
    )
    assert report["counts"]["quotes_not_found"] == 0
    assert report["counts"]["quotes_from_question"] == 1
    assert report["quotes"][0]["status"] == "from_question"
    assert "quotes the question" in report["summary"]


def test_a_quote_of_the_question_still_has_to_match_it():
    """Only a real quote of the question is excused; an invented one is not."""
    report = verify_citations(
        'You asked "something entirely different", which I cannot place.',
        EVIDENCE,
        query="Laya in Docker",
    )
    assert report["counts"]["quotes_not_found"] == 1
    assert report["counts"]["quotes_from_question"] == 0


def test_without_a_query_the_behaviour_is_unchanged():
    report = verify_citations('It says "processed within five business days" [1].', EVIDENCE)
    assert report["counts"]["quotes_from_question"] == 0
    assert report["counts"]["quotes_not_found"] == 0


def test_the_endpoint_accepts_the_query():
    """The GUI sends the question so the false alarm cannot reach a reader."""
    from fastapi.testclient import TestClient
    from laya_rag import api
    from laya_rag.config import EngineConfig
    import tempfile, pathlib as _p

    app = api.create_app(EngineConfig(index_dir=_p.Path(tempfile.mkdtemp()) / "i", warmup=False))
    response = TestClient(app).post("/v1/verify", json={
        "answer": 'Your question is "Laya in Docker", which is a topic.',
        "passages": [{"text": REFUND}],
        "query": "Laya in Docker",
    })
    assert response.status_code == 200
    assert response.json()["counts"]["quotes_not_found"] == 0

def test_a_quote_that_absorbed_sentence_punctuation_is_still_found():
    """Quotations routinely absorb the comma that follows them."""
    report = verify_citations(
        'The policy says "processed within five business days," exactly [1].', EVIDENCE
    )
    assert report["counts"]["quotes_not_found"] == 0
    assert report["quotes"][0]["status"] == "found"


def test_a_quote_of_the_question_that_absorbed_a_comma_is_excused():
    """The exact shape that produced a false alarm in a real answer."""
    report = verify_citations(
        'Your question is "Laya in Docker," which is a topic heading.', EVIDENCE,
        query="Laya in Docker",
    )
    assert report["counts"]["quotes_from_question"] == 1
    assert report["counts"]["quotes_not_found"] == 0


def test_stripping_punctuation_does_not_stop_a_fabrication_being_caught():
    """Only the ends are ignored: the middle still has to match."""
    report = verify_citations(
        'It claims "processed within two hours," which is wrong [1].', EVIDENCE
    )
    assert report["counts"]["quotes_not_found"] == 1
    assert report["quotes"][0]["status"] == "not_found"


def test_a_fabrication_that_merely_extends_a_real_span_is_caught():
    report = verify_citations(
        'It says "processed within five business days and never otherwise" [1].', EVIDENCE
    )
    assert report["counts"]["quotes_not_found"] == 1
