"""Tests for the citation-first extractive composer.

The structural guarantee is the point: every emitted sentence must exist verbatim in a retrieved
passage, and with nothing to emit the composer abstains. Citations must be document numbers that
`verify_citations` can check.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from laya_rag import cli  # noqa: E402
from laya_rag.engine import RagEngine  # noqa: E402
from laya_rag.extractive import ABSTENTION, compose_extractive, split_sentences  # noqa: E402

from laya_rag.evidence import group_passages_by_document  # noqa: E402

from .support import StubModel, make_config  # noqa: E402


def _p(path: str, text: str, score: float, chunk_id: str = "c") -> dict:
    return {"path": path, "text": text, "relevance": score, "chunk_id": chunk_id}


def test_split_sentences_keeps_punctuation_and_trailing_runs():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert split_sentences("No terminal punctuation") == ["No terminal punctuation"]


def test_every_emitted_sentence_is_verbatim():
    passages = [
        _p("a.md", "Refunds take five days. Contact support after ten.", 0.9),
        _p("b.md", "Shipping is free over fifty dollars.", 0.4),
    ]
    composed = compose_extractive(passages, max_sentences=3)
    texts = [p["text"] for p in passages]
    assert composed.sentences
    for sentence in composed.sentences:
        assert any(sentence.text in text for text in texts)
        assert f"[{sentence.citation}]" in composed.answer


def test_citations_follow_document_first_appearance():
    passages = [
        _p("a.md", "First fact.", 0.9),
        _p("b.md", "Second fact.", 0.8),
        _p("a.md", "Another fact from a.", 0.7),
    ]
    composed = compose_extractive(passages, max_sentences=5)
    by_text = {sentence.text: sentence for sentence in composed.sentences}
    assert by_text["First fact."].citation == 1
    assert by_text["Second fact."].citation == 2
    assert by_text["Another fact from a."].citation == 1
    assert by_text["Another fact from a."].path == "a.md"


def test_duplicate_sentences_are_emitted_once():
    passages = [
        _p("a.md", "The same sentence.", 0.9),
        _p("b.md", "The same sentence.", 0.8),
    ]
    composed = compose_extractive(passages, max_sentences=5)
    assert [s.text for s in composed.sentences] == ["The same sentence."]


def test_no_passages_abstains():
    composed = compose_extractive([], max_sentences=3)
    assert composed.abstained is True
    assert composed.answer == ABSTENTION
    assert composed.sentences == []


def test_max_sentences_is_respected_and_best_first():
    passages = [
        _p("a.md", "Low one. Low two.", 0.1),
        _p("b.md", "High one.", 0.9),
    ]
    composed = compose_extractive(passages, max_sentences=1)
    assert [s.text for s in composed.sentences] == ["High one."]


def test_min_score_drops_low_documents():
    passages = [_p("a.md", "Low.", 0.1), _p("b.md", "High.", 0.9)]
    composed = compose_extractive(passages, min_score=0.5)
    assert [s.text for s in composed.sentences] == ["High."]


def test_the_answer_is_clean_to_the_project_s_own_citation_checker():
    """The contract the finding claims: `verify_citations` runs over the result, no parsing.

    A trailing marker would be orphaned by `split_claims`, which splits *after* the terminal
    punctuation, so the sentence it supports would read as uncited and the last marker would
    dangle. This is the acceptance test for the marker leading instead.
    """
    from laya_rag.verify import verify_citations

    passages = [
        _p("a.md", "Refunds take five days. Contact support after ten.", 0.9),
        _p("b.md", "Shipping is free over fifty dollars.", 0.4),
    ]
    composed = compose_extractive(passages, max_sentences=3)
    merged = [text for _path, text in group_passages_by_document(
        (passage["path"], passage["text"]) for passage in passages
    )]
    report = verify_citations(composed.answer, merged)
    assert report["counts"]["claims"] == len(composed.sentences)
    assert report["counts"]["uncited"] == 0, report["claims"]
    assert report["clean"] is True, report["claims"]


def test_the_composed_answer_never_carries_half_of_a_quotation():
    """A chunk is a slice, so it can cut a quotation; half of a pair is not emittable.

    The checker pairs an opening mark with the next closing mark anywhere in the answer, so an
    orphaned half pairs across the `[n]` markers and the answer fails the quoted-span check with
    every citation in range. Measured on the MultiHop-RAG row: 16 of 400 answers, none of them a bad
    citation, and none with an uncited claim.
    """
    text = '…and I am picking Connor Bedard over Auston Matthews,” Turk said. The pick was obvious.'
    composed = compose_extractive([_p("a.md", text, 0.9)], max_sentences=5)
    assert composed.sentences
    for sentence in composed.sentences:
        assert sentence.text.count('"') % 2 == 0, sentence.text
        assert sentence.text.count("“") == sentence.text.count("”"), sentence.text


def test_splitting_is_not_quote_aware_because_the_checker_splits_there_too():
    """Suppressing the boundary inside a quotation makes the sentences after it uncited.

    `verify_citations` splits the answer into claims at the boundary either way, so a run that spans
    it is emitted with one `[n]` and checked as several claims — 24 uncited sentences of 1,637,
    traded for the quote failures it was meant to fix. Both checks need the same boundary rule.
    """
    text = 'He said “stop. Now.” Then he left.'
    assert split_sentences(text) == ["He said “stop.", "Now.”", "Then he left."]
    # …and the two halves of the quotation are refused, so only the clean tail is emitted.
    composed = compose_extractive([_p("a.md", text, 0.9)], max_sentences=5)
    assert [s.text for s in composed.sentences] == ["Then he left."]


def test_a_quoted_span_survives_the_round_trip_through_the_checker():
    from laya_rag.verify import verify_citations

    text = ('The prosecutor said “he was not what he appeared to be.” '
            'What followed was a thirty minute story.')
    composed = compose_extractive([_p("a.md", text, 0.9)], max_sentences=2)
    report = verify_citations(composed.answer, [text])
    assert report["counts"]["uncited"] == 0
    assert report["counts"]["quotes_not_found"] == 0, report["quotes"]
    assert report["clean"] is True, report["quotes"]


def test_a_span_that_carries_half_a_quotation_is_not_emitted():
    """A chunk is a slice, so it can cut a quotation; half of a pair is not emittable.

    The checker pairs an opening mark with the next closing mark anywhere in the answer, so the
    orphaned half pairs across the `[n]` markers and the answer fails the quoted-span check with
    every citation in range — 16 of 400 answers on the MultiHop-RAG row, none of them a bad
    citation.
    """
    from laya_rag.verify import verify_citations

    text = ('…and I am picking Connor Bedard over Auston Matthews,” Turk said. '
            'The pick was obvious.')
    composed = compose_extractive([_p("a.md", text, 0.9)], max_sentences=5)
    for sentence in composed.sentences:
        assert sentence.text.count("“") == sentence.text.count("”"), sentence.text
    report = verify_citations(composed.answer, [text])
    assert report["counts"]["uncited"] == 0
    assert report["counts"]["quotes_not_found"] == 0, report["quotes"]
    assert report["clean"] is True


def test_a_quotation_that_does_not_span_a_boundary_is_still_emitted():
    passage = 'He said “stop now” and then he left. The room was quiet.'
    composed = compose_extractive([_p("a.md", passage, 0.9)], max_sentences=5)
    assert any("stop now" in sentence.text for sentence in composed.sentences)


def test_every_marker_leads_the_sentence_it_supports():
    passages = [_p("a.md", "First fact. Second fact.", 0.9)]
    composed = compose_extractive(passages, max_sentences=2)
    assert composed.answer.startswith("[1] First fact.")
    for sentence in composed.sentences:
        assert f"[{sentence.citation}] {sentence.text}" in composed.answer


# ── engine and CLI ───────────────────────────────────────────────────────

def _corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir(parents=True, exist_ok=True)
    (root / "refunds.md").write_text("Refunds are processed within five business days.\n")
    (root / "shipping.md").write_text("Shipping is free over fifty dollars.\n")
    return root


def test_engine_query_extractive_is_verbatim_and_cited(tmp_path):
    config = make_config(
        tmp_path, relevance_threshold=0.0, answerability_threshold=0.0
    )
    engine = RagEngine(config, model=StubModel())
    engine.index_paths([_corpus(tmp_path)], rebuild=True)
    payload = engine.query_extractive("refund processing time", max_sentences=3)
    assert payload["sentences"]
    assert not payload["abstained"]
    assert "[" in payload["answer"]
    passages = engine.query("refund processing time").passages
    for sentence in payload["sentences"]:
        assert any(sentence["text"] in passage["text"] for passage in passages)


def test_extract_parser_collects_the_options():
    args = cli.build_parser().parse_args(
        ["extract", "why?", "--max-sentences", "2", "--tenant", "acme", "--json"]
    )
    assert args.query == "why?"
    assert args.max_sentences == 2
    assert args.tenant == "acme"
    assert args.json is True


# ── the coverage objective: select by fact, not by parent rank ────────────

_TOPICAL = {
    "path": "a.md",
    "relevance": 0.95,
    "chunk_id": "a1",
    "text": (
        "This section explains the general approach to the topic in broad terms. "
        "It discusses the background and motivation at length without giving a figure."
    ),
}
_FACTUAL = {
    "path": "b.md",
    "relevance": 0.70,
    "chunk_id": "b1",
    "text": (
        "The measured nDCG@10 was 0.7417 in 2023, against 0.7264 before the change. "
        "The improvement of 0.0153 was significant at p=0.023."
    ),
}


def _digits(sentences):
    return sum(sum(char.isdigit() for char in sentence.text) for sentence in sentences)


def test_a_rank_ordered_composer_prefers_the_best_document_and_its_facts_are_incidental():
    """The shipped behaviour, pinned: it is what every recorded end-to-end row was measured with."""
    answer = compose_extractive([_TOPICAL, _FACTUAL], query="What nDCG@10 was measured?",
                                max_sentences=2)
    assert [s.path for s in answer.sentences] == ["a.md", "a.md"]
    assert _digits(answer.sentences) == 0


def test_the_coverage_objective_reaches_the_facts_the_rank_order_never_shows():
    """Answer nugget recall is 0.1906 against a pool carrying 0.7438 — three quarters of the facts
    in the evidence never reach the answer. The shipped selection has no objective, so it emits the
    best document's first sentences; the objective emits the sentence with the facts in it."""
    answer = compose_extractive([_TOPICAL, _FACTUAL], query="What nDCG@10 was measured?",
                                max_sentences=2, coverage=True)
    assert answer.sentences[0].path == "b.md"
    assert _digits(answer.sentences) >= 6
    assert _digits(answer.sentences) > _digits(
        compose_extractive([_TOPICAL, _FACTUAL], query="What nDCG@10 was measured?",
                           max_sentences=2).sentences
    )


def test_the_objective_keeps_every_guarantee_the_shipped_path_has():
    """A different selection is not licence for a different contract."""
    passages = [_TOPICAL, _FACTUAL]
    answer = compose_extractive(passages, query="What nDCG@10 was measured?", max_sentences=5,
                                coverage=True)
    for sentence in answer.sentences:
        # verbatim, in one of the passages
        assert any(sentence.text in str(p["text"]) for p in passages)
        # in-range marker, and pointing at the document it quotes
        assert 1 <= sentence.citation <= len(passages)
        assert passages[sentence.citation - 1]["path"] == sentence.path


def test_the_objective_abstains_when_nothing_qualifies():
    assert compose_extractive([], coverage=True).abstained is True
    blocked = dict(_FACTUAL)
    empty = compose_extractive([blocked], min_score=0.99, coverage=True)
    assert empty.abstained is True


def test_duplicate_sentences_cannot_fill_the_budget_under_the_objective():
    """Near-duplicates are the reason a five-sentence extract can carry one fact."""
    repeated = {
        "path": "c.md",
        "relevance": 0.9,
        "chunk_id": "c1",
        "text": "The figure was 42 percent. The figure was 42 percent again.",
    }
    answer = compose_extractive([repeated], query="What was the figure?", max_sentences=3,
                                coverage=True)
    assert len(answer.sentences) == 1


def test_per_character_selection_is_opt_in_and_changes_the_choice():
    """Prediction (c)'s implementation: score new facts *per character*, without touching the default.

    Under the shipped objective a fact-dense sentence wins regardless of length, which is how the coverage arm
    came to emit 2.1x the text of the shipped order. Scoring per character is the registered remedy — and it is
    opt-in because the queued full-set row is measuring the shipped objective.
    """
    from laya_rag.extractive import _select_by_coverage, ExtractedSentence

    short_dense = ExtractedSentence(text="Revenue rose 12%.", citation=1, path=None, chunk_id=None, score=0.9)
    long_dense = ExtractedSentence(
        text="Revenue rose 12% and the company said that margins improved across every region "
             "it operates in during the period under review. " * 3,
        citation=2, path=None, chunk_id=None, score=0.9,
    )
    candidates = [long_dense, short_dense]

    absolute = _select_by_coverage(candidates, 1)
    per_char = _select_by_coverage(candidates, 1, per_char=True)
    assert absolute[0] is long_dense, "the absolute objective prefers the longer, denser sentence"
    assert per_char[0] is short_dense, "the per-character objective prefers the efficient one"
