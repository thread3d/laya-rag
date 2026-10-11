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
