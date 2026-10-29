"""Tests for the engine's extractive end-to-end row.

The row's whole value is that every component is the engine's, so the two things worth pinning are
the ones that would silently make it someone else's: an answer that is *not* a verbatim span of the
returned passages must fail the citation check rather than pass it, and the retrieval half must be
the benchmark's own metric over the same passages — not a re-derivation.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))


def _load():
    name = "laya_rag_engine_extractive"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "engine_extractive.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ex = _load()


def _payload(answer: str, sentences, text: str, *, abstained=False, query="who?"):
    return {
        "query": query,
        "answer": answer,
        "abstained": abstained,
        "sentences": sentences,
        "answerable": True,
        "answerability": 0.8,
        "passages": [{"path": "/docs/000000.txt", "text": text, "chunk_id": "a:0000"}],
    }


def _item(*facts: str):
    return {"evidence_list": [{"fact": fact} for fact in facts]}


def test_a_verbatim_answer_that_cites_cleanly_scores_clean():
    fact = "Refunds are processed within five business days."
    row = ex.score_one(
        _payload("[1] Refunds are processed within five business days.",
                 [{"text": fact, "citation": 1, "path": "/docs/000000.txt", "chunk_id": "a:0000"}],
                 fact),
        _item(fact),
    )
    assert row["citation_clean"] is True
    assert row["citation_uncited"] == 0
    assert row["nugget_recall"] == pytest.approx(1.0)


def test_an_uncited_answer_is_not_clean():
    """The contract the row rests on: below 1.0 on attribution is a bug, and it must show up."""
    fact = "Refunds are processed within five business days."
    row = ex.score_one(
        _payload("Refunds are processed within five business days.", [], fact),
        _item(fact),
    )
    assert row["citation_clean"] is False
    assert row["citation_uncited"] == 1


def test_an_answer_quoting_something_not_in_the_evidence_is_not_clean():
    """A span that is not in the passages cannot be verified, so the row must not call it clean."""
    fact = "Refunds are processed within five business days."
    row = ex.score_one(
        _payload('[1] "Refunds take a fortnight" is the policy.', [
            {"text": "Refunds take a fortnight is the policy.", "citation": 1},
        ], fact),
        _item(fact),
    )
    assert row["citation_clean"] is False
    assert row["citation_unfound_quotes"] >= 1


def test_the_retrieval_half_is_the_benchmark_s_metric():
    """A gold fact present in the returned passage is a hit; one that is not, is not."""
    hit = ex.score_one(
        _payload("[1] Refunds are processed within five business days.", [], 
                 "Refunds are processed within five business days."),
        _item("Refunds are processed within five business days."),
    )
    miss = ex.score_one(
        _payload("[1] Something else entirely about shipping.", [],
                 "Something else entirely about shipping."),
        _item("Refunds are processed within five business days."),
    )
    assert hit["Hits@10"] == 1.0
    assert miss["Hits@10"] == 0.0


def test_answer_nugget_recall_is_the_answer_s_not_the_passage_s():
    """The point of the metric: the passage covers the fact, a short extract of it may not."""
    fact = "Refunds arrive in 5 days; shipping is free over $50."
    row = ex.score_one(
        _payload("[1] Refunds arrive in 5 days.", [], fact + " " + fact),
        _item(fact),
    )
    assert row["Hits@10"] == 1.0                 # the passage covers it
    assert row["nugget_recall"] < 1.0           # the answer does not


def test_a_query_with_no_nuggets_reports_none_rather_than_zero():
    row = ex.score_one(_payload("[1] A.", [], "A."), _item("a"))
    assert row["nugget_recall"] is None


def test_summarize_counts_abstention_and_skips_null_recalls():
    # The answer has to be long enough to be a claim: `split_claims` drops a fragment of fewer
    # than three words, and a report with no claims is not a clean one.
    fact = "Refunds are processed within five business days."
    rows = [
        {**ex.score_one(_payload(f"[1] {fact}", [], fact), _item(fact))},
        {
            "abstained": True, "citation_clean": False, "citation_uncited": 0, "citation_claims": 0,
            "citation_invalid": 0, "citation_unfound_quotes": 0, "nugget_recall": None,
            "nuggets": 0, "Hits@10": 0.0, "Hits@4": 0.0, "MAP@10": 0.0, "MRR@10": 0.0,
        },
    ]
    summary = ex.summarize(rows)
    assert summary["queries"] == 2
    assert summary["answered"] == 1
    assert summary["abstention_rate"] == pytest.approx(0.5)
    # Attribution is a property of an answer: the rate is over the one that was emitted, while
    # `citation_clean_all` carries the abstention's uncited sentence.
    assert summary["citation_clean_rate"] == pytest.approx(1.0)
    assert summary["citation_clean_all"] == pytest.approx(0.5)
    assert summary["answer_nugget_recall"] == pytest.approx(1.0)


def test_summarize_of_nothing_is_empty_not_a_crash():
    assert ex.summarize([]) == {"queries": 0}


def test_render_states_that_a_citation_score_below_one_is_a_bug():
    text = ex.render({"summary": {
        "queries": 2, "answered": 1, "abstention_rate": 0.5, "citation_clean_rate": 1.0,
        "citation_clean_all": 1.0, "uncited_sentences": 0,
        "claims": 3, "answer_nugget_recall": 0.4,
        "Hits@10": 0.9, "Hits@4": 0.7, "MAP@10": 0.3, "MRR@10": 0.6,
    }})
    assert "1.0 by construction" in text
    assert "answer nugget recall" in text


def test_render_handles_a_missing_nugget_reading():
    text = ex.render({"summary": {
        "queries": 1, "answered": 0, "abstention_rate": 1.0, "citation_clean_rate": None,
        "citation_clean_all": 1.0, "uncited_sentences": 0,
        "claims": 0, "answer_nugget_recall": None,
        "Hits@10": 0.0, "Hits@4": 0.0, "MAP@10": 0.0, "MRR@10": 0.0,
    }})
    assert "| n/a |" in text


def test_a_row_records_the_composer_objective_it_ran_under():
    """A fingerprint is a hash: `947d10ba…` does not tell a reader which arm produced the row.

    The objective went into the resume fingerprint first (so a resumed run cannot mix arms) and only
    later into the settings block. That is the fifth instance of the same class in this programme — a
    field that describes a run without saying what actually ran — and the cheapest defence is to assert
    the field exists before any row is quoted from it.
    """
    import inspect

    source = inspect.getsource(__import__("engine_extractive"))
    assert source.count('"extract_coverage": os.environ.get("LAYA_RAG_EXTRACT_COVERAGE", "")') == 2, (
        "both the META record and the final report must carry the composer objective"
    )


def test_the_answerability_lever_is_in_the_fingerprint_and_the_record():
    """A configuration that changes which rows abstain must reach both, or a resume mixes two arms.

    This is the same rule the composer's objective follows, and the programme has paid for it once already:
    a fingerprint that ignored `LAYA_RAG_EXTRACT_COVERAGE` reported two identical arms to four decimals.
    """
    import inspect

    source = inspect.getsource(__import__("engine_extractive"))
    assert 'honour_answerability=bool(args.honour_answerability)' in source, "not in the fingerprint"
    assert source.count('"honour_answerability": bool(args.honour_answerability)') == 2, (
        "both the META record and the final report must carry it"
    )
    assert "honour_answerability=args.honour_answerability" in source, "not passed to query_extractive"


def test_the_row_can_be_scored_at_document_level_as_well_as_chunk_level():
    """The published table scores **documents**; the engine returns chunks.

    This corpus chunks 609 documents into 4,592, so ten chunk slots hold a fraction of one document's text
    where the published rows hold ten whole documents. A chunk-level row and a document-level row therefore
    answer different questions in the same metric's clothes. Both are recorded: the chunk row is the
    system's own, the document row is the one that can be placed beside a published one.
    """
    import inspect

    module = _load()
    source = inspect.getsource(module)
    assert "doc_texts = [corpus_document_text(document_id) for document_id in document_ids]" in source
    assert "ranking_from_passages(passages, corpus_document_ids())[:10]" in source, (
        "the collapse to distinct documents must reuse the harness's own path-to-id mapping"
    )
    assert '"doc_passages": len(document_ids)' in source, "how many documents the ten chunks covered"
    # the keys are built by an f-string, so assert on the pattern rather than on expanded literals
    assert 'f"doc_{key}"' in source, "no document-level counterpart keys"
    assert "for key in METRIC_KEYS" in source


def test_the_corpus_map_is_the_text_the_engine_indexed():
    module = _load()
    text = module.corpus_text()
    assert len(text) == 609, f"expected the benchmark's 609 documents, got {len(text)}"
    assert all(value.strip() for value in text.values()), "a document with no text"
