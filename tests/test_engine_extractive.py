"""Tests for the engine's extractive end-to-end row.

The row's whole value is that every component is the engine's, so the two things worth pinning are
the ones that would silently make it someone else's: an answer that is *not* a verbatim span of the
returned passages must fail the citation check rather than pass it, and the retrieval half must be
the benchmark's own metric over the same passages — not a re-derivation.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

# Where the harness looks for MultiHop-RAG. `eval/` is gitignored, so this holds the benchmark only
# on a host that has downloaded it — never on a fresh checkout, which is what CI is.
BENCH = ROOT / "eval" / "benchmarks" / "multihop-rag"


def _load():
    name = "laya_rag_engine_extractive"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "engine_extractive.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ex = _load()


@pytest.fixture()
def corpus(tmp_path, monkeypatch):
    """A three-document corpus in the benchmark's on-disk shape, so the scoring tests read nothing
    that a checkout does not have.

    `score_one` needs the corpus for one thing: `ranking_from_passages` maps a returned passage path
    (`/docs/000000.txt`) back to a corpus id and the document row scores that id's text. A
    hand-written three-document corpus exercises exactly that path. The eight scoring tests used to
    read `eval/benchmarks/multihop-rag/corpus.json` directly, which exists on a machine that has run
    the benchmark download and nowhere else: they passed locally and the first CI push failed on
    all of them with `FileNotFoundError`. A test that needs a 2 GB download to state that an
    uncited sentence is not clean is measuring the download.
    """
    raw = [
        {"url": "https://example.invalid/a", "title": "Refunds",
         "body": "Refunds are processed within five business days."},
        {"url": "https://example.invalid/b", "title": "Shipping",
         "body": "Shipping is free over $50."},
        {"url": "https://example.invalid/c", "title": "Returns",
         "body": "Returns take a fortnight."},
    ]
    (tmp_path / "corpus.json").write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(ex, "ROOT_BENCH", tmp_path)
    monkeypatch.setattr(ex, "_CORPUS_TEXT", None)
    return tmp_path


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


def test_a_verbatim_answer_that_cites_cleanly_scores_clean(corpus):
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


def test_an_uncited_answer_is_not_clean(corpus):
    """The contract the row rests on: below 1.0 on attribution is a bug, and it must show up."""
    fact = "Refunds are processed within five business days."
    row = ex.score_one(
        _payload("Refunds are processed within five business days.", [], fact),
        _item(fact),
    )
    assert row["citation_clean"] is False
    assert row["citation_uncited"] == 1


def test_an_answer_quoting_something_not_in_the_evidence_is_not_clean(corpus):
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


def test_the_retrieval_half_is_the_benchmark_s_metric(corpus):
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


def test_answer_nugget_recall_is_the_answer_s_not_the_passage_s(corpus):
    """The point of the metric: the passage covers the fact, a short extract of it may not."""
    fact = "Refunds arrive in 5 days; shipping is free over $50."
    row = ex.score_one(
        _payload("[1] Refunds arrive in 5 days.", [], fact + " " + fact),
        _item(fact),
    )
    assert row["Hits@10"] == 1.0                 # the passage covers it
    assert row["nugget_recall"] < 1.0           # the answer does not


def test_a_query_with_no_nuggets_reports_none_rather_than_zero(corpus):
    row = ex.score_one(_payload("[1] A.", [], "A."), _item("a"))
    assert row["nugget_recall"] is None


def test_summarize_counts_abstention_and_skips_null_recalls(corpus):
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


def test_the_corpus_map_is_the_text_the_engine_indexed(corpus):
    """`url -> title + body`, which is what `write_corpus` writes and what the document row scores.

    The engine indexes `a["title"] + " " + a["body"]`; if this map keyed the body alone, the document row
    would score a text the engine never saw and the placement against the published table would be
    between two different corpora.
    """
    text = ex.corpus_text()
    assert set(text) == {
        "https://example.invalid/a", "https://example.invalid/b", "https://example.invalid/c",
    }
    assert text["https://example.invalid/a"] == (
        "Refunds Refunds are processed within five business days."
    ), "the map dropped the title, which the engine's own corpus carries"
    assert all(value.strip() for value in text.values()), "a document with no text"


def test_the_downloaded_benchmark_corpus_is_the_size_the_findings_quote():
    """609 documents: the number `README.md` and `docs/findings-engine-multihop.md` quote.

    Skips rather than fails on a checkout without the benchmark, the way every other `eval/`-reading
    gate in this suite does — `eval/` is gitignored, and a test whose verdict depends on whether the
    author has run the download is not a test.
    """
    if not (BENCH / "corpus.json").is_file():
        pytest.skip("no downloaded MultiHop-RAG corpus (eval/ is gitignored)")
    text = ex.corpus_text()
    assert len(text) == 609, f"expected the benchmark's 609 documents, got {len(text)}"
    assert all(value.strip() for value in text.values()), "a document with no text"


def test_the_placement_finding_states_which_lever_each_arm_has():
    """A quota and a wider pool are not interchangeable, and the finding must say so per arm.

    `rerank_coverage_min_docs` is applied inside `if cross_scores is not None`, so the extractive row — which
    has no cross-encoder by default — cannot reach it; the only lever there is a wider pool. And with a
    cross-encoder, `resolved_rerank_coverage_lambda` auto-selects 0.7, so every CE row measured here is
    coverage-reordered as well as cross-encoded. A finding that said "a quota or a wider pool" without that
    distinction would send a reader to a setting their arm cannot use.
    """
    engine_source = (Path(__file__).resolve().parents[1] / "src" / "laya_rag" / "engine.py").read_text()
    assert "if cross_scores is not None and lambda_ > 0.0 and candidates:" in engine_source
    finding = (Path(__file__).resolve().parents[1] / "docs" / "findings-multihop-placement.md").read_text()
    assert "Without a cross-encoder" in finding
    assert "resolved_rerank_coverage_lambda` returns **0.7**" in finding


def test_the_row_records_what_citation_clean_does_not_cover():
    """`clean` is structural; the advisory content layer never folds into it, by design.

    A wrong number, a wrong citation index and a fabricated figure all leave `clean` True — the verifier says so
    in its own docstring ("reported in a separate `content` block and never change the `clean` verdict"). Every
    finding in this programme quotes `citation_clean_all`, and none of them quoted the content layer, so a reader
    could reasonably infer that 1.0000 meant the numbers were checked. It does not.
    """
    import inspect

    module = _load()
    source = inspect.getsource(module)
    assert '"content_clean": bool(citation["content"]["clean"])' in source
    assert '"content_clean_all": round(sum(1 for r in rows if r.get("content_clean")) / total, 4)' in source
    assert '"negation_flags": sum(int(r.get("negation_flags") or 0) for r in rows)' in source


def test_rows_are_stamped_with_the_version_that_wrote_them():
    """Absence of a settings field is indistinguishable from `false` unless the row says what wrote it.

    The full-set row demonstrates the problem: it started before `honour_answerability` and the `doc_*` fields
    existed, so its settings say nothing about either. A reader comparing it to the coverage arm — which has both —
    cannot tell whether the difference is a configuration or an era. The stamp is the fix: a row records the version
    that wrote it, so a field's absence can be read as "not yet possible" rather than "not set".
    """
    import inspect

    source = inspect.getsource(__import__("engine_extractive"))
    assert "def _tool_version()" in source
    assert source.count('"tool_version": _tool_version()') == 2, (
        "both the META record and the final report must carry the stamp"
    )
