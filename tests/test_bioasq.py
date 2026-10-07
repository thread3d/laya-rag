"""Offline tests for the BioASQ ingest.

BioASQ's corpus is the one input that cannot be fetched from a public mirror, and
`docs/findings-bioasq-unavailable.md` records a public substitute that is 813,939 documents short yet
still holds 1,533 of the 1,549 judged documents — close enough to look right. So the point of these
tests is not that the happy path writes files; it is that a **short corpus or an unresolvable
judgement refuses to produce a report**, which is the failure the tool exists to catch.
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


def _load():
    name = "laya_rag_bioasq"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "bioasq.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _corpus_json(tmp_path: Path, articles: list[dict]) -> Path:
    """An `allMeSH_2020.json` with the real file's shape: a header line and one article per line."""
    path = tmp_path / "allMeSH_2020.json"
    body = ",\n".join(json.dumps(a) for a in articles)
    path.write_text('{"articles":[\n' + body + "\n]}\n", encoding="utf-8")
    return path


def _golden(tmp_path: Path, questions: list[dict]) -> Path:
    directory = tmp_path / "Task8BGoldenEnriched"
    directory.mkdir()
    half = len(questions) // 2 or len(questions)
    for index, chunk in enumerate((questions[:half], questions[half:])):
        if not chunk:
            continue
        (directory / f"8B{index + 1}_golden.json").write_text(
            json.dumps({"questions": chunk}), encoding="utf-8"
        )
    return directory


def _additional(tmp_path: Path, rows: list[tuple[str, str, str]]) -> Path:
    path = tmp_path / "additional_passages.tsv"
    path.write_text("".join(f"{p}\t{t}\t{x}\n" for p, t, x in rows), encoding="utf-8")
    return path


def _article(pmid: str, title: str, abstract: str) -> dict:
    return {"pmid": pmid, "title": title, "abstractText": abstract,
            "journal": "J", "meshMajor": ["X"], "year": "2019"}


def test_build_writes_the_three_files_in_the_harness_layout(tmp_path):
    bioasq = _load()
    source = _corpus_json(tmp_path, [
        _article("1", "First title", "First abstract."),
        _article("2", "Second title", "Second abstract."),
    ])
    golden = _golden(tmp_path, [{"id": "q1", "body": "a question?", "documents": [
        "http://www.ncbi.nlm.nih.gov/pubmed/1"]}])
    additional = _additional(tmp_path, [("9", "Manual title", "Manual abstract.")])
    out = tmp_path / "out"

    summary = bioasq.build(source=source, additional=additional, golden=golden, out=out,
                           expect_articles=2, expect_queries=1)

    assert summary["documents"] == 3          # two articles + one manual-add
    assert summary["additional"] == 1
    assert summary["queries"] == 1
    assert summary["judgements"] == 1

    corpus = [json.loads(line) for line in (out / "corpus.jsonl").read_text().splitlines()]
    assert [row["_id"] for row in corpus] == ["9", "1", "2"], "manual-add rows go in first"
    assert corpus[1] == {"_id": "1", "title": "First title", "text": "First abstract."}
    assert json.loads((out / "queries.jsonl").read_text().splitlines()[0]) == {
        "_id": "q1", "text": "a question?"}
    assert (out / "qrels" / "test.tsv").read_text().strip() == "q1\t1\t1"


def test_a_short_corpus_refuses_to_finish(tmp_path):
    """The whole point: 813,939 documents short looked like a result once already."""
    bioasq = _load()
    source = _corpus_json(tmp_path, [_article("1", "t", "a")])
    golden = _golden(tmp_path, [{"id": "q1", "body": "b", "documents": [".../pubmed/1"]}])
    additional = _additional(tmp_path, [])

    with pytest.raises(AssertionError, match="short"):
        bioasq.build(source=source, additional=additional, golden=golden,
                     out=tmp_path / "out", expect_articles=14_913_939, expect_queries=1)


def test_an_unresolvable_judgement_refuses_to_finish(tmp_path):
    """A judged document the corpus cannot supply means the manual-add list is incomplete."""
    bioasq = _load()
    source = _corpus_json(tmp_path, [_article("1", "t", "a")])
    golden = _golden(tmp_path, [{"id": "q1", "body": "b", "documents": [
        ".../pubmed/1", ".../pubmed/999"]}])
    additional = _additional(tmp_path, [])

    with pytest.raises(AssertionError, match="absent from the corpus"):
        bioasq.build(source=source, additional=additional, golden=golden,
                     out=tmp_path / "out", expect_articles=1, expect_queries=1)


def test_a_manual_add_document_satisfies_a_judgement_the_corpus_lacks(tmp_path):
    """The 749 spreadsheet rows exist for exactly this: judged, but not in the corpus."""
    bioasq = _load()
    source = _corpus_json(tmp_path, [_article("1", "t", "a")])
    golden = _golden(tmp_path, [{"id": "q1", "body": "b", "documents": [
        ".../pubmed/1", ".../pubmed/999"]}])
    additional = _additional(tmp_path, [("999", "Manual", "Supplied by the sheet.")])

    summary = bioasq.build(source=source, additional=additional, golden=golden,
                           out=tmp_path / "out", expect_articles=1, expect_queries=1)
    assert summary["documents"] == 2 and summary["judgements"] == 2


def test_titles_and_abstracts_are_separated_and_sanitised(tmp_path):
    bioasq = _load()
    source = _corpus_json(tmp_path, [_article("1", "A\ttitle\nhere", "Body\nwith\ttabs")])
    golden = _golden(tmp_path, [{"id": "q1", "body": "b", "documents": [".../pubmed/1"]}])
    additional = _additional(tmp_path, [])

    bioasq.build(source=source, additional=additional, golden=golden,
                 out=tmp_path / "out", expect_articles=1, expect_queries=1)
    row = json.loads((tmp_path / "out" / "corpus.jsonl").read_text().splitlines()[0])
    assert row["title"] == "A title here" and row["text"] == "Body with tabs"


def test_the_zip_member_streams_the_same_as_the_plain_file(tmp_path):
    """The download arrives as a zip, so the tool reads the member rather than demanding an unzip."""
    import zipfile

    bioasq = _load()
    inner = _corpus_json(tmp_path, [_article("1", "t", "a"), _article("2", "u", "b")])
    archive = tmp_path / "allMeSH_2020.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(inner, "allMeSH_2020.json")

    lines = list(bioasq._iter_source_lines(archive))
    assert lines[0].strip() == '{"articles":['
    assert len([line for line in lines[1:] if line.strip() not in {"]}", "]"}]) == 2


def test_duplicate_pmids_are_dropped_and_reported(tmp_path):
    """The real dump repeats 84 pmids; the reference preprocessing keeps the first occurrence.

    So the assertion is on lines *parsed* — the property of the download — while the deduplicated
    total is reported separately, which keeps an 84-record dedup from reading as a short download.
    """
    bioasq = _load()
    source = _corpus_json_for_dupes(tmp_path)
    golden = _golden(tmp_path, [{"id": "q1", "body": "b", "documents": [".../pubmed/1"]}])
    summary = bioasq.build(source=source, additional=_additional(tmp_path, []), golden=golden,
                           out=tmp_path / "out", expect_articles=3, expect_queries=1)
    assert summary["parsed"] == 3 and summary["duplicates"] == 1
    assert summary["documents"] == 2
    ids = [json.loads(line)["_id"] for line in
           (tmp_path / "out" / "corpus.jsonl").read_text().splitlines()]
    assert ids == ["1", "2"], "the first occurrence wins"


def _corpus_json_for_dupes(tmp_path: Path) -> Path:
    return _corpus_json(tmp_path, [
        _article("1", "first", "a"),
        _article("2", "other", "b"),
        _article("1", "duplicate pmid", "c"),
    ])


def test_golden_batches_are_read_in_filename_order(tmp_path):
    bioasq = _load()
    golden = _golden(tmp_path, [{"id": "a", "body": "1", "documents": []},
                                {"id": "b", "body": "2", "documents": []}])
    assert [q["id"] for q in bioasq.golden_questions(golden)] == ["a", "b"]


def test_the_array_closer_shares_the_last_articles_line():
    """The real dump ends `...}]}`, not with a line of its own.

    This is the failure that costs a fifteen-million-line run: `json.loads` succeeds on every
    article but the last, then reports "Extra data" on the final line, which reads like a corrupt
    download rather than an off-by-one in the parser.
    """
    bioasq = _load()
    article = {"pmid": "7", "title": "t", "abstractText": "a"}
    assert bioasq._parse_article_line(json.dumps(article) + "]}") == article
    assert bioasq._parse_article_line(json.dumps(article) + ",") == article
    assert bioasq._parse_article_line("]}") is None
    assert bioasq._parse_article_line("") is None


def test_the_array_closer_on_the_last_line_does_not_lose_the_article(tmp_path):
    bioasq = _load()
    path = tmp_path / "allMeSH_2020.json"
    path.write_text(
        '{"articles":[\n'
        + json.dumps(_article("1", "t", "a")) + ",\n"
        + json.dumps(_article("2", "u", "b")) + "]}\n",
        encoding="utf-8",
    )
    golden = _golden(tmp_path, [{"id": "q1", "body": "b", "documents": [".../pubmed/1"]}])
    summary = bioasq.build(source=path, additional=_additional(tmp_path, []), golden=golden,
                           out=tmp_path / "out", expect_articles=2, expect_queries=1)
    assert summary["articles"] == 2
    ids = [json.loads(line)["_id"] for line in
           (tmp_path / "out" / "corpus.jsonl").read_text().splitlines()]
    assert ids == ["1", "2"], "the final article must survive its shared line"


def test_the_dump_is_read_as_latin1_not_utf8(tmp_path):
    """The dump is not valid UTF-8; reading it as UTF-8 would mangle every accented abstract."""
    bioasq = _load()
    path = tmp_path / "allMeSH_2020.json"
    article = {"pmid": "1", "title": "Caf\u00e9 study",
               "abstractText": "na\u00efve r\u00e9sum\u00e9"}
    path.write_bytes(
        b'{"articles":[\n'
        + json.dumps(article, ensure_ascii=False).encode("latin-1") + b"]}\n"
    )
    golden = _golden(tmp_path, [{"id": "q1", "body": "b", "documents": [".../pubmed/1"]}])
    bioasq.build(source=path, additional=_additional(tmp_path, []), golden=golden,
                 out=tmp_path / "out", expect_articles=1, expect_queries=1)
    row = json.loads(
        (tmp_path / "out" / "corpus.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert row["title"] == "Caf\u00e9 study" and row["text"] == "na\u00efve r\u00e9sum\u00e9"
    assert "\ufffd" not in (row["title"] + row["text"])
