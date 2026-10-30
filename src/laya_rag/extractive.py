"""A citation-first extractive composer: every emitted sentence is a verbatim span.

The engine's credibility hole is "no end-to-end ranking", and its measured citation mechanics are a
negative: over the pilot's 14 answers **no answer met the deterministic citation bar**, dominated by
39 of 81 sentences carrying no citation. A generative composer buys fluency with a *probabilistic*
attribution guarantee. This buys a **structural** one: a sentence that is not present verbatim in a
retrieved passage is never emitted, and with nothing to emit the composer abstains. The cost is
extractive prose, which is the honest trade for a system that cannot generate.

The output is also the structured citation contract: each emitted sentence carries the document
citation number it must be checked against, its source path and chunk id, so
:func:`laya_rag.verify.verify_citations` can be run over the result without parsing prose.

    from laya_rag.extractive import compose_extractive
    composed = compose_extractive(result.passages, query=query)
    composed.answer        # "Refunds take five days [1]. ..."
    composed.sentences[0]  # ExtractedSentence(text, citation=1, path=..., chunk_id=...)

No generator, no API, no model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .evidence import group_passages_by_document

#: A run ends at terminal punctuation followed by any closing quote or bracket, at a line break, or
#: at the end of the text. The closing class is not cosmetic — see :func:`split_sentences`.
_SENTENCE = re.compile(r"""[^.!?\n]+[.!?]+["'”’)\]]*|[^.!?\n]+$""")
_WHITESPACE = re.compile(r"\s+")

#: What the composer says when no passage yields an emittable sentence.
ABSTENTION = "The retrieved evidence does not contain a sentence that answers the question."


def split_sentences(text: str) -> list[str]:
    """Sentence runs, trimmed; a run with no terminal punctuation is still a sentence.

    A run ends at terminal punctuation (with any closing quote or bracket that follows it), at a
    line break, or at the end of the text. **It is deliberately not quote-aware.** Suppressing the
    boundary *inside* a quotation was tried and is wrong: `verify_citations` splits the answer into
    claims at that boundary, so sentences after it would carry no citation and the answer would fail
    the uncited check instead — 24 uncited sentences of 1,637 on the MultiHop-RAG row, traded for
    the quote failures it was meant to fix. Both checks have to be satisfied by the *same* boundary
    rule, and this is it: split where the checker splits, then refuse to emit a run that carries half
    of a quoted pair (:func:`_quotes_balanced`), which is what a chunk boundary can produce.
    """
    return [
        match.group(0).strip()
        for match in _SENTENCE.finditer(text or "")
        if match.group(0).strip()
    ]


def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip().casefold()


def _quotes_balanced(text: str) -> bool:
    """Whether a span carries both marks of every quotation it contains.

    `verify_citations` pairs an opening mark with the next closing one **anywhere in the answer**,
    so a sentence holding one mark of a pair leaves the other to pair with a mark in a *later*
    sentence — and the span between them contains the `[n]` markers this composer inserts, which no
    passage contains. The answer then fails the quoted-span check with every citation in range.

    A chunk boundary can do this to us: a chunk is a slice of a document, so it can begin or end
    inside a quotation. Measured on the MultiHop-RAG extractive row: 16 of 400 answers failed the
    deterministic check and **none** of them had an uncited claim or a citation out of range — every
    failure was half a quotation. A span that cannot be quoted cannot be evidence, so it is dropped
    like any other candidate.
    """
    return (
        text.count('"') % 2 == 0
        and text.count("“") == text.count("”")
        and text.count("`") % 2 == 0
    )


def _score(passage: Mapping[str, Any]) -> float:
    for key in ("relevance", "recall_score", "score"):
        value = passage.get(key)
        if value is not None:
            return float(value)
    return 0.0


@dataclass(frozen=True)
class ExtractedSentence:
    """One emitted sentence and the citation it must satisfy."""

    text: str
    citation: int
    path: str | None
    chunk_id: str | None
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "citation": self.citation,
            "path": self.path,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
        }


@dataclass
class ExtractiveAnswer:
    """The composed answer, or an abstention, plus its structured citations."""

    answer: str
    abstained: bool
    sentences: list[ExtractedSentence] = field(default_factory=list)
    considered: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "abstained": self.abstained,
            "sentences": [sentence.as_dict() for sentence in self.sentences],
            "considered": self.considered,
        }


#: Tokens that carry a fact rather than a topic: a number, a date, a percentage, an acronym, a
#: proper noun. Not an entity linker and not a model — the same "programmatic nugget" idea
#: `tools/nugget_recall.py` scores against, at the sentence rather than the fact level.
_NUGGET = re.compile(
    r"(?:\b\d[\d,.]*(?:%|st|nd|rd|th|s|px|kg|km|gb|mb|x)?\b)"      # numbers, ordinals, units
    r"|(?:\b[A-Z][A-Za-z0-9&+.-]*[A-Z0-9][A-Za-z0-9&+.-]*\b)"       # Capitalised / ACRONYM
    r"|(?:[%$€£])"                                                 # bare symbols
)


def _fact_terms(text: str) -> set[str]:
    """The fact-bearing tokens of a sentence, lowercased so a repeat counts once."""
    return {match.group(0).lower() for match in _NUGGET.finditer(text)}


def _content_terms(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]{3,}", text.lower())}


#: Two sentences sharing this much of their content vocabulary are the same sentence for the
#: purpose of an answer. Without the guard the novelty term is permissive enough that a
#: near-duplicate differing by one word ("...42 percent." / "...42 percent again.") is selected,
#: and five such sentences carry one fact between them — which is the failure the objective
#: exists to remove, reproduced by the objective itself.
_DUPLICATE_JACCARD = 0.7


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _select_by_coverage(
    candidates: Sequence[ExtractedSentence],
    budget: int,
    *,
    per_char: bool = False,
    max_chars: int = 0,
) -> list[ExtractedSentence]:
    """Greedy max-coverage over *facts*, not over the parent document's rank.

    The shipped order is "the sentences of the best-ranked document, in document order", which
    measures answer nugget recall **0.1906** against a pool carrying **0.7438** of the same gold
    content (`docs/findings-extractive-row.md`) — three quarters of the evidence's facts never
    reach the answer. The selection has no objective of its own, so the sentence it keeps is the
    first one in the first passage rather than the one carrying the fact.

    The objective is submodular greedy coverage over the same candidate sentences, under the same
    verbatim and citation constraints, scored on what a gold nugget actually is:

    * **new facts** — numbers, dates, acronyms and proper nouns not yet used. A sentence about the
      right subject with no number in it cannot answer a question about a number;
    * **new content** — remaining content terms, so near-duplicate sentences of one passage cannot
      fill a five-sentence budget;
    * **parent relevance** — the retrieval score the sentence inherits, as a tie-break so a
      fact-dense sentence from an irrelevant document does not outrank a fact-dense relevant one.

    ``per_char`` divides that value by the sentence's length — the registered remedy for the length confound
    (prediction (c) in `docs/findings-extractive-coverage.md`): under the shipped objective a fact-dense sentence
    wins regardless of how long it is, and the arm measured 2.1x the text of the shipped order. It is **opt-in and
    changes nothing by default**, because the queued full-set row is measuring the shipped objective and a
    changed default under a running measurement is the provenance failure this repository catalogs five times.

    Query terms are deliberately *not* the coverage unit: every candidate comes from a passage
    retrieved for the query, so all of them contain its terms, and covering them again measures
    nothing. An earlier probe of this file's selection did exactly that and showed the shipped
    order covering 0–5 query terms where greedy coverage over the same candidates covered 2–7
    (`.scratch/extractive-probe.json`) — that was a symptom, not the target.
    """
    selected: list[ExtractedSentence] = []
    covered_facts: set[str] = set()
    covered_terms: set[str] = set()
    chosen: set[str] = set()
    remaining = list(candidates)
    spent_chars = 0
    while remaining and len(selected) < budget and (max_chars <= 0 or spent_chars < max_chars):
        best: ExtractedSentence | None = None
        best_value = float("-inf")
        for sentence in remaining:
            key = _normalise(sentence.text)
            if key in chosen:
                continue
            facts = _fact_terms(sentence.text)
            terms = _content_terms(sentence.text)
            if any(
                _jaccard(terms, _content_terms(other.text)) >= _DUPLICATE_JACCARD
                for other in selected
            ):
                continue  # the same sentence carrying the same facts
            value = (
                len(facts - covered_facts)
                + 0.25 * len(terms - covered_terms)
                + 0.5 * sentence.score
            )
            if per_char:
                # The registered remedy for the length confound (prediction (c)): score *new facts per
                # character* rather than new facts. Under the shipped objective a fact-dense sentence wins
                # regardless of how long it is, and the arm measured 2.1x the text of the shipped order —
                # per thousand characters it was the less efficient of the two. This is opt-in and does not
                # touch the default, because the queued full-set row is measuring the shipped objective.
                value = value / max(1, len(sentence.text))
            if value > best_value:
                best, best_value = sentence, value
        if best is None:  # every remaining candidate is a duplicate
            break
        if max_chars > 0 and selected and spent_chars + len(best.text) > max_chars:
            # The character budget is a *stop* rule on the absolute objective, not a per-character score:
            # it takes the most fact-dense sentences by absolute value and stops when the text is spent,
            # which selects *fewer long* sentences rather than *more short* ones. Prediction (c) tested the
            # other reading and collapsed; this one is prediction (d).
            break
        spent_chars += len(best.text)
        chosen.add(_normalise(best.text))
        covered_facts |= _fact_terms(best.text)
        covered_terms |= _content_terms(best.text)
        selected.append(best)
        remaining = [s for s in remaining if _normalise(s.text) != _normalise(best.text)]
    return selected


def compose_extractive(
    passages: Sequence[Mapping[str, Any]],
    *,
    query: str | None = None,
    max_sentences: int = 5,
    min_score: float = 0.0,
    coverage: bool = False,
    coverage_per_char: bool = False,
    max_chars: int = 0,
) -> ExtractiveAnswer:
    """Emit up to ``max_sentences`` verbatim sentences, best-scored first, each cited.

    A candidate sentence is kept only if it appears **verbatim** in one of the passage texts —
    that is the structural guarantee, and it survives the document merge the same way
    :func:`build_messages` presents evidence. Documents are numbered in first-appearance order,
    so ``[n]`` means the same thing here, in the prompt, and in `verify_citations`; the marker
    is emitted *before* the sentence it supports, for the reason given where the answer is
    assembled below.

    ``coverage`` selects by a submodular fact-coverage objective instead of by the parent
    document's rank (see :func:`_select_by_coverage`). It is opt-in because the shipped order is
    what every recorded end-to-end row was measured with; the guarantees — verbatim span, in-range
    citation, abstention when nothing qualifies — are identical either way.

    One visible difference: markers can then appear out of numerical order in the answer, because
    a fact-dense sentence from the second document may be selected first. The numbers are **not**
    renumbered, because ``[n]`` has to keep meaning the same document here, in the composed prompt
    and in `verify_citations`; a renumbering would make the answer's citations agree with
    themselves and disagree with the evidence list they are checked against.
    """
    if max_sentences < 1:
        raise ValueError("max_sentences must be positive")
    if not passages:
        return ExtractiveAnswer(answer=ABSTENTION, abstained=True)

    # Per-document best score, so a sentence inherits its document's rank.
    best: dict[str, float] = {}
    for passage in passages:
        key = str(passage.get("path") or "")
        best[key] = max(best.get(key, float("-inf")), _score(passage))

    merged = group_passages_by_document(
        [(passage.get("path"), str(passage.get("text") or "")) for passage in passages]
    )
    candidates: list[ExtractedSentence] = []
    for index, (path, text) in enumerate(merged, start=1):
        document_score = best.get(path, 0.0)
        if document_score < min_score:
            continue
        for sentence in split_sentences(text):
            # The structural guarantee: only a span that exists verbatim in a passage.
            source = next(
                (
                    passage
                    for passage in passages
                    if sentence in str(passage.get("text") or "")
                ),
                None,
            )
            if source is None or not _quotes_balanced(sentence):
                continue
            candidates.append(
                ExtractedSentence(
                    text=sentence,
                    citation=index,
                    path=path or None,
                    chunk_id=source.get("chunk_id"),
                    score=document_score,
                )
            )

    if coverage:
        # An objective of its own, opt-in: the shipped order is what every recorded row was
        # measured with, and `findings-extractive-row.md` quotes it. Selecting by coverage changes
        # *which* sentences are emitted, never the guarantees — every sentence is still a verbatim
        # span of a cited passage, and the abstention path is untouched.
        kept = _select_by_coverage(
            candidates, max_sentences, per_char=coverage_per_char, max_chars=max_chars
        )
    else:
        candidates.sort(key=lambda sentence: -sentence.score)
        seen: set[str] = set()
        kept = []
        for sentence in candidates:
            key = _normalise(sentence.text)
            if key in seen:
                continue
            seen.add(key)
            kept.append(sentence)
            if len(kept) >= max_sentences:
                break

    if not kept:
        return ExtractiveAnswer(answer=ABSTENTION, abstained=True, considered=len(candidates))
    # The marker **leads** its sentence rather than trailing it. `verify_citations` splits an
    # answer into claims *after* the terminal punctuation, so a trailing "[1]" is orphaned onto
    # the next claim — the sentence it was meant to support reads as uncited and the final marker
    # as a dangling citation. Emitting it first is what makes the structured citation contract
    # checkable without parsing prose, and it matches how `build_messages` numbers the evidence
    # the model is shown: "[1]" opens both the prompt's document block and the sentence quoting it.
    answer = " ".join(f"[{sentence.citation}] {sentence.text}" for sentence in kept)
    return ExtractiveAnswer(
        answer=answer, abstained=False, sentences=kept, considered=len(candidates)
    )
