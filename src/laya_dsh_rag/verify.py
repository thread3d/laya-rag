"""Deterministic checks on composed prose.

This module answers one narrow question: **does the generated answer cite the
evidence it was given, and are the spans it quotes actually present there?** It makes
no claim about whether the answer is *true*, and nothing here should be presented as
if it did.

The distinction is the whole point. A fabricated quote, a citation to a passage that
does not exist, or an assertion with no citation at all are facts about the text, and
they can be decided mechanically with no model and no threshold — so a check like
this cannot itself be wrong in the way a model can. Whether a correctly-cited
sentence is a *fair reading* of its source is a different question, and it is left to
Laya (:meth:`laya_dsh_rag.engine.RagEngine.adjudicate_claims`).

Because these checks are deterministic they need no calibration, which is what makes
them safe to show immediately: there is no probability to misread.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

#: Shorter sentences are usually fragments ("See [1].") and carry no checkable claim.
CLAIM_MIN_CHARS = 15

#: Shorter quoted spans produce nothing but noise ("`a`", `"it"`).
QUOTE_MIN_CHARS = 4

_FENCED = re.compile(r"```.*?```", re.DOTALL)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CITATION = re.compile(r"\[(\d+)\]")
_BACKTICK_QUOTE = re.compile(r"`([^`\n]+)`")
_DOUBLE_QUOTE = re.compile(r"[\"“]([^\"“”\n]{%d,})[\"”]" % QUOTE_MIN_CHARS)

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Fold text to a comparable form: case, whitespace, and quote style.

    Used only to decide whether a quoted span appears in a passage. It is
    deliberately conservative: it does not stem, reorder, or fuzzy-match, so a
    positive result means the span really is there rather than merely similar.
    """
    folded = (
        str(text)
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00a0", " ")
    )
    return _WHITESPACE.sub(" ", folded).strip().casefold()


def extract_citations(text: str) -> list[int]:
    """Return the distinct evidence indices a span cites, in order of appearance."""
    seen: list[int] = []
    for match in _CITATION.finditer(str(text)):
        index = int(match.group(1))
        if index not in seen:
            seen.append(index)
    return seen


def extract_quotes(text: str) -> list[str]:
    """Return spans the answer presents as verbatim: backticks and double quotes.

    Fenced code blocks are excluded — they are handled separately because an answer
    legitimately contains whole commands lifted from a passage.
    """
    stripped = _FENCED.sub(" ", str(text))
    found: list[str] = []
    for pattern in (_BACKTICK_QUOTE, _DOUBLE_QUOTE):
        for match in pattern.finditer(stripped):
            candidate = match.group(1).strip()
            if len(candidate) < QUOTE_MIN_CHARS:
                continue
            if candidate not in found:
                found.append(candidate)
    return found


def split_claims(answer: str) -> list[str]:
    """Split composed prose into the sentences and bullets that assert something.

    Headings, fenced code blocks and bare list markers are dropped: they are
    structure or content, not claims that can be cited or checked. So are lead-ins
    that end in a colon ("From the repository root:"), which introduce a block
    rather than assert anything — flagging those as uncited claims would be noise,
    and a noisy check is one nobody trusts. Markdown link syntax is left intact so a
    citation inside a link is still visible to :func:`extract_citations`.
    """
    without_code = _FENCED.sub("\n", str(answer))
    claims: list[str] = []
    for raw_line in without_code.splitlines():
        line = raw_line.strip()
        if not line or _HEADING.match(line):
            continue
        if set(line) <= set("|-: "):  # markdown table rule
            continue
        line = _BULLET.sub("", line)
        for piece in _SENTENCE_SPLIT.split(line):
            claim = piece.strip()
            if claim.endswith(":"):
                continue
            # Ignore fragments that survive only as punctuation or a lone citation.
            body = _CITATION.sub("", claim).strip()
            if len(body) < CLAIM_MIN_CHARS or len(body.split()) < 3:
                continue
            claims.append(claim)
    return claims


def verify_citations(answer: str, passages: Sequence[str]) -> dict[str, Any]:
    """Check the answer's citations and quoted spans against the evidence.

    Quoted spans are checked against the evidence as a whole, whichever sentence
    they appear in and whether or not that sentence cites anything. Where a quote
    does turn up is reported, so an uncited sentence that is nevertheless faithful
    is visible as such rather than merely flagged.

    Returns a JSON-ready report. ``clean`` means every claim cites in-range evidence
    and every quoted span was found; it does **not** mean the answer is correct.
    """
    texts = list(passages)
    passage_count = len(texts)

    claims: list[dict[str, Any]] = []
    totals = {
        "claims": 0, "cited": 0, "uncited": 0, "invalid_citations": 0,
        "quotes_checked": 0, "quotes_found": 0, "quotes_not_found": 0,
    }

    for claim in split_claims(answer):
        cited = extract_citations(claim)
        invalid = [index for index in cited if index < 1 or index > passage_count]

        issues: list[str] = []
        if not cited:
            issues.append("uncited")
        if invalid:
            issues.append("citation out of range")

        totals["claims"] += 1
        totals["cited"] += 1 if cited and not invalid else 0
        totals["uncited"] += 0 if cited else 1

        claims.append({
            "text": claim,
            "citations": cited,
            "invalid_citations": invalid,
            "uncited": not cited,
            "issues": issues,
        })

    quotes: list[dict[str, Any]] = []
    for quote in extract_quotes(answer):
        hits = [position + 1 for position, text in enumerate(texts)
                if normalize(quote) and normalize(quote) in normalize(text)]
        totals["quotes_checked"] += 1
        if hits:
            totals["quotes_found"] += 1
            quotes.append({"text": quote, "status": "found", "found_in": hits})
        else:
            totals["quotes_not_found"] += 1
            quotes.append({"text": quote, "status": "not_found", "found_in": []})

    # Scan for out-of-range citations across the whole answer, not just the sentences
    # that survive claim splitting: a short fragment such as "See [7]." is not a
    # checkable claim, but a citation pointing outside the evidence is still a defect
    # and must not disappear with it.
    invalid_citations = [
        index for index in extract_citations(answer) if index < 1 or index > passage_count
    ]
    totals["invalid_citations"] = len(invalid_citations)

    return {
        "claims": claims,
        "quotes": quotes,
        "invalid_citations": invalid_citations,
        "counts": totals,
        "passages": passage_count,
        "clean": (
            totals["claims"] > 0
            and totals["uncited"] == 0
            and totals["invalid_citations"] == 0
            and totals["quotes_not_found"] == 0
        ),
        "summary": _summarise(totals, passage_count),
    }


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    """``3 claims`` / ``1 claim``, so the summary never reads like a template."""
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def _summarise(totals: dict[str, int], passage_count: int) -> str:
    """One line a reader can act on, phrased so it cannot be read as 'true'."""
    claims = totals["claims"]
    if claims == 0:
        return "Nothing in the answer was checkable as a claim."

    parts: list[str] = []
    uncited = totals["uncited"]
    if uncited == 0:
        parts.append("every claim cites evidence")
    elif uncited == claims:
        parts.append("no claim cites evidence")
    else:
        parts.append(f"{uncited} of {_plural(claims, 'claim')} cite no evidence")

    invalid = totals["invalid_citations"]
    if invalid:
        where = "the evidence" if passage_count == 1 else f"the {passage_count} passages"
        parts.append(f"{_plural(invalid, 'citation')} {'points' if invalid == 1 else 'point'} outside {where}")

    checked = totals["quotes_checked"]
    missing = totals["quotes_not_found"]
    if checked and missing == 0:
        parts.append("every quoted span appears verbatim")
    elif missing:
        noun = "quoted span" if missing == 1 else "quoted spans"
        parts.append(f"{missing} of {checked} {noun} {'is' if missing == 1 else 'are'} not in the evidence")

    return "Citation check: " + "; ".join(parts) + "."
