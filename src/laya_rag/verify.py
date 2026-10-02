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
Laya (:meth:`laya_rag.engine.RagEngine.adjudicate_claims`).

A second, **advisory** layer checks whether a cited claim's numbers, dates and units
actually occur in the evidence it cites, and flags a negation that the evidence lacks.
Those checks are still deterministic, but unlike the citation check they have legitimate
exceptions (rounding, "five" for "5", a range endpoint), so they are reported in a
separate ``content`` block and never change the ``clean`` verdict. A noisy check is one
a reader learns to ignore.

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

#: Punctuation that a quotation absorbs from the sentence around it. A quote written
#: as "five business days," is still a verbatim quote of "five business days".
_SURROUNDING = " \t.,;:!?\"'()[]{}—–-"


def _quote_variants(needle: str) -> list[str]:
    """A normalised quote, plus itself with surrounding punctuation stripped.

    Quotations routinely absorb the comma or full stop that follows them. Trying the
    trimmed form as a *fallback* keeps this a literal comparison — only the ends are
    ignored, never the middle — while removing a false alarm that would otherwise fire
    on correctly-quoted text.
    """
    variants = [needle]
    trimmed = needle.strip(_SURROUNDING)
    if trimmed and trimmed != needle:
        variants.append(trimmed)
    return variants


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


# ── content consistency (advisory, still deterministic) ──────────────────
#
# Citation mechanics prove a cited span *exists*; they say nothing about whether the
# claim's numbers, dates and units agree with it. These checks are still mechanical and
# model-free, but unlike the citation check they are **advisory**: a numeric mismatch has
# legitimate exceptions (rounding, "five" vs "5", a range's endpoints), so they are
# reported beside the verdict, never folded into ``clean``. A noisy check is one a reader
# learns to ignore, which is worse than no check.

_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?")
_DATE_ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_DATE_SLASH = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
_DATE_MONTH_FIRST = re.compile(rf"\b{_MONTH}\.?\s+\d{{1,2}},?\s+\d{{4}}\b", re.I)
_DATE_DAY_FIRST = re.compile(rf"\b\d{{1,2}}\s+{_MONTH}\.?,?\s+\d{{4}}\b", re.I)
_MEASUREMENT = re.compile(
    r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*"
    r"(%|percent|business\s+days?|days?|hours?|minutes?|seconds?|weeks?|months?|years?|[kmgt]i?b|bytes?)\b",
    re.I,
)
_NEGATION = re.compile(
    r"\b(?:not|no|never|cannot|can't|won't|doesn't|don't|isn't|aren't|without|none|nor)\b",
    re.I,
)
_CONTENT_WORD = re.compile(r"[a-z][a-z0-9]{2,}")
#: Words that carry no topical meaning for the negation heuristic.
_CONTENT_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "has",
    "have", "had", "its", "into", "than", "then", "but", "not", "you", "your", "per",
})

#: Unit family -> the canonical spellings that belong together. A claim is only flagged
#: when its own unit is absent from the evidence *and* the evidence uses a sibling unit,
#: which is the "days" vs "business days" case rather than a mention of two different
#: quantities.
_UNIT_CANON = {
    "day": ("time", "day"), "days": ("time", "day"),
    "business day": ("time", "business day"), "business days": ("time", "business day"),
    "hour": ("time", "hour"), "hours": ("time", "hour"),
    "minute": ("time", "minute"), "minutes": ("time", "minute"),
    "second": ("time", "second"), "seconds": ("time", "second"),
    "week": ("time", "week"), "weeks": ("time", "week"),
    "month": ("time", "month"), "months": ("time", "month"),
    "year": ("time", "year"), "years": ("time", "year"),
    "byte": ("size", "byte"), "bytes": ("size", "byte"),
    "kb": ("size", "kb"), "kib": ("size", "kb"),
    "mb": ("size", "mb"), "mib": ("size", "mb"),
    "gb": ("size", "gb"), "gib": ("size", "gb"),
    "tb": ("size", "tb"), "tib": ("size", "tb"),
    "%": ("rate", "percent"), "percent": ("rate", "percent"),
}


def _canonical_number(token: str) -> str:
    """Fold a numeric literal so ``1,200`` and ``1200`` compare equal."""
    body = token.rstrip("%").replace(",", "")
    if "." in body:
        body = body.rstrip("0").rstrip(".")
    return (body or "0") + ("%" if token.endswith("%") else "")


def extract_numbers(text: str, *, distinctive: bool = True) -> list[str]:
    """Numeric literals, optionally only the ones worth checking.

    ``distinctive`` keeps multi-digit, decimal or percent values and drops bare single
    digits, which are mostly list counts ("3 steps") and would generate false alarms.
    """
    out: list[str] = []
    for match in _NUMBER.finditer(str(text)):
        token = match.group(0)
        digits = token.rstrip("%").replace(",", "").replace(".", "")
        if distinctive and len(digits) < 2 and "." not in token and not token.endswith("%"):
            continue
        canonical = _canonical_number(token)
        if canonical not in out:
            out.append(canonical)
    return out


def extract_dates(text: str) -> list[str]:
    """Date literals in ISO, slash, or month/day/year form, normalised for comparison."""
    found: list[str] = []
    for pattern in (_DATE_ISO, _DATE_SLASH, _DATE_MONTH_FIRST, _DATE_DAY_FIRST):
        for match in pattern.finditer(str(text)):
            value = _WHITESPACE.sub(" ", match.group(0)).strip().rstrip(",").casefold()
            value = value.replace(".", "")
            if value not in found:
                found.append(value)
    return found


def extract_units(text: str) -> list[tuple[str, str]]:
    """``(family, canonical)`` for every measured unit in the text."""
    out: list[tuple[str, str]] = []
    for match in _MEASUREMENT.finditer(str(text)):
        unit = _WHITESPACE.sub(" ", match.group(2)).strip().casefold().rstrip(".")
        canonical = _UNIT_CANON.get(unit)
        if canonical and canonical not in out:
            out.append(canonical)
    return out


def _content_words(text: str) -> set[str]:
    return {
        word for word in _CONTENT_WORD.findall(normalize(text))
        if word not in _CONTENT_STOPWORDS
    }


def content_issues(claim: str, cited_texts: Sequence[str]) -> list[dict[str, Any]]:
    """Advisory mismatches between a claim and the passages it cites.

    Each issue is ``{"kind", "detail", "severity"}`` with severity ``"advisory"``: the
    kinds are a number or date absent from the cited evidence, a unit that disagrees with
    the evidence's unit of the same family, and a possible negation flip. None of them
    changes the ``clean`` verdict.
    """
    evidence = " \n ".join(str(text) for text in cited_texts)
    if not evidence.strip():
        return []

    issues: list[dict[str, Any]] = []
    evidence_numbers = set(extract_numbers(evidence))
    for number in extract_numbers(claim):
        if number not in evidence_numbers:
            issues.append({
                "kind": "number_not_in_evidence",
                "detail": number,
                "severity": "advisory",
            })

    evidence_dates = set(extract_dates(evidence))
    for date in extract_dates(claim):
        if date not in evidence_dates:
            issues.append({
                "kind": "date_not_in_evidence",
                "detail": date,
                "severity": "advisory",
            })

    evidence_units = extract_units(evidence)
    for family, canonical in extract_units(claim):
        if (family, canonical) in evidence_units:
            continue
        siblings = sorted({unit for fam, unit in evidence_units if fam == family})
        if siblings:
            issues.append({
                "kind": "unit_mismatch",
                "detail": f"claim says {canonical}, evidence says {', '.join(siblings)}",
                "severity": "advisory",
            })

    # A negation in the claim but none in the evidence, with enough shared vocabulary to
    # be about the same thing. Two content tokens is a deliberately high bar: below it the
    # two sentences are usually not comparable at all.
    if _NEGATION.search(claim) and not _NEGATION.search(evidence):
        shared = _content_words(claim) & _content_words(evidence)
        if len(shared) >= 2:
            issues.append({
                "kind": "possible_negation_flip",
                "detail": ", ".join(sorted(shared)[:4]),
                "severity": "advisory",
            })
    return issues


def verify_citations(
    answer: str,
    passages: Sequence[str],
    query: str = "",
) -> dict[str, Any]:
    """Check the answer's citations and quoted spans against the evidence.

    Quoted spans are checked against the evidence as a whole, whichever sentence
    they appear in and whether or not that sentence cites anything. Where a quote
    does turn up is reported, so an uncited sentence that is nevertheless faithful
    is visible as such rather than merely flagged.

    ``query`` is optional and exists to remove a false alarm: an answer that quotes
    *the question back* is not quoting the evidence, and flagging that as "not found"
    would train a reader to ignore the check. Such spans are reported as
    ``from_question`` instead.

    Returns a JSON-ready report. ``clean`` means every claim cites in-range evidence
    and every quoted span was found; it does **not** mean the answer is correct.
    ``content`` carries the advisory number/date/unit/negation checks separately and
    never changes ``clean``.
    """
    texts = list(passages)
    passage_count = len(texts)
    query_normalized = normalize(query)

    claims: list[dict[str, Any]] = []
    totals = {
        "claims": 0, "cited": 0, "uncited": 0, "invalid_citations": 0,
        "quotes_checked": 0, "quotes_found": 0, "quotes_not_found": 0,
        "quotes_from_question": 0,
    }
    content_totals = {
        "number_not_in_evidence": 0,
        "date_not_in_evidence": 0,
        "unit_mismatch": 0,
        "possible_negation_flip": 0,
    }

    for claim in split_claims(answer):
        cited = extract_citations(claim)
        invalid = [index for index in cited if index < 1 or index > passage_count]
        valid = [index for index in cited if index not in invalid]

        issues: list[str] = []
        if not cited:
            issues.append("uncited")
        if invalid:
            issues.append("citation out of range")

        totals["claims"] += 1
        totals["cited"] += 1 if cited and not invalid else 0
        totals["uncited"] += 0 if cited else 1

        advisory = content_issues(claim, [texts[index - 1] for index in valid])
        for issue in advisory:
            content_totals[issue["kind"]] += 1

        claims.append({
            "text": claim,
            "citations": cited,
            "invalid_citations": invalid,
            "uncited": not cited,
            "issues": issues,
            "content_issues": advisory,
        })

    quotes: list[dict[str, Any]] = []
    for quote in extract_quotes(answer):
        variants = [v for v in _quote_variants(normalize(quote)) if v]
        hits: list[int] = []
        for variant in variants:
            hits = [position + 1 for position, text in enumerate(texts)
                    if variant in normalize(text)]
            if hits:
                break
        totals["quotes_checked"] += 1
        if hits:
            totals["quotes_found"] += 1
            quotes.append({"text": quote, "status": "found", "found_in": hits})
            continue
        # The answer quoting the question is expected, not a fabrication.
        if query_normalized and any(v in query_normalized for v in variants):
            totals["quotes_from_question"] += 1
            quotes.append({"text": quote, "status": "from_question", "found_in": []})
            continue
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

    content_clean = not any(content_totals.values())
    return {
        "claims": claims,
        "quotes": quotes,
        "invalid_citations": invalid_citations,
        "counts": totals,
        "content": {
            "counts": content_totals,
            "clean": content_clean,
            "summary": _summarise_content(content_totals),
        },
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
    from_question = totals["quotes_from_question"]
    if checked and missing == 0:
        if from_question:
            parts.append("every quoted span either appears in the evidence or quotes the question")
        else:
            parts.append("every quoted span appears verbatim")
    elif missing:
        noun = "quoted span" if missing == 1 else "quoted spans"
        parts.append(f"{missing} of {checked} {noun} {'is' if missing == 1 else 'are'} not in the evidence")

    return "Citation check: " + "; ".join(parts) + "."


def _summarise_content(counts: dict[str, int]) -> str:
    """One advisory line; never phrased as a correctness verdict."""
    parts: list[str] = []
    if counts["number_not_in_evidence"]:
        noun = "number" if counts["number_not_in_evidence"] == 1 else "numbers"
        parts.append(f"{counts['number_not_in_evidence']} cited {noun} not in the cited evidence")
    if counts["date_not_in_evidence"]:
        noun = "date" if counts["date_not_in_evidence"] == 1 else "dates"
        parts.append(f"{counts['date_not_in_evidence']} cited {noun} not in the cited evidence")
    if counts["unit_mismatch"]:
        parts.append(f"{_plural(counts['unit_mismatch'], 'unit mismatch', 'unit mismatches')}")
    if counts["possible_negation_flip"]:
        parts.append(
            f"{_plural(counts['possible_negation_flip'], 'possible negation flip')}"
        )
    if not parts:
        return (
            "Content check: the numbers, dates and units of every cited claim were found "
            "in the evidence it cites."
        )
    return "Content check (advisory): " + "; ".join(parts) + "."
