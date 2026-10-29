"""What `clean` certifies, pinned as a table — because every published row depends on the answer.

`verify_citations` has two layers and its docstring says so: structural checks that decide `clean`, and an
**advisory** content layer that never folds into it. The individual behaviours are tested granularly in
`test_verify.py`; what was not tested is the *semantics as a whole*, which is the thing a reader of this
repository's tables needs and the thing a future change is most likely to break.

Two failure modes this catches, both of which would silently invalidate recorded numbers:

* making `clean` stricter — folding the content layer in — turns every published `citation_clean_all 1.0000`
  into a claim about a different property, and the rows are not re-measurable on demand;
* making it looser — missing a quote that is not in the evidence, or accepting an out-of-range citation — would
  let the engine's central guarantee weaken without a single number moving.

The table is the audit recorded in `docs/findings-citation-mechanics.md`, so the document and the code cannot
drift apart.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laya_rag.verify import verify_citations  # noqa: E402

PASSAGES = [
    "The engine reads 0.4799 nDCG@10 over the complete BEIR-18, above SPLADE's 0.4740.",
    "The answerability gate halves the false-accept rate: 30.7% against 69.3%.",
]

#: (case, answer, expected `clean`, expected content-clean) — `None` means "not asserted".
CASES = (
    ("verbatim and correctly cited",
     "[1] The engine reads 0.4799 nDCG@10 over the complete BEIR-18, above SPLADE's 0.4740.", True, True),
    ("one number changed",
     "[1] The engine reads 0.4979 nDCG@10 over the complete BEIR-18, above SPLADE's 0.4740.", True, False),
    ("wrong citation index",
     "[2] The engine reads 0.4799 nDCG@10 over the complete BEIR-18, above SPLADE's 0.4740.", True, None),
    ("polarity flipped",
     "[1] The engine reads 0.4799 nDCG@10 over the complete BEIR-18, below SPLADE's 0.4740.", True, None),
    ("two sentences, one fabricated",
     "[1] The answerability gate halves the false-accept rate: 30.7% against 69.3%. [2] The engine reads 0.99.", True, False),
    ("quote absent from the evidence",
     '[1] The report says "0.51 over BEIR-18" as measured.', False, None),
    ("no citation at all",
     "The engine reads 0.4799 nDCG@10 over the complete BEIR-18.", False, None),
)


def test_the_two_layers_are_as_documented():
    """Structural failures break `clean`; content failures do not. Anything else is a redefinition."""
    offenders = []
    for name, answer, expected_clean, expected_content in CASES:
        report = verify_citations(answer, PASSAGES)
        if report["clean"] is not expected_clean:
            offenders.append(f"{name}: clean={report['clean']}, expected {expected_clean}")
        if expected_content is not None and report["content"]["clean"] is not expected_content:
            offenders.append(
                f"{name}: content.clean={report['content']['clean']}, expected {expected_content}"
            )
    assert not offenders, "the citation checker's semantics changed:\n  " + "\n  ".join(offenders)


def test_a_content_failure_is_reported_even_though_it_does_not_break_clean():
    """The advisory layer must *fire* — a check that never fires is not a check, however advisory.

    This is the half that makes the recorded `content_clean_all` fields worth having: the layer is not decorative,
    it catches a changed number and a fabricated figure.
    """
    changed = verify_citations(
        "[1] The engine reads 0.4979 nDCG@10.", ["The engine reads 0.4799 nDCG@10."]
    )
    assert changed["content"]["counts"]["number_not_in_evidence"] >= 1
    assert changed["clean"] is True

    fabricated = verify_citations("[1] The score is 0.99.", ["The score is 0.31."])
    assert fabricated["content"]["counts"]["number_not_in_evidence"] >= 1


def test_a_polarity_flip_is_outside_what_the_checks_can_see():
    """Stated as a limit rather than left to be discovered: antonyms are not negators.

    `_NEGATION` matches not/no/never/cannot/without/none/nor. "Above" versus "below" is invisible to it, so an
    inverted claim passes both layers. The defence is structural (the engine emits verbatim spans, which cannot be
    inverted) rather than semantic — and this test exists so that nobody reads the passing verdict as a semantic
    guarantee.
    """
    report = verify_citations(
        "[1] The engine reads 0.4799 nDCG@10 over the complete BEIR-18, below SPLADE's 0.4740.",
        ["The engine reads 0.4799 nDCG@10 over the complete BEIR-18, above SPLADE's 0.4740."],
    )
    assert report["clean"] is True
    assert report["content"]["clean"] is True
    assert report["content"]["counts"]["possible_negation_flip"] == 0
