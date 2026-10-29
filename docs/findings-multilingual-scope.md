# R7 — what "multilingual" currently means, measured

`docs/best-in-the-world-2026-10.md` R7 asked for one multilingual collection, on the grounds that a
CPU-only multilingual retrieval row appears in no table the 2026 survey found. This is not that. It is
the cheaper and more urgent question underneath it: **the README claims the default checkpoint "handles
non-English text", and nothing in this repository has ever tested what that means for retrieval.**

The distinction matters because the claim is about the *checkpoint* and the engine's shipped recall is
lexical. A lexical channel cannot match a German query to an English passage; a checkpoint that reads
German does not change that.

## The probe

Five languages — the English original, German, French, Spanish, Japanese — for three questions whose
answers live in specific documents of the indexed corpus, asked against the **same English index** under
two configurations. Target recall@20, i.e. whether the document that answers the English question appears
anywhere in the returned evidence:

| configuration | en | de | fr | es | ja |
| --- | ---: | ---: | ---: | ---: | ---: |
| shipped (lexical, `DENSE_WEIGHT=0`) | 0.67 | **0.33** | **0.33** | 0.00 | **0.33** |
| dense forced on (`DENSE_WEIGHT=1`) | 0.67 | 0.00 | 0.00 | 0.00 | 0.00 |

Three questions per language, so each cell is 0/1/2/3 — a probe, not a measurement. But the direction is
unambiguous and it is not the direction anyone would guess.

## Two findings, one of them backwards

**Non-English retrieval works only by lexical accident.** With the shipped channel the German, French
and Japanese queries each find the right document once out of three, and the mechanism is visible in the
tokens: `nDCG@10`, `BEIR`, `SciFact`, `answerability`, `Hop` survive translation intact, and BM25 matches
them. Spanish found nothing because its paraphrase kept none of them. This is not multilingual retrieval;
it is proper-noun leakage.

**Forcing the dense channel on makes non-English retrieval worse, not better.** At `DENSE_WEIGHT=1` the
encoder contributes 20 candidates per query and non-English recall falls to **zero** in every language
while English is unchanged. That is consistent with everything this repository already knows about this
encoder — nDCG@10 ≈ 0.014 on SciFact against BM25's 0.654, which is why the dense channel ships off — and
it adds one specific fact: **the multilingual checkpoint's encoder is not cross-lingual for retrieval
either.** It reads non-English text; it does not place German and English sentences near each other.

So the accurate statement, and the one the README now carries: *the checkpoint is multilingual; the
retrieval is English, and its non-English behaviour is accidental token overlap.* The dense channel does
not rescue it and currently makes it worse.

## What would change it

A **retrieval-trained multilingual bi-encoder** through the existing `EMBED_MODEL` path — the repository
already measured that a retrieval-trained encoder is what makes the dense channel worth fusing
(SciFact 0.654 → 0.732 with `bge-small`), and the multilingual members of that family are the obvious
candidates. That is a `EMBED_MODEL` configuration, not a code change, and it needs its own measurement
per language before any claim is made.

## Why R7's original request is not met, stated plainly

R7 asked for one multilingual collection (MIRACL, MMTEB, a non-English retrieval task). The smallest
MIRACL collection that carries a published leaderboard row — Swahili — is **106,000 documents**, which at
this host's measured rate for the 322M multilingual encoder (7.9 chunks/s, from the 4,592-chunk index
build) is roughly **28 hours** of embedding for one language, before any query is asked. The larger
collections are an order of magnitude beyond that. That is a hardware limit, not a design one, and the
honest output is to say so and record the probe above rather than publish a truncation dressed as a
measurement — the mistake this repository quarantined once already when a 40,000-document TREC-COVID
subsample read 0.3062 against the full collection's 0.6188.

## What this changes

- **The README's capability sentence.** "Handles non-English text" was true of the checkpoint and
  unqualified in a paragraph about the engine. It now says what the probe measured.
- **The claim the programme can make.** Not "multilingual retrieval" — that is unmeasured and currently
  false. What is defensible: the engine runs, chunks, indexes and gates in any language, and its
  *retrieval* is English until a retrieval-trained multilingual encoder is configured and measured.
- **A next step with a number attached.** `EMBED_MODEL` set to a multilingual retrieval encoder, then the
  same probe — five languages, same three questions — as the cheap falsifier before any MIRACL run is
  worth its 28 hours.
