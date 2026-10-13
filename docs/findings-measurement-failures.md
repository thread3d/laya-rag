# How this repository's measurements failed, in one place

Every entry below is a **concrete instance from this project**, not a hypothetical. They are collected
here because they are the most transferable thing the programme has produced: a benchmark harness is
mostly machinery for avoiding these, and each one was found by *using* the machinery rather than by
reading it.

The organising observation, which took eleven instances to state plainly:

> **A measurement that did not measure what it claims is indistinguishable from one that did, until
> someone re-derives it from a different direction.**

Nothing in a table contradicts a wrong number. That is why every entry ends with what, if anything,
catches it now — and why four of the entries are checks that did not exist before this sweep.

---

## 1. A flag that is parsed and never read

**Looks like:** a documented knob that does nothing. No error, no warning; the command runs.

**Instance.** `tools/mteb_search.py --first-stage-only` printed a banner and set nothing on the adapter.
For four rounds, every "first-stage-only" MTEB run still ran the 322M decision on every candidate —
because `main` printed the message and never assigned the flag. When it was finally caught, **18
recorded reports turned out to be their first stage's ordering wearing a reranker's name.**

**Caught by:** dumping stacks during a five-query probe (`engine.query → rerank → score_relevance →
predict_many`), after two wrong hypotheses about why the run was slow.

**Now:** `tests/test_tools_flags_are_used.py` fails on any `add_argument` whose `dest` is never read.
It found four more in the same pass, and all five are resolved — three removed (their mechanism did not
exist), one wired, one reclassified as a default-mode alias.

## 2. A measurement attributed to a mechanism that was not running

**Looks like:** a number in a table, a justification in a README, a default with a reason.

**Instance.** `rerank(order="recall")` **sorted by the first-stage score** instead of preserving the
ordering it was handed. The coverage pass permutes the pool and then asks for `recall` precisely so its
order survives, so the sort undid every coverage decision and made a configured cross-encoder's scores
irrelevant. The README's `RERANK_COVERAGE_LAMBDA` default was justified by numbers taken through that
path.

**Caught by:** two different cross-encoders producing **byte-identical** metrics — 200 of 200 rankings —
which is not a plausible coincidence.

**Now:** the sort preserves the order it is given (three pinning tests), the README row is marked
suspended, and §P4 records the re-take that would reinstate or drop the default.

## 3. A claim of absence

**Looks like:** "no X exists", "X is unmeasured", "three pairs are pending".

**Instances.** *"No MTEB score exists"* in four places after scores existed. *"FEVER blocked by the
in-memory index"* after the on-disk BM25F landed. *"Three more task pairs are half-collected"* when
ArguAna was complete. None of these is a wrong number, so no gate could see any of them.

**Caught by:** re-reading the documents against their own findings, six separate times.

**Now:** nothing automatic — and that is a considered choice. A claim about what does not exist has no
artefact to compare against, so the only defence is that §10.5 exists to be re-read.

## 4. A level read across protocols

**Looks like:** "our score is above published BM25".

**Instance.** Reading MTEB's SciFact `nDCG@10 0.69482` against BEIR-convention baselines (BM25 0.665,
`bge-small` 0.72). MTEB's corpus carries titles and uses MTEB's evaluator; the two differ by more than
the retriever. **I published that comparison and then removed it.**

**Caught by:** asking what convention each number came from, one commit later.

**Now:** a baseline flag evaluates a stock model *through the same harness*, and the finding says the
level is a measured 0.018 below it rather than a cross-protocol inference.

## 5. A gate that cannot see what it guards

**Looks like:** a green gate. It runs, it exits 0, it checks 43 sites.

**Instance.** Perturbing `0.7085` in the **summary blockquote at the top of §1** — the first number a
reader meets — did not fail `consistency-check`. The gate was fine; that line simply was not one of its
sites. It could have drifted silently while every checked site stayed true.

**Caught by:** running the gate's own negative test instead of trusting its green.

**Now:** the blockquote is anchored (**43 → 47 sites**) and the perturbation fails as it should.

## 6. A verifier that had stopped running

**Looks like:** a passing gate whose code path exits early.

**Instance.** `tools/leakage_audit.py` raised an uncaught `UnicodeDecodeError` on a 24 GB latin-1 dump
and had been aborting for months. The dilution rule it exists to enforce was not running at all when it
was first looked at.

**Caught by:** reading the audit's code, then planting the signature it looks for.

**Now:** it catches a planted dilution and exits 2. A *fixed* audit is not the same as a *detecting*
one — which is why the plant is part of the record.

## 7. A gate that is green because it had nothing to do

**Looks like:** the same thing as a passing gate.

**Instance.** `make audit-verify` exits 0 with *"no log at eval/audit.jsonl (nothing to verify)"*. No
target produces that log and none exists, so the gate is **vacuous** on this checkout.

**Caught by:** running every cheap check target rather than the six that were familiar.

**Now:** recorded *and* distinguished from the previous entry: that verifier is tested, including a
tampered-record case that must exit 1. It is **idle, not suspect** — and "vacuous" on its own reads as
"suspect", so the sentence has to say which.

## 8. Silence mistaken for a hang

**Looks like:** a long job with no output. Three separate phases did this: the MTEB search, the MTEB
index, and `beir.py`'s on-disk BM25 build. Each cost rounds.

**Instance.** The MTEB adapter reported progress every 25 queries in a driver that could call `search`
once; the on-disk build printed one line and then nothing for ~30 minutes at 2.7M documents.

**Caught by:** `faulthandler` stack dumps, and by the observation that a phase reporting nothing is
indistinguishable from a phase that is stuck.

**Now:** every long phase reports — search, index, and the on-disk build — and §10.6 records what a
healthy run looks like, including that **33 processes is not a runaway if they are the run's children**.

**And the inverse:** a wrapper that filtered a long run's output through `grep` hid a working run's
progress lines, reproducing this same failure in a new place. *The instruments only help if the wrapper
does not discard them.*

## 9. A patch that doubles its own text

**Looks like:** correct prose with one sentence repeated.

**Instance.** Three commits needed a follow-up for a patch that missed or duplicated its anchor,
leaving this in the document:

```
**The re-takes beyond SciFact.****The re-takes beyond SciFact.**
```

**Caught by:** a new check, `tests/test_docs_no_doubled_text.py` — after a first version that split on
sentence boundaries and therefore could not see two copies joined with **no whitespace**.

## 10. A number that nothing counts

**Looks like:** a helpful exact figure in a comment.

**Instance.** The newcomer runbook said `make test # 1251 tests` and `make claim-check # 58 prose
claims`; the real figures were 1,718 and 65. Separately, **I reported an ETA "~2.5 h remaining" for
four rounds when the counter said ten hours.**

**Caught by:** reading the counter instead of the recollection. *Do not write a number where a checked
number would do, and do not write an exact number where the thing it counts changes weekly.*

## 11. A subset read as the member

**Looks like:** a report in the right directory with the right shape.

**Instance.** A one-sub-forum CQADupStack run was written to `eval/beir/`. Two authorities scan that
directory's top level, so the member appeared to have a dense channel (moving §1.1's lexical-only count
8 → 7 on a twelfth of the corpus) and `bm25-engine` appeared to disagree with itself.

**Caught by:** both gates failing at once.

**Now:** subset runs go to `eval/beir/subsets/`, and the rule is in the operating notes. Neither
authority is wrong — they cannot tell a subset from the member, which is why the path has to say it.

## 12. Checkers that are wrong about what they check

**Looks like:** a new gate with findings.

**Instances.** The `make`-target check produced **nineteen English false positives** ("make up", "make
for") before being scoped to code. The flag check produced **four** against a tool that declares its
flags with single quotes. The knob comparison reported two offenders that were its own prefix bug.

**Caught by:** examining each check's own first output instead of trusting the count.

**Now:** this is stated as a rule in §10.3 — **checking a checker is a different job from running it** —
and two classes (paths, configuration completeness) are deliberately *not* automated because a checker
that guesses intent is worse than none.

---

## 13. A correct number, over-generalised

**Looks like:** a measurement. Every figure in it is real, and the sentence around it is not.

This one is different from the twelve above: nothing is mis-measured, mis-attributed or missing. The
*sentence* claims more than the *data* supports, which is why no numeric gate can see it — the gate checks
the number, and the number is right.

**Instances, all from the last five rounds of this session, all caught by exercising the thing from a
second direction:**

* **"the model answers two orders of magnitude below a cross-encoder"** — true of the two hand-picked
  passages it was measured on (0.0505), false of the model: the engine's own chunk of a relevant passage
  scores **0.4455**. Corrected by running the engine path rather than the direct probe.
* **"a run at the default threshold would find nothing"** — true of everything measured, which is not the
  same as true. Now states the measured range: the highest score observed is 0.4455 against a 0.5 cut, and
  whether anything clears it is unmeasured.
* **"a listwise ranker is worse"** — true under the instruction arm 2 used, and the instruction is a
  **7×** variable. Narrowed to "much worse *under a web-search instruction*", with the re-run named.
* and, from the other side, **a speedup of "3×"** that was really **1.2×** — not over-generalised but
  mis-*controlled*: the first timing included the model load. A measurement that includes its own setup is
  the same family of error with the sign reversed.

**The tell is that the sentence survives no matter which example you pick, and then one more example
breaks it.** Three of these four were fixed by *running something the sentence did not cover*; none was
fixed by re-reading, which is why this class is now in a catalogue about measurement rather than in a list
of writing tips.

## What they cost, counted rather than remembered

`git rev-list --count` over the session's span: **133 commits.** The per-class figures below are
`git log | grep -ci` keyword counts, so they are indicative rather than exact — a commit touching two
classes counts twice, and one whose message does not name a class counts not at all:

| class | commits |
| --- | ---: |
| MTEB (silence, the inert flag, the placement) | **30** |
| the reranker (the ordering defect, both arms, the re-takes) | **15** |
| on-disk BM25F | 11 |
| flags | 5 |
| consistency-check | 3 |
| doubled text | 3 |
| leakage audit | 2 |
| inert-flag check | 2 |

The two classes at the top are **34% of the session's commits**, and both are of the same shape: a
number that looked right and was not. That is the argument for the checks in one line — not that the
failures were embarrassing, but that **two of them cost a third of the work**, and each was caught by
re-deriving it from a direction nobody had used the first time.

## The methods, in the order they proved useful

1. **Perturb the claim.** If a gate cannot be made to fail, it is not a gate.
2. **Re-derive from a different direction.** Two models producing identical rankings, a rate measured
   two ways, a sum recomputed from its parts.
3. **Plant the defect the check exists for.** A fixed audit is not a detecting audit.
4. **Read the counter, not the recollection.** Every mis-stated number here was a memory, not a
   measurement.
5. **Ask who a warning is for.** A correct constant attached to the wrong consumer is still a wrong
   sentence — the threshold that would empty a service's results does not apply to a benchmark runner
   that sets it to zero itself.
6. **Distinguish the two kinds of green, and the two kinds of red.** Vacuous is not suspect. A refuted
   mechanism is not a refuted observation.
7. **Write the refutation down next to the surviving claim.** Seven mechanisms of mine were refuted by
   my own probes in this sweep; each is recorded where the claim lives, because the next reader
   otherwise re-derives them.
