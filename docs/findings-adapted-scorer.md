# Findings: adapting a cross-encoder on the corpus's own conventions made it worse

ROADMAP Phase 2.2, and the third Phase 2 negative. Phase 2.1 falsified a topical late-interaction
head and 2.3 a relational NLI head; both borrowed a model trained on something else, so the
remaining explanation was that the head needed training on **this task's** conventions. This is
that attempt, and it fails too.

Reproduce:

```sh
.venv/bin/python tools/fetch_musique.py --split train
make order-train                 # ~3 min on MPS: 20k pairs from 4000 MuSiQue train queries
make order-eval-zero-shot        # the base architecture, unadapted
make order-eval-adapted          # the adapted model, same pool
make order-compare               # paired, adapted versus zero-shot
```

## Training

Mined from the MuSiQue **train** split, which ships gold supporting paragraphs, so no retrieval and
no LLM is involved:

* positive — a paragraph the record marks `is_supporting`;
* negative — a sampled non-supporting paragraph from the *same record's* candidate list, i.e. a
  same-topic distractor rather than a random corpus paragraph.

20,000 pairs from 4,000 queries (19,000 train / 1,000 held out), base
`cross-encoder/ms-marco-MiniLM-L-6-v2` (22M), 1 epoch, 594 steps, 157 s, held-out loss **0.1813**.
The final model is a `sentence_transformers` `CrossEncoder` at `eval/order-ce-musique`.

## Result — MuSiQue validation, 2,417 queries, pool 20

| arm | Hits@10 | Hits@4 | MAP@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| fused first stage | 0.9520 | 0.8879 | 0.3852 | 0.7824 |
| `ce-cached` (`bge-reranker-base`, pool 20) | 0.9586 | 0.9053 | **0.4121** | **0.8325** |
| **zero-shot** MiniLM | **0.9648** | **0.9110** | 0.4035 | 0.8146 |
| **adapted** MiniLM | 0.9603 | 0.8875 | 0.3701 | 0.7511 |

Paired over the queries, adapted against its own zero-shot base:

| metric | zero-shot | adapted | delta | 95% interval | p |
| --- | --- | --- | --- | --- | --- |
| Hits@10 | 0.9648 | 0.9603 | −0.0046 | [−0.0108, +0.0017] | 0.167 (ns) |
| Hits@4 | 0.9110 | 0.8875 | **−0.0236** | [−0.0360, −0.0112] | 0.0002 |
| MAP@10 | 0.4035 | 0.3701 | **−0.0334** | [−0.0395, −0.0273] | 0.0000 |
| MRR@10 | 0.8146 | 0.7511 | **−0.0635** | [−0.0766, −0.0504] | 0.0000 |

And against the plain first stage, the adapted model is **below** it on MAP@10
(**−0.0151 [−0.0222, −0.0082]**) and MRR@10 (**−0.0313 [−0.0460, −0.0165]**), while gaining on
Hits@10 alone.

**The answer to Phase 2 is negative.** Three ordering heads — topical late interaction, relational
NLI, and a cross-encoder adapted on the corpus's own train split — all lose to the existing
first-stage-plus-coverage rule, and two of the three lose to the simplest baseline available.

## Why it may have failed

Recorded as hypotheses, not findings:

- **A small, narrow signal against a large general one.** The base is MS MARCO-trained on millions
  of pairs; 19,000 same-corpus pairs is a much smaller and narrower supervision, and it appears to
  wash out general relevance rather than sharpen task relevance.
- **The negatives teach the wrong contrast.** Same-record distractors are *topically* similar, so
  the easiest way to separate them is topical discrimination — the very thing the multi-hop bridge
  needs the model to look past.
- **This repeats a known local negative.** The project already measured full-split fine-tuning on
  NFCorpus with mined hard negatives scoring *below* zero-shot (−0.0100). Two independent
  adaptations now lose to their zero-shot bases, which is a pattern rather than an accident.

## What this does not show

- **Not an exhaustive adaptation sweep.** One base architecture (MiniLM-L6), one recipe (4,000
  queries, same-record distractors, 1 epoch, BCE). The roadmap imagined MuSiQue + 2Wiki + HotpotQA
  train splits and a GPU. Cross-corpus training, hard-negative mining from the retrieval pool, and
  a listwise or pairwise objective are all untried.
- **Not a fair `ce-cached` comparison.** `ce-cached` here reranks the top-20, so it scores 0.4121
  against the harness's depth-100 `ce` row of 0.4032; the pool differs. The adapted-versus-zero-shot
  rows — the point of the experiment — share one pool and one architecture.
- **Not a claim that the first stage cannot be improved.** `fused` reproduces the harness's 0.3852
  exactly, so the evaluator is faithful; what is falsified is that these three ordering heads
  improve on it.

## Method notes

- **A step checkpoint belongs to the run that wrote it.** The first full training run printed
  `resumed from step-000005.pt` — a leftover from a 40-query smoke run, which silently contaminated
  the adaptation. Step checkpoints now carry a `jobkit` fingerprint of the run's settings and refuse
  to resume under a different one; `--fresh` discards them. A test pins the selection.
- **A limited evaluation must score against the subset it ranked.** `evalkit.evaluate_rankings`
  averages over every query in the qrels, so `--limit` without restricting the qrels dilutes every
  number toward zero — the same trap Phase 2.3 hit. `order_eval.py` scores only the queries it
  reranked.
