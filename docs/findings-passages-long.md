# Passage-level BM25 on the two genuinely long corpora

`SCORE-LIFT.md` §P7. Passage retrieval was written off on SciFact, NFCorpus and TREC-COVID, and
`RANKINGS.md` recorded the position precisely: for long documents it is "**untested, not
disproven**". The two corpora that statement was written for — Robust04 (466 words/doc) and
TREC-NEWS (635) — were reached from the recovered mirror only afterwards. SPLADE-v3's authors
independently report long-document passage decomposition as a real, fixable failure mode, so this
was the standing caveat worth converting into a result.

The earlier rejections do not test the hypothesis: SciFact and NFCorpus are ~230-word abstracts and
TREC-COVID's are *shorter* than NFCorpus's, so none of them is a long document.

## What was run

```sh
.venv/bin/python tools/beir.py --datasets robust04,trec-news \
    --retrievers bm25-engine --passages 600 --out eval/beir/passages-long
```

Same channel, same tokeniser, same `k1`/`b`/stopwords/stemming as the `bm25-engine` row; the only
difference is that each document is cut into 600-character pieces and the document's score is the
**max** over its pieces (`passage_rankings`, the same function that produced the earlier rejections).

## Result: they lose, and further

| corpus | documents | `bm25-engine` | `bm25-passages@600` | delta |
| --- | ---: | ---: | ---: | ---: |
| Robust04 | 528,155 | **0.4077** | 0.3522 | **−0.0555** |
| TREC-NEWS | 594,977 | **0.3351** | 0.2456 | **−0.0895** |

Every supporting metric agrees in sign: Robust04 MRR@10 0.6886 → 0.5908, MAP@100 0.1986 → 0.1741,
Recall@100 0.3821 → 0.3614; TREC-NEWS 0.7599 → 0.6323, 0.2214 → 0.1594, 0.4461 → 0.3737.

**§P7's expected sign was wrong, and the size of the loss grows with document length** (+0.01 to
+0.04 was predicted; −0.056 at 466 words and −0.090 at 635). That is the opposite of the hypothesis
that decomposition helps most where documents are longest.

## Reading

**The technique is closed for this engine at every document length it has been tested** — ~230-word
abstracts (SciFact, NFCorpus, TREC-COVID), 466 words, and 635 words. That is a cleaner statement
than the caveat it replaces, and it is a *result* rather than a gap: three corpora at three lengths
agree, with the loss scaling in the direction that refutes the mechanism.

What the mechanism would have needed is not what was tested. A fixed 600-character cut ignores the
document: it splits a 200-word news item into one piece and a 2,000-word article into four, so
max-pooling over pieces is a different retriever per document rather than a uniform one. The second
half of §P7's falsifier is therefore the experiment that follows, and it is now **implemented**:

- `--passages p90` — each document's budget is the **p90 of its own sentence lengths**, clamped to
  `[max(200, ceiling/4), ceiling]`. A document whose sentences are 90% short gets short passages,
  which is the point of sizing from the document rather than from a constant.
- `--passages-aggregate max|sum` — how a document's pieces become one score. `max` keeps the best
  piece, which is the historical behaviour and the one every refutation used; `sum` adds them
  (CombSUM over passages) and therefore favours longer documents, which have more pieces. That bias
  is what the knob exists to measure.

Nine tests pin both mechanisms, including that `max` still orders by each document's best piece and
that an unknown aggregation is rejected loudly.

**The measurement is pending, and the shape of the wait is itself a finding.** The `--passages p90`
run **completed Robust04** and is still working through TREC-NEWS; it writes its report when the last
dataset finishes. An earlier note here said Robust04's rows were lost by stopping the run — that was
wrong in an instructive way: the shell wrapper was killed and the Python process survived, and *had*
it been killed, `tools/beir.py`'s single write-at-the-end **would** have discarded every finished
dataset. The passage rows have no per-dataset checkpoint, where the engine row
(`--lexical-checkpoint`) and the dense pass (`--embed-chunk`) both do. So two things are true at once: the number is
coming, and the reporting path was fragile enough that a sweep of this size could lose finished
work. **That is fixed**: `tools/beir.py` now writes the report after **every** dataset, printing
which dataset landed and how many are in, so a kill costs at most the dataset in flight. The
markdown and console table are still rendered once at the end — the JSON is the artefact that has to
survive. Two tests pin it, one of them by making a later dataset fail and asserting the earlier
one's rows are on disk. Until the number lands: **"fixed-size passage decomposition is refuted;
adaptive decomposition is implemented, tested and pending"**.

One cost note that *is* supported by the run's progress: the adaptive budget raises the piece count
(a 300-character floor against a 600-character cut is roughly twice the passages) on top of the
full-document index every run builds regardless, so the sweep is slower — but this round did not
measure by how much, and does not claim to.

## What this does not show

- **Lexical only.** The dense and fused channels are not tested with passages here. The dense
  channel's chunking is a separate mechanism with its own measurements
  (`docs/findings-chunk-merge.md`), and a fused passage row is not claimed.
- **One passage size per corpus.** 600 characters was chosen to match the earlier rejections; other
  fixed sizes are unrun, and §P7's own reading is that the *fixed* form is what fails.
- **No interval.** 249 and 57 queries respectively, point estimates only. The margins (−0.056,
  −0.090) are far larger than the MDE this repository quotes for its multi-hop protocol, but that
  MDE is for a different query count and nothing here is bracketed.

## Verification

No new code: this uses `tools/beir.py --passages`, which already had the `bm25-passages@N` path and
its own tests. The reports are `eval/beir/passages-long.json` / `.md`, and the two new rows changed
the §1.1 "rows measured" column for Robust04 and TREC-NEWS — which `make consistency-check` caught
verbatim, as designed, before the table was regenerated.
