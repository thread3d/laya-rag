# Findings: adapting Laya's encoder for retrieval (Bet 3, measured)

This is the pre-registered long shot from `best-in-the-world/README.md` §4 Bet 3 and
`docs/study-laya-finetuning.md`: Laya's shipped encoder is a *decision* encoder, mean-pooled and
not retrieval-trained, so the engine's own dense channel is worth nDCG@10 **0.0141** on SciFact
against BM25's **0.6539**. The question was whether training that encoder on retrieval pairs makes
the Laya-native dense channel real, and whether the shipped default can then beat lexical-only.

**Answer: the mechanism is confirmed — the dense channel rises 17–33× and becomes a net positive
in fusion at a tuned weight. The strict bar is not met: it does not approach `bge-small`'s hybrid,
and the same checkpoint can no longer judge.** Across SciFact/NFCorpus/ArguAna the best per-corpus
adaptation lifts the fused mean nDCG@10 from **0.4725 to 0.5101 (+0.0376)** and the engine's own
SciFact number from **0.6702 to 0.6765**, but a single encoder neither reaches `bge-small` nor keeps
Laya's calibrated relevance head: AUC falls **0.8008 → 0.577–0.633**. The judge collapse is the more
consequential finding, because Laya's differentiator is the calibrated decision, not the embedding.

As a **shipped default** the adapted encoder improves the true lexical-only baseline on all five
BEIR datasets (4-dataset deployable mean **0.3918 → 0.4132**, after Run D) but is a **regression
against the `bge-small` configuration it replaces (0.4689)**, and it loses to BM25 on MultiHop-RAG
ordering and BRIGHT. It is therefore shipped as the default *dense channel* with the `bge-small`
configuration kept and drawn beside it as the higher-scoring alternative — the refresh below
documents both.

Reproduce:

```sh
# 104 min on MPS, 180k gold pairs from three train splits
nohup .venv/bin/python tools/laya_encoder_finetune.py \
    --datasets msmarco,hotpotqa,fever --per-dataset-cap 60000 --max-pairs 180000 \
    --epochs 1 --batch-size 32 --checkpoint-steps 500 \
    --checkpoint-dir eval/laya-encoder-checkpoints/runA \
    --save-model eval/laya-encoder-runA --export-hf eval/laya-encoder-runA-hf \
    --out eval/laya-encoder-runA > eval/laya-encoder-runA.log 2>&1 &

LAYA_RAG_CHECKPOINT=eval/laya-encoder-runA .venv/bin/python tools/beir.py \
    --datasets scifact,nfcorpus,arguana --retrievers bm25,dense,hybrid --dense-weight 0.3 \
    --out eval/beir/adapted-runA
```

## Training

`tools/laya_encoder_finetune.py` — symmetric InfoNCE over in-batch negatives, temperature 0.05,
query/document max length 64/128, batch 32, AdamW, 1 epoch. **180,000 gold pairs** drawn from the
`train` splits already in `eval/beir`: MS MARCO 60k, HotpotQA 60k, FEVER 60k (SciFact/NFCorpus
train splits are deliberately excluded, so the SciFact/NFCorpus rows below are *out-of-domain*
adaptation). The decision head is frozen; only the 306.9M-parameter encoder trains.
5,625 steps, 6,258 s (**104 min**, 1.11 s/step), MPS, loss 4.2977 → 0.0098. The export writes back
the full Laya checkpoint layout and a plain HF encoder directory (mean pooling).

## Result 1 — the dense channel becomes real

nDCG@10, `tools/beir.py`, same corpus/qrels/evaluator for every row:

| dataset | raw Laya `dense` | **adapted `dense`** | ×  | bm25 | bm25-engine | `bge-small` `biencoder` |
| --- | --- | --- | --- | --- | --- | --- |
| SciFact | 0.0141 | **0.4711** | 33× | 0.6539 | 0.6762 | 0.7200 |
| NFCorpus | 0.0146 | **0.2440** | 17× | 0.3042 | 0.3244 | 0.3386 |
| ArguAna | ~0.014 | **0.4683** | ~33× | 0.3935 | 0.4169 | 0.6377 |

The earlier linear ceiling — a ridge projection reached 0.0810 and teacher-documents control
0.1327 on SciFact — is exceeded by 3.5–5.8×. Nonlinear encoder training reaches what no linear map
could. On **ArguAna the adapted dense channel alone beats both BM25 arms**, which is the first time
a Laya-native representation has won a BEIR row here.

## Result 2 — fusion is weight-sensitive, and at 0.3 it is a net win

The engine's `resolved_dense_weight` is 1.0 whenever a dense model is configured, and 1.0 was the
wrong weight here: the dense channel is real but still weaker than BM25, so equal-weight RRF
dilutes the stronger channel. Hybrid nDCG@10 against the dense weight:

| `--dense-weight` | SciFact | NFCorpus |
| --- | --- | --- |
| 0.0 (lexical only) | 0.6762 | 0.3244 |
| **0.3** | **0.6799** | **0.3341** |
| 0.5 | 0.6751 | 0.3314 |
| 0.7 | 0.6596 | 0.3269 |
| 1.0 | 0.6459 | 0.3214 |

At **0.3** the adapted encoder improves both corpora over lexical-only — **SciFact +0.0037**,
**NFCorpus +0.0097** — and the ordering is monotone in the weight, so the gain is the dense
channel's and not noise. On ArguAna the dense channel is stronger than BM25, and there the
*opposite* weight is right: hybrid is **0.4890** at weight 1.0 but only **0.4599** at 0.3, below
dense alone (0.4683). **The optimal weight is corpus-dependent** — ≈0.3 where BM25 dominates, ≈1.0
where the dense channel does — which is the project's recurring routing conclusion, now for the
adapted channel.

**Consequence for the engine:** `DENSE_WEIGHT=auto → 1.0` is the wrong default for an adapted
Laya encoder. A tuned value near 0.3 is what makes the channel pay, and `tools/beir.py`'s
hard-coded 1.0 had hidden that.

## Result 3 — the judged trade-off: one checkpoint cannot both retrieve and judge

The decision head's weights are frozen, but it reads the encoder's hidden states, which moved. The
engine's own relevance question (`Does passage help answer query?`) on 300 SciFact test queries
(1 positive + 2 unjudged per query, `tools/laya_judge_check.py`):

| checkpoint | AUC | accuracy @0.5 | ECE | mean P(positive) | mean P(negative) |
| --- | --- | --- | --- | --- | --- |
| shipped `multilingual` | **0.8008** | 0.7567 | 0.1087 | 0.5537 | 0.1945 |
| adapted (Run A) | 0.5772 | 0.6344 | 0.1386 | 0.2825 | 0.2368 |
| adapted (Run B) | 0.6333 | 0.6433 | 0.1132 | 0.3746 | 0.2723 |

**Relevance discrimination falls from 0.80 to 0.58 AUC.** The encoder adaptation buys retrieval
quality at the cost of the decision head. This falsifies the second selling point of the bet as
stated — *"one checkpoint for encode, judge and abstain"* — unless the head is also retrained or
recalibrated.

**The engine can keep both today, with no code change:** load the adapted encoder as the external
dense channel (`EMBED_MODEL=eval/laya-encoder-runA-hf`) and keep `CHECKPOINT=multilingual` as the
decision model. The retrieval ordering does not need the head at all when thresholds are zero, and
the judge stays at AUC 0.80.

## Run B — in-domain data helps in-domain, and costs out-of-domain

Run B warm-starts from Run A (`--init-from`) and continues for one epoch on **255,000 pairs**
that add the evaluation corpora's own train splits: MS MARCO 100k, HotpotQA 40k, FEVER 40k,
**NFCorpus 60k**, FiQA 14,166, **SciFact 919**; LR 1.5e-5, 7,968 steps, 150 min. Best hybrid
nDCG@10 per corpus (weights are chosen per corpus, as the sweep shows they must be):

| dataset | lexical-only | Run A best | **Run B best (weight)** | `bge-small` hybrid-biencoder |
| --- | --- | --- | --- | --- |
| SciFact | 0.6762 | 0.6799 (0.3) | **0.6823 (0.1)** | 0.7041 |
| NFCorpus | 0.3244 | 0.3341 (0.3) | **0.3559 (0.5)** | 0.3510 |
| ArguAna | 0.4169 (bm25-engine) | **0.4890 (1.0)** | 0.4456 (0.3) | 0.6012 |

Two readings:

1. **The two in-domain corpora improve, and NFCorpus clears `bge-small`.** `0.3559` is above the
   retrieval-trained 33M bi-encoder's own hybrid row (0.3510), which no Laya representation had
   reached. Adding a corpus's train split does what it should.
2. **The out-of-domain corpora get worse.** SciFact's *dense* row falls 0.4711 → 0.4212 and
   ArguAna's 0.4683 → 0.4043: 60k NFCorpus pairs at a comparable count to MS MARCO pulled the
   encoder toward medical-abstract relevance. The best *single* model is therefore still Run A;
   Run B is a per-corpus specialist, and the two together are what a router would choose between.

The likely mechanism is a false-negative artifact of the mix, not just domain shift: NFCorpus
averages ~42 judgements per query, so several of a query's *true* positives land in the same batch
and MNRL penalises them as negatives. Run B's batch loss never falls below ~0.1 and ends at 0.42,
against Run A's 0.0098 on a one-positive-per-query mix. Filtering in-batch false negatives (or
sampling one positive per query) is the untried fix.

## Run C — masking false negatives does not fix the domain shift

Run C is Run B's exact mix and warm start with `--mask-false-negatives` on (253,738 known
positive pairs): 255k pairs, 7,968 steps, 153 min, loss 1.10 → 0.35. Best hybrid per corpus:

| dataset | lexical-only | Run B best | **Run C best (weight)** | ArguAna note |
| --- | --- | --- | --- | --- |
| SciFact | 0.6762 | 0.6823 (0.1) | **0.6834 (0.1)** | — |
| NFCorpus | 0.3244 | 0.3559 (0.5) | **0.3578 (0.5)** | above `bge-small` hybrid 0.3510 |
| ArguAna | 0.4169 | 0.4456 (0.3) | 0.4460 (0.3) | still below Run A's 0.4890 |

The mask buys a little in-domain (SciFact +0.0011, NFCorpus +0.0019 over Run B) but **does not
restore the out-of-domain rows**: ArguAna dense is 0.4052 against Run A's 0.4683, and SciFact
dense 0.4092 against 0.4711. The regression is therefore **domain shift / forgetting from the
in-domain pairs, not the in-batch false-negative artifact** — a clean negative for the mask as the
fix, though it is still the more correct objective and is kept behind the flag. Judge AUC 0.6258
(Run A 0.5772, Run B 0.6333), still far below the shipped 0.8008.

## The best measured configuration

Best per-corpus hybrid nDCG@10, each at its own weight:

| dataset | shipped lexical-only | best adapted | gain | `bge-small` hybrid-biencoder |
| --- | --- | --- | --- | --- |
| SciFact | 0.6762 | **0.6834** (Run C, w=0.1) | +0.0072 | 0.7041 |
| NFCorpus | 0.3244 | **0.3578** (Run C, w=0.5) | +0.0334 | 0.3510 |
| ArguAna | 0.4169 | **0.4890** (Run A, w=1.0) | +0.0721 | 0.6012 |
| **mean** | 0.4725 | **0.5101** | **+0.0376** | — |

Engine-level SciFact: **0.6702 → 0.6765** with Run A at weight 0.3 and the shipped judge kept.

The deployable recipe: **route** between Run A (general/out-of-domain — ArguAna) and Run C
(in-domain — SciFact/NFCorpus), fuse the dense channel at the corpus's tuned weight (≈0.1–0.5,
never the 1.0 auto-default), load the adapted encoder through `EMBED_MODEL`, and keep
`CHECKPOINT=multilingual` as the judge.

## The two adapted encoders do not ensemble

The project's largest measured retrieval win is a three-model **dense ensemble** (+0.0431 BEIR-5
mean), so the obvious next move was to RRF the two adapted Laya encoders
(`--biencoder-model eval/laya-encoder-runA-hf --ensemble-models eval/laya-encoder-runB-hf`,
`rrf_k=60`):

| dataset | Run A `biencoder` | Run B `biencoder` | A+B `dense-ensemble` | lexical-only |
| --- | --- | --- | --- | --- |
| SciFact | 0.4711 | 0.4212 | 0.4629 | 0.6762 |
| NFCorpus | 0.2442 | 0.2845 | **0.2845** | 0.3244 |
| ArguAna | 0.4675 | 0.4043 | 0.4399 | 0.4169 |

The ensemble helps only NFCorpus and hurts SciFact and ArguAna; `hybrid-ensemble` is below
lexical-only on all three. **Two checkpoints of the same architecture trained on overlapping data
are not diverse enough to ensemble** — the bge ensemble works because its members differ in size
(33M/110M/335M) and training, not just in the last 60k pairs. A useful side effect: the
`biencoder` row for `eval/laya-encoder-runA-hf` reproduces the Laya-native `dense` rows almost
exactly (SciFact 0.4711 vs 0.4711, NFCorpus 0.2442 vs 0.2440), so the exported HF directory and
the `EMBED_MODEL` serving path are faithful despite the tokenizer regex warning.

## The adapted encoder is now the shipped dense channel

The engine no longer ships dense-off. `EngineConfig.__post_init__` adopts a trained retrieval
adaptation when one is on disk at `models/laya-retrieval/` (the marker `laya_retrieval.json` is
what identifies it), and `resolved_dense_weight` fuses it at `ADAPTED_DENSE_WEIGHT` — **0.3**
by default, replacing the historical `auto → 1.0` that overdiluted it. A fresh clone has nothing
at that path and behaves exactly as before, so this is a default that is *earned* by running
`make laya-finetune`, not a hard-coded path in a gitignored tree. The decision checkpoint is
unchanged: the judge still comes from `multilingual`, which is why the relevance AUC above is
measured for the served configuration and stays at 0.8008.

Three tests pin it (`tests/test_retrieval_config.py`): the artefact is adopted and fused at 0.3;
an explicit `EMBED_MODEL` wins and keeps the stock 1.0; the weight is configurable. Harnesses
accept a `laya-adapted` alias (`beir.model_aliases`) so a report row reads
`laya-adapted dense@ov200` rather than a path, and the alias only exists when the artefact does.

## Refreshed benchmarks with the adapted encoder as the shipped dense channel

**Run D is the shipped encoder.** It warm-starts from Run A and continues for one epoch on a much
larger mix — **MS MARCO 400k + HotpotQA 30k + FEVER 30k = 460,000 pairs** (Run A used 180k) — at
**batch 64** rather than 32, so each step has ~63 in-batch negatives instead of 31, with the
false-negative mask on and LR lowered to 1e-5 for the continuation. 7,187 steps, 2.94 h on MPS,
loss 0.036 → 0.016. It improves on Run A on **every** BEIR row and on MultiHop and 2Wiki:

| | SciFact dense | NFCorpus dense | ArguAna dense | MultiHop hybrid MAP@10 | 2Wiki hybrid Δ |
| --- | --- | --- | --- | --- | --- |
| Run A | 0.4711 | 0.2442 | 0.4675 | 0.2381 | +0.0130 |
| **Run D** | **0.5314** | **0.2700** | **0.4748** | **0.2490** | **+0.0145** |

It is not uniformly better: FRAMES dense nDCG@10 slips 0.6839 → 0.6737 (recall@1 rises
0.7469 → 0.7543), and BRIGHT is unchanged in character (pony dense 0.0362 → 0.0208, still far
below BM25). The deployable 4-dataset BEIR mean moves **0.4092 → 0.4132**, still under
`bge-small`'s **0.4689**.

With that model packaged at `models/laya-retrieval` (weight 0.3), every encoder-affected
benchmark was re-run and the charts regenerated. The result is **not a uniform improvement**,
and the documentation keeps both configurations side by side rather than hiding the second one.

**BEIR-5** (`eval/beir/five-adapted.json`, bi-encoder and its fusion), with **Run D** as the
shipped encoder:

| dataset | lexical-only | adapted `biencoder` | adapted hybrid (w=0.3) | `bge-small` hybrid (previous) |
| --- | --- | --- | --- | --- |
| SciFact | 0.6762 | 0.5314 | **0.6880** | 0.7413 |
| NFCorpus | 0.3244 | 0.2700 | **0.3389** | 0.3673 |
| ArguAna | 0.4169 | 0.4748 | **0.4618** | 0.5679 |
| FiQA | 0.2386 | 0.2167 | **0.2801** | 0.3817 |
| SciDocs | 0.1497 | 0.1291 | **0.1640** | 0.1992 |

The adapted configuration beats lexical-only on **all five** — the 4-dataset deployable mean
rises **0.3918 → 0.4132** — but it is **below the `bge-small` configuration it would replace**
(**0.4689**). The charts therefore draw the bge row beside it, labelled as the higher-scoring
alternative.

**MultiHop-RAG** (official protocol): the adapted hybrid improves with Run D but is still
**below every existing arm** — MAP@10 **0.2490** at overlap 200 (0.2616 at overlap 0) against
BM25's 0.2534, `bge-small` hybrid's 0.2510 and `bge-large` hybrid's 0.2545; the adapted dense row
is worse still (0.1814). Hits@10 does improve (0.8164 against BM25's 0.7823), so the loss is in
ordering, not in finding one piece of evidence.

**2WikiMultiHopQA**: the adapted `biencoder` is **significantly worse** than `bm25-engine`
(MAP@10 **−0.0501**), while the adapted `hybrid-biencoder` is **significantly better**
(**+0.0145**, 0.5183 against 0.5037). One corpus, opposite signs — the routing pattern again.

**FRAMES**: adapted dense nDCG@10 **0.6737** beats lexical 0.6586, below `bge-small` dense 0.7505
and the 3-model ensemble 0.8013.

**BRIGHT**: the adapted dense channel is **far worse than BM25** on the subsets measured (pony:
dense 0.0208 against BM25 0.1221), consistent with the topical-retrieval weakness the whole
programme keeps hitting.

**MuSiQue**: the adapted encoder's `fused` first stage reads MAP@10 **0.3545** and its second-stage
arms 0.3772–0.4136, **byte-identical to the Run A run** — the RRF top-10 there is insensitive to
which encoder feeds the dense channel, because BM25 dominates the fusion. Reported as measured,
with that explanation rather than as an encoder effect.

Two operational findings from the same runs. The 307M encoder is heavy enough that FRAMES
**OOM'd MPS** at batch 32 with the model's advertised sequence budget (an 8 GB attention buffer);
`tools/frames_eval.py` now takes `--batch-size` / `--max-length`, and the run needs 8 / 256. And
the `bge-small` comparison rows had to be kept in the charts deliberately: a full replacement
would have *lowered* every headline deployable number.

## Run E — the `english` checkpoint, measured and not shipped

The clearest untried lever was Laya's other checkpoint: `english` is ModernBERT-large (395M) against
the multilingual mmBERT-base (307M), and it can be trained by the same recipe. It cannot warm-start
from Run D — different architecture and hidden width (1024 against 768) — so it trains from the base
on the same 460k-pair mix, batch 64, one epoch, 7,187 steps.

| | SciFact dense | NFCorpus dense | ArguAna dense | fused mean | s/step |
| --- | --- | --- | --- | --- | --- |
| Run D (multilingual, shipped) | 0.5314 | 0.2700 | **0.4748** | 0.4962 | **1.47** |
| **Run E (english)** | **0.5523** | **0.2799** | 0.4546 | **0.4981** | 4.61 |

It wins on SciFact (+0.0209) and NFCorpus (+0.0099) and loses on ArguAna (−0.0202); the fused
three-dataset mean moves **+0.0019**, inside the 0.002 margin `eval/pick_model.py` will swap a shipped
encoder on, so **Run D stays the default**. The other half of the result is the cost: ModernBERT-large
runs at **4.61 s/step against 1.47** on MPS, so the run took 9.2 h rather than the 3.7 h a
parameter-count extrapolation predicts. Bucketed padding (`--pad-bucket`, the new default) keeps the
per-shape MPS graph cache bounded without paying for absent tokens, but recovered only ~3% here
because these batches already sit near 128 tokens.

## The BEIR-18 pass — seven cells up, and the caption corrected

The figure's engine row is a **best-per-dataset over everything measured here**, so improving it means
improving individual cells. Two moves, both cheap relative to the multi-million-document members:

**Title-weighted BM25** (`--title-boost 2`, the setting the canonical five-small report used — the
tool keys the `bm25-title` row off the boost, not off the retriever name):

| dataset | before | after |
| --- | --- | --- |
| FEVER | 0.6323 | **0.6749** |
| NQ | 0.3118 | **0.3264** |
| DBPedia-entity | 0.3053 | **0.3166** |
| HotpotQA | 0.6352 | **0.6563** |
| CQADupStack | 0.2988 | **0.3046** |
| Climate-FEVER | 0.1610 | **0.1657** |
| ROBUST04 / Signal-1M / Quora / Touché-2020 | — | no-op or worse |

FEVER and HotpotQA are the two the first pass culled on cost, and they are the two largest gains:
one index build and one scoring pass each (~3 h per row on this host) against ~37 h for a dense
pass over 5.4M documents. In both, `bm25-engine` reproduces its recorded value exactly (0.6323 and
0.6352), which is what makes the new rows comparable rather than merely higher.

CQADupStack needed a tooling fix first: `tools/beir.py` refuses it (twelve sub-collections behind one
BEIR name) and `tools/cqadupstack.py`, the adapter that exists for exactly that reason, had no
title-boost flag, so the member could not be given the row the others had. It now takes
`--title-boost` and emits a `bm25-title` row through the same `beir.bm25_rankings` path, with the
`bm25` and `bm25-engine` rows reproducing exactly (0.2988 and 0.2943).

**The adapted dense channel, fused at weight 0.3**, on the members whose corpora this host can embed:

| dataset | before | adapted dense | **fused** |
| --- | --- | --- | --- |
| TREC-COVID | 0.6188 | 0.5040 | **0.6871** |
| ROBUST04 | 0.4077 | 0.3688 | **0.4405** |
| TREC-NEWS | 0.3351 | 0.3353 | **0.3745** |

Together the engine's **BEIR-18 best-per-dataset macro moves 0.4665 → 0.4799 (+0.0134)**, with nine of
the eighteen cells carrying a gain. The unaffected cells are unchanged. The matrix figure's caption
was also wrong and is corrected: it said dashes marked datasets this engine had not run, but the grid
is six systems across eighteen datasets and **none of the 108 cells is missing** — all five published
systems report all 18, and so does the engine — so the figure draws no dashes and no longer claims to.

**What is still not run, and why.** `bioasq` (14.9M documents) is the only one of the eighteen with no
title row: ~5 h of index build for a cell already at 0.5193 against the best published system's 0.504.
`msmarco` has no title row either, but it is the collection the BEIR encoders are trained on rather
than one of the 18, so it cannot move this row at all. Those are now the only two gaps; FEVER,
HotpotQA and CQADupStack were closed by the runs above. Dense rows on any multi-million-document member
remain out of reach here — ~40 documents/s means ~37 h per corpus — so those cells carry a lexical
best no matter what.

## What the pre-registered bar says

The Bet 3 falsification was: engine-level BEIR nDCG@10 on SciFact / NFCorpus / ArguAna with
`EMBED_MODEL=""`, adapted, beats the lexical-only default **and approaches `bge-small`'s hybrid
(0.732 / 0.364)**.

- **Beats lexical-only: yes, at weight 0.3** (+0.0037 SciFact, +0.0097 NFCorpus; +0.095 ArguAna).
- **Approaches `bge-small`: no.** 0.68 vs 0.732 on SciFact, 0.334 vs ~0.351 on NFCorpus.
- **Plateaus near the 0.1327 projection ceiling: no** — it is 3.5–5.8× past it, so the
  representation was not the hard limit the projection result suggested.

**Engine-level, not just the replica.** `tools/engine_beir.py` on SciFact (300 queries, thresholds
0, `FUSION_K=5`): lexical-only (`EMBED_MODEL=""`, `DENSE_WEIGHT=0`) reads **0.6702**, and the adapted
encoder loaded as the external dense channel (`EMBED_MODEL=eval/laya-encoder-runA-hf`,
`DENSE_WEIGHT=0.3`, `CHECKPOINT=multilingual` so the judge is untouched) reads **0.6765
(+0.0063)**. The engine moves the same way as the harness. Two caveats: the `EMBED_MODEL` path loads
the exported tokenizer through `AutoTokenizer`, which emits a known regex warning, so this is a
lower bound on the Laya-native tokenizer the harness used; and the native adapted checkpoint was
not run here because at threshold 0 the head cannot affect retrieval ordering — its cost is the
separate judge measurement above.

So this is a **partial positive**: the architectural claim ("the shipped default does not use the
system's own model") is now measurably false — it can, and it helps — but the encoder is not yet
competitive with a retrieval-trained 33M model, and it pays for the gain with the judge.

## What this does not show

- **The answerability gate was re-fitted on the shipped encoder's pool, and it did not move.** On
  the 150+150 protocol `gate-features.json` uses, the 13-feature gate reads ROC-AUC **0.8098**
  [0.7577, 0.8537], ECE 0.0782 and risk@80 0.4099, against the shipped pool's **0.8106** / 0.0695 /
  0.4097 — every interval overlaps, so swapping the dense channel neither breaks nor improves the
  gate. The full 601-query re-fit was attempted first and abandoned: the null half takes ~13 s per
  query against ~1.7 s for the answerable half, so 301 nulls alone are ~20 min of engine calls and
  the run was stopped after the positive half had not finished. The served configuration keeps
  `CHECKPOINT=multilingual` as the judge regardless, so the shipped reliability row is the one to
  quote; the re-fit lives in `eval/calibration-runD.*`, `eval/answerability-features-runD.json` and
  `docs/charts/reliability-runD.png`.
- **One seed, one recipe family.** Five runs (A external; B and C adding in-domain pairs; D a larger
  external mix at batch 64; E the `english` checkpoint, not shipped), one epoch each, and D uses
  ~460k of ~1M available gold pairs, with no explicit hard negatives. A GPU-scale run with real hard
  negatives remains the untried version of the recipe.
- **The judge collapse is measured, not fixed.** A multi-task run (retrieval + relevance decision
  on the shared encoder) is the untried route to making `EMBED_MODEL=""` competitive *and* safe.
- **The per-corpus routing is measured, not implemented.** The engine has no router that picks
  Run A vs Run C or the fusion weight; today it is a deployment choice made by hand.
- **The low Run A training loss (0.0098) is not a success signal.** In-batch negatives are easy once
  the encoder picks up lexical overlap; it is reported only to show the run converged.

## Method notes

- **Embedding caches are now keyed by content, not only by name.** The cache filename comes from
  a model's directory *basename*, so repackaging a retrained encoder at the same
  `models/laya-retrieval` path kept the name and silently reused the previous model's vectors.
  The first Run D refresh produced MultiHop and BRIGHT figures that were actually Run A's (a
  second sweep then removed the *new* caches rather than the old ones, because the `find`
  predicate was inverted). `tools/beir.py` now writes a `model_fingerprint` — a hash over the
  directory's `(relative path, size, mtime)` — into the cache's `.meta.json` and refuses to reuse
  a cache whose fingerprint differs; `LayaModel.weights_path()` exposes the directory to
  fingerprint. Caches written before this are adopted once and stamped, so the existing corpus
  caches were not thrown away. Four tests pin it, including the legacy-adoption path. The lesson
  that survives the fix: a metric that is byte-identical across two different encoders is a cache
  hit until proven otherwise — which is how the MuSiQue `fused` row was caught and explained.
- **Fixed-width padding is required on MPS, not an optimisation.** The trainer's first large run
  died at ~step 150 with `MPS backend out of memory (allocated 19 GB, other allocations 35 GB)`.
  The 35 GB was torch's per-shape graph cache: dynamic padding gives every batch a different
  width, so the cache grows without bound. The probe never showed it because the probe already
  pads to a fixed length. `tools/laya_encoder_finetune.py` now pads to a constant width by
  default (`--dynamic-padding` opts out) and releases the allocator every 50 steps
  (`--empty-cache-steps`); the same run then held flat and got *faster* (1.50 s/step against
  1.9–2.3 s/step while thrashing).
- **`LambdaLR` needs a factor, not a learning rate.** The first version passed an absolute LR, so
  the schedule stepped to `lr²` ≈ 4e-12 and the loss barely moved. Caught in the smoke run.
- **`LayaEmbedder.embed_queries` was missing `fixed_length`.** `beir.dense_rankings` always passes
  it, so the Laya `dense` row was broken in the current tree; fixed for interface parity with
  `BiEncoder`.
- **An `--out` stem with a decimal collides.** `Path("...-w0.3").with_suffix(".json")` rewrites
  `-w0.json`, so the first weight sweep overwrote its own reports; use a decimal-free tag.
- **Streaming positives.** MS MARCO's corpus is 8.8M documents; the trainer reads `corpus.jsonl`
  once and keeps only the judged ones, so a training load does not materialise the collection.
