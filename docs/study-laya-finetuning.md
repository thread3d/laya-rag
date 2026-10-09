# Feasibility study: fine-tuning the Laya engine to raise this project's retrieval scores

Status: **study only — no training has been run.** This document answers *where* fine-tuning can
move a retrieval number in this repository, *what* has already been measured (so it is not
retried), what is feasible on this host, and the ordered plan with a pre-registered falsification
bar for each step. The decision of which step to run first is the reader's.

Everything here is either a citation to a measurement already in the repository or a probe run
for this study; the probes are labelled and reproducible below.

---

## 1. The one-line finding

**The headline BEIR nDCG@10 does not use Laya's decision head at all.** It comes from
`tools/beir.py`, which fuses BM25 with a *retrieval-trained* bi-encoder (`bge-small` by default)
by RRF; Laya's `noul` head only filters and gates inside the engine. So "fine-tuning the Laya
engine" can raise a retrieval score through exactly two surfaces:

1. **Laya's encoder**, if it becomes the dense channel (`dense` / `hybrid` rows, and the engine's
   `EMBED_MODEL=""` path). Today that channel is worth nDCG@10 **0.0141** on SciFact against
   BM25's **0.6539** (`eval/beir/report-final.md`, `eval/beir/projection-scifact.json`), which is
   why it ships with `DENSE_WEIGHT=auto → 0.0`.
2. **Laya's decision head**, which only enters the *engine-level* and multi-hop paths, where it is
   a filter/gate rather than an orderer — and where four separate stronger-scorer adaptations have
   already lost (section 3).

The highest-expected-value fine-tuning is therefore **retrieval adaptation of Laya's encoder**,
not the decision head. The project already pre-registered this as *Bet 3*; this study adds the
feasibility proof, the data inventory, and the costs.

---

## 2. Trainable surface and the score map

| measured number | produced by | trainable Laya parameters in the path |
| --- | --- | --- |
| BEIR `nDCG@10` (headline, `report-final.md`) | `tools/beir.py`: BM25 + bi-encoder + RRF | **encoder** (the `dense` row); head not used |
| Engine BEIR (`engine-beir/*.json`) | `RagEngine` with `EMBED_MODEL` set or unset | **encoder** (dense channel) + **head** (filter) |
| MultiHop-RAG / MuSiQue / 2Wiki `MAP@10` | `tools/multihop_*.py`, `engine_multihop.py` | **encoder** (hybrid), **head** (filter); order comes from RRF/CE/coverage |
| Answerability AUC / ECE / risk@80 | `tools/multihop_gate.py` | **head** (the `answerability` `noul` question) |
| FRAMES / BRIGHT | `tools/frames_eval.py`, `tools/bright.py` | **encoder** (dense), head (filter) |

Components and sizes (measured from the vendored checkpoints):

| checkpoint | encoder | encoder params | total params | hidden |
| --- | --- | --- | --- | --- |
| `laya` (english, `CHECKPOINT=english`) | `answerdotai/ModernBERT-large` | 394.8M | 421.3M | 1024 |
| `multilingual` (**engine default**) | `jhu-clsp/mmBERT-base` | 306.9M | 321.9M | 768 |
| `typed-decisions` | ModernBERT-large | 394.8M | 421.3M | 1024 |

The encoder is a plain HuggingFace `AutoModel` inside `DecisionModel.encoder`
(`vendor/laya/laya/common.py:693`), trained by a 2-layer decision head. `embed_fn_from_agent`
mean-pools `encoder(...).last_hidden_state` with the attention mask
(`vendor/laya/laya/shortlist.py:208`). Both the encoder and the head are ordinary `nn.Module`s
with full gradients, and `TrainConfig.freeze_encoder` already exists
(`vendor/laya/laya/train.py:107`). **Nothing in the vendored package prevents encoder training.**

---

## 3. What has already been measured (do not retry these)

| attempt | measured result | verdict |
| --- | --- | --- |
| `bge-small` gold-pair fine-tune, full NFCorpus split + 5,180 naive mined hard negatives | nDCG@10 **0.3385 → 0.3285 (−0.0100)** | negative; sparse NFCorpus qrels make "hard" negatives near-relevant |
| `bge-small` 20k pairs, in-batch negatives | 0.3385 → 0.3477 (+0.0085) | positive but small |
| Ensemble distillation → `bge-small`, teacher-mined positives (7,770 pairs) | 0.3385 → **0.3505** declared CLS / **0.3606** mean pooling; teacher ensemble 0.3798 | **works**; recovers ~29% of the ensemble's gain |
| Cross-encoder adapted on MuSiQue's own train split | MAP@10 **0.4035 → 0.3701 (−0.0334)** | negative |
| Laya's head as a reranker | SciFact nDCG@10 **0.654 → 0.470** | negative; it is a filter, not an orderer |
| `bge-reranker-base` over fused top-100 | −0.027 … **−0.318** (ArguAna 0.568 → 0.250) on all five BEIR sets | negative |
| ColBERT MaxSim ordering head | MAP@10 −0.0430; −0.0257 with recall fixed | negative |
| NLI stance on ArguAna | 0.3767 vs dense 0.6467 (−0.2700) | negative |
| Laya projection head (closed-form ridge on **frozen** embeddings) | SciFact **0.0141 → 0.0810 (5.7×)**; teacher-docs-only control **0.1327** | positive; proves the representation is *linearly* reachable |
| 12-feature answerability gate (trained here) | AUC **0.8244** / ECE **0.0892** vs the shipped scalar's 0.5087 / 0.3085 | the one head-side training that pays off, for the gate |

**The pattern to respect:** every "trust a stronger *pointwise scorer* more" adaptation has lost.
Every *representation-side* adaptation (projection, distillation) has gained. That asymmetry is the
argument for adapting the encoder.

---

## 4. Is training feasible on this host? (measured, not assumed)

This host: Intel Xeon W-3275, 56 cores, 256 GB RAM, **no CUDA**; torch picks **MPS** (AMD Radeon
Pro). The project runs one heavy job at a time under `eval/.heavy-job.lock`, and the host has
crashed under concurrent corpus jobs, so runs must checkpoint.

**Probe (this study, `multilingual` encoder, MPS):**

| measurement | result |
| --- | --- |
| `laya.load` (multilingual) | 8.3 s |
| encoder parameters | 306.9 M |
| training step, batch 8 (query 192 + doc 256, two forwards + backward + AdamW) | 15.6 s first step (MPS compile), then **0.62 s/step** |
| MPS allocated during training | **3.75 GB** |
| checkpoint save (`laya.train.save_checkpoint`, fp16 state dict) | 1.4 s |
| save → `laya.load` → `predict` round-trip | **works**; returns a valid `noul` probability |

Projected cost at 0.62 s/step, effective batch 8 (grad-accumulation does not change FLOPs):

| training set | pairs | steps/epoch | hours/epoch (unoptimised) |
| --- | --- | --- | --- |
| NFCorpus | 110,575 | 13,822 | ≈ 2.4 h |
| FiQA | 14,167 | 1,771 | ≈ 0.3 h |
| MS MARCO | 532,752 | 66,594 | ≈ 11.5 h |
| HotpotQA | 170,001 | 21,250 | ≈ 3.7 h |
| FEVER | 140,086 | 17,511 | ≈ 3.0 h |

These are *upper* estimates from one probe batch shape; a throughput sweep (doc truncation,
batch size, `torch.compile`) is the first thing to run. A 100–200k-pair subset at 256 tokens is a
2–5 h run, and checkpointing every few hundred steps makes it crash-safe. **Training is feasible;
the constraint is wall-clock and the one-job rule, not the model.**

---

## 5. Data inventory — corrects a widely repeated claim

The project's own notes say "only NFCorpus ships a local BEIR train split". **That is wrong for the
current `eval/beir` tree.** Six datasets carry real train qrels, and the queries in
`queries.jsonl` cover them:

| dataset | train judgments | distinct train queries | test queries | corpus docs | usable for |
| --- | --- | --- | --- | --- | --- |
| MS MARCO | 532,752 | 502,939 | 43 | 8,841,823 | encoder adaptation (largest, canonical) |
| HotpotQA | 170,001 | 85,000 | 7,405 | 5,233,329 | encoder adaptation, multi-hop-ish |
| FEVER | 140,086 | 109,810 | 6,666 | 5,416,568 | encoder adaptation (claim→evidence) |
| NFCorpus | 110,576 | 2,590 | 323 | 3,633 | adaptation + teacher distillation (caches exist) |
| FiQA | 14,167 | 5,500 | 648 | 57,638 | adaptation + distillation (caches exist) |
| SciFact | 920 | 809 | 300 | 5,183 | tiny; dev/validation only |

Train and test query ids are **disjoint in all six** (checked directly: overlap 0 in every one), so
training on these and evaluating on the test splits is the standard BEIR protocol and is **not**
leakage. Teacher **document** embedding caches already
exist for NFCorpus/FiQA/ArguAna/SciDocs (`eval/beir/cache/*.bge-*.npy`), so ensemble distillation
on those corpora needs no corpus re-embed — only a query pass per teacher. For MS
MARCO/HotpotQA/FEVER the gold train pairs are enough and no teacher is needed.

---

## 6. Candidate interventions

Each entry states mechanism, evidence, data, code path, expected effect, cost, and the bar that
would falsify it. Effect sizes are estimates unless marked *measured*.

### I1 — Retrieval adaptation of Laya's encoder on the in-repo train splits (recommended primary)

**What.** Train `DecisionModel.encoder` with a contrastive objective (InfoNCE / MNRL) on
`(query, positive document)` pairs drawn from the six train splits, keeping the decision head
frozen and copying it back into a servable Laya checkpoint. Both towers share the one encoder, so
the exported artefact is a normal Laya checkpoint.

**Why.** The encoder is a strong general model (ModernBERT, 8192-token window) whose document
embeddings are *linearly* reachable to 0.0810; the missing piece is that it was never trained for
retrieval. Distillation into the 33M `bge-small` already gained +0.0120, and the student here is
~10× larger. The dense channel is currently worth ~0.014, so the headroom above it is large.

**Data.** MS MARCO (533k) + HotpotQA (170k) + FEVER (140k) + NFCorpus (110k) + FiQA (14k). Start
with a 100–200k stratified subset; scale if the learning curve is still rising.

**Code path.** Existing `tools/beir.py` (`load_corpus`, `load_queries`, `load_qrels(split="train")`)
for data; the "dense" retriever already routes through `LayaEmbedder` →
`LayaModel.embed` → the checkpoint encoder (`tools/beir.py:292`), so evaluating the adapted
encoder is `tools/beir.py --retrievers bm25,dense,hybrid`.

**Expected effect.** Dense SciFact 0.0141 → **0.25–0.50** (estimate); NFCorpus 0.0146 → **0.15–0.30**.
The harder question is whether it beats BM25 (0.6539 / 0.3042) — **that is genuinely uncertain**,
and it is the pre-registered Bet 3 bar.

**Cost.** 2–5 h for a 150k-pair epoch on MPS; ~1 h to wire the trainer and a checkpoint format.

**Falsification (pre-registered).** Engine-level BEIR nDCG@10 on SciFact / NFCorpus / ArguAna with
`EMBED_MODEL=""`: does the adapted encoder beat the lexical-only default (BM25 0.6539 / 0.3042) and
approach the `bge-small` retrieval-trained rows — SciFact hybrid **0.7264–0.732**, NFCorpus dense
**0.3386**, ArguAna dense **0.6467**? A plateau near the **0.1327** projection ceiling means the
representation itself is the limit, not the query side — and that converts the central
architectural claim from "unproven" to "measured".

### I2 — Ensemble distillation into Laya's encoder (highest expected gain where caches exist)

**What.** Run the existing distillation recipe
(`tools/finetune.py --teacher-models bge-small,bge-base,bge-large --teacher-positives 3`) with
Laya's encoder as the **student** instead of `bge-small`. Positive documents are the teachers'
RRF-fused top-3; the student learns the 478M ensemble's notion of relevance.

**Why.** *Measured*: the same recipe gained +0.0120/+0.0221 on NFCorpus and is the only fine-tuning
here that improved a retrieval score. Teacher caches already exist for NFCorpus/FiQA/ArguAna/SciDocs,
so the data is nearly free. The 307M student has far more capacity than the 33M one that already
worked.

**Data.** Teacher rankings over the train queries of NFCorpus (and FiQA once caches are extended);
positives are teacher-fused, so no qrels are needed beyond identifying train queries.

**Expected effect.** NFCorpus **0.35–0.38** (the teacher ensemble is 0.3798) — i.e. an owned,
Laya-native encoder at or above `bge-small`'s 0.3385. This is the most likely *positive* result in
the study.

**Cost.** ~1–2 h compute once the Laya student trainer exists (I1 builds it); teacher mining is
minutes from cache.

**Falsification.** The distilled student fails to beat the zero-shot student's 0.3385 NFCorpus
nDCG@10 with a paired bootstrap interval excluding zero. A student that merely reproduces the
teacher's *documents* but not its *queries* is the same plateau as I1 and is reported as such.

### I3 — Bet 3 literal: query side against a frozen document side, pool-mined negatives

**What.** Freeze a copy of the encoder for documents, train only a query tower, with negatives
mined from the retrieval pool rather than at random.

**Why.** It is the exact pre-registered wording, and pool-mined negatives are the untested axis
that the failed cross-encoder adaptation flagged (`docs/findings-adapted-scorer.md`). Pool-mined
negatives can be reused across all interventions.

**Non-obvious caveat (important).** The engine serves **one** encoder for both sides
(`LayaModel.embed`/`embed_query` are the same function). A query-only adaptation therefore cannot
be served through the native path unless either (a) the engine gains an asymmetric query adapter,
or (b) the document index is built once with a frozen encoder and queries are served with the
adapted one — which breaks the "same model both sides" invariant and the index/model pairing. I3 is
a **mechanism probe**, not a shipping plan; I1/I2 are the servable forms.

**Cost.** Cheapest training of the three (query tower only, document embeddings cached once).

**Falsification.** Frozen-doc + adapted-query fails to beat the frozen encoder's own projection
result (0.0810 SciFact).

### I4 — Query-side linear adapter on cached embeddings (near-free mechanism probe)

**What.** A closed-form or logistic/ridge map fitted on **query** embeddings only (the existing
`tools/projection_head.py` fits one map for both sides) against the teacher's space, or against a
retrieval objective on the cached Laya document vectors
(`eval/beir/cache/nfcorpus.laya-multilingual.npy` already exists).

**Why.** Hours, no encoder training, and it answers "is the query side the bottleneck?" before any
GPU time is spent. It is the natural follow-up to the measured 0.0141 → 0.0810 → 0.1327 chain.

**Expected effect.** Bounded by the 0.1327 teacher-documents control if the map is linear and
documents stay frozen.

**Cost.** Minutes. **No heavy job.**

### I5 — Decision-head fine-tune via `laya-train` (low expected retrieval gain)

**What.** Adapt the relevance `noul` question (`Does passage help answer query?`) on in-domain
pairs with `laya-train` / `laya.train.finetune` (`--freeze-encoder` or full).

**Why considered.** It is the literal "fine-tune Laya" and the shipped head is only **0.6575**
accuracy (multilingual, 0.625 english) on Laya's own MS MARCO relevance benchmark
(`vendor/laya/research/results/app_benchmark_results.json`), with ECE 0.1718 and Brier 0.489.

**Why it is not the primary.** The head is the *filter*, and the engine already discards its
ordering. A better-calibrated filter can change which passages survive the threshold, but the
project measured that the filter path is not where the score is lost; four stronger scorers have
already failed as rankers. **Expected effect on nDCG@10: ~0**; possible effect on gate
precision/recall and on `RERANK_THRESHOLD` behaviour (which could be a real, if smaller, win).

### I6 — Bridge-conditioned head fine-tune for multi-hop (targets the biggest gap, high risk)

**What.** Fine-tune the head on the project's *own* bridge-conditioned wording
`s(q, b, c)` using MuSiQue's train split (gold supporting paragraphs give a real bridge), then use
it in the multi-hop selector. The mechanism is validated (+0.0293 AUC over a shuffled-bridge
control) but the *head* was measured to be the limit (`README.md` §bridge, `findings-selector-ltr.md`).

**Why it might work where others failed.** All prior multi-hop head attempts borrowed an
off-the-shelf scorer; none trained Laya's own head on this task's conditioning. This is the only
untried version, and it attacks the project's largest measured headroom (ordering **+0.1254**).

**Cost.** MuSiQue train fetch + ~1 h training; scoring is the existing harness.

**Falsification.** Held-out MAP@10 must beat `covquota-3-0.7` (**0.3280** subset / **0.3192**
full) with a paired interval excluding zero, on **both** MultiHop-RAG and MuSiQue.

### I7 — Multilingual run (the differentiator, later)

Train on a multilingual retrieval set (MIRACL or similar) so the `multilingual` checkpoint's
100+ language window is a real retrieval advantage. Not local; needs a download decision. This is
where Bet 9's strategic value lives, but it is downstream of I1 proving the mechanism.

---

## 7. Recommended sequence, with decision points

| stage | work | stop condition |
| --- | --- | --- |
| **0. Measurement + probes** (hours, mostly light) | Record the engine baselines the falsification needs: `tools/beir.py --retrievers bm25,dense` on SciFact/NFCorpus/ArguAna, and `tools/engine_beir.py` with `EMBED_MODEL=""`. Run I4 (query-side linear adapter) and a 1k-step throughput sweep of the encoder trainer. | If I4 cannot beat 0.0141 with any linear query map, the bottleneck is the frozen representation and I1's ceiling is low — but I1 can still reshape *both* sides, so continue only if the throughput sweep is acceptable. |
| **1. Primary adaptation** (I1) | Train the encoder on a 150k-pair stratified subset with step checkpoints; evaluate the `dense` and `hybrid` rows after each checkpoint. | Stop early if the dense row flattens for two checkpoints. |
| **2. Distillation** (I2) | Run the ensemble-student recipe on NFCorpus (and FiQA if caches extend), reusing the I1 trainer. | A paired bootstrap against the zero-shot student. |
| **3. Decide on head work** (I5/I6) | Only if stage 1/2 show the encoder is not the binding constraint, or if multi-hop is the priority. | I6's covquota bar. |

**Pre-registered overall falsification (Bet 3).** Engine-level BEIR nDCG@10 on SciFact, NFCorpus and
ArguAna with `EMBED_MODEL=""` must beat the lexical-only default (BM25 0.6539 / 0.3042 / ~0.417) and
approach `bge-small`'s retrieval-trained rows (SciFact hybrid 0.7264–0.732, NFCorpus dense 0.3386,
ArguAna dense 0.6467). A plateau near 0.1327 is a **valuable negative**: it says the system's
differentiator is the calibrated judge, not the encoder, and it should be written down rather than
explained away.

---

## 7b. Stage 0 results (measured on this host, 2026-10-08)

**I4 query-side linear adapter** (`tools/laya_query_adapter.py`, `eval/beir/query-adapter-*.md`):

| arm | SciFact nDCG@10 | NFCorpus nDCG@10 |
| --- | --- | --- |
| `raw_laya` | 0.0141 | 0.0146 |
| shared doc-fitted map (existing projection) | 0.0810 | 0.0321 |
| `teacher_docs_laya_q_proj` (existing control) | 0.1327 | 0.0420 |
| **query-only adapter + teacher docs** | **0.1262** | **0.1008** |
| `teacher_full` (ceiling) | 0.7200 | 0.3388 |

**The linear route is exhausted at ~0.10–0.13.** A supervised query-only map helps a lot on
NFCorpus (0.0420 → 0.1008, 2.4×) but is *below* the existing control on SciFact, and both sit an
order of magnitude under the teacher. Nothing linear closes that gap; only nonlinear encoder
training can.

**Throughput sweep** (`tools/laya_finetune_probe.py`, worst-case max-length padding, two encoder
passes + backward + AdamW): batch 8/128 0.633 s/step, **batch 32/128 0.832 s/step (0.026 s/pair,
the best)**, batch 32/256 1.351 s/step; memory 3.8–4.9 GB. Real training measured 0.95–1.1 s/step
at batch 32 with dynamic padding and the optimiser state, so ~90 min per 180k-pair epoch.

**De-risk run** — 400 steps, 20k NFCorpus gold pairs, exported at `eval/laya-encoder-derisk`:

| eval | baseline | adapted | note |
| --- | --- | --- | --- |
| NFCorpus `dense` (in-domain) | 0.0146 | **0.2017** | 13.8×; ~5× above the 0.042 linear control |
| NFCorpus `hybrid` (RRF k=60, weight 1.0) | 0.3042 (`bm25`) | **0.3171** | dense now *adds* to BM25 on this corpus |
| SciFact `dense` (out-of-domain) | 0.0141 | **0.2846** | 20× from a 20k-pair NFCorpus-only run |
| SciFact `hybrid` (weight 1.0) | 0.6539 (`bm25`) | 0.5284 | **fusion hurts here** — the dense channel is still weaker than BM25 |

So the mechanism is proven: full encoder training escapes the linear ceiling by 5–20×, and it
generalises to an unseen corpus. The open problem is not "does it learn" but "does it get strong
enough that RRF weight 1.0 does not dilute a strong lexical channel". Run A trains on the external
mix; the next measurement is the dense/hybrid rows, and a fusion-weight sweep rather than the
hard-coded 1.0.

**Results (measured, three runs).** Run A (180k MS MARCO/HotpotQA/FEVER gold pairs, 1 epoch,
104 min) lifts the dense row SciFact **0.4711**, NFCorpus **0.2440**, ArguAna **0.4683** (raw
0.014). Runs B and C (warm-started from A, adding in-domain NFCorpus/FiQA/SciFact pairs; C also
masks in-batch false negatives) specialise: best per-corpus fused nDCG@10 is **SciFact 0.6834
(Run C, w=0.1), NFCorpus 0.3578 (Run C, w=0.5), ArguAna 0.4890 (Run A, w=1.0)** against
lexical-only 0.6762 / 0.3244 / 0.4169 — a **mean 0.4725 → 0.5101 (+0.0376)**. The engine-level
falsification passes in the replica's direction on SciFact (0.6702 → **0.6765**). Three boundaries
were measured: the optimum fusion weight is per-corpus (≈0.1–0.5, never the 1.0 auto-default);
in-domain training costs the out-of-domain rows and the false-negative mask does not recover them;
and two adapted encoders do not ensemble. The deciding cost is the judge: the same checkpoint's
relevance AUC falls **0.8008 → 0.577–0.633**, so `EMBED_MODEL=""` is not safe without retraining
the head — serve the adapted encoder via `EMBED_MODEL` and keep the shipped checkpoint as the judge.
Full detail, caveats and method notes:
[`findings-encoder-adaptation.md`](findings-encoder-adaptation.md).

---

## 7c. Evaluation runbook after a training run

All of these take the heavy-job lock, so they run **one at a time**. `tools/beir.py` reuses the
embedding cache keyed by `laya-<checkpoint basename>`, so only the first run on a dataset pays
the embedding pass.

```sh
CKPT=eval/laya-encoder-runA

# 1. Dense and fused rows for the adapted checkpoint (the score the falsification is about).
LAYA_RAG_CHECKPOINT=$CKPT .venv/bin/python tools/beir.py \
    --datasets scifact,nfcorpus,arguana --retrievers bm25,dense,hybrid \
    --dense-weight 1.0 --out eval/beir/adapted-runA

# 2. Fusion-weight sweep. The de-risk run showed weight 1.0 can dilute a strong lexical
#    channel, so the fused number is only meaningful with the weight reported.
for w in 0.0 0.3 0.5 0.7 1.0; do
  LAYA_RAG_CHECKPOINT=$CKPT .venv/bin/python tools/beir.py \
      --datasets scifact,nfcorpus --retrievers hybrid --dense-weight $w \
      --out eval/beir/adapted-runA-w$w
done

# 3. Engine-level falsification: lexical-only vs raw-Laya-fused vs adapted-fused. The engine
#    resolves the dense weight to 0 for Laya's own encoder, so DENSE_WEIGHT must be forced to
#    actually fuse it; `--embed-model ""` is what selects the Laya-native path.
LAYA_RAG_DENSE_WEIGHT=0 .venv/bin/python tools/engine_beir.py \
    --dataset scifact --embed-model "" --tag lexical --out eval/engine-beir
LAYA_RAG_DENSE_WEIGHT=1 .venv/bin/python tools/engine_beir.py \
    --dataset scifact --embed-model "" --tag laya-raw --out eval/engine-beir
LAYA_RAG_CHECKPOINT=$CKPT LAYA_RAG_DENSE_WEIGHT=1 .venv/bin/python tools/engine_beir.py \
    --dataset scifact --embed-model "" --tag adapted --out eval/engine-beir

# 4. Judge damage: the head's weights are frozen but its inputs moved.
.venv/bin/python tools/laya_judge_check.py --dataset scifact --out eval/judge-original
LAYA_RAG_CHECKPOINT=$CKPT .venv/bin/python tools/laya_judge_check.py \
    --dataset scifact --out eval/judge-adapted
```

Training runs are launched the same way, in the background with resumable state:

```sh
nohup .venv/bin/python tools/laya_encoder_finetune.py \
    --datasets msmarco,hotpotqa,fever --per-dataset-cap 60000 --max-pairs 180000 \
    --epochs 1 --batch-size 32 --checkpoint-steps 500 \
    --checkpoint-dir eval/laya-encoder-checkpoints/runA \
    --save-model eval/laya-encoder-runA --export-hf eval/laya-encoder-runA-hf \
    --out eval/laya-encoder-runA > eval/laya-encoder-runA.log 2>&1 &
```

`--resume-checkpoint` restores encoder + optimiser + step from `latest.pt`, and refuses if any
fingerprinted setting changed.

---

## 8. Implementation sketch

New tool `tools/laya_encoder_finetune.py` (mirroring `tools/finetune.py`'s interface so the two are
directly comparable):

- **Load**: `laya.load(models_dir/"laya", subfolder=..., device="mps")`; keep `agent.model.encoder`
  and `agent.tok`; freeze `agent.model.head`/`scorer`.
- **Pairs**: gold pairs from `beir.load_qrels(split="train")`, or teacher positives via the
  existing `finetune.ensemble_positive_pairs` (model-agnostic) and `beir.cached_embed`.
- **Objective**: symmetric InfoNCE over in-batch negatives at temperature 0.05 (the probe's
  objective), plus optional pooled hard negatives (`finetune.mine_hard_negatives` generalised to
  the Laya encoder).
- **Checkpointing** (required: host has crashed before): save a resumable state
  (`encoder.pt` + optimiser + step + a settings fingerprint, refusing to resume under changed
  settings — the pattern `tools/order_train.py` and `tools/finetune.py` already use) every N steps
  to `eval/laya-encoder-checkpoints/<dataset>/`.
- **Export**: write a servable Laya checkpoint by copying the original `model.safetensors` and
  replacing the `encoder.*` tensors, then `laya.train.save_checkpoint` layout
  (`model.safetensors`, `encoder/`, `tokenizer/`, `rl_agent_config.json`) — the round-trip is
  verified in section 4. Also export a plain HF encoder dir + `1_Pooling/config.json` (mean) so
  `EMBED_MODEL=<dir>` works with no engine change.
- **Serve the native path**: `LayaModel._resolve_spec` currently accepts only the four checkpoint
  names. To run the falsification's `EMBED_MODEL=""`, allow `CHECKPOINT` to be a filesystem path to
  a checkpoint directory (a small, testable change in `src/laya_rag/config.py` /
  `src/laya_rag/model.py`).
- **Makefile**: `laya-finetune` (train), `laya-finetune-eval` (BEIR dense/hybrid + engine rows),
  `laya-finetune-verify` (round-trip fidelity, like `distill_verify.py`).
- **Heavy job**: wrap in `heavylock.heavy_job_guard("laya_encoder_finetune.py")` and checkpoint, per
  the one-heavy-job rule.

---

## 9. Risks

- **The ceiling may be low.** The frozen-representation control is 0.1327; whether full-encoder
  training escapes it is the open question. This is the main risk and the reason stage 0 exists.
- **Encoder training can damage the judge.** The frozen decision head reads encoder hidden states;
  changing the encoder shifts them. Measure relevance AUC and answerability AUC (via
  `tools/multihop_gate.py`) on the adapted checkpoint, not just nDCG.
- **Domain overfit.** Train on a mix, not one corpus; report cross-corpus rows.
- **Cost/benefit.** 2–5 h per run on MPS, serialised by the heavy-job lock.
- **The prior is strong.** If I1/I2 fail, that is a *result* about the architecture (the
  differentiator is the judge, not the encoder), and it belongs in `FINDINGS.md` as a negative.

---

## Appendix — measured anchors used above

| number | value | source |
| --- | --- | --- |
| Laya encoder dense SciFact | 0.0141 | `eval/beir/report-final.md`, `eval/beir/projection-scifact.json` |
| ridge projection (frozen) | 0.0810 | `eval/beir/projection-scifact.json` |
| teacher-documents control | 0.1327 | `eval/beir/projection-scifact.json` |
| BM25 SciFact / NFCorpus | 0.6539 / 0.3042 | `eval/beir/report-final.md` |
| `bge-small` bi-encoder SciFact / NFCorpus | 0.7200 / 0.3386 | `eval/beir/report-final.md` |
| engine hybrid SciFact | 0.7264 (0.7443 with sparse) | `eval/engine-beir/scifact-base.json`, `docs/findings-engine-parity.md` |
| ensemble+BM25 BEIR-5 mean | 0.495 (SciFact 0.767, NFCorpus 0.387, ArguAna 0.651, SciDocs 0.225, FiQA 0.444) | `RANKINGS.md` |
| NFCorpus fine-tune full | 0.3385 → 0.3285 | `eval/finetune-full.json` |
| NFCorpus distillation | 0.3385 → 0.3505 / 0.3606; ensemble 0.3798 | `eval/distill-eval.json`, `tools/distill_eval.py` |
| MultiHop best / engine default MAP@10 | 0.3192 / 0.3128 | `docs/FINDINGS.md` (cross-encoder × coverage; engine-level MultiHop) |
| answerability scalar vs 12-feature gate | AUC 0.5087 / 0.8244; ECE 0.3085 / 0.0892 | `docs/findings-calibration-row.md` |

The probes are reproducible as one script:

```sh
.venv/bin/python tools/laya_finetune_probe.py --checkpoint multilingual --out /tmp/laya-probe.json
```

Its `multilingual` / MPS output, as run for this study:

```json
{"load_seconds": 8.304}
{"device": "mps", "encoder_parameters": 306939648, "batch_size": 8, "query_len": 192,
 "doc_len": 256, "cold_step_seconds": 15.554, "warm_step_seconds_mean": 0.62,
 "warm_step_seconds": [0.644, 0.607, 0.61], "loss_first": 2.4433, "loss_last": 2.0399,
 "mps_allocated_mb": 3750.0}
{"save_seconds": 1.358, "files": ["encoder", "model.safetensors", "rl_agent_config.json",
 "tokenizer"], "reload_probability": 0.8563, "reload_ok": true}
```
