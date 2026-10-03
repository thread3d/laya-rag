# Implementation plan — closing the seven gaps

Companion to [`IMPROVEMENTS.md`](IMPROVEMENTS.md). That document diagnoses and ranks; this
one is the execution plan: what will be changed, in what order, how it is verified, and what
cannot be done offline on this machine. Status is updated as work lands.

The forward programme — the six steps toward a winnable "best in the world" claim, with the
resumability contract for long jobs — is [`ROADMAP.md`](ROADMAP.md).

Guiding rules:

- **Every code change ships with an offline test.** The suite is `make test` (315 tests at the
  start of this work).
- **Nothing measured is silently reinterpreted.** New behaviour is opt-in and defaults to the
  current behaviour, so existing benchmark numbers stay comparable.
- **External-resource work is separated**, not faked: model downloads, training runs and
  leaderboard submissions are listed in Phase 5 and reported as remaining.

---

## Phase 0 — baseline and guardrails

- [x] Offline suite green before editing: **344 passed, 4 deselected**.
- [x] Cached models recorded: bge-small/base/large, bge-reranker-base, ms-marco-MiniLM.
      Not cached (external work): SPECTER2, NLI, HHEM.

## Phase 1 — decision layer (IMPROVEMENTS §5, §2)

| Item | File | Change | Test | Status |
| --- | --- | --- | --- | --- |
| 1.1 | `src/laya_rag/rerank.py` | `blend_scores` + `blend` on `rerank` (1.0 = historical, 0.0 = first stage) | `tests/test_rerank_blend.py` | done |
| 1.2 | `src/laya_rag/config.py` | `RERANK_BLEND` | config tests | done |
| 1.3 | `src/laya_rag/engine.py` | blend passthrough; `answerability_measured` on result + response; gate in `stats()` | engine tests | done |
| 1.4 | `tools/beir.py` | `crossencoder_rankings(..., blend)` + `--rerank-blend` (RRF blend) | harness tests | done |
| 1.5 | `src/laya_rag/answerability.py` | pure-NumPy feature gate (9 signals), fit/save/load/AUC | `tests/test_answerability.py` | done |
| 1.6 | `src/laya_rag/rerank.py` | optional `gate` override; short-circuit marked *not measured* | as above | done |
| 1.7 | `tools/answerability_train.py` | fit/evaluate/persist the gate on a labelled gate run | `tests/test_answerability_train.py` | done |

## Phase 2 — multi-hop set coverage (IMPROVEMENTS §1)

| Item | File | Change | Test | Status |
| --- | --- | --- | --- | --- |
| 2.1 | `tools/multihop_coverage.py` | `doc_quota_select` | `tests/test_coverage_select.py` | done |
| 2.2 | `tools/multihop_coverage.py` | `facility_location_select` (sentence-atom submodular coverage) | as above | done |
| 2.3 | `tools/multihop_coverage.py` | `pool_ceiling_MAP@10` reorder-ceiling diagnostic | as above | done |
| 2.4 | run | held-out MAP@10 for the new families | `coverage-set.json` | done — `facility-0.5` **0.2835** (+0.0288) |
| 2.5 | `src/laya_rag/engine.py` | ship the coverage objective: `RECALL_COVERAGE_LAMBDA` + `_coverage_select` on the fused pool | `tests/test_recall_coverage.py` | done |

## Phase 3 — scale (IMPROVEMENTS §6)

| Item | File | Change | Test | Status |
| --- | --- | --- | --- | --- |
| 3.1 | `src/laya_rag/engine.py` | preallocate the embedding matrix instead of list+`vstack` | existing index tests | done |
| 3.2 | on-disk backend wiring | first-class backend + incremental update | — | deferred (Phase 5) |

## Phase 4 — SciDocs representation (IMPROVEMENTS §7)

| Item | File | Change | Test | Status |
| --- | --- | --- | --- | --- |
| 4.1 | `tools/beir.py` | `--dataset-models`, `--dense-text-only-datasets` | harness tests | done |
| 4.2 | run | SciDocs sweep | measured | done — query instruction **0.1973 → 0.2052** |

## Phase 5 — completion pass

| Item | Change | Status |
| --- | --- | --- |
| 5.1 §6 on-disk engine backend | `OnDiskBM25` + `OnDiskVectorIndex`, `INDEX_BACKEND`, engine build/query wiring, deferred text | done (`src/laya_rag/ondisk.py`, `tests/test_ondisk_engine.py`) |
| 5.2 §6 quantized ANN | `build_ann` (IVF+PQ) with exact mmap fallback, engine `ANN` wiring | done |
| 5.3 §4 projection head | `LinearProjection` ridge distillation + `evaluate_projected` | done (`tools/projection_head.py`, 16 tests) |
| 5.4 §7 NLI answerability | `NLIAnswerability`, precedence over the gate | done (injected-scorer tests) |
| 5.5 §7 domain encoder | SPECTER download + `--dataset-models` + `--biencoder-pooling` | download done; SciDocs run in progress |
| 5.6 §1 MultiHop rerank | `--limit-queries`/`--rerank-targets`; 400-query subset | done — MAP@10 **+0.0464**, Hits@4 **+0.0600**, all significant |
| 5.7 §3 generation/placement | target table + scaffold + doc | **prepared only**, no submission |
| 5.8 §6/§7 remaining | streaming ingestion / term-dictionary merge; BRIGHT/LoTTE/LegalBench-RAG/FRAMES/RGB; full BEIR 18 | external (network/disk/hours) |


---

## Status log

- Phase 0: done (baseline 344).
- Phase 1: done (blend, gate, trainer, measurement flag, stats). Answerability gate measured
  on 300 real MultiHop queries: held-out AUC **0.815** vs the scalar's **0.520**.
- Phase 2: done. `facility-0.5` is the best pass at **MAP@10 0.2835** (held-out +0.0288 vs
  0.2546), ahead of `cluster-dedup-0.9` (0.2790). Pool reorder ceiling **0.609** corrected
  (0.7307 as reported then; the denominator used the pool's matched facts, not the true gold count).
- Phase 3.1: done; Phase 5 on-disk backend + quantized ANN done.
- Phase 4: flags done; SciDocs query instruction **0.1973 → 0.2052**; SPECTER pooling fix
  (`--biencoder-pooling cls`) added after the default mean pooling scored poorly.
- Phase 5.3/5.4: projection head and NLI paths implemented and tested.
- Phase 5.3 measured: SciFact projection lifts Laya nDCG@10 **0.0141 → 0.0810** (control cap
  0.1327). Phase 5.5 measured: SPECTER needs CLS pooling (mean 0.0943 → CLS 0.1475 dense) but
  is still a negative against bge-small's 0.1973 — the `--biencoder-pooling` fix is the keeper.
- Phase 5.6: MultiHop rerank subset — MAP@10 0.2522 → **0.2986** (+0.0464 [+0.0275, +0.0655]),
  Hits@4 0.6625 → **0.7225**, Hits@10 → **0.8600**, MRR@10 → **0.6010** (400 queries, seed 17).
- Test suite after implementation: **472 passed, 4 deselected**.

## Phase 6 — README known-limits pass

| Item | Change | Where | Status |
| --- | --- | --- | --- |
| 6.1 | prune chunks of files deleted/unreadable since the last run on a directory re-index; `prune` flag on the CLI and API | `src/laya_rag/engine.py`, `cli.py`, `api.py` | done |
| 6.2 | default `CHUNK_OVERLAP_CHARS` 200 → 400 and sentence-aware overlap (never split a sentence that fits a chunk) | `src/laya_rag/chunking.py`, `config.py` | done |
| 6.3 | token-capped on-disk build block + heap k-way run merge | `src/laya_rag/ondisk.py`, `tools/ondisk.py` | done |
| 6.4 | update README Known limits with the measured outcomes | `README.md`, `IMPROVEMENTS.md` | done |

Measured: the 290-character boundary fact is intact and retrieved at every split tested; on-disk
build peak RSS 1,045 MB → **453 MB** on a 30k × 200-token synthetic corpus at unchanged speed.
Test suite: **481 passed, 4 deselected**.

**Correction (2026-10-03).** The ceiling quoted as 0.7307 was mis-normalised — it divided by the gold
facts the pool matched, not the query's true gold count — and is **0.609** corrected. It also *rises*
with pool depth (0.659 at 500) rather than falling, and 31.9% of depth-100 pools are missing a gold
fact. The conclusion below that the gap is purely selection therefore overstates it: recall is a live
lever. See the README diagnostics subsection.
