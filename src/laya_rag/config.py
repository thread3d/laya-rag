"""Configuration for the Laya RAG engine.

Every value here is deployment-varying and therefore overridable through the
environment. Defaults target this repository's local checkout: the Laya submodule
under ``vendor/laya``, with the checkpoints it downloads symlinked into ``models/``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

# <repo>/src/laya_rag/config.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]

ENV_PREFIX = "LAYA_RAG_"

# Fields that must never be serialised verbatim. ``as_dict`` feeds ``/v1/stats`` and
# the CLI's ``--json`` output, so a secret here would land in a response body or a
# log line. Report that one is configured without disclosing it.
REDACTED_FIELDS: frozenset[str] = frozenset({"api_key"})

# Where a stock Ollama serves its OpenAI-compatible API. This is the default endpoint
# for answer composition, so a local Ollama is used without any environment at all.
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"

# The composition model this repository defaults to. Composition is the one step that
# needs a generative model, and the default names the one installed in the local
# Ollama on this project's development host; ``COMPOSE_MODEL`` overrides it. A name
# that is not actually installed fails loudly on first use, which is why the default
# is a model that is known to be present rather than a guess.
DEFAULT_COMPOSE_MODEL = "ornith-1.5:35b"

# Values of ``COMPOSE_MODEL`` that mean "do not compose". They let the switch be turned
# off explicitly now that a model name is defaulted, instead of requiring the variable
# to stay unset.
COMPOSE_DISABLED_VALUES: frozenset[str] = frozenset({"none", "off", "disabled", "false"})


def _env(name: str) -> str | None:
    value = os.environ.get(ENV_PREFIX + name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{ENV_PREFIX}{name} must be a boolean, got {raw!r}")


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_PREFIX}{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_PREFIX}{name} must be a number, got {raw!r}") from exc


@dataclass
class EngineConfig:
    """Resolved engine configuration.

    ``checkpoint`` names one of the bundled Laya checkpoints. ``multilingual`` is
    the default because it is the smallest (322M) and the only one that reads long
    documents up to 8192 tokens, which is what chunking feeds it.
    """

    models_dir: Path = field(
        default_factory=lambda: Path(_env("MODELS") or REPO_ROOT / "models")
    )
    index_dir: Path = field(default_factory=lambda: Path(_env("INDEX_DIR") or REPO_ROOT / "indexes" / "default"))
    # Retrieval storage backend. "memory" is the default vector index + Python-postings
    # BM25. "ondisk" streams vectors to a memory-mapped file and packs BM25 postings into
    # one mmap, so query-time residency no longer scales with corpus size (the route to
    # the large BEIR corpora). The on-disk backend is append-only and rebuilds on change.
    index_backend: str = field(default_factory=lambda: _env("INDEX_BACKEND") or "memory")
    checkpoint: str = field(default_factory=lambda: _env("CHECKPOINT") or "multilingual")
    # "auto" defers to Laya's CUDA -> MPS -> CPU selection. On this host that picks
    # MPS (Metal), which measured 1.6-3x faster than CPU in the Laya checkout.
    device: str = field(default_factory=lambda: _env("DEVICE") or "auto")

    # Encoder limits. max_len governs the state (document) budget; head_max_len the
    # option/instructions budget. Laya defaults to 512/192.
    max_len: int = field(default_factory=lambda: _env_int("MAX_LEN", 2048))
    head_max_len: int = field(default_factory=lambda: _env_int("HEAD_MAX_LEN", 192))
    embed_max_length: int = field(default_factory=lambda: _env_int("EMBED_MAX_LENGTH", 512))
    embed_batch_size: int = field(default_factory=lambda: _env_int("EMBED_BATCH_SIZE", 16))

    # The dense-channel embedder. Empty keeps Laya's own mean-pooled encoder, which is
    # not retrieval-trained: measured on BEIR it scores nDCG@10 ~0.014 where BM25 scores
    # ~0.65. Setting this to a retrieval-trained model (e.g. "BAAI/bge-small-en-v1.5")
    # replaces the dense channel and makes fusing it worthwhile. Laya still owns the
    # decisions either way.
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL") or "")
    # BGE-family models are trained with [CLS] pooling, not mean pooling; "auto" picks
    # CLS for them and mean otherwise. Their retrieval query instruction is optional for
    # v1.5 and measured slightly better off on BEIR SciFact, so it defaults to off.
    embed_pooling: str = field(default_factory=lambda: _env("EMBED_POOLING") or "auto")
    embed_query_instruction: bool = field(
        default_factory=lambda: _env_bool("EMBED_QUERY_INSTRUCTION", False)
    )

    # Optional retrieval-trained cross-encoder that orders and filters the candidate
    # evidence, replacing Laya's relevance ordering (Laya keeps the answerability gate,
    # and the adjudication path is unchanged). Off by default because it is a second
    # model to download; measured it lifts lexical recall from 0.654 to 0.680 on BEIR
    # SciFact and 0.304 to 0.339 on NFCorpus.
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL") or "")

    # torch thread oversubscription hurts on this host (56 logical cores); 16 is the
    # measured sweet spot recorded in the Laya checkout's LOCAL_SETUP.md.
    torch_threads: int = field(default_factory=lambda: _env_int("TORCH_THREADS", 16))

    # Chunking. Character-based with paragraph preference; sized to sit well inside
    # max_len so a rerank call sees the whole chunk plus the query. The overlap is the
    # containment guarantee: a fact no longer than `chunk_overlap_chars` is intact in at
    # least one chunk whatever the split distance (a 290-character fact measured lost at
    # 200, which is why the default is 400). `chunk_text` additionally never splits a
    # sentence that fits in one chunk, so prose facts longer than the overlap survive too.
    chunk_chars: int = field(default_factory=lambda: _env_int("CHUNK_CHARS", 1200))
    # Derive the chunk size from each corpus's own document lengths instead of using the
    # fixed default: the 90th percentile, clamped into [CHUNK_CHARS, CHUNK_CHARS_AUTO_MAX]
    # so deriving can only chunk *less* aggressively than before. Measured at the engine
    # level on SciFact: 0.7264 -> 0.7417 nDCG@10 (p=0.023), with 9,986 -> 5,976 chunks; and
    # on the vendored docs corpus MRR 0.6429 -> 0.7143 with 459 -> 220 chunks. See
    # docs/findings-chunk-merge.md.
    chunk_chars_auto: bool = field(default_factory=lambda: _env_bool("CHUNK_CHARS_AUTO", True))
    chunk_chars_auto_max: int = field(
        default_factory=lambda: _env_int("CHUNK_CHARS_AUTO_MAX", 2000)
    )
    # Which quantile of the document lengths to derive from. The clamp makes p90 and p95
    # identical on short documents, while p75 chunks more; the value is heuristic and
    # swept in docs/findings-chunk-merge.md.
    chunk_chars_auto_percentile: float = field(
        default_factory=lambda: _env_float("CHUNK_CHARS_AUTO_PERCENTILE", 0.9)
    )
    chunk_overlap_chars: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP_CHARS", 400))
    min_chunk_chars: int = field(default_factory=lambda: _env_int("MIN_CHUNK_CHARS", 80))
    # Keep a document of at most chunk_chars + overlap_chars whole instead of splitting it
    # into two heavily overlapping chunks. Measured at the engine level on SciFact
    # (median document 1,426 chars) this did **not** recover the whole-document gain:
    # 0.7236 against 0.7264 for the default chunking and 0.7396 for whole-document
    # chunking, and whole-document beat this rule significantly (+0.0160, p=0.016). The
    # loss therefore lives in documents longer than the merge window, or in candidate
    # diversity, not in short ones. Off by default; see docs/findings-chunk-merge.md.
    chunk_merge_short: bool = field(
        default_factory=lambda: _env_bool("CHUNK_MERGE_SHORT", False)
    )

    # Structural context prefix (RESEARCH.md Tier 1.6). When true, every chunk carries a
    # breadcrumb (document path and first heading) that the lexical and dense channels
    # index but the returned evidence does not contain. It is off by default so existing
    # indexes and their rankings are unchanged; turning it on requires a re-index.
    context_prefix: bool = field(default_factory=lambda: _env_bool("CONTEXT_PREFIX", False))

    @property
    def resolved_chunk_overlap(self) -> int:
        """The overlap actually used, never wider than the chunk.

        The default 400 is larger than a deliberately small ``CHUNK_CHARS``; a user who
        sets only the chunk size should get a valid split rather than a `ValueError`
        from the chunker, so the overlap is clamped to ``chunk_chars - 1``.
        """
        return max(0, min(self.chunk_overlap_chars, self.chunk_chars - 1))

    # Retrieval funnel: dense candidates -> Laya re-ranked -> evidence passages.
    retrieve_k: int = field(default_factory=lambda: _env_int("RETRIEVE_K", 20))
    rerank_k: int = field(default_factory=lambda: _env_int("RERANK_K", 5))
    # Promote the best passage of each distinct document into the first N kept slots. The
    # engine counts *passages* against RERANK_K, so an overlapping chunking can spend
    # several slots on one file. It guarantees prefix diversity — the first N delivered
    # passages are N documents — and on SciFact (300 queries, top_k=10) it left
    # document-level nDCG **exactly unchanged**, because the top-10 already held 5–10
    # distinct documents; it can add documents that a shorter top_k would have cut. It is a
    # presentation control, while the fact-level gain the harness measured on MultiHop
    # (doc-quota +0.0173 MAP@10) comes from spending slots on new facts. 0 keeps the pure
    # passage order.
    rerank_min_docs: int = field(default_factory=lambda: _env_int("RERANK_MIN_DOCS", 0))
    # Bounded cache of recent query embeddings. A service sees the same question
    # repeatedly (retries, refreshes, shared links) and the query embedding is the one
    # dense cost on an otherwise lexical path; 0 disables the cache. It holds vectors, not
    # results, so a re-index can never be hidden by it.
    query_cache_size: int = field(default_factory=lambda: _env_int("QUERY_CACHE_SIZE", 128))

    # English stemming for the lexical channel. Measured on BEIR it closes most of the gap
    # to the published, stemmed Lucene BM25 baseline: SciFact nDCG@10 0.654 -> 0.677,
    # NFCorpus 0.304 -> 0.324. Set false for corpora where exact terms matter more than
    # morphology (source code, identifiers); the setting is stored with the index, so an
    # existing unstemmed index keeps working and its queries stay unstemmed.
    lexical_stem: bool = field(default_factory=lambda: _env_bool("LEXICAL_STEM", True))

    # Which BM25 implementation the lexical channel uses. "bm25" is the historical
    # per-document postings index; "eager" precomputes each posting's final weight and
    # scores with vectorised NumPy accumulation. The ranking is identical for the same
    # k1/b/stem/stopword settings (pinned by tests and a parity bench); "eager" trades a
    # slightly larger build for far less per-query Python and a smaller resident index.
    lexical_backend: str = field(default_factory=lambda: _env("LEXICAL_BACKEND") or "bm25")

    @property
    def resolved_lexical_backend(self) -> str:
        value = (self.lexical_backend or "").strip().lower()
        if value in {"bm25", "eager"}:
            return value
        raise ValueError(
            f"{ENV_PREFIX}LEXICAL_BACKEND must be 'bm25' or 'eager', got {self.lexical_backend!r}"
        )

    # Reciprocal-rank-fusion weights. Plain RRF assumes the two channels are
    # comparably good; measured, they are not. Laya's mean-pooled encoder scores
    # nDCG@10 ~0.014 on BEIR where BM25 scores ~0.654, and every non-zero dense weight
    # lowered the score — including on this project's own planted corpus, where lexical
    # recall ranks the gold passage first for every query and fusion drops it. So the
    # default is "auto": off for Laya's encoder, on for a configured retrieval-trained
    # embedder, which is the only case where the dense channel earns a weight.
    # -1 means auto; anything >= 0 is used verbatim.
    dense_weight: float = field(default_factory=lambda: _env_float("DENSE_WEIGHT", -1.0))
    # Reciprocal-rank-fusion constant for the first stage (dense + lexical). Lower values
    # weight the very top of each ranking more heavily. Full 2,255-query MultiHop
    # measurement: `k=5` beats the historical default 60 on MAP@10 by
    # +0.0040 [+0.0012, +0.0069] (p=0.008), with a non-significant Hits@10 cost
    # (-0.0053, p=0.10) — and the harness that produced the MultiHop numbers already used
    # 5, so this aligns the engine with its own measurement.
    fusion_k: int = field(default_factory=lambda: _env_int("FUSION_K", 5))
    # Approximate dense search. The exact scan is fast to a few hundred thousand vectors and
    # linear after that; tools/ann_bench.py measures the recall/latency trade. Off by
    # default so small corpora keep exact results.
    ann: bool = field(default_factory=lambda: _env_bool("ANN", False))
    ann_nprobe: int = field(default_factory=lambda: _env_int("ANN_NPROBE", 16))
    lexical_weight: float = field(default_factory=lambda: _env_float("LEXICAL_WEIGHT", 1.0))
    # BM25's term-frequency saturation and length normalisation. The defaults are Lucene's
    # (and match ``laya_rag.lexical.K1/B``); the published BM25 baselines on BEIR/Anserini
    # use 0.9/0.4, which is what tools/beir.py ran — so the engine's lexical channel could
    # not previously reproduce the harness rows it was compared against. Exposed to measure
    # and, if warranted, align.
    bm25_k1: float = field(default_factory=lambda: _env_float("BM25_K1", 1.2))
    bm25_b: float = field(default_factory=lambda: _env_float("BM25_B", 0.75))
    # A learned-sparse (SPLADE-style) third channel. Unset means the engine indexes and
    # searches exactly as before. Measured on the BEIR harness it beats the stemmed lexical
    # channel by +0.03 to +0.14 and is the first channel addition that lifts the fusion
    # consistently (+0.004 to +0.023), at the cost of one extra model at index and query
    # time. Indexing encodes every chunk with it, so it is opt-in and rebuilds are slower.
    sparse_model: str = field(default_factory=lambda: _env("SPARSE_MODEL") or "")
    sparse_weight: float = field(default_factory=lambda: _env_float("SPARSE_WEIGHT", 1.0))
    sparse_top_k: int = field(default_factory=lambda: _env_int("SPARSE_TOP_K", 128))
    sparse_batch_texts: int = field(default_factory=lambda: _env_int("SPARSE_BATCH_TEXTS", 16))

    @property
    def sparse_enabled(self) -> bool:
        return bool((self.sparse_model or "").strip())
    # Near-duplicate suppression on the fused candidate list. Chunk overlap makes a
    # strongly-matching passage arrive several times and spend several recall slots, which
    # measured on MultiHop-RAG is where the multi-hop MAP@10 loss lives: skipping a fused
    # candidate whose cosine to an already-kept chunk is >= this threshold lifted MAP@10
    # +0.024 (0.2546 -> 0.2790) and Hits@10 +0.052. 0.0 disables it; 0.9 is the measured
    # setting, and it is opt-in because it needs the dense index and only pays off when the
    # corpus has heavy chunk overlap or near-duplicate documents.
    recall_dedup_threshold: float = field(
        default_factory=lambda: _env_float("RECALL_DEDUP_THRESHOLD", 0.0)
    )
    # Evidence-set coverage selection on the fused candidate list: a sentence-atom
    # submodular max-coverage pass that spends the recall budget on *new facts* rather
    # than on a second near-duplicate of one already-kept passage. Measured on
    # MultiHop-RAG it is the best coverage pass (MAP@10 0.2546 -> 0.2835, held-out
    # +0.0288), ahead of near-duplicate suppression (0.2790). 0.0 disables it; the
    # lambda balances fused relevance against marginal atom coverage (0.5 is the
    # measured setting). Opt-in because it only pays off on multi-fact corpora.
    recall_coverage_lambda: float = field(
        default_factory=lambda: _env_float("RECALL_COVERAGE_LAMBDA", 0.0)
    )
    # The dense-gram redundancy term inside the coverage objective. Full 2,255-query
    # measurement at the recommended lambda 0.5: dropping it lifts MAP@10 0.2835 -> 0.2850
    # (+0.0015 [+0.0002, +0.0029], p=0.032) with Hits@4 0.7197 -> 0.7326 and Hits@10
    # 0.8585 -> 0.8723, so it defaults to false; at lambda 0.7 the penalty does help
    # (+0.0036), so it stays exposed. Only relevant when RECALL_COVERAGE_LAMBDA > 0.
    recall_coverage_redundancy: bool = field(
        default_factory=lambda: _env_bool("RECALL_COVERAGE_REDUNDANCY", False)
    )

    @property
    def resolved_dense_weight(self) -> float:
        """The dense RRF weight actually used: explicit, or auto from the embedder."""
        if self.dense_weight >= 0.0:
            return self.dense_weight
        return 1.0 if (self.embed_model or "").strip() else 0.0

    @property
    def resolved_index_backend(self) -> str:
        """``memory`` or ``ondisk``; anything else is a configuration error."""
        value = (self.index_backend or "").strip().lower()
        if value in {"memory", "ondisk"}:
            return value
        raise ValueError(
            f"{ENV_PREFIX}INDEX_BACKEND must be 'memory' or 'ondisk', got {self.index_backend!r}"
        )

    # What the returned evidence is ranked by. "recall" keeps the retrieval order and
    # uses the relevance score only to filter and gate; "relevance" orders by that score;
    # "auto" (the default) picks relevance when a cross-encoder is configured — a trained
    # ranker's ordering is worth using — and recall otherwise, because Laya's decision is
    # a stronger filter than ranker (reordering by it lowered nDCG@10 on both BEIR
    # datasets: SciFact 0.654 -> 0.470, NFCorpus 0.304 -> 0.283).
    rerank_order: str = field(default_factory=lambda: _env("RERANK_ORDER") or "auto")

    @property
    def resolved_rerank_order(self) -> str:
        value = (self.rerank_order or "").strip().lower()
        if value in {"recall", "relevance"}:
            return value
        return "relevance" if (self.rerank_model or "").strip() else "recall"

    # How far a supplied second-stage score may move the first-stage order, in [0, 1].
    # 1.0 orders entirely by the decision (the historical behaviour); 0.0 keeps the
    # recall order. Measured, a pointwise cross-encoder trusted fully lost nDCG@10 on
    # all five BEIR corpora (ArguAna 0.568 -> 0.250); blending bounds that damage, so a
    # second stage can be used as a nudge rather than a replacement. It is ignored when
    # the coverage pass below is active, because a blend reorders by a mixed score and
    # would discard the coverage order; coverage forces a pure replacement.
    rerank_blend: float = field(default_factory=lambda: _env_float("RERANK_BLEND", 1.0))

    # Coverage selection over the cross-encoder-scored pool. When a cross-encoder is
    # configured and this resolves > 0, the candidate pool is reordered by sentence-atom
    # max-coverage using the cross-encoder score as the relevance term, before the
    # relevance filter and the top-k cut. Measured against a full cross-encoder re-sort,
    # first on the deterministic 400-query subset (MAP@10 0.2986 -> 0.3206 at lam 0.7)
    # and then confirmed on the full 2,255-query protocol: MAP@10 0.2948 -> 0.3165
    # (+0.0217 [+0.0190, +0.0242], p=0.0005) and Hits@4 0.7348 -> 0.7694
    # (+0.0346 [+0.0235, +0.0457]). -1 means auto: 0.7 when a cross-encoder is configured
    # (the measured setting, and the same "auto" shape DENSE_WEIGHT/RERANK_ORDER use) and
    # 0.0 otherwise, so a caller who configures a second stage gets the measured-best
    # configuration by default; >= 0 is used verbatim, so 0.0 still disables it explicitly.
    rerank_coverage_lambda: float = field(
        default_factory=lambda: _env_float("RERANK_COVERAGE_LAMBDA", -1.0)
    )

    @property
    def resolved_rerank_coverage_lambda(self) -> float:
        """The coverage weight actually used: explicit, or 0.7 auto with a cross-encoder."""
        if self.rerank_coverage_lambda >= 0.0:
            return self.rerank_coverage_lambda
        return 0.7 if (self.rerank_model or "").strip() else 0.0
    # Document quota for the coverage pass above: the first N selections are the best
    # cross-encoder candidate of distinct documents, then coverage fills the rest. This
    # stops one article occupying the top-k with overlapping chunks, which is the
    # multi-hop failure mode. Full-protocol (2,255 queries) with lambda 0.7, against
    # coverage alone: MAP@10 0.3165 -> 0.3192 (+0.0027 [+0.0002, +0.0052], p=0.037) but
    # Hits@4 0.7694 -> 0.7588 (-0.0106). It is a real trade, not a free win, so it is
    # off by default; set 3 when MAP@10 matters more than Hits@4.
    rerank_coverage_min_docs: int = field(
        default_factory=lambda: _env_int("RERANK_COVERAGE_MIN_DOCS", 0)
    )

    # Retrieval-time trust (src/laya_rag/trust.py). A JSON manifest maps path globs to a
    # tier — quarantined < untrusted < trusted — and a query may exclude candidates below
    # QUERY_MIN_TRUST. An empty manifest means no filtering at all, so the shipped
    # behaviour is unchanged; when a manifest is set, anything it does not name falls to
    # TRUST_DEFAULT_TIER (untrusted by default: an allow-list is the safe reading).
    trust_manifest: str = field(default_factory=lambda: _env("TRUST_MANIFEST") or "")
    trust_default_tier: str = field(
        default_factory=lambda: _env("TRUST_DEFAULT_TIER") or "untrusted"
    )
    query_min_trust: str = field(default_factory=lambda: _env("QUERY_MIN_TRUST") or "")

    # Claim adjudication is linear in claims x candidates. Measured warm on five
    # claims: 21.6s at retrieve_k=20 against 8.7s at 10, for the same verdicts on the
    # sample. Claims are short, so a smaller pool is the better default here even
    # though the query path uses the wider one.
    claim_retrieve_k: int = field(default_factory=lambda: _env_int("CLAIM_RETRIEVE_K", 10))
    claim_max: int = field(default_factory=lambda: _env_int("CLAIM_MAX", 25))

    # Laya emits calibrated probabilities, so these are real thresholds, not scores.
    relevance_threshold: float = field(default_factory=lambda: _env_float("RELEVANCE_THRESHOLD", 0.5))
    # Minimum score a *substituted* second stage (a cross-encoder) must reach to be kept.
    # RELEVANCE_THRESHOLD is calibrated for Laya's probabilities and is far too strict for
    # another model's sigmoid: on the MultiHop CE pool only ~11% of the top-100 clear 0.5,
    # and within the engine's top-20 pool 83 of 400 queries would keep nothing at all. The
    # experiments that measured the second stage never thresholded it, so this defaults to
    # 0.0 (keep the pool, let the ordering and top-k decide). Laya's own path still uses
    # RELEVANCE_THRESHOLD.
    rerank_threshold: float = field(default_factory=lambda: _env_float("RERANK_THRESHOLD", 0.0))
    # The answerability decision boundary. The gate module warns that its class-weighted
    # intercept makes 0.5 non-optimal and that a threshold should be chosen on held-out
    # data — but measured on 300 labelled MultiHop queries over 200 stratified splits, a
    # threshold learned on each training half averaged 0.470 (sd 0.060) and changed
    # held-out balanced accuracy by +0.0004 (0.7424 -> 0.7428); the optimum is flat. So
    # 0.5 stays, and no artifact-threshold plumbing was added. See
    # docs/findings-gate-calibration.md.
    answerability_threshold: float = field(default_factory=lambda: _env_float("ANSWERABILITY_THRESHOLD", 0.5))
    # Optional persisted multi-feature answerability gate (src/laya_rag/answerability.py).
    # The shipped decision is a single scalar that measured at chance out of domain
    # (ROC-AUC 0.51 on MultiHop-RAG) and cannot be fixed by rescaling one number. A gate
    # trained on a domain's own null/answerable split over the retrieval signals the
    # engine already has is the portable replacement. Empty disables it.
    answerability_gate_path: str = field(
        default_factory=lambda: _env("ANSWERABILITY_GATE_PATH") or ""
    )
    # Optional NLI cross-encoder used as an answerability scorer, e.g.
    # "cross-encoder/nli-deberta-v3-base". Empty keeps Laya's decision (or the gate).
    # This is a different model with its own calibration, so it is explicit, not default.
    answerability_nli_model: str = field(
        default_factory=lambda: _env("ANSWERABILITY_NLI_MODEL") or ""
    )

    # Answer composition. This is the one optional step in the pipeline that is not
    # Laya's: a generative model writes prose from evidence the engine already
    # returned. Laya cannot generate text, so everything above is calibrated and
    # nothing here is. Off unless the caller explicitly asks for it.
    #
    # Both are this project's own settings and both default to a local Ollama: the
    # endpoint below is the one Ollama installs with, and the model is the one this
    # host has pulled. Set COMPOSE_MODEL to another model, or to "none" / "off" /
    # "disabled" to turn composition off so the interface disables the switch and says
    # why rather than offering a model that is not there.
    compose_base_url: str = field(
        default_factory=lambda: _env("COMPOSE_BASE_URL") or DEFAULT_OLLAMA_BASE_URL
    )
    compose_model: str = field(
        default_factory=lambda: _env("COMPOSE_MODEL") or DEFAULT_COMPOSE_MODEL
    )
    # Which route to call. Ollama's native /api/chat honours `think: false`; its
    # OpenAI-compatible /v1 route silently ignores it, which measured 561-614 chars of
    # reasoning on every call, roughly 3x the latency, and once a whole token budget
    # spent thinking with no answer written at all. "openai" exists for endpoints that
    # are not Ollama.
    compose_api: str = field(default_factory=lambda: _env("COMPOSE_API") or "ollama")
    compose_timeout_s: int = field(default_factory=lambda: _env_int("COMPOSE_TIMEOUT_S", 300))
    # A full answer over five passages measured ~520 completion tokens, and the model
    # can spend a few hundred more reasoning even with thinking disabled: at 600,
    # two runs in three hit the ceiling and stopped mid-answer.
    compose_max_tokens: int = field(default_factory=lambda: _env_int("COMPOSE_MAX_TOKENS", 1200))

    # Service
    host: str = field(default_factory=lambda: _env("HOST") or "127.0.0.1")
    port: int = field(default_factory=lambda: _env_int("PORT", 8077))
    api_key: str | None = field(default_factory=lambda: _env("API_KEY"))

    warmup: bool = field(default_factory=lambda: _env_bool("WARMUP", True))

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in REDACTED_FIELDS:
                out[f.name] = "<redacted>" if value else None
                continue
            out[f.name] = str(value) if isinstance(value, Path) else value
        # `dense_weight` is -1 to mean "auto"; report what auto resolved to as well, so a
        # status endpoint never shows a negative weight as if it were in use.
        out["dense_weight_resolved"] = self.resolved_dense_weight
        # The overlap is clamped below `chunk_chars`; report the value actually used.
        out["chunk_overlap_resolved"] = self.resolved_chunk_overlap
        # `rerank_coverage_lambda` is -1 to mean "auto"; report what auto resolved to.
        out["rerank_coverage_lambda_resolved"] = self.resolved_rerank_coverage_lambda
        return out

    def ensure_dirs(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
