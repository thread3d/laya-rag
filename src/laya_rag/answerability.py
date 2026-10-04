"""A portable, feature-based answerability gate.

The out-of-domain answerability decision currently rides on a single scalar
probability from Laya's decision head. That is a problem for ranking: a monotone
recalibration of one scalar cannot change the *order* of two candidates, so when
Laya's head is miscalibrated out of domain the gate collapses to a coin flip
(measured ROC-AUC ~0.51). No threshold or temperature fixes an ordering.

This module replaces that scalar with a small logistic gate over cheap retrieval
and evidence signals — relevance spread, recall, source diversity, evidence
volume, and query-term coverage. Those features can separate in-domain from
out-of-domain questions even when the underlying probability cannot, and a linear
gate over them is:

* pure NumPy, so the package keeps its no-scikit-learn dependency set;
* fully offline and deterministic (zero-initialised gradient descent);
* small enough to train on a few hundred labelled queries and persist as JSON.

``fit`` standardises its inputs and weights the classes inversely to their
frequency, so a gate trained on a mostly-answerable or mostly-unanswerable split
is not dominated by the majority class. Because the intercept is weighted for
that rebalanced distribution, threshold selection should be done on held-out
data (see :func:`evaluate_gate`) rather than assumed to be 0.5. The learned
parameters are stored alongside the feature names, and ``load`` selects those
columns **by name** from the current vector: a gate trained before the
evidence-shape features were appended keeps working, while a file naming a
feature this build does not define fails loudly instead of scoring the wrong
columns.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .lexical import STOPWORDS, tokenize

GATE_VERSION = 1

# The exact column order every vector in this module uses. Appending is safe: a persisted
# artefact names its own columns and the gate selects them by name, so a nine-feature gate
# keeps loading against the longer vector. Removing or reordering is a breaking change.
FEATURE_NAMES: tuple[str, ...] = (
    "max_relevance",
    "mean_relevance",
    "min_relevance",
    "max_recall",
    "mean_recall",
    "distinct_document_ratio",
    "passage_count",
    "log_evidence_chars",
    "query_term_coverage",
    # Evidence *shape* signals, appended after the nine score features. Measured on 300
    # labelled MultiHop queries over 200 stratified splits, adding these three raised the
    # gate's ROC-AUC 0.8146 -> 0.8290 (+0.0144 [+0.0123, +0.0163], p<0.0001), cut risk@80
    # 0.4069 -> 0.4009, and lowered AURC/AUGRC, at a non-significant calibration cost
    # (ECE 0.0650 -> 0.0683, p=0.085). An earlier experiment with *derived* columns of the
    # nine moved every metric the wrong way, so these are new information, not more
    # freedom for the linear model.
    "max_passage_coverage",
    "coverage_spread",
    "pairwise_redundancy",
)

#: Names of the appended evidence-shape features, in :data:`FEATURE_NAMES` order.
SHAPE_FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES[-3:]

_SHAPE_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: Any) -> set[str]:
    """Lowercase alphanumeric tokens of length >= 3, the unit the shape features use."""
    return {token for token in _SHAPE_TOKEN.findall(str(text).lower()) if len(token) >= 3}


def _text_of(passage: Any) -> str:
    """The text of a passage given as an object, a mapping, or a bare string."""
    if isinstance(passage, str):
        return passage
    for attribute in ("text", "passage", "content"):
        value = getattr(passage, attribute, None)
        if isinstance(value, str):
            return value
    if isinstance(passage, Mapping):
        for key in ("text", "passage", "content"):
            if isinstance(passage.get(key), str):
                return passage[key]
    return ""


def max_passage_coverage(query: str, passages: Sequence[Any]) -> float:
    """The best single passage's fraction of the query's content terms (0 when empty)."""
    wanted = _tokens(query)
    if not wanted:
        return 0.0
    best = 0.0
    for passage in passages or []:
        best = max(best, len(wanted & _tokens(_text_of(passage))) / len(wanted))
    return best


def coverage_spread(query: str, passages: Sequence[Any]) -> float:
    """Passages a greedy set cover needs to cover every query term.

    ``0.0`` when the query has no content terms or the evidence covers none of them; a
    larger value means the answer is spread thin across passages. Deterministic.
    """
    remaining = _tokens(query)
    if not remaining:
        return 0.0
    texts = list(passages or [])
    used = 0
    while remaining and texts:
        best_index, best_gain = None, 0
        for index, passage in enumerate(texts):
            gain = len(remaining & _tokens(_text_of(passage)))
            if gain > best_gain:
                best_index, best_gain = index, gain
        if best_index is None:
            break
        remaining -= _tokens(_text_of(texts[best_index]))
        texts.pop(best_index)
        used += 1
    return float(used)


def pairwise_redundancy(passages: Sequence[Any]) -> float:
    """Mean Jaccard overlap between every pair of passages (0 below two passages)."""
    texts = [_tokens(_text_of(passage)) for passage in passages or []]
    if len(texts) < 2:
        return 0.0
    total, pairs = 0.0, 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            union = texts[i] | texts[j]
            if union:
                total += len(texts[i] & texts[j]) / len(union)
            pairs += 1
    return total / pairs if pairs else 0.0


def _as_float(value: Any, default: float = 0.0) -> float:
    """Coerce a possibly-missing value to a finite float, else ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _as_text(value: Any) -> str:
    """Coerce a possibly-missing value to text, with ``None`` becoming ``""``."""
    return "" if value is None else str(value)


def _field(obj: Any, name: str, default: Any) -> Any:
    """Read ``name`` from an object attribute or a mapping.

    The engine returns passages as attribute-bearing ``ScoredPassage`` objects, but the
    HTTP/JSON surface and the saved reports carry the same fields as plain dicts. Both
    must produce identical features, so a mapping is read by key and anything else by
    attribute.
    """
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _query_term_coverage(query: str, evidence: str) -> float:
    """Fraction of distinct non-stopword query terms present in ``evidence``."""
    terms = {token for token in tokenize(query) if token not in STOPWORDS}
    if not terms:
        return 0.0
    present = set(tokenize(evidence))
    return len(terms & present) / len(terms)


def extract_features(query: str, passages: Sequence[Any]) -> np.ndarray:
    """Describe one query/evidence set as a vector in :data:`FEATURE_NAMES` order.

    ``passages`` are objects with ``.text`` and ``.relevance`` (the engine's
    ``ScoredPassage``) plus ``.recall_score`` and either ``.path`` or
    ``.chunk_id``. Every attribute access is defensive: a missing or malformed
    field becomes ``0.0`` or ``""`` rather than an exception, because this gate
    must not become a new failure point in the pipeline.

    Returns a 1-D ``float64`` array of length ``len(FEATURE_NAMES)``.
    """
    items = [] if passages is None else list(passages)
    count = len(items)
    relevance: list[float] = []
    recalls: list[float] = []
    documents: list[str] = []
    texts: list[str] = []
    for passage in items:
        relevance.append(_as_float(_field(passage, "relevance", 0.0)))
        recalls.append(_as_float(_field(passage, "recall_score", 0.0)))
        # A path identifies the document; chunk id is the fallback identity when a
        # caller's passage has no path. Both missing means one anonymous document.
        document = _as_text(_field(passage, "path", ""))
        if not document:
            document = _as_text(_field(passage, "chunk_id", ""))
        documents.append(document)
        texts.append(_as_text(_field(passage, "text", "")))

    if count:
        rel = np.asarray(relevance, dtype=np.float64)
        rec = np.asarray(recalls, dtype=np.float64)
        max_relevance = float(rel.max())
        mean_relevance = float(rel.mean())
        min_relevance = float(rel.min())
        max_recall = float(rec.max())
        mean_recall = float(rec.mean())
        distinct_document_ratio = len(set(documents)) / count
    else:
        max_relevance = mean_relevance = min_relevance = 0.0
        max_recall = mean_recall = distinct_document_ratio = 0.0

    # Separated by spaces so adjacent passage boundaries cannot glue two words into
    # one token and hide a genuine query-term match.
    evidence = " ".join(texts)
    total_chars = float(sum(len(text) for text in texts))

    return np.array(
        [
            max_relevance,
            mean_relevance,
            min_relevance,
            max_recall,
            mean_recall,
            distinct_document_ratio,
            float(count),
            math.log1p(total_chars),
            _query_term_coverage(query, evidence),
            max_passage_coverage(query, texts),
            coverage_spread(query, texts),
            pairwise_redundancy(texts),
        ],
        dtype=np.float64,
    )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function, clipped so extreme logits cannot overflow ``exp``."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


class LogisticGate:
    """A tiny logistic regression over :data:`FEATURE_NAMES`, fit by NumPy only.

    The model is deliberately linear and low-dimensional: with a few hundred
    labels a linear gate generalises, while anything larger would overfit the
    small labelled split this is meant to train on. Inputs are standardised with
    the training statistics before the dot product, which keeps the single
    learning rate meaningful across features on wildly different scales
    (``passage_count`` versus a cosine in ``[0, 1]``).
    """

    FEATURE_NAMES = FEATURE_NAMES

    def __init__(
        self,
        mean: Any = None,
        scale: Any = None,
        coef: Any = None,
        intercept: float = 0.0,
        *,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        self._mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self._scale = None if scale is None else np.asarray(scale, dtype=np.float64)
        self._coef = None if coef is None else np.asarray(coef, dtype=np.float64)
        self._intercept = float(intercept)
        # The columns this gate actually uses, in its own order. A persisted artefact from
        # an earlier build names fewer columns; keeping the names lets it select them from
        # the current vector instead of failing or silently scoring the wrong ones.
        self.feature_names: tuple[str, ...] = (
            tuple(feature_names) if feature_names is not None else FEATURE_NAMES
        )

    # -- training --------------------------------------------------------
    def fit(
        self,
        X: Any,
        y: Any,
        *,
        epochs: int = 4000,
        lr: float = 0.5,
        l2: float = 1e-3,
    ) -> "LogisticGate":
        """Fit by gradient descent on the class-weighted mean log-loss plus L2.

        Class weights are ``n / (2 * count)``, so the average sample weight is one
        and the two classes contribute equally regardless of prevalence. The
        intercept is never penalised. ``coef`` starts at zero and no sampling or
        shuffling takes place, so repeated fits reproduce the same parameters up to
        the platform's floating-point rounding.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        # A bare 1-D X is a single column of samples; a bare y is a flat label list.
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or X.shape[0] == 0 or X.size == 0:
            raise ValueError(f"X must be a non-empty 2-D array, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]} labels")
        if epochs < 1:
            raise ValueError(f"epochs must be positive, got {epochs}")

        classes, counts = np.unique(y, return_counts=True)
        if classes.shape[0] < 2:
            raise ValueError(
                f"y must contain two classes to fit a gate, got only {classes.tolist()}"
            )

        n_samples, n_features = X.shape
        # A matrix whose width matches a prefix of FEATURE_NAMES is named accordingly (so
        # it persists and reloads); anything wider — an experiment over derived columns —
        # gets placeholder names it cannot persist, which is what the load-time name check
        # is for.
        self.feature_names = (
            FEATURE_NAMES[:n_features]
            if n_features <= len(FEATURE_NAMES)
            else tuple(f"feature_{index}" for index in range(n_features))
        )
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        # A constant column has no scale to divide by; leaving it at 1 means it
        # still contributes its (zero-centred) value instead of blowing up.
        scale = np.where(std > 0.0, std, 1.0)
        Z = (X - mean) / scale

        weight_for = {
            cls: float(n_samples) / (classes.shape[0] * count)
            for cls, count in zip(classes.tolist(), counts.tolist())
        }
        sample_weight = np.array([weight_for[value] for value in y.tolist()], dtype=np.float64)

        coef = np.zeros(n_features, dtype=np.float64)
        intercept = 0.0
        for _ in range(int(epochs)):
            logits = (Z * coef).sum(axis=1) + intercept
            error = (_sigmoid(logits) - y) * sample_weight
            # Mean loss over samples; the L2 term is 0.5 * l2 * ||coef||^2, whose
            # gradient is l2 * coef. The intercept has no penalty term. Reductions are
            # elementwise rather than BLAS mat-vecs, so the sum order is fixed and a
            # repeated fit does not depend on BLAS threading.
            gradient = (Z * error[:, None]).sum(axis=0) / n_samples
            coef -= lr * (gradient + l2 * coef)
            intercept -= lr * float(error.mean())

        self._mean = mean
        self._scale = scale
        self._coef = coef
        self._intercept = float(intercept)
        return self

    # -- prediction ------------------------------------------------------
    def _standardize(self, X: Any) -> np.ndarray:
        if self._mean is None or self._scale is None or self._coef is None:
            raise ValueError("LogisticGate is not fitted; call fit() or load() first")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            # One feature vector, not a column of samples.
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"X must be 1-D or 2-D, got shape {X.shape}")
        if X.shape[1] == len(self.feature_names):
            selected = X
        elif X.shape[1] == len(FEATURE_NAMES):
            # A full-width vector from ``extract_features``: take the columns this gate
            # named, in the gate's order.
            indices = [FEATURE_NAMES.index(name) for name in self.feature_names]
            selected = X[:, indices]
        else:
            raise ValueError(
                f"X has {X.shape[1]} features but the gate expects "
                f"{len(self.feature_names)} (or the full {len(FEATURE_NAMES)})"
            )
        if selected.shape[1] != self._coef.shape[0]:
            raise ValueError(
                f"X has {selected.shape[1]} features but the gate expects {self._coef.shape[0]}"
            )
        return (selected - self._mean) / self._scale

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return ``P(answerable)`` for each row of ``X`` as a 1-D array."""
        Z = self._standardize(X)
        if Z.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        # Elementwise sum rather than ``Z @ coef``: the summation order is fixed, so a
        # save/load round trip reproduces predictions to floating-point rounding.
        return _sigmoid((Z * self._coef).sum(axis=1) + self._intercept)

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        """Return a boolean ``P(answerable) >= threshold`` for each row of ``X``."""
        return self.predict_proba(X) >= float(threshold)

    # -- persistence -----------------------------------------------------
    def save(self, path: Any) -> None:
        """Write the fitted parameters to ``path`` as JSON, atomically."""
        if self._mean is None or self._scale is None or self._coef is None:
            raise ValueError("LogisticGate is not fitted; nothing to save")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": GATE_VERSION,
            "feature_names": list(self.feature_names),
            "mean": self._mean.tolist(),
            "scale": self._scale.tolist(),
            "coef": self._coef.tolist(),
            "intercept": self._intercept,
        }
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(target)

    @classmethod
    def load(cls, path: Any) -> "LogisticGate":
        """Read a persisted gate, rejecting a stale or mismatched artefact."""
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        version = payload.get("version")
        if version != GATE_VERSION:
            raise ValueError(
                f"answerability gate version {version!r} is unsupported "
                f"(expected {GATE_VERSION}); retrain the gate"
            )
        names = payload.get("feature_names")
        known = set(FEATURE_NAMES)
        if names is None or not names or any(name not in known for name in names):
            unknown = [name for name in (names or []) if name not in known]
            raise ValueError(
                "answerability gate feature names are not known to this build "
                f"({unknown or names!r}); retrain the gate"
            )
        if len(set(names)) != len(names):
            raise ValueError("answerability gate feature names contain duplicates; retrain")
        gate = cls(
            mean=payload["mean"],
            scale=payload["scale"],
            coef=payload["coef"],
            intercept=payload.get("intercept", 0.0),
            feature_names=names,
        )
        width = len(names)
        if any(array.shape != (width,) for array in (gate._mean, gate._scale, gate._coef)):
            raise ValueError(
                f"answerability gate parameters are not {width}-dimensional; "
                "the artefact is corrupt"
            )
        return gate

    def __repr__(self) -> str:
        fitted = "fitted" if self._coef is not None else "unfitted"
        return f"LogisticGate(features={len(self.feature_names)}, {fitted})"


def roc_auc(y_true: Any, scores: Any) -> float:
    """Tie-averaged rank AUC (Mann-Whitney U), 0.5 when one class is absent.

    ``scores`` are treated as an answerability signal and the larger label is the
    positive class. Ties receive the average of the ranks they span, which is
    exactly the 0.5 credit the rank-sum formula needs to stay in ``[0, 1]``.
    """
    y = np.asarray(y_true).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y.shape[0] != s.shape[0]:
        raise ValueError(f"got {y.shape[0]} labels for {s.shape[0]} scores")
    n = y.shape[0]
    if n == 0:
        return 0.5

    classes = np.unique(y)
    if classes.shape[0] < 2:
        return 0.5
    positive = y == classes[-1]
    n_pos = int(np.count_nonzero(positive))
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Average ranks over ties: sort, find each run of equal scores, and assign the
    # midpoint of the ranks that run occupies.
    order = np.argsort(s, kind="stable")
    ordered = s[order]
    ranks = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        end = start + 1
        while end < n and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end + 1)
        start = end

    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate_gate(y_true: Any, scores: Any, threshold: float = 0.5) -> dict[str, float]:
    """Summarise a gate's ranking and thresholded behaviour on labelled data.

    ``balanced_accuracy`` is the mean of per-class recall, so it does not reward
    a gate that simply predicts the majority class.
    """
    y = np.asarray(y_true).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y.shape[0] != s.shape[0]:
        raise ValueError(f"got {y.shape[0]} labels for {s.shape[0]} scores")
    predictions = s >= float(threshold)
    classes = np.unique(y)
    positive_label = classes[-1] if classes.shape[0] else 0.0
    # Recall for a class is the fraction of its members the gate labels as that
    # class, so the negative class contributes its predicted-negative rate (TNR).
    recalls = [
        float(np.mean(predictions[y == label] == (label == positive_label)))
        for label in classes
    ]
    balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0
    return {
        "auc": float(roc_auc(y, s)),
        "balanced_accuracy": balanced_accuracy,
        "n": float(y.shape[0]),
    }


# ── natural-language-inference answerability ─────────────────────────────
#
# A second, model-based path to the same decision. Laya's scalar and the feature gate
# both decide from retrieval/evidence signals; an NLI cross-encoder asks the sharper
# question "does this evidence entail an answer to the query?" and can be swapped per
# domain by pointing ``LAYA_RAG_ANSWERABILITY_NLI_MODEL`` at a checkpoint. It is a
# different model with a different calibration, so it is an explicit alternative, never
# a silent default.

#: Label spellings that mean "entailed" across the common NLI checkpoint families.
_ENTAILMENT_MARKERS = ("entail", "label_2", "true")
_CONTRADICTION_MARKERS = ("contradict", "label_0")


def _resolve_contradiction_index(id2label: Mapping[Any, Any]) -> int | None:
    """Pick the contradiction logit index, or ``None`` when the head has no such column.

    ``None`` rather than a guess: a single-label head scores entailment only, and inventing a
    contradiction column for it would turn "the corpus is silent about this" into "the corpus
    refutes this", which is the opposite claim.
    """
    if id2label:
        for index, label in id2label.items():
            if any(marker in str(label).strip().lower() for marker in _CONTRADICTION_MARKERS):
                return int(index)
        if len(id2label) == 1:
            return None
    return 0


def _resolve_entailment_index(id2label: Mapping[Any, Any]) -> int:
    """Pick the entailment logit index from a model's ``id2label`` map.

    Falls back by arity: 3-way NLI is ``(contradiction, neutral, entailment)`` and 2-way
    is ``(not-entailment, entailment)``, so the last index is the entailment in both. A
    single-label head is treated as the entailment score.
    """
    if id2label:
        for index, label in id2label.items():
            text = str(label).strip().lower()
            if any(marker in text for marker in _ENTAILMENT_MARKERS):
                return int(index)
        if len(id2label) == 1:
            return int(next(iter(id2label)))
    return 2


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def _passage_text(passage: Any) -> str:
    """Text of a passage whether it is a ScoredPassage, a Hit, or a mapping.

    ``ScoredPassage`` and JSON rows carry ``text`` directly; a retrieval ``Hit`` carries
    it under ``chunk``. Both must read the same, so an NLI scorer cannot silently see an
    empty premise for one of them.
    """
    text = _field(passage, "text", None)
    if text:
        return _as_text(text)
    chunk = _field(passage, "chunk", None)
    if chunk is not None:
        return _as_text(_field(chunk, "text", ""))
    return ""


class NLIAnswerability:
    """Score ``P(evidence entails an answer to query)`` with an NLI cross-encoder.

    ``scorer`` is an injection seam for tests (and for callers that already have a
    model): when given, it is called as ``scorer(premise, hypothesis) -> float`` and no
    weights are loaded. Production use lazily loads ``model_id`` on first score.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "auto",
        max_length: int = 512,
        scorer: Any | None = None,
        verdict_scorer: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_length = int(max_length)
        self._scorer = scorer
        #: Entailment-only seam. `verdict` needs both columns, so it takes its own seam rather
        #: than treating `1 - entailment` as a contradiction, which is not what an NLI head says.
        self._verdict_scorer = verdict_scorer
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device = device
        self.entailment_index = 2
        self.contradiction_index: int | None = 0

    def ensure_loaded(self) -> "NLIAnswerability":
        if self._scorer is not None or self._verdict_scorer is not None or self._model is not None:
            return self
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "an NLI answerability model needs torch and transformers; install the "
                "engine's default dependencies"
            ) from exc
        device = self.device
        if device in ("auto", ""):
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = (
            AutoModelForSequenceClassification.from_pretrained(self.model_id).to(device).eval()
        )
        self._torch = torch
        self._device = device
        self.entailment_index = _resolve_entailment_index(
            getattr(self._model.config, "id2label", {}) or {}
        )
        self.contradiction_index = _resolve_contradiction_index(
            getattr(self._model.config, "id2label", {}) or {}
        )
        return self

    def verdict(self, query: str, passages: Sequence[Any]) -> dict[str, float]:
        """Both NLI columns for the (evidence, claim) pair: entailment and contradiction.

        ``score`` answers "does the evidence entail an answer to this?"; a claim the corpus
        *refutes* and one the corpus is merely *silent* about both score low on entailment, and
        only the contradiction column tells them apart.
        """
        premise = "\n\n".join(_passage_text(passage) for passage in passages)
        if not premise.strip():
            return {"entailment": 0.0, "contradiction": 0.0}
        if self._verdict_scorer is not None:
            columns = self._verdict_scorer(premise, str(query)) or {}
            return {
                "entailment": float(columns.get("entailment", 0.0)),
                "contradiction": float(columns.get("contradiction", 0.0)),
            }
        if self._scorer is not None:
            # The entailment-only seam has no contradiction column; report it as unknown.
            return {"entailment": float(self._scorer(premise, str(query))), "contradiction": 0.0}
        self.ensure_loaded()
        torch = self._torch
        assert torch is not None
        encoded = self._tokenizer(
            premise, str(query), truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**encoded, return_dict=True).logits.float().cpu().numpy()
        probabilities = _softmax(np.asarray(logits, dtype=np.float64)).reshape(-1)
        entailment = float(probabilities[min(self.entailment_index, probabilities.shape[-1] - 1)])
        contradiction = 0.0
        if self.contradiction_index is not None:
            contradiction = float(
                probabilities[min(self.contradiction_index, probabilities.shape[-1] - 1)]
            )
        return {"entailment": entailment, "contradiction": contradiction}

    def score(self, query: str, passages: Sequence[Any]) -> float:
        """Return ``P(entailment)`` for the query against the assembled evidence."""
        premise = "\n\n".join(_passage_text(passage) for passage in passages)
        if not premise.strip():
            return 0.0
        if self._scorer is not None:
            return float(self._scorer(premise, str(query)))
        self.ensure_loaded()
        torch = self._torch
        assert torch is not None
        encoded = self._tokenizer(
            premise,
            str(query),
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**encoded, return_dict=True).logits.float().cpu().numpy()
        probabilities = _softmax(np.asarray(logits, dtype=np.float64))
        index = min(self.entailment_index, probabilities.shape[-1] - 1)
        return float(probabilities.reshape(-1)[index])

    def info(self) -> dict[str, Any]:
        return {
            "kind": "nli-answerability",
            "model": self.model_id,
            "loaded": self._model is not None or self._scorer is not None,
        }
