"""Evaluation harness for the Laya RAG engine.

The unit suite proves plumbing; this module exists to answer the two questions it
deliberately does not:

* **ability** — does hybrid recall actually find the labelled passage, does Laya's
  relevance decision separate it from the negatives by a usable margin, and are the
  shipped thresholds defensible on a labelled set?
* **limits** — at what evidence size, batch shape, corpus size and concurrency does
  the pipeline stop behaving, and which stage costs what?

Everything here is pure and offline, so the metrics can be unit-tested with no
checkpoints. ``tools/eval.py`` and ``tools/bench.py`` drive it against real weights.

Two corpora are defined. ``planted`` is generated here so every query has exactly one
unambiguous gold file and the unanswerable questions are provably absent from the
text; ``docs`` labels queries against the vendored Laya documentation, which is the
corpus this repository actually indexes by default.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence


# ── queries ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EvalQuery:
    """One labelled query: what to ask, what should be found, and whether it is fair."""

    id: str
    query: str
    gold: tuple[str, ...] = ()
    answerable: bool = True
    category: str = "topical"

    def matches(self, path: str) -> bool:
        """Whether an indexed chunk path is one of this query's gold files."""
        return any(fragment in path for fragment in self.gold)


def parse_queries(text: str) -> list[EvalQuery]:
    """Parse a JSONL query set, one object per line, ``#`` comments ignored."""
    out: list[EvalQuery] = []
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {number}: not JSON ({exc})") from exc
        out.append(
            EvalQuery(
                id=str(row["id"]),
                query=str(row["query"]),
                gold=tuple(row.get("gold") or ()),
                answerable=bool(row.get("answerable", True)),
                category=str(row.get("category", "topical")),
            )
        )
    return out


def dump_queries(queries: Sequence[EvalQuery]) -> str:
    """Serialise a query set back to JSONL."""
    return "\n".join(
        json.dumps(
            {
                "id": q.id,
                "query": q.query,
                "gold": list(q.gold),
                "answerable": q.answerable,
                "category": q.category,
            },
            ensure_ascii=False,
        )
        for q in queries
    )


# ── corpora ──────────────────────────────────────────────────────────────

def planted_corpus() -> dict[str, str]:
    """A small corpus with one unambiguous fact per file and no accidental overlap.

    Deliberately mixed: prose policies, rare identifiers (``whsec_``, ``SH-``), code
    and CLI instructions, two non-English files, and one file that is topical noise
    for every query so an unanswerable question still has candidates to reject.
    """
    return {
        "billing/refunds.md": (
            "# Refund policy\n\n"
            "Refunds are processed within five business days to the original payment method. "
            "If the refund has not arrived after ten business days, contact billing support.\n\n"
            "Partial refunds are available for annual plans within thirty days of purchase. "
            "Monthly plans are not eligible for partial refunds.\n"
        ),
        "billing/invoices.md": (
            "# Invoices\n\n"
            "Invoices are emailed on the first business day of each month. Every invoice carries "
            "a VAT identifier in the format `EU-XX-000000000` when the account has one on file.\n\n"
            "Invoice PDFs stay downloadable for seven years from the billing history page.\n"
        ),
        "shipping/rates.md": (
            "# Shipping rates\n\n"
            "Shipping is free on orders over fifty dollars in the continental United States. "
            "International shipping is calculated at checkout by destination and weight.\n\n"
            "Oversized items ship by freight and cannot be expedited.\n"
        ),
        "shipping/tracking.md": (
            "# Tracking\n\n"
            "Every parcel gets a tracking number in the format `SH-0000-XX`. The carrier receives "
            "the parcel within forty-eight hours of the order leaving the warehouse.\n\n"
            "Tracking events can lag by up to six hours before they appear in the interface.\n"
        ),
        "security/passwords.md": (
            "# Passwords\n\n"
            "Reset your password from Settings, then Security, then Reset Password. Passwords must "
            "be at least twelve characters long.\n\n"
            "An account locks after five failed sign-in attempts and unlocks automatically after "
            "fifteen minutes.\n"
        ),
        "security/two_factor.md": (
            "# Two-factor authentication\n\n"
            "Two-factor authentication supports authenticator applications and hardware security "
            "keys. Ten single-use recovery codes are issued when it is enabled.\n\n"
            "Administrators can require two-factor authentication for an entire workspace.\n"
        ),
        "api/rate_limits.md": (
            "# API rate limits\n\n"
            "The API allows 600 requests per minute per API key. A burst allowance of 100 requests "
            "is available above that ceiling.\n\n"
            "Exceeding the limit returns HTTP 429 with a `Retry-After` header in seconds.\n"
        ),
        "api/webhooks.md": (
            "# Webhooks\n\n"
            "Every webhook is signed with a per-endpoint secret that begins with `whsec_`. Verify "
            "the signature before trusting a payload.\n\n"
            "Failed deliveries are retried five times with exponential backoff, then parked.\n"
        ),
        "deploy/docker.md": (
            "# Docker deployment\n\n"
            "Run the service without installing Python on the host:\n\n"
            "```bash\ndocker compose run --build --rm laya\n```\n\n"
            "Allow 8 GB of RAM and 10 GB of free disk for the CPU image.\n"
        ),
        "deploy/kubernetes.md": (
            "# Kubernetes deployment\n\n"
            "The Helm chart is named `laya-rag` and runs three replicas by default. Its readiness "
            "probe calls the `/health` endpoint on port 8077.\n\n"
            "Set `resources.limits.memory` to at least 4Gi for the multilingual checkpoint.\n"
        ),
        "i18n/de_hilfe.md": (
            "# Rückerstattungen\n\n"
            "Eine Rückerstattung wird innerhalb von fünf Werktagen auf die ursprüngliche "
            "Zahlungsmethode zurückgebucht.\n\n"
            "Bei Jahresplänen ist eine teilweise Rückerstattung innerhalb von dreißig Tagen möglich.\n"
        ),
        "i18n/es_ayuda.md": (
            "# Envíos\n\n"
            "El envío es gratuito para pedidos superiores a cincuenta dólares en los Estados "
            "Unidos continentales.\n\n"
            "Los envíos internacionales se calculan por destino y peso.\n"
        ),
        "misc/company.md": (
            "# About the company\n\n"
            "The company was founded in 2017 and is headquartered in a converted warehouse. It "
            "publishes an annual letter describing product direction.\n\n"
            "The engineering blog covers release notes and infrastructure notes.\n"
        ),
    }


def planted_queries() -> list[EvalQuery]:
    """Labelled queries over :func:`planted_corpus`, including unanswerable controls."""
    return parse_queries(
        "\n".join([
            '{"id":"q-refund","query":"How long does a refund take?","gold":["billing/refunds.md"],"category":"topical"}',
            '{"id":"q-refund-partial","query":"Can I get a partial refund on an annual plan?","gold":["billing/refunds.md"],"category":"topical"}',
            '{"id":"q-invoice-vat","query":"What format does the VAT identifier use?","gold":["billing/invoices.md"],"category":"exact-term"}',
            '{"id":"q-shipping-free","query":"When is shipping free?","gold":["shipping/rates.md"],"category":"topical"}',
            '{"id":"q-tracking-format","query":"What is the tracking number format?","gold":["shipping/tracking.md"],"category":"exact-term"}',
            '{"id":"q-password-reset","query":"How do I reset my password?","gold":["security/passwords.md"],"category":"topical"}',
            '{"id":"q-lockout","query":"How many failed sign-in attempts lock an account?","gold":["security/passwords.md"],"category":"topical"}',
            '{"id":"q-2fa-hardware","query":"Does two-factor authentication support hardware security keys?","gold":["security/two_factor.md"],"category":"topical"}',
            '{"id":"q-recovery-codes","query":"How many recovery codes are issued?","gold":["security/two_factor.md"],"category":"topical"}',
            '{"id":"q-ratelimit","query":"What is the API rate limit per key?","gold":["api/rate_limits.md"],"category":"exact-term"}',
            '{"id":"q-http429","query":"Which HTTP status is returned when the rate limit is exceeded?","gold":["api/rate_limits.md"],"category":"exact-term"}',
            '{"id":"q-webhook-secret","query":"What prefix does a webhook signing secret have?","gold":["api/webhooks.md"],"category":"exact-term"}',
            '{"id":"q-webhook-retries","query":"How many times are failed webhook deliveries retried?","gold":["api/webhooks.md"],"category":"topical"}',
            '{"id":"q-docker","query":"How do I run Laya in Docker?","gold":["deploy/docker.md"],"category":"code"}',
            '{"id":"q-k8s-probe","query":"Which endpoint does the Kubernetes readiness probe call?","gold":["deploy/kubernetes.md"],"category":"code"}',
            '{"id":"q-de-hilfe","query":"Wie lange dauert eine Rückerstattung?","gold":["i18n/de_hilfe.md"],"category":"multilingual"}',
            '{"id":"q-es-envio","query":"¿Cuánto cuesta el envío internacional?","gold":["i18n/es_ayuda.md"],"category":"multilingual"}',
            '{"id":"n-ceo","query":"What is the chief executive\'s home address?","gold":[],"answerable":false,"category":"unanswerable"}',
            '{"id":"n-headcount","query":"How many employees work in the Reykjavik office?","gold":[],"answerable":false,"category":"unanswerable"}',
            '{"id":"n-crypto-refund","query":"What is the refund policy for cryptocurrency payments?","gold":[],"answerable":false,"category":"unanswerable"}',
            '{"id":"n-mobile-lang","query":"Which programming language is the mobile application written in?","gold":[],"answerable":false,"category":"unanswerable"}',
            '{"id":"n-ceo-salary","query":"What salary does the chief executive receive?","gold":[],"answerable":false,"category":"unanswerable"}',
            # Hard negatives: topically adjacent enough to clear the relevance gate, but
            # the corpus does not actually answer them. Without these the answerability
            # decision is never reached on a negative, and its abstention behaviour is
            # unmeasured. See the "hard-negative" category in the report.
            '{"id":"hn-giftcard","query":"What is the refund policy for gift cards?","gold":[],"answerable":false,"category":"hard-negative"}',
            '{"id":"hn-intl-cost","query":"How much does international shipping cost to Germany?","gold":[],"answerable":false,"category":"hard-negative"}',
            '{"id":"hn-vat-countries","query":"Which countries require a VAT identifier?","gold":[],"answerable":false,"category":"hard-negative"}',
            '{"id":"hn-key-models","query":"Which hardware security key models are supported?","gold":[],"answerable":false,"category":"hard-negative"}',
        ])
    )


def docs_queries() -> list[EvalQuery]:
    """Labelled queries over the vendored Laya documentation this repo indexes by default."""
    return parse_queries(
        "\n".join([
            '{"id":"d-docker","query":"How do I run Laya in Docker?","gold":["docs/docker.md"],"category":"topical"}',
            '{"id":"d-langchain","query":"How do I use Laya with LangChain?","gold":["docs/langchain.md"],"category":"topical"}',
            '{"id":"d-structured","query":"How does structured output with Pydantic work?","gold":["docs/structured.md"],"category":"topical"}',
            '{"id":"d-hooks","query":"What hooks does Laya expose for a prediction?","gold":["docs/hooks"],"category":"topical"}',
            '{"id":"d-evals","query":"How do I evaluate a Laya model?","gold":["docs/evals.md"],"category":"topical"}',
            '{"id":"d-benchmarks","query":"What do the benchmark results show?","gold":["docs/benchmarks.md"],"category":"topical"}',
            '{"id":"d-adoption","query":"What is staged adoption?","gold":["docs/staged-adoption.md"],"category":"topical"}',
            '{"id":"dn-gpu-vendor","query":"Which GPU vendor does the company recommend in Iceland?","gold":[],"answerable":false,"category":"unanswerable"}',
            '{"id":"dn-pricing","query":"How much does a Laya enterprise licence cost per year?","gold":[],"answerable":false,"category":"unanswerable"}',
        ])
    )


# ── ranking metrics ──────────────────────────────────────────────────────

def first_match_rank(paths: Sequence[str], gold: Sequence[str]) -> int | None:
    """1-based rank of the first path matching ``gold``, or ``None`` if absent."""
    if not gold:
        return None
    for position, path in enumerate(paths, start=1):
        if any(fragment in path for fragment in gold):
            return position
    return None


def rank_summary(ranks: Sequence[int | None], k: int) -> dict[str, Any]:
    """Recall, MRR, P@k and rank spread for one retrieval channel."""
    total = len(ranks)
    found = [rank for rank in ranks if rank is not None]
    return {
        "n": total,
        "found": len(found),
        "recall": _round(len(found) / total) if total else 0.0,
        "mrr": _round(sum(1.0 / rank for rank in found) / total) if total else 0.0,
        f"precision@{k}": _round(sum(1 for rank in found if rank <= k) / total) if total else 0.0,
        "median_rank": median(found) if found else None,
        "worst_rank": max(found) if found else None,
    }


def margin_summary(margins: Sequence[float | None]) -> dict[str, Any]:
    """How far the gold passage's relevance sits above the best negative, per query."""
    present = [m for m in margins if m is not None]
    if not present:
        return {"n": 0, "mean": None, "median": None, "min": None, "negative": 0}
    return {
        "n": len(present),
        "mean": _round(sum(present) / len(present)),
        "median": _round(median(present)),
        "min": _round(min(present)),
        "negative": sum(1 for m in present if m < 0),
    }


# ── calibration ──────────────────────────────────────────────────────────

def reliability_bins(
    probabilities: Sequence[float], labels: Sequence[int], n_bins: int = 10
) -> list[dict[str, Any]]:
    """Bucket ``(probability, outcome)`` pairs and report confidence vs accuracy.

    A well-calibrated probability of 0.8 should be right about 80% of the time, so
    ``gap`` is the distance between the two. This is the check behind the project's
    claim that its thresholds are thresholdable at all.
    """
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for probability, label in zip(probabilities, labels):
        index = min(n_bins - 1, max(0, int(float(probability) * n_bins)))
        buckets[index].append((float(probability), int(label)))
    out: list[dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        confidence = sum(p for p, _ in bucket) / len(bucket)
        accuracy = sum(y for _, y in bucket) / len(bucket)
        out.append({
            "bin": index,
            "range": f"{index / n_bins:.1f}-{(index + 1) / n_bins:.1f}",
            "count": len(bucket),
            "confidence": _round(confidence),
            "accuracy": _round(accuracy),
            "gap": _round(abs(accuracy - confidence)),
        })
    return out


def expected_calibration_error(
    probabilities: Sequence[float], labels: Sequence[int], n_bins: int = 10
) -> float:
    """Count-weighted mean confidence/accuracy gap, in [0, 1]. Lower is better."""
    total = len(probabilities)
    if not total:
        return 0.0
    return _round(
        sum(bin_["count"] / total * bin_["gap"] for bin_ in reliability_bins(probabilities, labels, n_bins))
    )


def threshold_sweep(
    probabilities: Sequence[float],
    labels: Sequence[int],
    thresholds: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Micro-averaged precision/recall/F1 for every candidate decision threshold."""
    if thresholds is None:
        thresholds = [round(step / 20, 2) for step in range(1, 20)]
    out: list[dict[str, Any]] = []
    for threshold in thresholds:
        true_positive = false_positive = false_negative = true_negative = 0
        for probability, label in zip(probabilities, labels):
            predicted = float(probability) >= threshold
            if predicted and label:
                true_positive += 1
            elif predicted and not label:
                false_positive += 1
            elif not predicted and label:
                false_negative += 1
            else:
                true_negative += 1
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.append({
            "threshold": threshold,
            "precision": _round(precision),
            "recall": _round(recall),
            "f1": _round(f1),
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "tn": true_negative,
        })
    return out


# ── selective prediction ─────────────────────────────────────────────────
#
# AUC and one-threshold balanced accuracy say nothing about the risk actually accepted
# at a deployable coverage. These treat the gate as a selective predictor: sort by score,
# accept a fraction, and measure the error rate among the accepted. `aurc` is the area
# under that curve, and `augrc` its generalization (NeurIPS 2024), which is the expected
# risk of undetected failures and does not credit coverage that is never taken.

def _positive_label(labels: Sequence[int]) -> int:
    return max(int(label) for label in labels) if labels else 1


def risk_coverage_curve(
    probabilities: Sequence[float], labels: Sequence[int]
) -> dict[str, list[float]]:
    """Risk (error rate among accepted) at every achievable coverage level.

    One point per query, from accepting only the highest-scored item to accepting all.
    ``risk`` is the fraction of accepted items whose label is *not* the positive class,
    so for the answerability gate it is the false-accept rate at that coverage.
    """
    if not probabilities:
        return {"coverage": [], "risk": []}
    positive = _positive_label(labels)
    order = sorted(range(len(probabilities)), key=lambda i: -float(probabilities[i]))
    ordered = [int(labels[i]) for i in order]
    n = len(ordered)
    negative = 0
    coverage: list[float] = []
    risk: list[float] = []
    for accepted, label in enumerate(ordered, start=1):
        if label != positive:
            negative += 1
        coverage.append(accepted / n)
        risk.append(negative / accepted)
    return {"coverage": [round(c, 6) for c in coverage], "risk": [round(r, 6) for r in risk]}


def _trapezoid(xs: Sequence[float], ys: Sequence[float]) -> float:
    return sum(
        (xs[i + 1] - xs[i]) * (ys[i + 1] + ys[i]) / 2.0 for i in range(len(xs) - 1)
    )


def aurc(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the risk-coverage curve; lower is better. 0.0 for an empty input."""
    curve = risk_coverage_curve(probabilities, labels)
    if len(curve["coverage"]) < 2:
        return 0.0
    return round(_trapezoid(curve["coverage"], curve["risk"]), 4)


def augrc(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the generalized risk-coverage curve (NeurIPS 2024); lower is better.

    ``GRC(c) = c * risk(c) + (1 - c)`` is the probability of an *undetected* failure at
    coverage ``c``, so unlike AURC it accounts for the cost of the coverage not taken.
    """
    curve = risk_coverage_curve(probabilities, labels)
    if len(curve["coverage"]) < 2:
        return 0.0
    grc = [
        risk_value * coverage + (1.0 - coverage)
        for coverage, risk_value in zip(curve["coverage"], curve["risk"])
    ]
    return round(_trapezoid(curve["coverage"], grc), 4)


def risk_at_coverage(
    probabilities: Sequence[float], labels: Sequence[int], coverage: float
) -> float:
    """False-accept rate when the top ``coverage`` fraction by score is accepted."""
    if not probabilities:
        return 0.0
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0, 1], got {coverage!r}")
    positive = _positive_label(labels)
    order = sorted(range(len(probabilities)), key=lambda i: -float(probabilities[i]))
    accepted = max(1, min(len(order), int(math.ceil(coverage * len(order)))))
    chosen = [int(labels[i]) for i in order[:accepted]]
    return round(sum(1 for label in chosen if label != positive) / accepted, 4)


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    if not probabilities:
        return 0.0
    return round(
        sum((float(p) - float(y)) ** 2 for p, y in zip(probabilities, labels)) / len(probabilities),
        4,
    )


def negative_log_likelihood(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    """Mean log loss, with the probability clipped away from the two poles."""
    if not probabilities:
        return 0.0
    total = 0.0
    for probability, label in zip(probabilities, labels):
        p = min(1.0 - 1e-12, max(1e-12, float(probability)))
        total += -(float(label) * math.log(p) + (1.0 - float(label)) * math.log(1.0 - p))
    return round(total / len(probabilities), 4)


def reliability_curve_equal_mass(
    probabilities: Sequence[float], labels: Sequence[int], n_bins: int
) -> list[dict[str, Any]]:
    """Equal-count bins (rather than equal-width), which are more stable on small sets."""
    if not probabilities or n_bins < 1:
        return []
    order = sorted(range(len(probabilities)), key=lambda i: float(probabilities[i]))
    total = len(order)
    out: list[dict[str, Any]] = []
    for b in range(n_bins):
        start = b * total // n_bins
        end = (b + 1) * total // n_bins
        if start >= end:
            continue
        chunk = order[start:end]
        confidence = sum(float(probabilities[i]) for i in chunk) / len(chunk)
        accuracy = sum(float(labels[i]) for i in chunk) / len(chunk)
        out.append({
            "count": len(chunk),
            "confidence": _round(confidence),
            "accuracy": _round(accuracy),
            "gap": _round(abs(accuracy - confidence)),
        })
    return out


def ece_sweep(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    min_bins: int = 2,
    max_bins: int = 30,
) -> dict[str, Any]:
    """ECE over a range of equal-mass bin counts, preferring a monotone curve.

    The bin count is a free parameter that changes the number, so reporting one value
    understates the uncertainty. This reports the whole curve and selects the largest
    bin count whose reliability curve is still non-decreasing in confidence — the finest
    resolution that has not started fitting noise.
    """
    curve: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None
    total = len(probabilities)
    for n_bins in range(max(1, min_bins), max_bins + 1):
        if n_bins > total:
            break
        bins = reliability_curve_equal_mass(probabilities, labels, n_bins)
        if not bins:
            continue
        ece = sum(b["count"] / total * b["gap"] for b in bins)
        accuracies = [b["accuracy"] for b in bins]
        monotone = all(
            accuracies[i + 1] >= accuracies[i] - 1e-9 for i in range(len(accuracies) - 1)
        )
        entry = {"bins": n_bins, "ece": _round(ece), "monotone": monotone}
        curve.append(entry)
        if monotone:
            chosen = entry
    if chosen is None and curve:
        chosen = curve[-1]
    return {
        "ece": chosen["ece"] if chosen else 0.0,
        "bins": chosen["bins"] if chosen else 0,
        "monotone": bool(chosen and chosen["monotone"]),
        "curve": curve,
    }


# ── standard IR metrics (TREC / BEIR) ────────────────────────────────────
#
# These are the metrics an IR leaderboard reports, so a number computed here can be
# read beside a published one. They are macro-averaged over queries, use graded
# relevance with the exponential gain convention (2**rel - 1), and are deliberately
# separate from the single-gold metrics above, which answer a different question.

def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    """Discounted cumulative gain of an already-ordered list of graded relevances."""
    return sum(
        (2.0 ** float(relevance) - 1.0) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances[:k], start=1)
    )


def ndcg_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    """nDCG@k against graded relevance; 0.0 when the query has no relevant document."""
    gains = [float(relevant.get(doc, 0.0)) for doc in ranked[:k]]
    ideal = sorted((float(v) for v in relevant.values() if v > 0), reverse=True)[:k]
    ideal_gain = dcg_at_k(ideal, k)
    return dcg_at_k(gains, k) / ideal_gain if ideal_gain > 0 else 0.0


def mrr_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    """Reciprocal rank of the first relevant document within the top ``k``."""
    for rank, doc in enumerate(ranked[:k], start=1):
        if float(relevant.get(doc, 0.0)) > 0:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    """Fraction of the query's relevant documents that appear in the top ``k``."""
    total = sum(1 for value in relevant.values() if value > 0)
    if not total:
        return 0.0
    return sum(1 for doc in ranked[:k] if float(relevant.get(doc, 0.0)) > 0) / total


def average_precision_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    """Average precision over the top ``k``, normalized by all relevant documents."""
    total = sum(1 for value in relevant.values() if value > 0)
    if not total:
        return 0.0
    hits = 0
    running = 0.0
    for rank, doc in enumerate(ranked[:k], start=1):
        if float(relevant.get(doc, 0.0)) > 0:
            hits += 1
            running += hits / rank
    return running / total


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, float]],
    k_values: Sequence[int] = (10, 100),
    *,
    ignore_identical_ids: bool = True,
) -> dict[str, Any]:
    """Macro-average standard IR metrics over every query that has a relevant document.

    Queries with no relevant document are skipped rather than scored zero, which is
    what ``pytrec_eval`` and the BEIR scripts do; scoring them zero would drag the mean
    down for reasons that have nothing to do with the retriever.

    ``ignore_identical_ids`` mirrors BEIR's ``EvaluateRetrieval`` argument of the same
    name, which defaults to ``True``: a retrieved document whose id equals the query id
    is dropped before scoring. It is a no-op on every BEIR corpus except ArguAna, whose
    queries *are* corpus documents -- 1,298 of its 1,406 test queries share an ``_id``
    with a document and all 1,406 share the text. A retriever therefore returns the
    query itself at rank 1 for every ArguAna query unless it is removed. BEIR removes
    it, so published ArguAna scores assume it is gone; leaving it in caps a
    single-relevant-document query at ``1/log2(3) = 0.631`` and cost this project
    0.10-0.19 nDCG@10 on that dataset alone.
    """
    per_query: list[dict[str, Any]] = []
    for query_id, relevant in qrels.items():
        if not any(value > 0 for value in relevant.values()):
            continue
        ranked = list(rankings.get(query_id, ()))
        if ignore_identical_ids:
            ranked = [doc for doc in ranked if doc != query_id]
        row: dict[str, Any] = {"query": query_id}
        for k in k_values:
            row[f"nDCG@{k}"] = ndcg_at_k(ranked, relevant, k)
            row[f"Recall@{k}"] = recall_at_k(ranked, relevant, k)
        row["MRR@10"] = mrr_at_k(ranked, relevant, 10)
        row["MAP@100"] = average_precision_at_k(ranked, relevant, 100)
        per_query.append(row)

    summary: dict[str, float] = {}
    if per_query:
        for key in per_query[0]:
            if key == "query":
                continue
            summary[key] = round(sum(row[key] for row in per_query) / len(per_query), 4)
    return {"queries": len(per_query), "summary": summary, "per_query": per_query}


# ── significance ─────────────────────────────────────────────────────────

def per_query_metric(per_query: Sequence[Mapping[str, Any]], metric: str) -> dict[str, float]:
    """``query_id -> metric`` for one metric, ready for :func:`paired_bootstrap`."""
    return {str(row["query"]): float(row[metric]) for row in per_query if metric in row}


def paired_bootstrap(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    samples: int = 10_000,
    seed: int = 20210617,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Paired bootstrap over per-query scores, to ask whether an A/B difference is real.

    Resampling *queries* rather than runs is what makes this paired: every resample
    scores both systems on the same query draw, so query difficulty cancels and only
    the difference between the systems is left. Reported differences on these datasets
    are routinely 0.004-0.013, which a few hundred queries cannot resolve without an
    interval, so a candidate that moves nDCG@10 by 0.005 is not evidence of anything
    until the interval excludes zero.

    ``p_value`` is the two-sided bootstrap achievement level -- twice the smaller tail
    mass at zero -- a resampling estimate, not an exact test. ``significant`` is the
    interval criterion (the whole interval on one side of zero) and is the one to quote.
    """
    shared = sorted(set(baseline) & set(candidate))
    if not shared:
        raise ValueError("paired_bootstrap needs at least one query scored by both systems")
    deltas = [float(candidate[q]) - float(baseline[q]) for q in shared]
    n = len(deltas)
    alpha = (1.0 - confidence) / 2.0

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a declared dependency
        np = None

    if np is not None:
        values = np.asarray(deltas, dtype=np.float64)
        rng = np.random.default_rng(seed)
        blocks = []
        remaining = samples
        # Chunked so the index matrix stays bounded on the 6k-query corpora.
        while remaining > 0:
            chunk = min(256, remaining)
            blocks.append(values[rng.integers(0, n, size=(chunk, n))].mean(axis=1))
            remaining -= chunk
        means = np.sort(np.concatenate(blocks))
        low = float(np.percentile(means, 100 * alpha))
        high = float(np.percentile(means, 100 * (1 - alpha)))
        below = float((means <= 0.0).mean())
        above = float((means >= 0.0).mean())
    else:  # pragma: no cover - exercised only without numpy
        rng = random.Random(seed)
        draws = []
        for _ in range(samples):
            draws.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
        draws.sort()
        low = draws[max(0, int(alpha * samples) - 1)]
        high = draws[min(samples - 1, int((1 - alpha) * samples))]
        below = sum(1 for value in draws if value <= 0.0) / samples
        above = sum(1 for value in draws if value >= 0.0) / samples

    return {
        "queries": n,
        "delta": round(sum(deltas) / n, 4),
        "ci_low": round(low, 4),
        "ci_high": round(high, 4),
        "p_value": round(min(1.0, 2.0 * min(below, above)), 4),
        "samples": samples,
        "significant": bool(low > 0.0 or high < 0.0),
    }


def permutation_test(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    samples: int = 10_000,
    seed: int = 20210617,
    alpha: float = 0.05,
    exact_max_n: int = 12,
) -> dict[str, Any]:
    """Paired randomization (sign-flip) test over per-query deltas.

    The bootstrap in :func:`paired_bootstrap` estimates an interval; this asks the
    complementary question — under the null that the two systems are interchangeable,
    how often would a mean delta this large arise? Each query's delta is exchangeable in
    sign, so the null distribution is the mean of randomly sign-flipped deltas.

    ``n <= exact_max_n`` enumerates all ``2**n`` sign assignments exactly (4096 at the
    default cut); larger sets draw ``samples`` random flips. The reported ``p_value``
    includes the observed assignment (``+1`` for every query), so it is never zero, and
    the exact variant carries no Monte-Carlo error.

    Both this and the bootstrap need *exchangeable* paired deltas: they are invalid if
    the same queries were used to choose the configuration being tested.
    """
    shared = sorted(set(baseline) & set(candidate))
    if not shared:
        raise ValueError("permutation_test needs at least one query scored by both systems")
    deltas = [float(candidate[q]) - float(baseline[q]) for q in shared]
    n = len(deltas)
    observed = sum(deltas) / n
    tolerance = 1e-12 * max(1.0, abs(observed))

    if n <= exact_max_n:
        from itertools import product

        extreme = 0
        total = 0
        for signs in product((-1.0, 1.0), repeat=n):
            mean = sum(s * d for s, d in zip(signs, deltas)) / n
            if abs(mean) >= abs(observed) - tolerance:
                extreme += 1
            total += 1
        p_value = extreme / total
        method = "exact"
        effective = total
    else:
        import numpy as np

        values = np.asarray(deltas, dtype=np.float64)
        rng = np.random.default_rng(seed)
        extreme = 0
        remaining = samples
        while remaining > 0:
            chunk = min(2048, remaining)
            signs = rng.choice(np.array([-1.0, 1.0]), size=(chunk, n))
            means = (signs * values).mean(axis=1)
            extreme += int(np.count_nonzero(np.abs(means) >= abs(observed) - tolerance))
            remaining -= chunk
        # A finite draw cannot evidence a p below 1/(samples+1); report that floor.
        p_value = (extreme + 1) / (samples + 1)
        method = "approximate"
        effective = samples

    return {
        "queries": n,
        "delta": round(observed, 4),
        "p_value": round(min(1.0, p_value), 4),
        "method": method,
        "draws": effective,
        "significant": bool(p_value <= alpha),
        "alpha": alpha,
    }


def paired_mde(
    deltas: Sequence[float], *, alpha: float = 0.05, power: float = 0.8
) -> float | None:
    """Smallest paired mean difference an n-query set can detect, at the given power.

    Two-sided normal approximation, ``(z_(1-alpha/2) + z_power) * sd(deltas) / sqrt(n)``.
    The SD is the *observed* per-query difference SD, so this answers "what could this
    comparison have resolved?" rather than "how many queries should I have run?" — report
    it as design guidance for the next run, and note it is post-hoc once the effect is
    known. ``None`` below two queries, where no SD exists.
    """
    values = [float(delta) for delta in deltas]
    n = len(values)
    if n < 2:
        return None
    from statistics import NormalDist, stdev

    sd = stdev(values)
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_power = NormalDist().inv_cdf(power)
    return _round((z_alpha + z_power) * sd / math.sqrt(n))


def benjamini_hochberg(
    pvalues: Mapping[str, float],
    *,
    q: float = 0.05,
) -> dict[str, Any]:
    """Step-up FDR control over a named family of p-values.

    A comparison grid tests dozens of (retriever x dataset x metric) cells at once, so
    a per-cell 5% level is not a 5% family error rate. Benjamini-Hochberg controls the
    expected *proportion* of false discoveries instead: sort p ascending, set the
    adjusted value of the i-th to ``min over k >= i of (m/k * p_(k))``, and reject every
    cell whose adjusted value is at or below ``q``. The family must be declared before
    looking at the results, or the denominator m is whatever the analyst chose after the
    fact.
    """
    names = list(pvalues)
    m = len(names)
    if m == 0:
        return {"q": q, "tests": 0, "adjusted": {}, "rejected": [], "any": False}
    order = sorted(names, key=lambda name: float(pvalues[name]))
    adjusted: dict[str, float] = {}
    running = 1.0
    for index in range(m - 1, -1, -1):
        name = order[index]
        value = float(pvalues[name]) * m / (index + 1)
        running = min(running, value)
        adjusted[name] = min(1.0, running)
    rejected = [name for name in names if adjusted[name] <= q]
    return {
        "q": q,
        "tests": m,
        "adjusted": {name: round(value, 4) for name, value in adjusted.items()},
        "rejected": rejected,
        "any": bool(rejected),
    }


def anchor_deviations(
    measured: Mapping[str, float],
    published: Mapping[str, float],
    *,
    tolerance: float = 0.05,
) -> list[dict[str, Any]]:
    """Compare a control retriever's scores against published baselines.

    A benchmark's control row is what keeps its evaluator honest: if plain BM25 lands
    where the published baseline says it should, the corpus, the qrels and the scoring
    are probably right, and a deviation is the harness asking for a look rather than
    the retriever underperforming. This turns that check into a reported one, because
    it was already stated informally and still failed to surface a 0.10 ArguAna gap.
    """
    rows: list[dict[str, Any]] = []
    for name in sorted(set(measured) & set(published)):
        delta = float(measured[name]) - float(published[name])
        rows.append({
            "dataset": name,
            "measured": round(float(measured[name]), 4),
            "published": round(float(published[name]), 4),
            "delta": round(delta, 4),
            "within_tolerance": abs(delta) <= tolerance,
        })
    return rows


# ── reporting ────────────────────────────────────────────────────────────

def _round(value: float | int | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_cell(v) for v in value)
    return str(value)


def markdown_table(rows: Sequence[dict[str, Any]]) -> str:
    """Render a list of flat dicts as one markdown table."""
    if not rows:
        return "_(no rows)_"
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def write_json(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path: Path | str, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def compare_reports(baseline: Any, current: Any, prefix: str = "") -> list[dict[str, Any]]:
    """Flat list of numeric metrics that differ between two report trees."""
    differences: list[dict[str, Any]] = []
    if isinstance(baseline, dict) and isinstance(current, dict):
        for key in baseline:
            if key in current:
                differences.extend(
                    compare_reports(baseline[key], current[key], f"{prefix}.{key}" if prefix else key)
                )
    elif isinstance(baseline, list) and isinstance(current, list) and len(baseline) == len(current):
        for index, (left, right) in enumerate(zip(baseline, current)):
            differences.extend(compare_reports(left, right, f"{prefix}[{index}]"))
    elif isinstance(baseline, (int, float)) and isinstance(current, (int, float)):
        if baseline != current:
            differences.append({
                "metric": prefix,
                "baseline": baseline,
                "current": current,
                "delta": _round(current - baseline),
            })
    return differences
