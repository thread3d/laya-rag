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
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Sequence


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
