"""
Pure aggregation helpers for the RAGAS baseline (used by evaluate.py).

Kept dependency-free (no ragas / datasets / app imports) so the baseline
scoring + artifact logic stays unit-testable without API keys or a live model.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

FAITHFULNESS_TARGET = 0.75
ANSWER_RELEVANCY_TARGET = 0.80


def per_route_breakdown(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Count ok / errored queries per route type (SQL / DOCUMENTS / HYBRID / AGENT)."""
    breakdown: dict[str, dict[str, int]] = {}
    for qt in Counter(r["query_type"] for r in results):
        group = [r for r in results if r["query_type"] == qt]
        errored = sum(1 for r in group if r["error"])
        breakdown[qt] = {"total": len(group), "ok": len(group) - errored, "errored": errored}
    return breakdown


def thresholds_met(scores: dict[str, Any]) -> bool:
    """True only if RAGAS ran and both targets are met.

    Unscored runs (no API keys / no retrieval queries) are not 'met' — the
    baseline gate must fail loudly rather than silently pass on missing data.
    """
    if "faithfulness" not in scores or "answer_relevancy" not in scores:
        return False
    return (
        scores["faithfulness"] > FAITHFULNESS_TARGET
        and scores["answer_relevancy"] > ANSWER_RELEVANCY_TARGET
    )


def build_metrics(results: list[dict[str, Any]], scores: dict[str, Any]) -> dict[str, Any]:
    """Compact, pinnable baseline artifact (separate from verbose query dump)."""
    return {
        "evaluation_date": datetime.utcnow().isoformat(),
        "total_queries": len(results),
        "evaluated_queries": scores.get("evaluated_queries", 0),
        "faithfulness": scores.get("faithfulness"),
        "answer_relevancy": scores.get("answer_relevancy"),
        "faithfulness_target": FAITHFULNESS_TARGET,
        "answer_relevancy_target": ANSWER_RELEVANCY_TARGET,
        "thresholds_met": thresholds_met(scores),
        "agent_validation": scores.get("agent_validation"),
        "per_route": scores.get("per_route", per_route_breakdown(results)),
        "ragas_error": scores.get("ragas_error"),
    }
