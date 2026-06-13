"""
Offline tests for the RAGAS evaluation aggregation helpers (evaluate.py).

These cover the pure scoring / artifact-building logic without any live
OpenAI / Pinecone / DB calls, so the baseline harness stays trustworthy even
though the full RAGAS run requires API keys + seeded data.
"""

from eval_metrics import (
    ANSWER_RELEVANCY_TARGET,
    FAITHFULNESS_TARGET,
    build_metrics,
    per_route_breakdown,
    thresholds_met,
)


def _results():
    return [
        {"query_type": "SQL", "error": None},
        {"query_type": "SQL", "error": "boom"},
        {"query_type": "DOCUMENTS", "error": None},
        {"query_type": "AGENT", "error": None},
    ]


def test_per_route_breakdown_counts_ok_and_errored():
    bd = per_route_breakdown(_results())
    assert bd["SQL"] == {"total": 2, "ok": 1, "errored": 1}
    assert bd["DOCUMENTS"] == {"total": 1, "ok": 1, "errored": 0}
    assert bd["AGENT"] == {"total": 1, "ok": 1, "errored": 0}


def test_thresholds_met_true_when_both_above_target():
    scores = {
        "faithfulness": FAITHFULNESS_TARGET + 0.05,
        "answer_relevancy": ANSWER_RELEVANCY_TARGET + 0.05,
    }
    assert thresholds_met(scores) is True


def test_thresholds_met_false_when_one_below():
    scores = {"faithfulness": 0.9, "answer_relevancy": ANSWER_RELEVANCY_TARGET - 0.1}
    assert thresholds_met(scores) is False


def test_thresholds_met_false_when_unscored():
    # No RAGAS run (no keys / no retrieval queries) must not silently pass.
    assert thresholds_met({"evaluated_queries": 0}) is False


def test_build_metrics_shape_and_gate():
    scores = {
        "faithfulness": 0.81,
        "answer_relevancy": 0.85,
        "evaluated_queries": 1,
    }
    m = build_metrics(_results(), scores)
    assert m["total_queries"] == 4
    assert m["faithfulness"] == 0.81
    assert m["thresholds_met"] is True
    assert m["faithfulness_target"] == FAITHFULNESS_TARGET
    assert "SQL" in m["per_route"]
    assert "evaluation_date" in m


def test_build_metrics_unscored_run_not_met():
    m = build_metrics(_results(), {"evaluated_queries": 0})
    assert m["faithfulness"] is None
    assert m["thresholds_met"] is False
