"""
Unit tests for the rule-based query router (app/services/router_service.py).

Covers the routing contract advertised in the README: questions are routed to
SQL / DOCUMENTS / HYBRID / AGENT, and AGENT questions dispatch to the correct
quality workflow. Pure functions — no API keys or network required.
"""

import pytest

from app.services.router_service import QueryRouter, detect_agent_workflow


@pytest.mark.parametrize(
    "question, expected",
    [
        ("How many CAPAs are open past their due date?", "SQL"),
        ("Which suppliers have more than 5 NCRs this year?", "SQL"),
        ("Average Cpk ranking by category", "SQL"),
    ],
)
def test_routes_structured_counts_to_sql(question, expected):
    assert QueryRouter.route(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "What is the containment procedure in the QMS manual?",
        "Explain the PFMEA process step for torque verification",
        "Define APQP",
    ],
)
def test_routes_policy_questions_to_documents(question):
    assert QueryRouter.route(question) == "DOCUMENTS"


@pytest.mark.parametrize(
    "question",
    [
        "Do a 5-why analysis for recurring torque failures at Station 12",
        "Build a fishbone for fastener torque failures",
        "Draft CAPA for weld delamination",
        "Generate 8D for the Station 12 torque issue",
    ],
)
def test_routes_agent_phrases_to_agent(question):
    assert QueryRouter.route(question) == "AGENT"


def test_hybrid_when_sql_and_doc_signals_combine():
    q = "Show defect counts and explain the containment procedure"
    assert QueryRouter.route(q) == "HYBRID"


def test_unmatched_question_defaults_to_documents():
    assert QueryRouter.route("Tell me something interesting") == "DOCUMENTS"


@pytest.mark.parametrize(
    "question, workflow",
    [
        ("draw a fishbone diagram", "fishbone"),
        ("ishikawa for paint defects", "fishbone"),
        ("draft 8d for the recall", "8d"),
        ("draft capa for supplier issue", "capa"),
        ("5-why on the leak", "five_why"),
        ("root cause analysis for the recurring fault", "five_why"),
    ],
)
def test_detect_agent_workflow(question, workflow):
    assert detect_agent_workflow(question) == workflow


def test_routing_confidence_reports_route_and_scores():
    info = QueryRouter.get_routing_confidence("How many open NCRs are there?")
    assert info["route"] == "SQL"
    assert info["keyword_matches"]["sql_keywords"] >= 1
    assert 0.0 <= info["confidence_scores"]["sql"] <= 1.0


def test_route_is_case_insensitive():
    assert QueryRouter.route("HOW MANY CAPAS ARE OPEN?") == "SQL"
