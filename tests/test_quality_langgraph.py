"""Offline unit tests for app/services/quality_langgraph.py (LLM mocked)."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from app.config import settings
from app.services import quality_langgraph as qlg


def _make_wf(monkeypatch, llm_json, rag_service=None, sql_service=None):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
    wf = qlg.QualityLangGraphWorkflows(rag_service, sql_service)
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps(llm_json)
    wf.client = MagicMock()
    wf.client.chat.completions.create = AsyncMock(return_value=resp)
    return wf


def test_safe_json_loads_bad_returns_empty():
    assert qlg._safe_json_loads("not json{") == {}
    assert qlg._safe_json_loads('{"a": 1}') == {"a": 1}


def test_run_five_why_clamps_confidence(monkeypatch):
    llm_json = {
        "problem_statement": "p",
        "whys": [
            {"level": 1, "why": "w1", "evidence_source": "s"},
            {"level": 2, "why": "w2", "evidence_source": "s"},
            {"level": 3, "why": "w3", "evidence_source": "s"},
        ],
        "root_cause": "rc",
        "recommended_action": "ra",
        "confidence_score": 1.5,
    }
    wf = _make_wf(monkeypatch, llm_json)
    output = asyncio.run(wf.run_five_why("p"))
    assert output["confidence_score"] == 1.0


def test_run_five_why_accepts_handoff_kwargs(monkeypatch):
    llm_json = {"problem_statement": "p", "whys": [], "root_cause": "rc"}
    wf = _make_wf(monkeypatch, llm_json)
    output = asyncio.run(
        wf.run_five_why("p", component="Telematics Control Unit", anomaly_label="cloud_sync")
    )
    assert output == llm_json


def test_run_fishbone_clamps_bone_weights(monkeypatch):
    llm_json = {
        "effect": "e",
        "bones": {
            "Man": [{"cause": "c", "evidence": "ev", "weight": 2.0}],
            "Machine": [],
        },
    }
    wf = _make_wf(monkeypatch, llm_json)
    output = asyncio.run(wf.run_fishbone("e", station="ST1"))
    assert output["bones"]["Man"][0]["weight"] == 1.0


def test_retrieve_sql_blocks_dangerous(monkeypatch):
    sql_service = MagicMock()
    sql_service.generate_sql_for_approval = AsyncMock(
        return_value={"sql": "DROP TABLE ncr", "query_id": "q1"}
    )
    sql_service.execute_approved_query = AsyncMock()
    wf = _make_wf(monkeypatch, {}, sql_service=sql_service)
    result = asyncio.run(wf._retrieve_sql_snippet("q"))
    assert result == ""
    sql_service.execute_approved_query.assert_not_awaited()


def test_retrieve_sql_runs_safe(monkeypatch):
    sql_service = MagicMock()
    sql_service.generate_sql_for_approval = AsyncMock(
        return_value={"sql": "SELECT 1", "query_id": "q1"}
    )
    sql_service.execute_approved_query = AsyncMock(
        return_value={"status": "executed", "results": [{"a": 1}]}
    )
    wf = _make_wf(monkeypatch, {}, sql_service=sql_service)
    result = asyncio.run(wf._retrieve_sql_snippet("q"))
    assert "a" in result
