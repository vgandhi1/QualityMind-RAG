"""
Tests for the CLaimLens -> QualityMind handoff contract on /quality/five-why.

Verifies ProblemBody accepts the full RcaHandoff contract (problem_statement,
part_number, anomaly_label, claim_count) and that part_number / anomaly_label
are threaded into the 5-Why workflow so retrieval can filter by part.
Pure unit test: the LangGraph workflow is mocked, no OpenAI / DB / network.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.quality_routes import ProblemBody, router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_problem_body_accepts_full_handoff_contract():
    body = ProblemBody(
        problem_statement="Recurring sync failure on telematics unit",
        part_number="TCU-0420",
        anomaly_label="cloud_sync",
        claim_count=3,
    )
    assert body.part_number == "TCU-0420"
    assert body.anomaly_label == "cloud_sync"
    assert body.claim_count == 3


def test_problem_body_defaults_contract_fields_optional():
    body = ProblemBody(problem_statement="Recurring sync failure")
    assert body.part_number is None
    assert body.anomaly_label is None
    assert body.claim_count is None


def test_five_why_threads_part_number_to_workflow(client):
    mock_wf = AsyncMock()
    mock_wf.run_five_why.return_value = {
        "problem_statement": "x",
        "whys": [],
        "root_cause": "y",
        "confidence_score": 0.5,
    }

    with patch("app.api.quality_routes._get_workflows_or_503", return_value=mock_wf):
        resp = client.post(
            "/quality/five-why",
            json={
                "problem_statement": "Recurring cloud_sync failure on telematics unit",
                "part_number": "TCU-0420",
                "anomaly_label": "cloud_sync",
                "claim_count": 3,
            },
        )

    assert resp.status_code == 200
    mock_wf.run_five_why.assert_awaited_once_with(
        "Recurring cloud_sync failure on telematics unit",
        part_number="TCU-0420",
        anomaly_label="cloud_sync",
    )
