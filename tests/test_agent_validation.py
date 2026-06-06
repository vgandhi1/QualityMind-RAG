"""
Unit tests for quality-agent structural validators (app/agent_validation.py).

These pin the contract shown in the README's Evaluation table without any LLM
call: 5-Why (keys + 3-5 whys + confidence in [0,1]), fishbone (all 6M bones),
CAPA fields, and 8D disciplines D1-D8.
"""

from app.agent_validation import (
    validate_8d,
    validate_capa,
    validate_fishbone,
    validate_five_why,
)


def _good_five_why():
    return {
        "problem_statement": "Recurring weld delamination",
        "whys": [
            {"level": 1, "why": "joint failing"},
            {"level": 2, "why": "amplitude drift"},
            {"level": 3, "why": "no SPC chart"},
        ],
        "root_cause": "missing control plan gate",
        "recommended_action": "update QMS procedure",
        "confidence_score": 0.82,
    }


def test_five_why_valid():
    assert validate_five_why(_good_five_why())["passed"] is True


def test_five_why_too_few_whys_fails():
    bad = _good_five_why()
    bad["whys"] = bad["whys"][:2]
    result = validate_five_why(bad)
    assert result["passed"] is False
    assert any("whys" in m for m in result["missing_keys"])


def test_five_why_confidence_out_of_range_fails():
    bad = _good_five_why()
    bad["confidence_score"] = 1.7
    result = validate_five_why(bad)
    assert result["passed"] is False
    assert any("confidence_score" in m for m in result["missing_keys"])


def test_five_why_missing_root_cause_fails():
    bad = _good_five_why()
    del bad["root_cause"]
    assert validate_five_why(bad)["passed"] is False


def _good_fishbone():
    return {
        "effect": "torque failures",
        "bones": {
            "Man": [{"cause": "training gap"}],
            "Machine": [{"cause": "calibration overdue"}],
            "Method": [{"cause": "WI ambiguity"}],
            "Material": [{"cause": "lot variation"}],
            "Measurement": [{"cause": "wrench resolution"}],
            "Environment": [{"cause": "temperature"}],
        },
    }


def test_fishbone_valid():
    assert validate_fishbone(_good_fishbone())["passed"] is True


def test_fishbone_missing_bone_fails():
    bad = _good_fishbone()
    del bad["bones"]["Environment"]
    result = validate_fishbone(bad)
    assert result["passed"] is False
    assert "bone:Environment" in result["missing_keys"]


def test_fishbone_empty_bone_fails():
    bad = _good_fishbone()
    bad["bones"]["Man"] = []
    assert validate_fishbone(bad)["passed"] is False


def test_8d_requires_all_disciplines():
    full = dict.fromkeys(["problem_statement", "d1_team", "d2_problem_desc", "d3_containment", "d4_root_cause", "d5_perm_action", "d6_implemented", "d7_prevention", "d8_closure"], "x")
    assert validate_8d(full)["passed"] is True
    del full["d8_closure"]
    assert validate_8d(full)["passed"] is False


def test_capa_requires_core_fields():
    full = {
        "problem_statement": "x", "root_cause": "x",
        "corrective_action": "x", "preventive_action": "x", "status": "open",
    }
    assert validate_capa(full)["passed"] is True
    del full["preventive_action"]
    assert validate_capa(full)["passed"] is False
