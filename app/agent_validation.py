"""
Structural validators for quality agent (5-Why / fishbone / CAPA / 8D) outputs.

Pure, dependency-free functions so they can be reused by the evaluation harness
(`evaluate.py`) and exercised by unit tests without pulling in RAGAS/OpenAI.
Each validator returns: {"workflow": str, "passed": bool, "missing_keys": list}.
"""



def check_keys(output: dict, required: list, workflow: str) -> dict:
    """Base check: every required key present and truthy."""
    missing = [k for k in required if k not in output or not output[k]]
    return {"workflow": workflow, "passed": len(missing) == 0, "missing_keys": missing}


def validate_five_why(output: dict, expected_keys: list | None = None) -> dict:
    required = expected_keys or [
        "problem_statement", "whys", "root_cause", "recommended_action", "confidence_score",
    ]
    result = check_keys(output, required, "five_why")
    whys = output.get("whys", [])
    if not isinstance(whys, list) or len(whys) < 3:
        result['passed'] = False
        result['missing_keys'].append("whys (min 3 levels required)")
    score = output.get("confidence_score", -1)
    try:
        in_range = 0.0 <= float(score) <= 1.0
    except (TypeError, ValueError):
        in_range = False
    if not in_range:
        result['passed'] = False
        result['missing_keys'].append("confidence_score out of [0,1] range")
    return result


def validate_fishbone(output: dict, expected_keys: list | None = None) -> dict:
    required_bones = ["Man", "Machine", "Method", "Material", "Measurement", "Environment"]
    result = check_keys(output, ["effect", "bones"], "fishbone")
    bones = output.get("bones", {})
    missing_bones = [b for b in required_bones if b not in bones or not bones[b]]
    if missing_bones:
        result['passed'] = False
        result['missing_keys'].extend([f"bone:{b}" for b in missing_bones])
    return result


def validate_8d(output: dict, expected_keys: list | None = None) -> dict:
    required = expected_keys or [
        "problem_statement", "d1_team", "d2_problem_desc", "d3_containment",
        "d4_root_cause", "d5_perm_action", "d6_implemented", "d7_prevention", "d8_closure",
    ]
    return check_keys(output, required, "8d")


def validate_capa(output: dict, expected_keys: list | None = None) -> dict:
    required = expected_keys or [
        "problem_statement", "root_cause", "corrective_action", "preventive_action", "status",
    ]
    return check_keys(output, required, "capa")
