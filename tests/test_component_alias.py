from app.services.component_alias import (
    alias_sql_hint,
    resolve_part_aliases,
)


def test_resolve_known_component():
    assert resolve_part_aliases("Telematics Control Unit") == ["TCU", "PCB-CTRL"]


def test_resolve_is_case_and_variant_tolerant():
    # Extraction variant + casing still resolves via substring tolerance.
    assert resolve_part_aliases("telematics unit") == ["TCU", "PCB-CTRL"]
    assert resolve_part_aliases("CONNECTIVITY GATEWAY") == ["GW", "PCB-CTRL"]


def test_resolve_unknown_returns_empty():
    assert resolve_part_aliases("Brake Caliper") == []
    assert resolve_part_aliases(None) == []
    assert resolve_part_aliases("") == []


def test_sql_hint_names_bom_part_numbers():
    hint = alias_sql_hint("Telematics Control Unit")
    assert "part_number IN ('TCU', 'PCB-CTRL')" in hint
    assert "ILIKE '%TCU%'" in hint
    # Descriptive name preserved so the agent knows the source mapping.
    assert "Telematics Control Unit" in hint


def test_sql_hint_empty_for_unknown():
    assert alias_sql_hint("Brake Caliper") == ""
    assert alias_sql_hint(None) == ""
