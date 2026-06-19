"""
Component → BOM part_number alias map.

CLaimLens hands off a **descriptive component** name (e.g. "Telematics Control
Unit") because the customer/warranty path never carries BOM identifiers. The
engineering quality DB (``defects`` / ``ncr`` / ``eight_d`` / ``inspection_results``)
is keyed on **BOM ``part_number``**. Text-to-SQL retrieval that filters on the
descriptive string therefore misses every BOM-keyed row.

This module bridges that gap: it resolves a descriptive component to its
candidate BOM ``part_number`` codes (or prefixes) so the SQL agent can build a
``part_number IN (...) OR part_number ILIKE '%...%'`` predicate instead of
matching free text. RAG semantic search still uses the descriptive name.

The map is intentionally small and synthetic — extend it as the BOM grows. Keys
are matched case-insensitively and substring-tolerant so minor extraction
variants ("telematics unit" vs "Telematics Control Unit") still resolve.
"""

from __future__ import annotations

# Descriptive component (canonical, lowercased) → BOM part_number codes / prefixes.
# Codes mirror the synthetic seed in data/generate_sample_data.py; PCB-CTRL is the
# electronics control board that backs the connected-vehicle modules.
COMPONENT_PART_ALIASES: dict[str, list[str]] = {
    "telematics control unit": ["TCU", "PCB-CTRL"],
    "connectivity gateway": ["GW", "PCB-CTRL"],
    "cellular modem": ["MODEM", "PCB-CTRL"],
    "electronic control unit": ["ECU", "PCB-CTRL"],
    "infotainment head unit": ["IVI", "PCB-CTRL"],
}


def _normalize(component: str | None) -> str:
    return (component or "").strip().lower()


def resolve_part_aliases(component: str | None) -> list[str]:
    """Return candidate BOM part_number codes for a descriptive component.

    Empty list when the component is unknown — callers should fall back to the
    descriptive name so behavior never regresses for unmapped components.
    """
    key = _normalize(component)
    if not key:
        return []
    if key in COMPONENT_PART_ALIASES:
        return COMPONENT_PART_ALIASES[key]
    # Token-subset tolerance for extraction variants (e.g. "telematics unit" →
    # "telematics control unit"). Require ≥2 overlapping tokens so a single
    # generic word like "unit" never matches an arbitrary component.
    key_tokens = set(key.split())
    if len(key_tokens) >= 2:
        for canonical, codes in COMPONENT_PART_ALIASES.items():
            if key_tokens <= set(canonical.split()):
                return codes
    return []


def alias_sql_hint(component: str | None) -> str:
    """SQL-filter hint for the Text-to-SQL agent.

    Returns a clause-style fragment naming the BOM ``part_number`` candidates so
    the agent joins engineering tables on real keys. Empty string when no alias
    is known (agent then filters on the descriptive component as before).
    """
    codes = resolve_part_aliases(component)
    if not codes:
        return ""
    quoted = ", ".join(f"'{c}'" for c in codes)
    likes = " OR ".join(f"part_number ILIKE '%{c}%'" for c in codes)
    return (
        f"Engineering tables are keyed on BOM part_number, not the descriptive "
        f"component name '{component}'. Filter using part_number IN ({quoted}) "
        f"OR ({likes})."
    )
