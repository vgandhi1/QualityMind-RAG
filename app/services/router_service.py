"""
Query Router Service — manufacturing quality + hybrid RAG/SQL (plan.md §7).
"""

from typing import Literal

QueryType = Literal["SQL", "DOCUMENTS", "HYBRID", "AGENT"]

# Agent sub-type lookup used by unified_query to dispatch directly to the right workflow
FISHBONE_PHRASES = {"fishbone", "ishikawa", "cause and effect", "categories of cause"}
EIGHT_D_PHRASES = {"draft 8d", "draft an 8d", "generate 8d"}
CAPA_PHRASES = {"draft capa", "write corrective action", "create problem report"}
FIVE_WHY_PHRASES = {"5-why", "5 why", "five why", "root cause analysis for"}


def detect_agent_workflow(question: str) -> str:
    """Return the specific agent workflow type for an AGENT-routed question."""
    q = question.lower()
    if any(p in q for p in FISHBONE_PHRASES):
        return "fishbone"
    if any(p in q for p in EIGHT_D_PHRASES):
        return "8d"
    if any(p in q for p in CAPA_PHRASES):
        return "capa"
    return "five_why"  # default for "5 why", "root cause analysis for", etc.


class QueryRouter:
    """Rule-based router: Text-to-SQL, Document RAG, Hybrid, or quality agent workflows."""

    AGENT_PHRASES = [
        *FIVE_WHY_PHRASES,
        *FISHBONE_PHRASES,
        *EIGHT_D_PHRASES,
        *CAPA_PHRASES,
    ]

    # Requires at least one quality-domain noun to avoid false positives on generic language
    SQL_KEYWORDS = [
        "how many",
        "count",
        "total",
        "sum",
        "average",
        "avg",
        "mean",
        "maximum",
        "max",
        "minimum",
        "min",
        "highest",
        "lowest",
        "which suppliers",
        "supplier quality",
        "capa",
        "ncr",
        "defect",
        "defects",
        "scrap rate",
        "yield",
        "first pass yield",
        "cpk",
        "cp ",
        "spc",
        "inspection result",
        "open capa",
        "overdue",
        "recurrence",
        "this month",
        "this year",
        "today",
        "yesterday",
        "last week",
        "last month",
        "last year",
        "more than",
        "less than",
        "greater than",
        "top failure",
        "bottom",
        "rank",
        "ranking",
        "worst supplier",
        "best supplier",
        "by segment",
        "by category",
        "by status",
        "group by",
        "failure mode",
        "failure modes",
        "detection station",
        "open ncr",
        "overdue capa",
        "severity score",
    ]

    DOCUMENT_KEYWORDS = [
        "what is",
        "what are",
        "define",
        "definition",
        "explain",
        "describe",
        "tell me about",
        "information about",
        "policy",
        "policies",
        "procedure",
        "procedures",
        "guideline",
        "guidelines",
        "rule",
        "rules",
        "regulation",
        "manual",
        "handbook",
        "documentation",
        "document",
        "reference",
        "instruction",
        "instructions",
        "how to",
        "how do",
        "how can",
        "how should",
        "when should",
        "where can",
        "who should",
        "according to",
        "based on",
        "mentioned in",
        "stated in",
        "document says",
        "documentation states",
        "understand",
        "clarify",
        "elaborate",
        "overview",
        "pfmea",
        "dfmea",
        "control plan",
        "work instruction",
        "qms",
        "8d report",
        "8d for",
        "containment",
        "detection method",
        "severity rating",
        "process step",
        "failure effect",
        "iatf",
        "iso 9001",
        "apqp",
        "ppap",
        "gauge r&r",
        "measurement system",
    ]

    HYBRID_KEYWORDS = [
        "and explain",
        "and describe",
        "and tell me",
        "also explain",
        "also describe",
        "show data and explain",
        "list and describe",
        "compare and explain",
        "analyze and describe",
        "defect counts and",
        "counts and explain",
        "open capas and",
        "ncr history and",
    ]

    @staticmethod
    def route(question: str) -> QueryType:
        question_lower = question.lower()

        if any(p in question_lower for p in QueryRouter.AGENT_PHRASES):
            return "AGENT"

        has_hybrid = any(k in question_lower for k in QueryRouter.HYBRID_KEYWORDS)
        has_sql = any(k in question_lower for k in QueryRouter.SQL_KEYWORDS)
        has_doc = any(k in question_lower for k in QueryRouter.DOCUMENT_KEYWORDS)

        if has_hybrid or (has_sql and has_doc):
            return "HYBRID"
        if has_sql:
            return "SQL"
        if has_doc:
            return "DOCUMENTS"
        return "DOCUMENTS"

    @staticmethod
    def get_routing_confidence(question: str) -> dict:
        question_lower = question.lower()
        agent_matches = sum(1 for p in QueryRouter.AGENT_PHRASES if p in question_lower)
        sql_matches = sum(1 for k in QueryRouter.SQL_KEYWORDS if k in question_lower)
        doc_matches = sum(1 for k in QueryRouter.DOCUMENT_KEYWORDS if k in question_lower)
        hybrid_matches = sum(1 for k in QueryRouter.HYBRID_KEYWORDS if k in question_lower)

        total = max(agent_matches + sql_matches + doc_matches + hybrid_matches, 1)
        route = QueryRouter.route(question)

        return {
            "question": question,
            "route": route,
            "confidence_scores": {
                "agent": round(agent_matches / total, 3),
                "sql": round(sql_matches / total, 3),
                "documents": round(doc_matches / total, 3),
                "hybrid": round(hybrid_matches / total, 3),
            },
            "keyword_matches": {
                "agent_phrases": agent_matches,
                "sql_keywords": sql_matches,
                "document_keywords": doc_matches,
                "hybrid_keywords": hybrid_matches,
            },
        }

    @staticmethod
    def explain_routing(question: str) -> str:
        route = QueryRouter.route(question)
        confidence = QueryRouter.get_routing_confidence(question)
        lines = [
            f"Question routed to: {route}",
            "",
            "Keyword matches:",
            f"- Agent phrases: {confidence['keyword_matches']['agent_phrases']}",
            f"- SQL keywords: {confidence['keyword_matches']['sql_keywords']}",
            f"- Document keywords: {confidence['keyword_matches']['document_keywords']}",
            f"- Hybrid keywords: {confidence['keyword_matches']['hybrid_keywords']}",
            "",
        ]
        if route == "AGENT":
            wf = detect_agent_workflow(question)
            lines.append(f"Structured quality workflow ({wf}).")
        elif route == "SQL":
            lines.append("Structured data query against quality database.")
        elif route == "DOCUMENTS":
            lines.append("Documentation or policy-style question.")
        else:
            lines.append("Combines database facts with document context.")
        return "\n".join(lines)
