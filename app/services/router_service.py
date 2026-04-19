"""
Query Router Service — manufacturing quality + hybrid RAG/SQL (plan.md §7).
"""

from typing import Literal


QueryType = Literal["SQL", "DOCUMENTS", "HYBRID", "AGENT"]


class QueryRouter:
    """Rule-based router: Text-to-SQL, Document RAG, Hybrid, or quality agent workflows."""

    AGENT_PHRASES = [
        "5-why",
        "5 why",
        "five why",
        "fishbone",
        "ishikawa",
        "cause and effect",
        "categories of cause",
        "draft capa",
        "draft 8d",
        "draft an 8d",
        "generate 8d",
        "write corrective action",
        "create problem report",
        "root cause analysis for",
    ]

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
        "list all",
        "show all",
        "find all",
        "get all",
        "display all",
        "list",
        "show",
        "find",
        "get",
        "display",
        "which suppliers",
        "supplier",
        "capa",
        "ncr",
        "defect",
        "defects",
        "scrap",
        "yield",
        "cpk",
        "cp ",
        "spc",
        "inspection",
        "open capa",
        "overdue",
        "last",
        "recent",
        "past",
        "previous",
        "this month",
        "this year",
        "today",
        "yesterday",
        "week",
        "month",
        "year",
        "more than",
        "less than",
        "greater than",
        "top",
        "bottom",
        "rank",
        "ranking",
        "best",
        "worst",
        "by segment",
        "by category",
        "by status",
        "group by",
        "per",
        "each",
        "every",
        "database",
        "table",
        "record",
        "row",
        "data",
        "station",
        "failure mode",
        "failure modes",
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
        "process",
        "guideline",
        "guidelines",
        "rule",
        "rules",
        "regulation",
        "guide",
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
        "detail",
        "overview",
        "summary",
        "summarize",
        "pfmea",
        "dfmea",
        "control plan",
        "work instruction",
        "qms",
        "8d for",
        "8d report",
        "containment",
        "detection method",
        "severity rating",
    ]

    HYBRID_KEYWORDS = [
        "and explain",
        "and describe",
        "and tell me",
        "also explain",
        "also describe",
        "also tell me",
        "show data and explain",
        "list and describe",
        "compare and explain",
        "analyze and describe",
        "defect counts and",
        "counts and explain",
        "open capas and",
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
            lines.append("Structured quality workflow (5-Why, fishbone, or CAPA/8D draft).")
        elif route == "SQL":
            lines.append("Structured data query.")
        elif route == "DOCUMENTS":
            lines.append("Documentation or policy-style question.")
        else:
            lines.append("Combines database facts with document context.")
        return "\n".join(lines)
