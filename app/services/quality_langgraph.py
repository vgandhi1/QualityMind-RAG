"""
LangGraph workflows for 5-Why, fishbone, and CAPA/8D drafts (plan.md §5).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph
from openai import AsyncOpenAI

from app.config import settings
from app.services.rag_service import RAGService
from app.services.sql_service import TextToSQLService

logger = logging.getLogger("rag_app.quality_langgraph")

_workflows_singleton: Optional["QualityLangGraphWorkflows"] = None


def init_quality_workflows(rag_service: Optional[RAGService], sql_service: Optional[TextToSQLService]) -> None:
    global _workflows_singleton
    _workflows_singleton = None
    if not settings.OPENAI_API_KEY:
        return
    try:
        _workflows_singleton = QualityLangGraphWorkflows(rag_service, sql_service)
        logger.info("Quality LangGraph workflows initialized")
    except Exception as e:
        logger.warning("Quality workflows not initialized: %s", type(e).__name__)


def get_quality_workflows() -> "QualityLangGraphWorkflows":
    if _workflows_singleton is None:
        raise RuntimeError("Quality workflows unavailable")
    return _workflows_singleton


class FiveWhyState(TypedDict, total=False):
    problem_statement: str
    rag_context: str
    sql_context: str
    output: dict[str, Any]


class FishboneState(TypedDict, total=False):
    effect: str
    station: str
    rag_context: str
    sql_context: str
    output: dict[str, Any]


class DraftState(TypedDict, total=False):
    mode: str
    problem_statement: str
    part_number: Optional[str]
    rag_context: str
    sql_context: str
    output: dict[str, Any]


class QualityLangGraphWorkflows:
    """Small LangGraph pipelines: retrieve context then structured JSON."""

    def __init__(
        self,
        rag_service: Optional[RAGService],
        sql_service: Optional[TextToSQLService],
    ) -> None:
        self.rag_service = rag_service
        self.sql_service = sql_service
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for quality agents")
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = "gpt-4o"

    async def _retrieve_rag(self, query: str, top_k: int = 6) -> str:
        if not self.rag_service:
            return ""
        try:
            res = await self.rag_service.get_similar_chunks(question=query, top_k=top_k)
            chunks = res.get("chunks") or []
            parts = []
            for i, ch in enumerate(chunks, 1):
                text = ch.get("text", "")[:1200]
                fn = ch.get("metadata", {}).get("filename", "unknown")
                parts.append(f"[{i}] {fn}: {text}")
            return "\n".join(parts)
        except Exception as e:
            logger.warning("RAG retrieval failed for quality workflow: %s", type(e).__name__)
            return ""

    async def _retrieve_sql_snippet(self, question: str) -> str:
        if not self.sql_service:
            return ""
        try:
            gen = await self.sql_service.generate_sql_for_approval(question)
            exec_res = await self.sql_service.execute_approved_query(gen["query_id"], approved=True)
            if exec_res.get("status") != "executed":
                return ""
            rows = exec_res.get("results") or []
            return json.dumps(rows[:30], default=str)
        except Exception as e:
            logger.warning("SQL auto-run failed for quality workflow: %s", type(e).__name__)
            return ""

    async def _node_five_why_gather(self, state: FiveWhyState) -> FiveWhyState:
        problem = state["problem_statement"]
        import asyncio

        rag_context, sql_context = await asyncio.gather(
            self._retrieve_rag(problem, top_k=6),
            self._retrieve_sql_snippet(
                f"Recent defects and failure modes related to: {problem}. Limit 20 rows."
            ),
        )
        return {**state, "rag_context": rag_context, "sql_context": sql_context}

    async def _node_five_why_synthesize(self, state: FiveWhyState) -> FiveWhyState:
        system = (
            "You are a senior manufacturing quality engineer. "
            "Return ONLY valid JSON matching this shape: "
            '{"problem_statement": str, "whys": [{"level": int, "why": str, "evidence_source": str}], '
            '"root_cause": str, "recommended_action": str, "confidence_score": number}. '
            "Use 3 to 5 whys. Ground each why in rag_context or sql_context when possible."
        )
        user = json.dumps(
            {
                "problem_statement": state.get("problem_statement", ""),
                "rag_context": state.get("rag_context", ""),
                "sql_context": state.get("sql_context", ""),
            }
        )
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        return {**state, "output": json.loads(content)}

    def build_five_why_graph(self):
        graph = StateGraph(FiveWhyState)
        graph.add_node("gather", self._node_five_why_gather)
        graph.add_node("synthesize", self._node_five_why_synthesize)
        graph.set_entry_point("gather")
        graph.add_edge("gather", "synthesize")
        graph.add_edge("synthesize", END)
        return graph.compile()

    async def run_five_why(self, problem_statement: str) -> dict[str, Any]:
        app = self.build_five_why_graph()
        final_state = await app.ainvoke({"problem_statement": problem_statement})
        return final_state.get("output") or {}

    async def _node_fishbone_gather(self, state: FishboneState) -> FishboneState:
        effect = state["effect"]
        station = state.get("station", "")
        rag_q = f"{effect} {station} PFMEA control plan work instruction causes"
        import asyncio

        rag_context, sql_context = await asyncio.gather(
            self._retrieve_rag(rag_q, top_k=6),
            self._retrieve_sql_snippet(
                f"Top failure modes at {station} if relevant for: {effect}. Group by failure_mode, last 180 days."
            ),
        )
        return {**state, "rag_context": rag_context, "sql_context": sql_context}

    async def _node_fishbone_synthesize(self, state: FishboneState) -> FishboneState:
        system = (
            "You are a manufacturing QE building an Ishikawa (6M) fishbone. "
            "Return ONLY JSON: "
            '{"effect": str, "bones": {"Man": [...], "Machine": [...], "Method": [...], '
            '"Material": [...], "Measurement": [...], "Environment": [...]}} '
            'Each bone entry: {"cause": str, "evidence": str, "weight": number 0-1}.'
        )
        user = json.dumps(
            {
                "effect": state.get("effect", ""),
                "station": state.get("station", ""),
                "rag_context": state.get("rag_context", ""),
                "sql_context": state.get("sql_context", ""),
            }
        )
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        return {**state, "output": json.loads(content)}

    def build_fishbone_graph(self):
        graph = StateGraph(FishboneState)
        graph.add_node("gather", self._node_fishbone_gather)
        graph.add_node("synthesize", self._node_fishbone_synthesize)
        graph.set_entry_point("gather")
        graph.add_edge("gather", "synthesize")
        graph.add_edge("synthesize", END)
        return graph.compile()

    async def run_fishbone(self, effect: str, station: Optional[str] = None) -> dict[str, Any]:
        app = self.build_fishbone_graph()
        final_state = await app.ainvoke({"effect": effect, "station": station or ""})
        return final_state.get("output") or {}

    async def _node_draft_gather(self, state: DraftState) -> DraftState:
        problem = state["problem_statement"]
        part = state.get("part_number") or ""
        import asyncio

        rag_context, sql_context = await asyncio.gather(
            self._retrieve_rag(f"{problem} {part} CAPA 8D PFMEA history", top_k=8),
            self._retrieve_sql_snippet(
                f"Recent defects, CAPAs, and 8Ds for part {part or 'N/A'} related to: {problem}"
            ),
        )
        return {**state, "rag_context": rag_context, "sql_context": sql_context}

    async def _node_draft_synthesize(self, state: DraftState) -> DraftState:
        mode = state.get("mode", "capa")
        if mode == "8d":
            system = (
                "Return ONLY JSON for an 8D draft with keys: "
                "report_number (placeholder string), part_number, problem_statement, "
                "d1_team, d2_problem_desc, d3_containment, d4_root_cause, d5_perm_action, "
                "d6_implemented, d7_prevention, d8_closure. "
                "Use evidence from context; do not invent real person names."
            )
        else:
            system = (
                "Return ONLY JSON for a CAPA draft with keys: "
                "capa_number (placeholder), title, problem_statement, root_cause, "
                "corrective_action, preventive_action, owner (generic role), "
                "status (open), due_date (ISO ~30 days out). "
                "Ground content in provided context where possible."
            )
        user = json.dumps(
            {
                "mode": mode,
                "problem_statement": state.get("problem_statement", ""),
                "part_number": state.get("part_number"),
                "rag_context": state.get("rag_context", ""),
                "sql_context": state.get("sql_context", ""),
            }
        )
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        return {**state, "output": json.loads(content)}

    def build_draft_graph(self):
        graph = StateGraph(DraftState)
        graph.add_node("gather", self._node_draft_gather)
        graph.add_node("synthesize", self._node_draft_synthesize)
        graph.set_entry_point("gather")
        graph.add_edge("gather", "synthesize")
        graph.add_edge("synthesize", END)
        return graph.compile()

    async def run_draft(self, mode: str, problem_statement: str, part_number: Optional[str]) -> dict[str, Any]:
        app = self.build_draft_graph()
        final_state = await app.ainvoke(
            {"mode": mode, "problem_statement": problem_statement, "part_number": part_number}
        )
        return final_state.get("output") or {}
