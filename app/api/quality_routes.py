"""
Manufacturing quality endpoints (plan.md section 6).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.agent_validation import (
    validate_8d,
    validate_capa,
    validate_fishbone,
    validate_five_why,
)
from app.config import settings
from app.services.quality_data_service import get_quality_data_service
from app.services.quality_langgraph import get_quality_workflows
from app.services.service_registry import get_rag_service
from app.utils import ErrorResponse, QueryValidator, ValidationError

logger = logging.getLogger("rag_app.quality_routes")

router = APIRouter(prefix="/quality", tags=["Quality"])

# Module-level OpenAI client singleton for SPC narrative generation
_narrative_client: AsyncOpenAI | None = None


def _get_narrative_client() -> AsyncOpenAI:
    global _narrative_client
    if _narrative_client is None:
        _narrative_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _narrative_client


class ProblemBody(BaseModel):
    problem_statement: str = Field(..., min_length=3, max_length=2000)


class FishboneBody(BaseModel):
    effect: str = Field(..., min_length=3, max_length=2000)
    station: str | None = Field(None, max_length=120)


class DraftBody(BaseModel):
    problem_statement: str = Field(..., min_length=3, max_length=2000)
    part_number: str | None = Field(None, max_length=64)


class PfmeaSearchBody(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)


class NcrHistoryBody(BaseModel):
    part_number: str | None = Field(None, max_length=64)
    limit: int = Field(100, ge=1, le=500)


class SpcSummaryBody(BaseModel):
    part_number: str = Field(..., min_length=1, max_length=64)
    characteristic: str | None = Field(None, max_length=200)
    station: str | None = Field(None, max_length=120)


def _validate_question(text: str) -> str:
    try:
        return QueryValidator.validate_question(text)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=ErrorResponse.validation_error(str(e))) from e


def _get_workflows_or_503():
    try:
        return get_quality_workflows()
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse.service_unavailable(
                "Quality workflows",
                "Configure OPENAI_API_KEY and RAG/SQL services.",
            ),
        )


def _dev_structure_check(workflow: str, output: dict) -> dict | None:
    """In development, attach offline structural validation for analytical iteration."""
    if settings.ENVIRONMENT != "development":
        return None
    validators = {
        "five_why": validate_five_why,
        "fishbone": validate_fishbone,
        "capa": validate_capa,
        "8d": validate_8d,
    }
    validator = validators.get(workflow)
    return validator(output) if validator else None


def _agent_response(payload_key: str, workflow: str, result: dict) -> dict[str, Any]:
    body: dict[str, Any] = {"status": "success", payload_key: result}
    check = _dev_structure_check(workflow, result)
    if check is not None:
        body["structure_check"] = check
    return body


@router.post("/five-why", status_code=status.HTTP_200_OK)
async def five_why(body: ProblemBody) -> dict[str, Any]:
    _validate_question(body.problem_statement)
    wf = _get_workflows_or_503()
    try:
        result = await wf.run_five_why(body.problem_statement)
        return _agent_response("analysis", "five_why", result)
    except Exception:
        logger.exception("five_why failed")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.internal_error("five_why", Exception("workflow error")),
        )


@router.post("/fishbone", status_code=status.HTTP_200_OK)
async def fishbone(body: FishboneBody) -> dict[str, Any]:
    _validate_question(body.effect)
    wf = _get_workflows_or_503()
    try:
        result = await wf.run_fishbone(body.effect, body.station)
        return _agent_response("fishbone", "fishbone", result)
    except Exception:
        logger.exception("fishbone failed")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.internal_error("fishbone", Exception("workflow error")),
        )


@router.post("/draft-capa", status_code=status.HTTP_200_OK)
async def draft_capa(body: DraftBody) -> dict[str, Any]:
    _validate_question(body.problem_statement)
    wf = _get_workflows_or_503()
    try:
        result = await wf.run_draft("capa", body.problem_statement, body.part_number)
        return _agent_response("draft", "capa", result)
    except Exception:
        logger.exception("draft_capa failed")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.internal_error("draft_capa", Exception("workflow error")),
        )


@router.post("/draft-8d", status_code=status.HTTP_200_OK)
async def draft_8d(body: DraftBody) -> dict[str, Any]:
    _validate_question(body.problem_statement)
    wf = _get_workflows_or_503()
    try:
        result = await wf.run_draft("8d", body.problem_statement, body.part_number)
        return _agent_response("draft", "8d", result)
    except Exception:
        logger.exception("draft_8d failed")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.internal_error("draft_8d", Exception("workflow error")),
        )


@router.get("/capa-status", status_code=status.HTTP_200_OK)
async def capa_status(
    supplier_id: int | None = Query(None, ge=1),
    overdue_only: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    try:
        svc = get_quality_data_service()
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse.service_unavailable(
                "Quality database",
                "Configure DATABASE_URL and apply schema/seed scripts.",
            ),
        )
    try:
        rows = await asyncio.to_thread(
            svc.capa_status, supplier_id=supplier_id, overdue_only=overdue_only, limit=limit
        )
        return {"status": "success", "count": len(rows), "capas": rows}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=ErrorResponse.validation_error(str(e))) from e
    except Exception:
        logger.exception("capa_status failed")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.internal_error("capa_status", Exception("database error")),
        )


@router.post("/pfmea-search", status_code=status.HTTP_200_OK)
async def pfmea_search(body: PfmeaSearchBody) -> dict[str, Any]:
    _validate_question(body.query)
    try:
        rag = get_rag_service()
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse.service_unavailable(
                "RAG service", "Configure OpenAI and Pinecone."
            ),
        )
    try:
        chunks = await rag.get_similar_chunks(question=body.query, top_k=body.top_k)
        return {
            "status": "success",
            "chunks": chunks.get("chunks", []),
            "total_found": chunks.get("total_found"),
        }
    except Exception:
        logger.exception("pfmea_search failed")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.internal_error("pfmea_search", Exception("retrieval error")),
        )


@router.post("/ncr-history", status_code=status.HTTP_200_OK)
async def ncr_history(body: NcrHistoryBody) -> dict[str, Any]:
    try:
        svc = get_quality_data_service()
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse.service_unavailable("Quality database", "Configure DATABASE_URL."),
        )
    try:
        rows = await asyncio.to_thread(
            svc.ncr_history, part_number=body.part_number, limit=body.limit
        )
        return {"status": "success", "count": len(rows), "ncrs": rows}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=ErrorResponse.validation_error(str(e))) from e
    except Exception:
        logger.exception("ncr_history failed")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.internal_error("ncr_history", Exception("database error")),
        )


@router.post("/spc-summary", status_code=status.HTTP_200_OK)
async def spc_summary(body: SpcSummaryBody) -> dict[str, Any]:
    try:
        svc = get_quality_data_service()
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse.service_unavailable("Quality database", "Configure DATABASE_URL."),
        )
    try:
        stats = await asyncio.to_thread(
            svc.spc_aggregate,
            part_number=body.part_number,
            characteristic=body.characteristic,
            station=body.station,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=ErrorResponse.validation_error(str(e))) from e
    except Exception:
        logger.exception("spc_aggregate failed")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.internal_error("spc_summary", Exception("database error")),
        )

    narrative = ""
    if settings.OPENAI_API_KEY:
        try:
            client = _get_narrative_client()
            resp = await client.chat.completions.create(
                model=settings.NARRATIVE_MODEL,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a manufacturing quality engineer. "
                            "Summarize SPC metrics clearly for operators and quality teams. "
                            "Highlight any Cpk values below 1.33 as requiring attention, "
                            "and values below 1.0 as critical. "
                            "Do not invent numbers; only interpret the JSON provided."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(stats, default=str),
                    },
                ],
            )
            narrative = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning("SPC narrative generation skipped: %s", type(e).__name__)

    return {"status": "success", "statistics": stats, "narrative": narrative}
