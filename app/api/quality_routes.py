"""
Manufacturing quality endpoints (plan.md section 6).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, status
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.config import settings
from app.services.quality_data_service import get_quality_data_service
from app.services.quality_langgraph import get_quality_workflows
from app.utils import QueryValidator, ValidationError, ErrorResponse

logger = logging.getLogger("rag_app.quality_routes")

router = APIRouter(prefix="/quality", tags=["Quality"])


class ProblemBody(BaseModel):
    problem_statement: str = Field(..., min_length=3, max_length=2000)


class FishboneBody(BaseModel):
    effect: str = Field(..., min_length=3, max_length=2000)
    station: Optional[str] = Field(None, max_length=120)


class DraftBody(BaseModel):
    problem_statement: str = Field(..., min_length=3, max_length=2000)
    part_number: Optional[str] = Field(None, max_length=64)


class PfmeaSearchBody(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)


class NcrHistoryBody(BaseModel):
    part_number: Optional[str] = Field(None, max_length=64)
    limit: int = Field(100, ge=1, le=500)


class SpcSummaryBody(BaseModel):
    part_number: str = Field(..., min_length=1, max_length=64)
    characteristic: Optional[str] = Field(None, max_length=200)
    station: Optional[str] = Field(None, max_length=120)


def _validate_question(text: str) -> str:
    try:
        return QueryValidator.validate_question(text)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=ErrorResponse.validation_error(str(e))) from e


@router.post("/five-why", status_code=status.HTTP_200_OK)
async def five_why(body: ProblemBody) -> dict[str, Any]:
    _validate_question(body.problem_statement)
    try:
        wf = get_quality_workflows()
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse.service_unavailable(
                "Quality workflows",
                "Configure OPENAI_API_KEY and RAG/SQL services for full context.",
            ),
        )
    try:
        result = await wf.run_five_why(body.problem_statement)
        return {"status": "success", "analysis": result}
    except Exception:
        logger.exception("five_why failed")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.internal_error("five_why", Exception("workflow error")),
        )


@router.post("/fishbone", status_code=status.HTTP_200_OK)
async def fishbone(body: FishboneBody) -> dict[str, Any]:
    _validate_question(body.effect)
    try:
        wf = get_quality_workflows()
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse.service_unavailable(
                "Quality workflows",
                "Configure OPENAI_API_KEY and RAG/SQL services for full context.",
            ),
        )
    try:
        result = await wf.run_fishbone(body.effect, body.station)
        return {"status": "success", "fishbone": result}
    except Exception:
        logger.exception("fishbone failed")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.internal_error("fishbone", Exception("workflow error")),
        )


@router.post("/draft-capa", status_code=status.HTTP_200_OK)
async def draft_capa(body: DraftBody) -> dict[str, Any]:
    _validate_question(body.problem_statement)
    try:
        wf = get_quality_workflows()
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse.service_unavailable("Quality workflows", "Configure OPENAI_API_KEY."),
        )
    try:
        result = await wf.run_draft("capa", body.problem_statement, body.part_number)
        return {"status": "success", "draft": result}
    except Exception:
        logger.exception("draft_capa failed")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.internal_error("draft_capa", Exception("workflow error")),
        )


@router.post("/draft-8d", status_code=status.HTTP_200_OK)
async def draft_8d(body: DraftBody) -> dict[str, Any]:
    _validate_question(body.problem_statement)
    try:
        wf = get_quality_workflows()
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse.service_unavailable("Quality workflows", "Configure OPENAI_API_KEY."),
        )
    try:
        result = await wf.run_draft("8d", body.problem_statement, body.part_number)
        return {"status": "success", "draft": result}
    except Exception:
        logger.exception("draft_8d failed")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.internal_error("draft_8d", Exception("workflow error")),
        )


@router.get("/capa-status", status_code=status.HTTP_200_OK)
async def capa_status(
    supplier_id: Optional[int] = Query(None, ge=1),
    overdue_only: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    try:
        svc = get_quality_data_service()
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse.service_unavailable(
                "Quality database", "Configure DATABASE_URL and apply schema/seed scripts.",
            ),
        )
    try:
        rows = svc.capa_status(supplier_id=supplier_id, overdue_only=overdue_only, limit=limit)
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
    from app.main import rag_service

    if not rag_service:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse.service_unavailable("RAG service", "Configure OpenAI and Pinecone."),
        )
    try:
        chunks = await rag_service.get_similar_chunks(question=body.query, top_k=body.top_k)
        return {"status": "success", "chunks": chunks.get("chunks", []), "total_found": chunks.get("total_found")}
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
        rows = svc.ncr_history(part_number=body.part_number, limit=body.limit)
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
        stats = svc.spc_aggregate(
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
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a manufacturing quality engineer. Summarize SPC metrics clearly for operators. "
                        "Do not invent numbers; only interpret the JSON provided.",
                    },
                    {
                        "role": "user",
                        "content": str(stats),
                    },
                ],
            )
            narrative = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning("SPC narrative generation skipped: %s", type(e).__name__)

    return {"status": "success", "statistics": stats, "narrative": narrative}
