"""
Service registry — neutral module written to by main.py on startup
and read by quality_routes.py, breaking the circular import.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.services.rag_service import RAGService

logger = logging.getLogger("rag_app.service_registry")

_rag_service: Any = None


def register_rag_service(svc: "RAGService") -> None:
    global _rag_service
    _rag_service = svc
    logger.debug("RAG service registered")


def get_rag_service() -> "RAGService":
    if _rag_service is None:
        raise RuntimeError("RAG service unavailable")
    return _rag_service
