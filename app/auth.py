"""Optional API-key authentication for production deployments."""

from __future__ import annotations

import hmac

from fastapi import HTTPException, Request, status

from app.config import settings

# Paths that remain public when API_KEY is configured (health probes, OpenAPI).
_PUBLIC_PATHS = frozenset({"/health", "/docs", "/redoc", "/openapi.json"})


async def api_key_middleware(request: Request, call_next):
    """Require X-API-Key header when settings.API_KEY is set."""
    if not settings.API_KEY:
        return await call_next(request)

    path = request.url.path.rstrip("/") or "/"
    if path in _PUBLIC_PATHS or path.startswith("/docs") or path.startswith("/redoc"):
        return await call_next(request)

    provided = request.headers.get("X-API-Key")
    # Constant-time comparison to avoid leaking the key via timing side-channel.
    if not provided or not hmac.compare_digest(provided, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Unauthorized", "message": "Invalid or missing API key"},
        )

    return await call_next(request)
