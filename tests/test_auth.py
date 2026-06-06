"""Optional API-key authentication tests (no httpx/TestClient required)."""

import asyncio

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.auth import api_key_middleware


def _make_request(path: str, headers: dict | None = None) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "query_string": b"",
    }
    return Request(scope)


def test_middleware_allows_health_when_key_required(monkeypatch):
    monkeypatch.setattr("app.config.settings.API_KEY", "secret")

    async def call_next(_request):
        from starlette.responses import JSONResponse

        return JSONResponse({"ok": True})

    response = asyncio.run(api_key_middleware(_make_request("/health"), call_next))
    assert response.status_code == 200


def test_middleware_rejects_missing_key(monkeypatch):
    monkeypatch.setattr("app.config.settings.API_KEY", "secret")

    async def call_next(_request):
        raise AssertionError("should not reach handler")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(api_key_middleware(_make_request("/query"), call_next))
    assert exc.value.status_code == 401


def test_middleware_passes_with_valid_key(monkeypatch):
    monkeypatch.setattr("app.config.settings.API_KEY", "secret")

    async def call_next(_request):
        from starlette.responses import JSONResponse

        return JSONResponse({"ok": True})

    response = asyncio.run(
        api_key_middleware(_make_request("/query", {"X-API-Key": "secret"}), call_next)
    )
    assert response.status_code == 200
