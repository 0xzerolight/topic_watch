"""Tests for CSRF error handling when request.form() raises an exception."""

import httpx
import pytest
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from app.web.csrf import CSRFMiddleware, verify_csrf

CSRF_TEST_TOKEN = "test-csrf-token-for-error-handling"


def _make_test_app() -> FastAPI:
    """Create a minimal FastAPI app with a protected POST endpoint."""
    test_app = FastAPI()
    test_app.add_middleware(CSRFMiddleware)

    @test_app.post("/protected")
    async def protected(_: None = Depends(verify_csrf)):
        return JSONResponse({"ok": True})

    return test_app


@pytest.fixture
def test_app() -> FastAPI:
    return _make_test_app()


# --- Baseline: valid form CSRF passes ---


async def test_valid_csrf_in_form_passes(test_app: FastAPI) -> None:
    """A matching CSRF token in form data allows the request through."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=test_app),
        base_url="http://test",
        cookies={"csrf_token": CSRF_TEST_TOKEN},
    ) as ac:
        response = await ac.post(
            "/protected",
            data={"csrf_token": CSRF_TEST_TOKEN, "other": "value"},
        )
    assert response.status_code == 200
    assert response.json() == {"ok": True}


# --- Malformed form body returns 403 ---


async def test_malformed_form_body_returns_403(test_app: FastAPI) -> None:
    """Content-type form with a body that causes request.form() to fail returns 403."""
    # Send a multipart/form-data content-type but with binary garbage as the
    # body and no boundary parameter — Starlette will raise when parsing it.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=test_app),
        base_url="http://test",
        cookies={"csrf_token": CSRF_TEST_TOKEN},
    ) as ac:
        response = await ac.post(
            "/protected",
            content=b"\x00\x01\x02\x03binary garbage",
            headers={"content-type": "multipart/form-data"},
        )
    assert response.status_code == 403
    assert response.json()["detail"] == "CSRF token invalid"


async def test_multipart_missing_boundary_returns_403(test_app: FastAPI) -> None:
    """multipart/form-data without a boundary parameter returns 403, not 500."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=test_app),
        base_url="http://test",
        cookies={"csrf_token": CSRF_TEST_TOKEN},
    ) as ac:
        response = await ac.post(
            "/protected",
            content=b"some body content",
            headers={"content-type": "multipart/form-data"},
            # No boundary= in the content-type → Starlette raises on form parse
        )
    assert response.status_code == 403
    assert response.json()["detail"] == "CSRF token invalid"


# --- HTMX header path still works ---


async def test_csrf_header_path_passes(test_app: FastAPI) -> None:
    """HTMX-style requests using the X-CSRF-Token header are accepted."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=test_app),
        base_url="http://test",
        cookies={"csrf_token": CSRF_TEST_TOKEN},
        headers={"X-CSRF-Token": CSRF_TEST_TOKEN},
    ) as ac:
        response = await ac.post("/protected", content=b"")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


async def test_csrf_header_mismatch_returns_403(test_app: FastAPI) -> None:
    """Mismatched X-CSRF-Token header returns 403."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=test_app),
        base_url="http://test",
        cookies={"csrf_token": CSRF_TEST_TOKEN},
        headers={"X-CSRF-Token": "wrong-token"},
    ) as ac:
        response = await ac.post("/protected", content=b"")
    assert response.status_code == 403
