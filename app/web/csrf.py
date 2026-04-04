"""CSRF protection using the double-submit cookie pattern.

The middleware sets a random CSRF token in a cookie on every response.
The verify_csrf dependency validates that POST/PUT/DELETE requests
include a matching token via either the X-CSRF-Token header (HTMX)
or a csrf_token form field (regular forms).
"""

import hmac
import secrets
from typing import Any, cast

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

COOKIE_NAME = "csrf_token"
HEADER_NAME = "x-csrf-token"
FORM_FIELD = "csrf_token"


class CSRFMiddleware(BaseHTTPMiddleware):
    """Sets a CSRF token cookie on every response if not already present."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        token = request.cookies.get(COOKIE_NAME)
        new_token = token is None
        if new_token:
            token = secrets.token_hex(32)

        request.state.csrf_token = token

        response = cast(Response, await call_next(request))

        if new_token and token is not None:
            secure = getattr(getattr(request.app.state, "settings", None), "secure_cookies", False)
            response.set_cookie(
                COOKIE_NAME,
                token,
                httponly=False,
                samesite="lax",
                secure=secure,
            )

        return response


async def verify_csrf(request: Request) -> None:
    """FastAPI dependency: validates CSRF token on unsafe requests."""
    cookie_token = request.cookies.get(COOKIE_NAME)
    if not cookie_token:
        raise HTTPException(status_code=403, detail="CSRF cookie missing")

    # Check header first (HTMX requests send this via hx-headers)
    submitted = request.headers.get(HEADER_NAME)

    # Fall back to form field (regular form submissions)
    if not submitted:
        content_type = request.headers.get("content-type", "")
        if "form" in content_type:
            try:
                form_data = await request.form()
                raw = form_data.get(FORM_FIELD)
                submitted = raw if isinstance(raw, str) else None
            except Exception as exc:
                raise HTTPException(status_code=403, detail="CSRF token invalid") from exc

    if not submitted or not hmac.compare_digest(cookie_token, submitted):
        raise HTTPException(status_code=403, detail="CSRF token invalid")
