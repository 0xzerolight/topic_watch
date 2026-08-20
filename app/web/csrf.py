"""CSRF protection using a signed double-submit cookie.

The middleware issues a CSRF token cookie and re-sends it on every response.
The verify_csrf dependency validates that POST/PUT/DELETE requests include a
matching token via either the X-CSRF-Token header (HTMX) or a csrf_token form
field (regular forms).

Two properties beyond plain double-submit (AUG-003):

* **The cookie is authenticated.** Its value is ``<payload>.<hmac>``, signed
  with a per-process secret, so a site sharing our registrable domain cannot
  plant a parent-``Domain`` cookie whose value it already knows and then submit
  the matching form token. The secret is per-process because the app is
  single-process by design (the scheduler cannot be run in more than one
  worker); a restart simply reissues tokens on the next page load.
* **Cross-site submissions are refused.** ``Sec-Fetch-Site`` distinguishes a
  same-origin mutation from a same-site sibling's, which cookie equality alone
  cannot. Requests without the header (non-browser clients, pre-2023 browsers)
  fall through to the double-submit check.

Cookies issued before the signature existed are unsigned. Rather than logging
every open session out mid-flight, an unsigned cookie is adopted: its value
becomes the payload of a freshly signed cookie, so the token embedded in the
already-rendered page keeps validating while the browser is upgraded in place.
"""

import hashlib
import hmac
import secrets
from typing import Any, cast

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

COOKIE_NAME = "csrf_token"
HEADER_NAME = "x-csrf-token"
FORM_FIELD = "csrf_token"

# Sec-Fetch-Site values that may carry a state-changing request: a same-origin
# fetch/form post, or a direct user navigation. "same-site" (a sibling
# subdomain) and "cross-site" are refused.
_ALLOWED_FETCH_SITES = frozenset({"same-origin", "none"})

# Signing key for the double-submit value. Regenerated per process; see module docstring.
_SECRET = secrets.token_bytes(32)


def sign_token(payload: str) -> str:
    """Return ``payload`` with its HMAC appended."""
    signature = hmac.new(_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}"


def issue_token() -> str:
    """Mint a fresh signed CSRF token."""
    return sign_token(secrets.token_hex(32))


def token_payload(value: str) -> str | None:
    """Return the payload of a correctly signed token, else ``None``.

    ``None`` also covers unsigned values (no separator) — callers distinguish
    those from a bad signature by looking for the separator themselves.
    """
    payload, separator, signature = value.rpartition(".")
    if not separator:
        return None
    expected = hmac.new(_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return payload if hmac.compare_digest(signature, expected) else None


class CSRFMiddleware(BaseHTTPMiddleware):
    """Issues, upgrades and re-sends the signed CSRF token cookie."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        received = request.cookies.get(COOKIE_NAME)
        if not received:
            token = issue_token()
        elif "." in received:
            # Signed: keep it when the signature holds, replace it when it does not
            # (a forged value, or one signed by a previous process).
            token = received if token_payload(received) is not None else issue_token()
        else:
            # Pre-signature cookie: adopt its value so the token already rendered
            # into the open page keeps validating, and sign it going forward.
            token = sign_token(received)

        request.state.csrf_token = token

        response = cast(Response, await call_next(request))

        # Re-sent on every response rather than only when newly minted, so that
        # flipping secure_cookies rewrites the attribute on a cookie the browser
        # already holds instead of waiting for it to expire (AUG-018). Static
        # assets are skipped: they are cacheable and carry no token.
        if not request.url.path.startswith("/static"):
            secure = getattr(getattr(request.app.state, "settings", None), "secure_cookies", False)
            response.set_cookie(
                COOKIE_NAME,
                token,
                httponly=False,
                samesite="lax",
                secure=bool(secure),
            )

        return response


async def verify_csrf(request: Request) -> None:
    """FastAPI dependency: validates CSRF token on unsafe requests."""
    fetch_site = request.headers.get("sec-fetch-site", "").lower()
    if fetch_site and fetch_site not in _ALLOWED_FETCH_SITES:
        raise HTTPException(status_code=403, detail="CSRF origin rejected")

    cookie_token = request.cookies.get(COOKIE_NAME)
    if not cookie_token:
        raise HTTPException(status_code=403, detail="CSRF cookie missing")

    payload = token_payload(cookie_token)
    if payload is None and "." in cookie_token:
        # Carries a signature that is not ours: forged, or minted by a process
        # that is gone. Either way it must not authorize a mutation.
        raise HTTPException(status_code=403, detail="CSRF token invalid")

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

    # The bare payload is accepted alongside the full cookie value so that a page
    # rendered before the cookie was signed still submits a token that matches.
    accepted = [cookie_token] if payload is None else [cookie_token, payload]
    if not submitted or not any(hmac.compare_digest(candidate, submitted) for candidate in accepted):
        raise HTTPException(status_code=403, detail="CSRF token invalid")
