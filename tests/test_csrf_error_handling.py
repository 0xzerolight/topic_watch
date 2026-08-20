"""Tests for CSRF error handling when request.form() raises an exception."""

import httpx
import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.web.csrf import COOKIE_NAME, CSRFMiddleware, issue_token, sign_token, token_payload, verify_csrf

CSRF_TEST_TOKEN = "test-csrf-token-for-error-handling"


def _make_test_app() -> FastAPI:
    """Create a minimal FastAPI app with a protected POST endpoint."""
    test_app = FastAPI()
    test_app.add_middleware(CSRFMiddleware)

    @test_app.post("/protected")
    async def protected(_: None = Depends(verify_csrf)):
        return JSONResponse({"ok": True})

    @test_app.get("/page")
    async def page(request: Request):
        return JSONResponse({"token": request.state.csrf_token})

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


# --- AUG-003: the cookie is signed ---


async def test_issued_cookie_is_signed(test_app: FastAPI) -> None:
    """A browser with no cookie gets an authenticated one, not a bare random value."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=test_app), base_url="http://test") as ac:
        response = await ac.get("/page")
    issued = response.cookies[COOKIE_NAME]
    assert issued == response.json()["token"]
    assert token_payload(issued) is not None


async def test_forged_signature_cannot_authorize_a_mutation(test_app: FastAPI) -> None:
    """AUG-003: a planted cookie whose value the attacker chose is refused."""
    forged = "attacker-chosen-payload.0123456789abcdef"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=test_app),
        base_url="http://test",
        cookies={COOKIE_NAME: forged},
        headers={"X-CSRF-Token": forged},
    ) as ac:
        response = await ac.post("/protected", content=b"")
    assert response.status_code == 403
    assert response.json()["detail"] == "CSRF token invalid"


async def test_signed_cookie_and_matching_token_pass(test_app: FastAPI) -> None:
    signed = sign_token("payload-under-our-own-key")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=test_app),
        base_url="http://test",
        cookies={COOKIE_NAME: signed},
        headers={"X-CSRF-Token": signed},
    ) as ac:
        response = await ac.post("/protected", content=b"")
    assert response.status_code == 200


# --- AUG-003: cross-site submissions ---


@pytest.mark.parametrize("site", ["same-site", "cross-site"])
async def test_cross_site_submission_is_rejected(test_app: FastAPI, site: str) -> None:
    """A sibling subdomain's POST is refused even when the tokens match."""
    signed = sign_token("some-payload")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=test_app),
        base_url="http://test",
        cookies={COOKIE_NAME: signed},
        headers={"X-CSRF-Token": signed, "Sec-Fetch-Site": site},
    ) as ac:
        response = await ac.post("/protected", content=b"")
    assert response.status_code == 403
    assert response.json()["detail"] == "CSRF origin rejected"


@pytest.mark.parametrize("site", ["same-origin", "none"])
async def test_same_origin_submission_is_accepted(test_app: FastAPI, site: str) -> None:
    signed = sign_token("some-payload")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=test_app),
        base_url="http://test",
        cookies={COOKIE_NAME: signed},
        headers={"X-CSRF-Token": signed, "Sec-Fetch-Site": site},
    ) as ac:
        response = await ac.post("/protected", content=b"")
    assert response.status_code == 200


# --- AUG-003 compatibility: in-flight unsigned cookies ---


async def test_existing_unsigned_cookie_is_rotated_not_rejected(test_app: FastAPI) -> None:
    """An open session keeps working: its unsigned cookie is upgraded in place.

    The token already rendered into the user's page is the unsigned value, so it
    becomes the payload of the signed replacement and still validates.
    """
    legacy = "a" * 64
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=test_app),
        base_url="http://test",
        headers={"Cookie": f"{COOKIE_NAME}={legacy}"},
    ) as ac:
        # The mutation the user was mid-way through still succeeds...
        accepted = await ac.post("/protected", data={"csrf_token": legacy})
        assert accepted.status_code == 200

        # ...and the browser is left holding a signed cookie built on that value.
        rotated = accepted.cookies[COOKIE_NAME]
        assert rotated != legacy
        assert token_payload(rotated) == legacy

    # The token already rendered into the open page still matches the rotated cookie.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=test_app),
        base_url="http://test",
        headers={"Cookie": f"{COOKIE_NAME}={rotated}"},
    ) as ac:
        still_valid = await ac.post("/protected", data={"csrf_token": legacy})
    assert still_valid.status_code == 200


async def test_unsigned_cookie_is_upgraded_on_a_plain_page_view(test_app: FastAPI) -> None:
    legacy = "b" * 64
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=test_app),
        base_url="http://test",
        cookies={COOKIE_NAME: legacy},
    ) as ac:
        response = await ac.get("/page")
    assert token_payload(response.cookies[COOKIE_NAME]) == legacy


async def test_stale_signature_reissues_instead_of_wedging(test_app: FastAPI) -> None:
    """A cookie signed by a previous process is replaced on the next page view."""
    stale = "some-payload.deadbeef"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=test_app),
        base_url="http://test",
        cookies={COOKIE_NAME: stale},
    ) as ac:
        response = await ac.get("/page")
    reissued = response.cookies[COOKIE_NAME]
    assert token_payload(reissued) is not None
    assert reissued != stale


# --- AUG-018: the secure-cookie toggle reaches an established browser ---


async def test_secure_toggle_upgrades_an_existing_cookie(test_app: FastAPI) -> None:
    """Turning secure_cookies on rewrites the attribute on a cookie already held."""

    class _Settings:
        secure_cookies = False

    test_app.state.settings = _Settings()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=test_app),
        base_url="http://test",
        cookies={COOKIE_NAME: issue_token()},
    ) as ac:
        before = await ac.get("/page")
        assert "secure" not in before.headers["set-cookie"].lower()

        test_app.state.settings.secure_cookies = True
        after = await ac.get("/page")

    assert "secure" in after.headers["set-cookie"].lower()


async def test_static_responses_do_not_set_the_cookie(test_app: FastAPI) -> None:
    @test_app.get("/static/thing.css")
    async def asset():
        return JSONResponse({"ok": True})

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=test_app), base_url="http://test") as ac:
        response = await ac.get("/static/thing.css")
    assert "set-cookie" not in response.headers
