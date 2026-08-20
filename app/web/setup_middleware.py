"""ASGI middleware to redirect all routes to /setup when the app is unconfigured."""

from starlette.responses import JSONResponse, RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send

ALLOWED_PREFIXES = ("/setup", "/health", "/static")

API_PREFIX = "/api"

#: Methods that carry no body and can safely be redirected as-is. Anything else
#: is sent on as a GET (see AUG-210).
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

_API_NOT_CONFIGURED = "Topic Watch is not configured yet. Complete the first-run setup at /setup."


def _is_exempt(path: str) -> bool:
    """True only if path equals an allowed prefix or is a sub-path of one.

    Segment-aware so /setupx, /healthz, /static-leak are NOT exempt — only an
    exact match (/setup) or a true sub-path (/setup/...) qualifies.
    """
    return any(path == prefix or path.startswith(prefix + "/") for prefix in ALLOWED_PREFIXES)


def _is_api(path: str) -> bool:
    """Segment-aware match for the JSON API, so /apix is not treated as /api."""
    return path == API_PREFIX or path.startswith(API_PREFIX + "/")


class SetupRedirectMiddleware:
    """Redirects all HTTP requests to /setup when app.state.setup_required is True.

    Exempt paths: /setup, /health, /static and their sub-paths (so the setup
    page itself, health checks, and CSS/JS assets still work). Matching is
    segment-aware: /setupx, /healthz, /static-leak are not exempt.

    Unsafe methods redirect with 303, not 307: 307 preserves the method and the
    body, so an unconfigured POST /topics was replayed into POST /setup, feeding
    an unrelated form to the wizard. API paths get an explicit JSON 503 instead
    of a redirect, because a redirect drops an automation client into an HTML
    form flow rather than telling it what is wrong (AUG-210).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            path: str = scope["path"]
            app_state = scope.get("app")
            if app_state is not None and getattr(app_state.state, "setup_required", False) and not _is_exempt(path):
                if _is_api(path):
                    api_response = JSONResponse({"detail": _API_NOT_CONFIGURED}, status_code=503)
                    await api_response(scope, receive, send)
                    return
                status_code = 307 if scope.get("method", "GET") in SAFE_METHODS else 303
                response = RedirectResponse(url="/setup", status_code=status_code)
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)
