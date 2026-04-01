"""ASGI middleware to redirect all routes to /setup when the app is unconfigured."""

from starlette.responses import RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send

ALLOWED_PREFIXES = ("/setup", "/health", "/static")


class SetupRedirectMiddleware:
    """Redirects all HTTP requests to /setup when app.state.setup_required is True.

    Exempt paths: /setup, /health, /static (so the setup page itself,
    health checks, and CSS/JS assets still work).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            path: str = scope["path"]
            app_state = scope.get("app")
            if (
                app_state is not None
                and getattr(app_state.state, "setup_required", False)
                and not any(path.startswith(prefix) for prefix in ALLOWED_PREFIXES)
            ):
                response = RedirectResponse(url="/setup", status_code=307)
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)
