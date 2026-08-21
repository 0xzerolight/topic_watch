"""FastAPI application entry point.

Configures the web application with Jinja2 templates, database
initialization, scheduler lifecycle, and route mounting.
Run with: uvicorn app.main:app
"""

import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app import __version__ as _app_version
from app.check_context import generate_check_id, request_id_var
from app.config import DEFAULT_CONFIG_PATH, load_settings, resolve_db_path
from app.crud import recover_stuck_topics
from app.database import get_db, init_db
from app.logging_config import setup_logging
from app.scheduler import start_scheduler, stop_scheduler
from app.web.api import router as api_router
from app.web.csrf import CSRFMiddleware
from app.web.host_allowlist import HostAllowlistMiddleware
from app.web.routers import router
from app.web.routers.templates import templates
from app.web.setup_middleware import SetupRedirectMiddleware

# Configured at import time, not from lifespan(): Uvicorn logs "Started server
# process [pid]" and "Waiting for application startup" through its own text
# handler BEFORE it ever awaits the ASGI lifespan startup event, so calling
# setup_logging() from inside lifespan() left both lines — and anything logged
# by an eagerly-imported dependency such as LiteLLM (imported transitively via
# app.scheduler above) — outside the configured JSON stream (AUG-249). Module
# import happens strictly before that point (uvicorn imports the app via
# ``config.load()`` before logging anything), and litellm is already imported
# above by the time this runs, so its logger exists to retarget.
setup_logging()

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

# A correlation id is copied into log lines and echoed back, so it needs a shape.
# Without one, tabs, C1 bytes and soft hyphens make log fields visually ambiguous
# and an arbitrarily long value is amplified into every record (AUG-331). The
# grammar covers what real proxies emit: nginx $request_id, Cloudflare Ray IDs,
# X-Ray trace ids, UUIDs.
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class RequestIdMiddleware:
    """Correlate each inbound request with an id surfaced to logs and echoed back.

    Reads the inbound ``X-Request-ID`` header when it matches
    :data:`REQUEST_ID_PATTERN` (so an upstream proxy's trace id is preserved),
    otherwise generates one — an id outside the grammar is replaced rather than
    reflected. Sets ``request_id_var`` for the request scope (the logging filter
    surfaces it as ``check_id``) and echoes the id back in the response header
    (OVH-043).

    Pure ASGI, not ``BaseHTTPMiddleware`` (AUG-271): ``BaseHTTPMiddleware``
    runs the downstream app in a separate task and sends the buffered
    response only after ``dispatch()`` returns, so resetting the context var
    in its ``finally`` happened before the response — and Uvicorn's access
    log line — were actually sent, losing correlation on exactly what this
    middleware exists to correlate. This middleware also sits *inside*
    ``ServerErrorMiddleware`` (added last = innermost of the user middleware
    stack), which unwinds past every inner middleware's own ``finally``
    before calling its registered handler — no middleware style fixes that by
    itself, so the id is additionally stashed on ``request.state`` (same
    ``scope``, so it survives into the ``Request`` that layer reconstructs)
    for ``unhandled_exception_handler`` to recover.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        inbound = Headers(scope=scope).get(REQUEST_ID_HEADER, "")
        request_id = inbound if REQUEST_ID_PATTERN.match(inbound) else generate_check_id()
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_var.set(request_id)

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).append(REQUEST_ID_HEADER, request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            request_id_var.reset(token)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: init DB, start scheduler on startup; stop on shutdown."""
    settings = load_settings()
    db_path = resolve_db_path(settings)
    init_db(db_path)
    app.state.settings = settings
    app.state.db_path = db_path
    app.state.config_path = DEFAULT_CONFIG_PATH
    app.state.setup_required = not settings.is_configured()

    if settings.is_configured():
        with get_db(db_path) as conn:
            recover_stuck_topics(conn)
        # Wire the app so scheduler jobs read live settings from app.state (OVH-015/036).
        start_scheduler(settings, db_path=db_path, app=app)
        logger.info("Topic Watch web UI started")
    else:
        logger.info("Topic Watch started in setup mode — visit /setup to configure")

    try:
        yield
    finally:
        # An exceptional or cancelled exit is thrown in at the ``yield``, so cleanup
        # placed after it was skipped entirely — leaving the only scheduler shutdown
        # hook uncalled and its timers and in-flight jobs alive past the failure
        # (AUG-265).
        stop_scheduler()
        logger.info("Topic Watch web UI stopped")


app = FastAPI(title="Topic Watch", version=_app_version, lifespan=lifespan)
app.add_middleware(CSRFMiddleware)
app.add_middleware(SetupRedirectMiddleware)
# Host check runs before routing, setup redirects and CSRF issuance, so a rebound
# attacker domain never receives a CSRF token or a redirect (AUG-002).
app.add_middleware(HostAllowlistMiddleware)
# Added last so it wraps everything: the correlation id is set before any other
# middleware/handler runs and is available to all of their log lines.
app.add_middleware(RequestIdMiddleware)
app.include_router(router)
app.include_router(api_router)
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")


def _wants_json(request: Request) -> bool:
    """Return True if the request is for the JSON API (not browser HTML)."""
    accept = request.headers.get("accept", "")
    return request.url.path.startswith("/api/") or ("application/json" in accept and "text/html" not in accept)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> HTMLResponse | JSONResponse:
    """Render HTTP errors as HTML for browsers, JSON for API clients."""
    if _wants_json(request):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    from app import __version__

    return templates.TemplateResponse(
        request,
        "error.html",
        {"status_code": exc.status_code, "detail": exc.detail, "version": __version__},
        status_code=exc.status_code,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> HTMLResponse | JSONResponse:
    """Render validation errors as HTML for browsers, JSON for API clients."""
    if _wants_json(request):
        return JSONResponse({"detail": exc.errors()}, status_code=422)

    from app import __version__

    return templates.TemplateResponse(
        request,
        "error.html",
        {"status_code": 422, "detail": "Invalid request", "version": __version__},
        status_code=422,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> HTMLResponse | JSONResponse:
    """Catch-all for unhandled errors: branded HTML for browsers, JSON for API clients.

    Logs the full exception server-side but never leaks the traceback or internal
    detail to the client (mirrors the two dual-render handlers above).

    Starlette's ``ServerErrorMiddleware`` calls this handler *outside*
    ``RequestIdMiddleware`` — after that middleware's own context has already
    unwound (see its docstring) — so ``request_id_var`` cannot be trusted here.
    Recover the id from ``request.state`` instead, restore it just for this log
    line, and attach the response header directly: this is the one handler
    ``send_with_request_id`` never wraps (AUG-271).
    """
    request_id = getattr(request.state, "request_id", None) or "-"
    token = request_id_var.set(request_id)
    try:
        logger.exception("Unhandled exception while handling %s %s", request.method, request.url.path)
    finally:
        request_id_var.reset(token)

    if _wants_json(request):
        response: HTMLResponse | JSONResponse = JSONResponse({"detail": "Internal server error"}, status_code=500)
    else:
        from app import __version__

        response = templates.TemplateResponse(
            request,
            "error.html",
            {"status_code": 500, "detail": "Something went wrong.", "version": __version__},
            status_code=500,
        )

    response.headers[REQUEST_ID_HEADER] = request_id
    return response
