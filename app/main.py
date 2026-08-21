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
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app import __version__ as _app_version
from app.check_context import generate_check_id, request_id_var
from app.config import DEFAULT_CONFIG_PATH, load_settings, resolve_db_path
from app.crud import prune_all_knowledge_revisions, recover_stuck_topics, reset_all_heartbeat_state
from app.database import get_db, init_db
from app.logging_config import setup_logging
from app.scheduler import start_scheduler, stop_scheduler
from app.web.api import router as api_router
from app.web.csrf import CSRFMiddleware
from app.web.host_allowlist import HostAllowlistMiddleware
from app.web.routers import router
from app.web.routers.templates import templates
from app.web.setup_middleware import SetupRedirectMiddleware

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

# A correlation id is copied into log lines and echoed back, so it needs a shape.
# Without one, tabs, C1 bytes and soft hyphens make log fields visually ambiguous
# and an arbitrarily long value is amplified into every record (AUG-331). The
# grammar covers what real proxies emit: nginx $request_id, Cloudflare Ray IDs,
# X-Ray trace ids, UUIDs.
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Correlate each inbound request with an id surfaced to logs and echoed back.

    Reads the inbound ``X-Request-ID`` header when it matches
    :data:`REQUEST_ID_PATTERN` (so an upstream proxy's trace id is preserved),
    otherwise generates one — an id outside the grammar is replaced rather than
    reflected. Sets ``request_id_var`` for the request scope (the logging filter
    surfaces it as ``check_id``) and echoes the id back in the response header
    (OVH-043).
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        inbound = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = inbound if REQUEST_ID_PATTERN.match(inbound) else generate_check_id()
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: init DB, start scheduler on startup; stop on shutdown."""
    setup_logging()
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
            # Revision pruning otherwise runs only when a topic is written, so
            # lowering knowledge_revision_limit left quiet and finished topics
            # holding their full snapshots forever — the configured cap silently
            # did not apply to them (AUG-034). One indexed pass here makes it.
            pruned = prune_all_knowledge_revisions(conn, settings.knowledge_revision_limit)
            if pruned:
                logger.info("Knowledge revision retention: pruned %d revision(s) over the limit", pruned)
            if settings.silence_heartbeat_checks <= 0:
                # The per-check reset only reaches a topic when that topic next
                # runs, so a latch parked while the feature was on survives being
                # switched off — and fires a phantom recovery when it comes back
                # (AUG-260). Reconcile once, here, where the effective setting is
                # first known.
                cleared = reset_all_heartbeat_state(conn)
                if cleared:
                    logger.info("Silence Heartbeat is off: cleared %d parked outage latch(es)", cleared)
            conn.commit()
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
    """
    logger.exception("Unhandled exception while handling %s %s", request.method, request.url.path)

    if _wants_json(request):
        return JSONResponse({"detail": "Internal server error"}, status_code=500)

    from app import __version__

    return templates.TemplateResponse(
        request,
        "error.html",
        {"status_code": 500, "detail": "Something went wrong.", "version": __version__},
        status_code=500,
    )
