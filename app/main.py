"""FastAPI application entry point.

Configures the web application with Jinja2 templates, database
initialization, scheduler lifecycle, and route mounting.
Run with: uvicorn app.main:app
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__ as _app_version
from app.config import DEFAULT_CONFIG_PATH, load_settings, resolve_db_path
from app.crud import recover_stuck_topics
from app.database import get_db, init_db
from app.logging_config import setup_logging
from app.scheduler import start_scheduler, stop_scheduler
from app.web.api import router as api_router
from app.web.csrf import CSRFMiddleware
from app.web.routers import router
from app.web.routers.templates import templates
from app.web.setup_middleware import SetupRedirectMiddleware

logger = logging.getLogger(__name__)


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
        # Wire the app so scheduler jobs read live settings from app.state (OVH-015/036).
        start_scheduler(settings, db_path=db_path, app=app)
        logger.info("Topic Watch web UI started")
    else:
        logger.info("Topic Watch started in setup mode — visit /setup to configure")

    yield
    stop_scheduler()
    logger.info("Topic Watch web UI stopped")


app = FastAPI(title="Topic Watch", version=_app_version, lifespan=lifespan)
app.add_middleware(CSRFMiddleware)
app.add_middleware(SetupRedirectMiddleware)
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
