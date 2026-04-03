"""FastAPI application entry point.

Configures the web application with Jinja2 templates, database
initialization, scheduler lifecycle, and route mounting.
Run with: uvicorn app.main:app
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import DEFAULT_CONFIG_PATH, load_settings, resolve_db_path
from app.crud import recover_stuck_topics
from app.database import get_db, init_db
from app.logging_config import setup_logging
from app.scheduler import start_scheduler, stop_scheduler
from app.web.csrf import CSRFMiddleware
from app.web.routes import router
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
        start_scheduler(settings, db_path=db_path)
        logger.info("Topic Watch web UI started")
    else:
        logger.info("Topic Watch started in setup mode — visit /setup to configure")

    yield
    stop_scheduler()
    logger.info("Topic Watch web UI stopped")


app = FastAPI(title="Topic Watch", lifespan=lifespan)
app.add_middleware(CSRFMiddleware)
app.add_middleware(SetupRedirectMiddleware)
app.include_router(router)
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")

_error_templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> HTMLResponse:
    """Render HTTP errors using the app's error template instead of raw JSON."""
    from app import __version__

    return _error_templates.TemplateResponse(
        request,
        "error.html",
        {"status_code": exc.status_code, "detail": exc.detail, "version": __version__},
        status_code=exc.status_code,
    )
