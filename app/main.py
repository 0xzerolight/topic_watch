"""FastAPI application entry point.

Configures the web application with Jinja2 templates, database
initialization, scheduler lifecycle, and route mounting.
Run with: uvicorn app.main:app
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import load_settings, resolve_db_path
from app.crud import recover_stuck_topics
from app.database import get_db, init_db
from app.logging_config import setup_logging
from app.scheduler import start_scheduler, stop_scheduler
from app.web.csrf import CSRFMiddleware
from app.web.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: init DB, start scheduler on startup; stop on shutdown."""
    setup_logging()
    settings = load_settings()
    db_path = resolve_db_path(settings)
    init_db(db_path)
    with get_db(db_path) as conn:
        recover_stuck_topics(conn)
    app.state.settings = settings
    app.state.db_path = db_path
    start_scheduler(settings, db_path=db_path)
    logger.info("Topic Watch web UI started")
    yield
    stop_scheduler()
    logger.info("Topic Watch web UI stopped")


app = FastAPI(title="Topic Watch", lifespan=lifespan)
app.add_middleware(CSRFMiddleware)
app.include_router(router)
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")
