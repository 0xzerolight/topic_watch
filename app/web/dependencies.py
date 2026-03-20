"""FastAPI dependencies for database access and settings."""

import sqlite3
from collections.abc import Generator

from fastapi import Request

from app.config import Settings
from app.database import get_db


def get_db_conn(request: Request) -> Generator[sqlite3.Connection, None, None]:
    """Yield a database connection per request with auto-commit/rollback."""
    db_path = getattr(request.app.state, "db_path", None)
    with get_db(db_path) as conn:
        yield conn


def get_settings(request: Request) -> Settings:
    """Get application settings from app state."""
    settings: Settings = request.app.state.settings
    return settings
