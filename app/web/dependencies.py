"""FastAPI dependencies for database access and settings."""

import sqlite3
from collections.abc import Generator

from fastapi import Request

from app.config import Settings
from app.database import get_db


def get_db_conn(request: Request) -> Generator[sqlite3.Connection, None, None]:
    """Yield a database connection for the handler with auto-commit/rollback.

    **Always depend on this with** ``Depends(get_db_conn, scope="function")``.

    An unscoped yield-dependency is torn down by FastAPI's *request* exit stack,
    which Starlette closes only after response background tasks have finished. A
    handler that enqueues a check or an init therefore kept its route connection
    — already finished with, doing nothing — alive for the whole minutes-long
    task, on top of the connections the task itself opens (AUG-209). Function
    scope closes it when the handler returns, before any background task starts.

    Safe for every handler here because none of them stream lazily from the
    connection: the export routes build their payload in memory first and hand
    ``StreamingResponse`` a finished string.
    """
    db_path = getattr(request.app.state, "db_path", None)
    with get_db(db_path) as conn:
        yield conn


def get_settings(request: Request) -> Settings:
    """Get application settings from app state."""
    settings: Settings = request.app.state.settings
    return settings
