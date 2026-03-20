"""Migration 001: Baseline.

Records the initial schema version. The base tables are created
by init_db() via CREATE TABLE IF NOT EXISTS.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """No-op — baseline schema already created by init_db."""
    pass
