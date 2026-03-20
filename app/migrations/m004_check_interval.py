"""Migration 004: Add check_interval_hours column to topics.

Allows per-topic check intervals. NULL means use the global default.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(topics)").fetchall()}
    if "check_interval_hours" not in columns:
        conn.execute("ALTER TABLE topics ADD COLUMN check_interval_hours INTEGER DEFAULT NULL")
