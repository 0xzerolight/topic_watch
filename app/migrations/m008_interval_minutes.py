"""Migration 008: Convert check_interval_hours to check_interval_minutes.

Adds check_interval_minutes column and migrates existing per-topic hour values
to minutes. The global default (config) stays in hours for backwards compatibility.
The old check_interval_hours column is kept (harmless, SQLite drop-column requires
newer versions).
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(topics)").fetchall()}
    if "check_interval_minutes" not in columns:
        conn.execute("ALTER TABLE topics ADD COLUMN check_interval_minutes INTEGER DEFAULT NULL")
        # Migrate existing per-topic hour values to minutes
        conn.execute(
            "UPDATE topics SET check_interval_minutes = check_interval_hours * 60 "
            "WHERE check_interval_hours IS NOT NULL"
        )
