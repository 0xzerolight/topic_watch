"""Migration 005: Add status_changed_at column to topics.

Tracks when a topic's status last changed, enabling detection of topics
stuck in RESEARCHING status during runtime (not just on server restart).
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(topics)").fetchall()}
    if "status_changed_at" not in columns:
        conn.execute("ALTER TABLE topics ADD COLUMN status_changed_at TEXT DEFAULT NULL")
        # Backfill existing rows: use created_at as a reasonable default
        conn.execute("UPDATE topics SET status_changed_at = created_at WHERE status_changed_at IS NULL")
