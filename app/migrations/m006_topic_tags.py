"""Migration 006: Add tags column to topics.

Enables organizing topics into user-defined categories for filtering.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(topics)").fetchall()}
    if "tags" not in columns:
        conn.execute("ALTER TABLE topics ADD COLUMN tags TEXT NOT NULL DEFAULT '[]'")
