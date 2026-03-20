"""Migration 003: Add feed_mode column to topics.

Supports automatic (Google News) and manual RSS feed sources.
Existing topics with feed_urls are set to 'manual', others to 'auto'.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    # Check if column already exists (schema may include it on fresh DBs)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(topics)").fetchall()}
    if "feed_mode" not in columns:
        conn.execute("ALTER TABLE topics ADD COLUMN feed_mode TEXT NOT NULL DEFAULT 'auto'")
    conn.execute("UPDATE topics SET feed_mode = 'manual' WHERE feed_urls != '[]'")
