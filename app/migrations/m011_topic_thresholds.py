"""Migration 011: Add per-topic confidence/relevance threshold overrides.

Both columns are nullable. NULL means "inherit the global
min_confidence_threshold / min_relevance_threshold from settings"; a concrete
value (0.0-1.0) overrides the global gate for that topic.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(topics)").fetchall()}
    if "confidence_threshold" not in columns:
        conn.execute("ALTER TABLE topics ADD COLUMN confidence_threshold REAL DEFAULT NULL")
    if "relevance_threshold" not in columns:
        conn.execute("ALTER TABLE topics ADD COLUMN relevance_threshold REAL DEFAULT NULL")
