"""Migration 023: Add per-topic importance notify threshold.

Nullable column. NULL means "no suppression" (notify on any importance); a
concrete value (1-5) suppresses notifications for new info whose LLM
importance score falls below it. There is deliberately no global setting: a
global default of 1 would be a functional no-op, so the gate falls back to a
literal 1 in the checker.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(topics)").fetchall()}
    if "importance_threshold" not in columns:
        conn.execute("ALTER TABLE topics ADD COLUMN importance_threshold INTEGER DEFAULT NULL")
