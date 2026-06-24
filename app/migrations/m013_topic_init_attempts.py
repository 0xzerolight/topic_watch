"""Migration 013: Add init_attempts counter to topics.

Legacy column. It was added to drive multi-round initialization (bounce a thin
topic back to NEW and retry across cycles when the LLM reported insufficient
data), but that retry behavior was removed in b3b994c: thin topics now reach
READY on the first pass with a baseline summary. The column is retained for
schema continuity and is reset to 0 on the READY transition; it no longer gates
any retry.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(topics)").fetchall()}
    if "init_attempts" not in columns:
        conn.execute("ALTER TABLE topics ADD COLUMN init_attempts INTEGER NOT NULL DEFAULT 0")
