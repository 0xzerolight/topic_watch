"""Migration 013: Add init_attempts counter to topics.

Tracks how many times a thin topic has been re-initialized after the LLM
reported insufficient data, so the scheduler's gradual NEW-topic init can
retry across cycles instead of marking a thin topic READY after one pass.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(topics)").fetchall()}
    if "init_attempts" not in columns:
        conn.execute("ALTER TABLE topics ADD COLUMN init_attempts INTEGER NOT NULL DEFAULT 0")
