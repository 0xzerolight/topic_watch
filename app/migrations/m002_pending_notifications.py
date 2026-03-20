"""Migration 002: Add pending_notifications table.

Stores notifications that failed to send for retry on subsequent
check cycles.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_notifications (
            id INTEGER PRIMARY KEY,
            topic_id INTEGER NOT NULL,
            check_result_id INTEGER,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL,
            retry_count INTEGER NOT NULL DEFAULT 0,
            max_retries INTEGER NOT NULL DEFAULT 3,
            FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
        )
    """)
