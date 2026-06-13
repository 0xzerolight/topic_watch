"""Migration 010: Add pending_webhooks table.

Stores webhook deliveries that failed to send, for retry on subsequent
check cycles. Mirrors the pending_notifications retry queue.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_webhooks (
            id INTEGER PRIMARY KEY,
            topic_id INTEGER NOT NULL,
            check_result_id INTEGER,
            url TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL,
            retry_count INTEGER NOT NULL DEFAULT 0,
            max_retries INTEGER NOT NULL DEFAULT 3,
            FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
        )
    """)
