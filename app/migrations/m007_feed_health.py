"""Migration 007: Add feed_health table for per-feed error tracking."""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feed_health (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feed_url TEXT UNIQUE NOT NULL,
            last_success_at TEXT,
            last_error_at TEXT,
            last_error_message TEXT,
            consecutive_failures INTEGER DEFAULT 0,
            total_fetches INTEGER DEFAULT 0,
            total_failures INTEGER DEFAULT 0
        )
    """)
