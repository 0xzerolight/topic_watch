"""Add source_provider column to articles table.

Tracks which news provider (e.g. 'bing_news', 'google_news') was used
to fetch each article. NULL for articles fetched before multi-provider
support and for MANUAL mode articles.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE articles ADD COLUMN source_provider TEXT")
