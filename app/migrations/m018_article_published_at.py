"""Add published_at column to articles table.

Stores the article's publication timestamp parsed from the RSS/Atom feed
(ISO-8601 TEXT, UTC). NULL for articles fetched before this migration and
for entries whose feed omitted a parseable date.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE articles ADD COLUMN published_at TEXT")
