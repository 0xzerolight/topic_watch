"""Migration 014: Add performance indexes to the articles table.

Turns three hot-path queries from full table scans (plus temp B-tree sorts)
into index searches:

- ``articles(content_hash)`` — cross-topic dedup (``find_article_by_hash``)
  ran a full scan of every article row on each new feed entry.
- ``articles(fetched_at)`` — retention sweep (``delete_old_articles``)
  scanned every row to find expired articles.
- ``articles(topic_id, fetched_at DESC)`` — topic article listing/export
  (``list_articles_for_topic``) materialised a temp B-tree to sort.

Indexes only; the query text is unchanged here.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_content_hash_lookup ON articles(content_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_fetched_at ON articles(fetched_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_topic_fetched_at ON articles(topic_id, fetched_at DESC)")
