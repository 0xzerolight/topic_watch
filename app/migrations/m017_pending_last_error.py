"""Migration 017: Scope pending notifications to a URL + record last error.

A notification batch that partially failed (one of N Apprise targets down) was
stored as a single row covering the whole batch. On retry the drain re-sent to
every configured URL, re-delivering to the channels that had already succeeded
(OVH-039). This adds:

  * ``url`` — the single failed target a row covers (NULL on legacy/whole-batch
    rows; the drain then falls back to all configured URLs). One row per failed
    URL means retry only re-hits the channel that actually failed.
  * ``last_error`` — the most recent failure reason, for operator diagnostics
    (distinguish a transient blip from a permanently-broken token/chat id).

Both nullable with no default so existing rows read back as NULL.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(pending_notifications)").fetchall()}
    if "url" not in columns:
        conn.execute("ALTER TABLE pending_notifications ADD COLUMN url TEXT")
    if "last_error" not in columns:
        conn.execute("ALTER TABLE pending_notifications ADD COLUMN last_error TEXT")
