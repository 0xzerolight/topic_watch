"""Migration 020: Add a seen_at column to check_results.

The dashboard "Ready · new info" badge is driven solely by the latest check's
``has_new_info``, which is write-once: nothing acknowledged it when the user
opened a topic and read the new info, so the badge persisted until a later check
happened to find nothing new. ``has_new_info`` cannot be cleared on view because
it also drives the detail-page history column and the Notify button.

This nullable column records when the user first opened a topic whose latest
check carried new info. The badge is gated on ``has_new_info AND seen_at IS
NULL``; opening the detail page stamps ``seen_at`` on that latest row.

Nullable with no default: existing rows read back as NULL (unseen), so any
genuinely-unacknowledged badge correctly persists until the topic is first
opened. No backfill.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(check_results)").fetchall()}
    if "seen_at" not in columns:
        conn.execute("ALTER TABLE check_results ADD COLUMN seen_at TEXT")
