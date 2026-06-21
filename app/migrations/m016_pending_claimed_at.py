"""Migration 016: Add a claimed_at column to both retry queues.

The notification/webhook retry drains snapshotted every pending row, released
the connection, then sent each item with no per-row claim. Two overlapping
drains — a scheduler tick racing a UI/CLI check-all, or a second process —
therefore double-delivered every queued item (OVH-017).

This nullable column lets a drainer atomically claim a row
(``UPDATE ... SET claimed_at=? WHERE id=? AND claimed_at IS NULL``) and act
only on rows it won, which fences the cross-process case the in-process
``asyncio.Lock`` cannot cover. It is cleared again when a failed item is
re-queued so the next cycle can re-claim it.

Nullable with no default: existing rows and unclaimed rows read back as NULL.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    for table in ("pending_notifications", "pending_webhooks"):
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if "claimed_at" not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN claimed_at TEXT")
