"""Migration 026: schema for the wave-A durability rework.

Landed as ONE migration because wave A is a single architectural unit: the
columns below are consumed by the phased ``check_topic`` pipeline (CAS knowledge
write + generation guard), by the failure-path semantics that follow it, and by
durable per-target delivery intents. Splitting them across three migrations would
rewrite the same tables three times for no benefit, and the C3 transaction body
already needs ``topics.generation`` + ``knowledge_states.version`` to exist.

Columns not yet read by any code path are inert — a NOT NULL DEFAULT costs one
table rewrite now instead of another one later.

Semantics:

* ``knowledge_states.version`` — compare-and-swap counter. A knowledge write
  bumps it and is rejected when the row moved since the snapshot was taken, so
  two overlapping checks can never interleave a lost update.
* ``topics.generation`` — an opaque, never-reused per-topic identity. ``topics.id``
  is a recyclable rowid, so a delete+recreate can hand a stale in-flight worker a
  live row; the generation guard makes that worker's apply a no-op. Backfilled
  with random hex for existing rows.
* ``articles.analysis_attempts`` — per-article analysis attempt counter, so a
  repeatedly-failing article is abandoned instead of retried forever.
* ``check_results.notify_disposition`` — why this check did (or did not) notify:
  'sent', 'suppressed_importance', 'below_confidence', 'below_relevance',
  'no_new_info', 'analysis_failed', 'pending'. NULL on pre-migration rows.
* ``pending_notifications`` / ``pending_webhooks`` gain the delivery-intent
  lifecycle: 'pending' | 'sending' | 'sent' | 'abandoned' | 'revoked', an
  immutable ``claim_token`` that fences a late apply from a stale worker, and a
  ``next_attempt_at`` due-time so a retry backoff survives a restart. Existing
  rows default to 'pending', which is correct — they are undelivered.

Every ALTER is guarded by a PRAGMA table_info check so a re-run after a partial
failure (``run_migrations`` records the version only after ``up()`` returns) is a
no-op rather than an ``OperationalError``: SQLite has no ``ADD COLUMN IF NOT
EXISTS``.
"""

import sqlite3

_ADDITIONS: dict[str, list[tuple[str, str]]] = {
    "knowledge_states": [
        ("version", "ALTER TABLE knowledge_states ADD COLUMN version INTEGER NOT NULL DEFAULT 0"),
    ],
    "topics": [
        ("generation", "ALTER TABLE topics ADD COLUMN generation TEXT NOT NULL DEFAULT ''"),
    ],
    "articles": [
        ("analysis_attempts", "ALTER TABLE articles ADD COLUMN analysis_attempts INTEGER NOT NULL DEFAULT 0"),
    ],
    "check_results": [
        ("notify_disposition", "ALTER TABLE check_results ADD COLUMN notify_disposition TEXT"),
    ],
    "pending_notifications": [
        ("status", "ALTER TABLE pending_notifications ADD COLUMN status TEXT NOT NULL DEFAULT 'pending'"),
        ("kind", "ALTER TABLE pending_notifications ADD COLUMN kind TEXT NOT NULL DEFAULT 'novelty'"),
        ("claim_token", "ALTER TABLE pending_notifications ADD COLUMN claim_token TEXT"),
        ("next_attempt_at", "ALTER TABLE pending_notifications ADD COLUMN next_attempt_at TEXT"),
        ("latch_value", "ALTER TABLE pending_notifications ADD COLUMN latch_value TEXT"),
        ("delivered_at", "ALTER TABLE pending_notifications ADD COLUMN delivered_at TEXT"),
    ],
    "pending_webhooks": [
        ("status", "ALTER TABLE pending_webhooks ADD COLUMN status TEXT NOT NULL DEFAULT 'pending'"),
        ("claim_token", "ALTER TABLE pending_webhooks ADD COLUMN claim_token TEXT"),
        ("next_attempt_at", "ALTER TABLE pending_webhooks ADD COLUMN next_attempt_at TEXT"),
        ("last_error", "ALTER TABLE pending_webhooks ADD COLUMN last_error TEXT"),
        ("delivered_at", "ALTER TABLE pending_webhooks ADD COLUMN delivered_at TEXT"),
    ],
}


def up(conn: sqlite3.Connection) -> None:
    for table, additions in _ADDITIONS.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for column, ddl in additions:
            if column not in existing:
                conn.execute(ddl)

    # Backfill the generation identity for rows that predate the column. Random
    # per row (not a constant) so two pre-migration topics never share one.
    conn.execute("UPDATE topics SET generation = lower(hex(randomblob(8))) WHERE generation = ''")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pending_notifications_due ON pending_notifications(status, next_attempt_at)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_webhooks_due ON pending_webhooks(status, next_attempt_at)")
