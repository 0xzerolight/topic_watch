"""Migration 030: durable command intents for accepted background checks.

``POST /topics/{id}/check`` and ``POST /topics/bulk-check`` acknowledged a check
after merely registering a Starlette background task, so a crash after the
response lost the accepted command with nothing on disk to say it was ever
asked for; a bulk request left an unknowable completed prefix (AUG-286). Each
accepted topic now gets one row here BEFORE the response, and the scheduler's
check cycle resumes rows a dead process left behind.

* ``request_id`` — the inbound web request's correlation id, for log and
  ledger correlation only. Not unique: a replayed request admits a second
  intent, which is harmless (it loses the guard, or is satisfied by the newer
  result), whereas refusing it would drop a real click from any client that
  reuses ``X-Request-ID``.
* ``baseline_check_id`` — ``MAX(check_results.id)`` at admission. Any newer row
  satisfies the intent, so a check that committed and then lost its apply is
  not run a second time.
* ``status`` — 'pending' | 'running' | 'done' | 'abandoned', with the same
  ``claim_token`` fence, wall-clock ``claimed_at`` lease and ``next_attempt_at``
  backoff the delivery intents use (m026). ``attempts`` counts claims, so a
  check that takes the process down is bounded by ``max_attempts`` too.

No topic generation is stored: ``topics.generation`` is never rewritten, and
a deleted-and-recreated topic cascade-deletes its intents before any claim.

``CREATE TABLE IF NOT EXISTS`` / ``CREATE INDEX IF NOT EXISTS`` make a re-run
after a partial failure a no-op; nothing here needs a PRAGMA guard.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS check_intents (
            id INTEGER PRIMARY KEY,
            request_id TEXT NOT NULL,
            topic_id INTEGER NOT NULL,
            baseline_check_id INTEGER,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 3,
            next_attempt_at TEXT,
            claimed_at TEXT,
            claim_token TEXT,
            check_result_id INTEGER,
            last_error TEXT,
            FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_check_intents_due ON check_intents(status, next_attempt_at)")
