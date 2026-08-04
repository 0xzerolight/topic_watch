"""Migration 025: Add the knowledge_revisions history table.

``knowledge_states`` remains the single current-state row read by the checker.
This table is history only: one row per persisted knowledge write (initial
research or a post-check update), so the UI can diff consecutive revisions.
Rows are only appended or pruned oldest-first — never rewritten.

``source`` is 'init' (initial research / re-initialize) or 'update' (a check
that found new info). ``change_note`` carries the novelty summary that prompted
an update; nothing else in the UI surfaces that string.

Backfills one 'init' revision per existing knowledge state, stamped with that
state's ``updated_at``, so pre-migration topics have a baseline to diff against.
Two guards:

* Skipped when the table is non-empty, so a hand re-run cannot duplicate it.
* The SELECT filters to states whose topic still exists. Connections run with
  ``PRAGMA foreign_keys=ON`` (app/database.py:98), so one orphaned row — from a
  pre-FK-era DB, a manual sqlite3 edit, or a partial restore — would otherwise
  raise IntegrityError, abort run_migrations, and leave the app unbootable.

The backfill is deliberately NOT wrapped in try/except. ``run_migrations``
records ``schema_version`` immediately after ``up()`` returns and commits per
migration (app/database.py:206-212), so swallowing an error here would mark
version 25 applied and lose every baseline permanently. Letting it raise is
strictly better: the CREATE statements autocommit (Python's sqlite3 opens an
implicit transaction only before DML), so the table survives while version 25 is
left unrecorded — the next boot re-runs ``up()``, the IF NOT EXISTS DDL no-ops,
the empty-table guard passes, and the backfill retries.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_revisions (
            id INTEGER PRIMARY KEY,
            topic_id INTEGER NOT NULL,
            summary_text TEXT NOT NULL,
            token_count INTEGER NOT NULL DEFAULT 0,
            source TEXT NOT NULL DEFAULT 'update',
            change_note TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
        )
    """)
    # Covering index: the timeline listing selects exactly these columns, so the
    # query is served from the index and never walks the overflow pages that a
    # ~40 KB summary_text spills onto.
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_knowledge_revisions_topic
            ON knowledge_revisions(topic_id, id DESC, created_at, token_count, source)
    """)

    if conn.execute("SELECT EXISTS(SELECT 1 FROM knowledge_revisions)").fetchone()[0]:
        return
    conn.execute("""
        INSERT INTO knowledge_revisions
            (topic_id, summary_text, token_count, source, change_note, created_at)
        SELECT topic_id, summary_text, token_count, 'init', NULL, updated_at
        FROM knowledge_states
        WHERE topic_id IN (SELECT id FROM topics)
    """)
