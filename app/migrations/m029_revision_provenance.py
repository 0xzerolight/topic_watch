"""Migration 029: record what each knowledge revision was derived from.

A revision stores text, a token count and a timestamp, which is not enough to
interpret either of them later:

* ``model`` — the configured LLM that wrote the summary, and therefore the
  tokenizer identity ``token_count`` is measured in. The model is switchable at
  any time and the counter falls back to a character estimate when a tokenizer is
  missing, so subtracting two counts produced under different models reports a
  unit change as knowledge growth (AUG-255). The timeline now shows a delta only
  when both sides name the same model.
* ``basis_hash`` — a fingerprint of the topic scope (name, description, feed
  mode, feed URLs, novelty instruction) the summary was derived from. Without it
  a topic edited after its baseline was built keeps comparing new articles
  against knowledge assembled for the old scope, with nothing on screen saying so
  (TW-AUD-017).

Both are nullable and stay NULL on every existing row, including the ones m025
backfilled from ``knowledge_states``: their provenance genuinely is unknown, and
inventing the current model for them would assert something false about history.

Each ALTER is guarded by a PRAGMA table_info check so a re-run after a partial
failure is a no-op rather than an ``OperationalError`` — SQLite has no
``ADD COLUMN IF NOT EXISTS``.
"""

import sqlite3

_COLUMNS = (
    ("model", "ALTER TABLE knowledge_revisions ADD COLUMN model TEXT"),
    ("basis_hash", "ALTER TABLE knowledge_revisions ADD COLUMN basis_hash TEXT"),
)


def up(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(knowledge_revisions)").fetchall()}
    for name, statement in _COLUMNS:
        if name not in existing:
            conn.execute(statement)
