"""Migration 022: Add per-topic novelty instruction.

Nullable free-text column. NULL/empty means "no topic-specific criteria"; a
value is injected into the novelty-detection prompt as user-defined criteria
for what counts as new. Length is capped at the form boundary
(``NOVELTY_INSTRUCTION_MAX_CHARS``), not in the schema.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(topics)").fetchall()}
    if "novelty_instruction" not in columns:
        conn.execute("ALTER TABLE topics ADD COLUMN novelty_instruction TEXT DEFAULT NULL")
