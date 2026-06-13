"""Migration 012: Add per-check token counts to check_results.

Records the total prompt/completion tokens consumed by a single check
(novelty analysis plus any knowledge init/update). Both default to 0 so
early-return checks that never call the LLM record 0.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(check_results)").fetchall()}
    if "prompt_tokens" not in columns:
        conn.execute("ALTER TABLE check_results ADD COLUMN prompt_tokens INTEGER NOT NULL DEFAULT 0")
    if "completion_tokens" not in columns:
        conn.execute("ALTER TABLE check_results ADD COLUMN completion_tokens INTEGER NOT NULL DEFAULT 0")
