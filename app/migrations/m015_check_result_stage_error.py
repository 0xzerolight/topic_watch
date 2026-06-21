"""Migration 015: Add a generic stage_error column to check_results.

The only typed error channel on check_results was notification_error, so the
three structurally distinct failure stages — scrape, LLM analysis, and
knowledge-update — collapsed into rows indistinguishable from a clean
"nothing new" check. This nullable column records a short, machine-readable
reason ('scrape_failed'/'analysis_failed'/'knowledge_update_failed') plus an
exception summary, set at each swallow site in the check pipeline.

Nullable with no default: existing rows and clean runs read back as NULL.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(check_results)").fetchall()}
    if "stage_error" not in columns:
        conn.execute("ALTER TABLE check_results ADD COLUMN stage_error TEXT")
