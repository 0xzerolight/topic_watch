"""Migration 024: Silence Heartbeat latch.

NULL means no "sources failing" alert is outstanding for the topic. The checker
stamps a UTC ISO timestamp when it announces an outage and clears it back to NULL
on the first healthy check afterwards, which is what makes the alert fire once
per outage instead of once per check. Written only through
``claim_heartbeat_alert`` / ``clear_heartbeat_alert`` (conditional UPDATEs), so
it is deliberately absent from the create_topic/update_topic column lists.
"""

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(topics)").fetchall()}
    if "heartbeat_alerted_at" not in columns:
        conn.execute("ALTER TABLE topics ADD COLUMN heartbeat_alerted_at TEXT DEFAULT NULL")
