"""Migration 027: canonicalize stored topic tags.

``Topic`` now canonicalizes tags on every construction (NFC, invisible
formatting characters removed, whitespace collapsed, stable deduplication), but
the dashboard chip filter and ``/api/v1/topics?tag=`` match with SQL equality
against the stored JSON. Rows written before that validator existed therefore
keep variants the filter can never select. Rewrite them once, here.

Rows whose ``tags`` column is not a JSON array are left alone: ``_safe_json``
already degrades those to an empty list on read, and rewriting them would
destroy whatever a future version or a hand edit put there.
"""

import json
import sqlite3

from app.models import normalize_tags


def up(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT id, tags FROM topics").fetchall()
    for row in rows:
        raw = row[1]
        if not raw:
            continue
        try:
            stored = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(stored, list):
            continue
        canonical = normalize_tags(str(tag) for tag in stored)
        if canonical == stored:
            continue
        conn.execute("UPDATE topics SET tags = ? WHERE id = ?", (json.dumps(canonical), row[0]))
