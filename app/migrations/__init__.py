"""Database migration registry.

Migrations are applied in version order. Each migration is a
(version, description, up_function) tuple.
"""

import sqlite3
from collections.abc import Callable

from app.migrations.m001_baseline import up as m001_up
from app.migrations.m002_pending_notifications import up as m002_up
from app.migrations.m003_feed_mode import up as m003_up
from app.migrations.m004_check_interval import up as m004_up
from app.migrations.m005_status_changed_at import up as m005_up
from app.migrations.m006_topic_tags import up as m006_up
from app.migrations.m007_feed_health import up as m007_up
from app.migrations.m008_interval_minutes import up as m008_up
from app.migrations.m009_article_provider import up as m009_up

MIGRATIONS: list[tuple[int, str, Callable[[sqlite3.Connection], None]]] = [
    (1, "baseline schema version", m001_up),
    (2, "add pending_notifications table", m002_up),
    (3, "add feed_mode column to topics", m003_up),
    (4, "add check_interval_hours column to topics", m004_up),
    (5, "add status_changed_at column to topics", m005_up),
    (6, "add tags column to topics", m006_up),
    (7, "add feed_health table", m007_up),
    (8, "add check_interval_minutes column to topics", m008_up),
    (9, "add source_provider column to articles", m009_up),
]
