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
from app.migrations.m010_pending_webhooks import up as m010_up
from app.migrations.m011_topic_thresholds import up as m011_up
from app.migrations.m012_check_result_tokens import up as m012_up
from app.migrations.m013_topic_init_attempts import up as m013_up
from app.migrations.m014_perf_indexes import up as m014_up
from app.migrations.m015_check_result_stage_error import up as m015_up
from app.migrations.m016_pending_claimed_at import up as m016_up
from app.migrations.m017_pending_last_error import up as m017_up
from app.migrations.m018_article_published_at import up as m018_up

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
    (10, "add pending_webhooks table", m010_up),
    (11, "add confidence/relevance threshold columns to topics", m011_up),
    (12, "add prompt/completion token columns to check_results", m012_up),
    (13, "add init_attempts column to topics", m013_up),
    (14, "add performance indexes to articles", m014_up),
    (15, "add stage_error column to check_results", m015_up),
    (16, "add claimed_at column to retry queues", m016_up),
    (17, "add url/last_error columns to pending_notifications", m017_up),
    (18, "add published_at column to articles", m018_up),
]
