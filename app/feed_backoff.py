"""Soft, self-healing exponential backoff for persistently-failing feeds.

Pure function over a ``FeedHealth`` row. Derives a skip window from
``consecutive_failures`` + ``last_error_at`` with NO stored state: one success
resets ``consecutive_failures`` (see ``upsert_feed_health_success``) which
immediately clears the backoff. The delay is capped so every feed is always
retried eventually — feeds are never permanently disabled.

AUTO-mode provider backoff is owned by ``ProviderRouter`` (3 fails -> 30 min
cooldown); this helper is applied to MANUAL feed URLs only.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models import FeedHealth

BACKOFF_BASE_MINUTES = 15
BACKOFF_CAP_HOURS = 24
BACKOFF_THRESHOLD = 3


def feed_backoff_until(
    health: FeedHealth | None,
    *,
    base_minutes: int = BACKOFF_BASE_MINUTES,
    cap_hours: int = BACKOFF_CAP_HOURS,
    threshold: int = BACKOFF_THRESHOLD,
) -> datetime | None:
    """Return the UTC time before which the feed should be skipped, or None.

    None means "fetch now": no health row, no recorded error, or fewer than
    ``threshold`` consecutive failures.
    """
    if health is None or health.last_error_at is None:
        return None
    if health.consecutive_failures < threshold:
        return None
    exponent = health.consecutive_failures - threshold
    delay_minutes = min(base_minutes * (2**exponent), cap_hours * 60)
    return health.last_error_at + timedelta(minutes=delay_minutes)
