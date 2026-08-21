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

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models import FeedHealth

logger = logging.getLogger(__name__)

BACKOFF_BASE_MINUTES = 15
BACKOFF_CAP_HOURS = 24
BACKOFF_THRESHOLD = 3


def feed_backoff_until(
    health: FeedHealth | None,
    *,
    now: datetime | None = None,
    base_minutes: int = BACKOFF_BASE_MINUTES,
    cap_hours: int = BACKOFF_CAP_HOURS,
    threshold: int = BACKOFF_THRESHOLD,
) -> datetime | None:
    """Return the UTC time before which the feed should be skipped, or None.

    None means "fetch now": no health row, no recorded error, or fewer than
    ``threshold`` consecutive failures.

    ``cap_hours`` is a cap on ELAPSED time, so the anchor is clamped to now: a
    failure stamped during a temporary forward clock jump (or restored from a
    machine whose clock ran fast) otherwise kept a manual feed skipped for the
    skew PLUS the delay, days past the configured cap, and a skipped feed never
    runs the health callback that would clear it (AUG-281). ``now`` is injectable
    so callers that already read the clock — and tests — can pass one.
    """
    if health is None or health.last_error_at is None:
        return None
    if health.consecutive_failures < threshold:
        return None
    exponent = health.consecutive_failures - threshold
    delay_minutes = min(base_minutes * (2**exponent), cap_hours * 60)
    anchor = health.last_error_at
    moment = now or datetime.now(UTC)
    if anchor > moment:
        logger.warning(
            "Feed health for %s carries a failure timestamped in the future (%s); backing off from now instead",
            health.feed_url,
            anchor.isoformat(),
        )
        anchor = moment
    return anchor + timedelta(minutes=delay_minutes)
