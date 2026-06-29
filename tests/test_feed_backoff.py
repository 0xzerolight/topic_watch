"""Tests for the soft exponential feed-backoff helper."""

from datetime import UTC, datetime, timedelta

from app.feed_backoff import feed_backoff_until
from app.models import FeedHealth


def _health(consecutive_failures: int, last_error_at: datetime | None) -> FeedHealth:
    return FeedHealth(
        feed_url="https://ex.com/feed",
        consecutive_failures=consecutive_failures,
        last_error_at=last_error_at,
    )


def test_none_health_not_backed_off():
    assert feed_backoff_until(None) is None


def test_below_threshold_not_backed_off():
    err = datetime(2026, 1, 1, tzinfo=UTC)
    assert feed_backoff_until(_health(2, err), threshold=3) is None


def test_no_error_timestamp_not_backed_off():
    assert feed_backoff_until(_health(5, None)) is None


def test_at_threshold_uses_base_delay():
    err = datetime(2026, 1, 1, tzinfo=UTC)
    # consecutive_failures == threshold -> exponent 0 -> base delay
    assert feed_backoff_until(_health(3, err), base_minutes=15, threshold=3) == err + timedelta(minutes=15)


def test_exponential_growth():
    err = datetime(2026, 1, 1, tzinfo=UTC)
    # exponent = 5 - 3 = 2 -> 15 * 4 = 60 min
    assert feed_backoff_until(_health(5, err), base_minutes=15, threshold=3) == err + timedelta(minutes=60)


def test_capped():
    err = datetime(2026, 1, 1, tzinfo=UTC)
    # huge failure count -> capped at cap_hours
    assert feed_backoff_until(_health(50, err), base_minutes=15, cap_hours=24, threshold=3) == err + timedelta(hours=24)
