"""Tests for provider routing with health-based cascade."""

from datetime import UTC, datetime, timedelta

from app.scraping.providers import BingNewsProvider, GoogleNewsProvider
from app.scraping.routing import (
    _FAILURE_THRESHOLD,
    _UNHEALTHY_COOLDOWN,
    ProviderRouter,
)


def _make_router() -> ProviderRouter:
    """Create a fresh router with default providers."""
    return ProviderRouter(providers=[BingNewsProvider(), GoogleNewsProvider()])


class TestGetProvider:
    def test_returns_first_when_all_healthy(self) -> None:
        router = _make_router()
        provider = router.get_provider()
        assert provider.name == "bing_news"

    def test_returns_second_when_first_unhealthy(self) -> None:
        router = _make_router()
        # Mark Bing as unhealthy (3 consecutive failures)
        for _ in range(_FAILURE_THRESHOLD):
            router.mark_unhealthy("bing_news")
        provider = router.get_provider()
        assert provider.name == "google_news"

    def test_returns_first_when_all_unhealthy(self) -> None:
        """When all providers are unhealthy, return first as best effort."""
        router = _make_router()
        for _ in range(_FAILURE_THRESHOLD):
            router.mark_unhealthy("bing_news")
            router.mark_unhealthy("google_news")
        provider = router.get_provider()
        assert provider.name == "bing_news"


class TestGetNextProvider:
    def test_returns_next_healthy(self) -> None:
        router = _make_router()
        bing = router.providers[0]
        next_provider = router.get_next_provider(bing)
        assert next_provider is not None
        assert next_provider.name == "google_news"

    def test_returns_none_for_last_provider(self) -> None:
        router = _make_router()
        google = router.providers[1]
        assert router.get_next_provider(google) is None

    def test_skips_unhealthy_next(self) -> None:
        router = _make_router()
        bing = router.providers[0]
        # Mark Google as unhealthy
        for _ in range(_FAILURE_THRESHOLD):
            router.mark_unhealthy("google_news")
        assert router.get_next_provider(bing) is None

    def test_falls_back_when_all_unhealthy(self) -> None:
        """When BOTH providers are unhealthy, get_next_provider still returns
        the other provider (best effort), matching get_provider's behaviour."""
        router = _make_router()
        bing = router.providers[0]
        for _ in range(_FAILURE_THRESHOLD):
            router.mark_unhealthy("bing_news")
            router.mark_unhealthy("google_news")
        next_provider = router.get_next_provider(bing)
        assert next_provider is not None
        assert next_provider.name == "google_news"


class TestMarkUnhealthy:
    def test_single_failure_stays_healthy(self) -> None:
        router = _make_router()
        router.mark_unhealthy("bing_news")
        # Still below threshold — should be healthy
        provider = router.get_provider()
        assert provider.name == "bing_news"

    def test_threshold_failures_becomes_unhealthy(self) -> None:
        router = _make_router()
        for _ in range(_FAILURE_THRESHOLD):
            router.mark_unhealthy("bing_news")
        provider = router.get_provider()
        assert provider.name == "google_news"

    def test_records_last_failure_time(self) -> None:
        router = _make_router()
        router.mark_unhealthy("bing_news")
        health = router._health["bing_news"]
        assert health.last_failure is not None
        assert health.consecutive_failures == 1


class TestMarkHealthy:
    def test_resets_failure_count(self) -> None:
        router = _make_router()
        router.mark_unhealthy("bing_news")
        router.mark_unhealthy("bing_news")
        router.mark_healthy("bing_news")
        assert "bing_news" not in router._health

    def test_noop_for_unknown_provider(self) -> None:
        router = _make_router()
        router.mark_healthy("nonexistent")  # Should not raise


class TestMonotonicHealthAccounting:
    """OVH-127: a stale success must not wipe a fresh cooldown.

    The read-fetch-mark sequence in ``_fetch_auto`` spans an ``await``: a check
    that started earlier can succeed and call ``mark_healthy`` *after* a
    concurrent check has just tripped the cooldown via ``mark_unhealthy``. The
    accounting must be monotonic — capture the health epoch before the fetch,
    then reset only if no failure was recorded in the interim.
    """

    def test_health_epoch_advances_on_failure(self) -> None:
        router = _make_router()
        e0 = router.health_epoch("bing_news")
        router.mark_unhealthy("bing_news")
        e1 = router.health_epoch("bing_news")
        assert e1 > e0

    def test_stale_mark_healthy_does_not_clear_fresh_cooldown(self) -> None:
        router = _make_router()
        # A check captures the epoch before its (slow) fetch.
        observed = router.health_epoch("bing_news")
        # Meanwhile, a concurrent check fails three times and trips cooldown.
        for _ in range(_FAILURE_THRESHOLD):
            router.mark_unhealthy("bing_news")
        assert router.get_provider().name == "google_news"  # cooldown engaged
        # The earlier (now stale) check finally returns success and marks healthy
        # using its captured epoch — it must NOT clobber the fresh cooldown.
        router.mark_healthy("bing_news", observed_epoch=observed)
        assert router.get_provider().name == "google_news"
        assert router._health["bing_news"].consecutive_failures >= _FAILURE_THRESHOLD

    def test_fresh_mark_healthy_still_resets(self) -> None:
        router = _make_router()
        router.mark_unhealthy("bing_news")
        router.mark_unhealthy("bing_news")
        # Success observed after those failures (current epoch) resets normally.
        observed = router.health_epoch("bing_news")
        router.mark_healthy("bing_news", observed_epoch=observed)
        assert "bing_news" not in router._health

    def test_mark_healthy_without_epoch_resets_unconditionally(self) -> None:
        """Back-compat: callers that don't pass an epoch keep the old reset."""
        router = _make_router()
        router.mark_unhealthy("bing_news")
        router.mark_healthy("bing_news")
        assert "bing_news" not in router._health

    def test_epoch_unknown_provider_is_zero(self) -> None:
        router = _make_router()
        assert router.health_epoch("nonexistent") == 0


class TestCooldownExpiry:
    def test_provider_reeligible_after_cooldown(self) -> None:
        router = _make_router()
        # Mark as unhealthy
        for _ in range(_FAILURE_THRESHOLD):
            router.mark_unhealthy("bing_news")

        # Simulate cooldown expiry by setting last_failure in the past
        health = router._health["bing_news"]
        health.last_failure = datetime.now(UTC) - _UNHEALTHY_COOLDOWN - timedelta(seconds=1)

        # Should be healthy again
        provider = router.get_provider()
        assert provider.name == "bing_news"

    def test_provider_stays_unhealthy_before_cooldown(self) -> None:
        router = _make_router()
        for _ in range(_FAILURE_THRESHOLD):
            router.mark_unhealthy("bing_news")

        # Set last failure to just recently
        health = router._health["bing_news"]
        health.last_failure = datetime.now(UTC) - timedelta(seconds=10)

        provider = router.get_provider()
        assert provider.name == "google_news"
