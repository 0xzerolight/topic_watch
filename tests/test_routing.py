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
