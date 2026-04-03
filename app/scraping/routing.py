"""Provider routing with health-based cascade.

Tracks per-provider health in-memory and selects the first healthy
provider for each check cycle. Separate from the per-URL ``feed_health``
table (which tracks individual feed URLs for the UI dashboard).
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from app.scraping.providers import BingNewsProvider, GoogleNewsProvider, NewsProvider

logger = logging.getLogger(__name__)

# Default provider priority: Bing first (no redirect resolution),
# Google second (best coverage but fragile).
DEFAULT_PROVIDERS: list[NewsProvider] = [BingNewsProvider(), GoogleNewsProvider()]

_UNHEALTHY_COOLDOWN = timedelta(minutes=30)
_FAILURE_THRESHOLD = 3


@dataclass
class _ProviderHealth:
    consecutive_failures: int = 0
    last_failure: datetime | None = None


@dataclass
class ProviderRouter:
    """Selects providers based on health state.

    Health is tracked in-memory: 3+ consecutive failures marks a
    provider unhealthy for 30 minutes. State resets on app restart
    (desirable, transient failures don't persist).
    """

    providers: list[NewsProvider] = field(default_factory=lambda: list(DEFAULT_PROVIDERS))
    _health: dict[str, _ProviderHealth] = field(default_factory=dict)

    def get_provider(self) -> NewsProvider:
        """Return the first healthy provider."""
        for provider in self.providers:
            if self._is_healthy(provider.name):
                return provider
        # All unhealthy — return first (best effort, cooldown will expire)
        return self.providers[0]

    def get_next_provider(self, after: NewsProvider) -> NewsProvider | None:
        """Return the next healthy provider after the given one, or None."""
        found = False
        for provider in self.providers:
            if found and self._is_healthy(provider.name):
                return provider
            if provider.name == after.name:
                found = True
        return None

    def mark_unhealthy(self, provider_name: str) -> None:
        """Record a failure for a provider."""
        health = self._health.setdefault(provider_name, _ProviderHealth())
        health.consecutive_failures += 1
        health.last_failure = datetime.now(UTC)
        logger.debug(
            "Provider %s: failure %d/%d",
            provider_name,
            health.consecutive_failures,
            _FAILURE_THRESHOLD,
        )

    def mark_healthy(self, provider_name: str) -> None:
        """Reset failure count for a provider on success."""
        if provider_name in self._health:
            del self._health[provider_name]

    def _is_healthy(self, provider_name: str) -> bool:
        health = self._health.get(provider_name)
        if not health or health.consecutive_failures < _FAILURE_THRESHOLD:
            return True
        # Cooldown expired — give it another chance
        return bool(health.last_failure and datetime.now(UTC) - health.last_failure > _UNHEALTHY_COOLDOWN)


# Module-level singleton — all callers import this.
# The scheduler, CLI, and web layer share the same instance.
router = ProviderRouter()
