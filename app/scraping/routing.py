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
    # Monotonic counter bumped on every failure. Lets a success that started
    # before a concurrent failure detect it is stale and skip the reset.
    epoch: int = 0


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
        """Return the next provider after the given one, or None.

        Prefers a healthy provider after ``after``. When every provider is
        unhealthy, falls back to the first OTHER provider (mirroring
        :meth:`get_provider`'s all-unhealthy best-effort behaviour) so the
        within-cycle retry still attempts the second provider during the
        shared 30-minute cooldown instead of silently returning nothing.
        """
        found = False
        for provider in self.providers:
            if found and self._is_healthy(provider.name):
                return provider
            if provider.name == after.name:
                found = True
        # No healthy successor. If ALL providers are unhealthy, fall back to
        # the first provider that isn't ``after`` (best effort, same as
        # get_provider's all-unhealthy path).
        if not any(self._is_healthy(p.name) for p in self.providers):
            for provider in self.providers:
                if provider.name != after.name:
                    return provider
        return None

    def health_epoch(self, provider_name: str) -> int:
        """Return the provider's current failure epoch (0 if never failed).

        Callers capture this *before* a fetch await and pass it back to
        :meth:`mark_healthy` so a success that raced with a concurrent failure
        can be recognised as stale and not clobber a just-tripped cooldown
        (OVH-127).
        """
        health = self._health.get(provider_name)
        return health.epoch if health else 0

    def mark_unhealthy(self, provider_name: str) -> None:
        """Record a failure for a provider."""
        health = self._health.setdefault(provider_name, _ProviderHealth())
        health.consecutive_failures += 1
        health.epoch += 1
        health.last_failure = datetime.now(UTC)
        logger.debug(
            "Provider %s: failure %d/%d",
            provider_name,
            health.consecutive_failures,
            _FAILURE_THRESHOLD,
        )

    def mark_healthy(self, provider_name: str, observed_epoch: int | None = None) -> None:
        """Reset failure count for a provider on success.

        Monotonic accounting (OVH-127): if ``observed_epoch`` is given and the
        provider's epoch has advanced since (a concurrent check failed during
        this fetch's await), the success is stale and the reset is skipped so a
        freshly-tripped cooldown is not wiped. ``observed_epoch=None`` keeps the
        unconditional legacy reset for non-overlapping callers.
        """
        health = self._health.get(provider_name)
        if health is None:
            return
        if observed_epoch is not None and health.epoch != observed_epoch:
            logger.debug(
                "Provider %s: stale success (epoch %d != %d); cooldown kept",
                provider_name,
                observed_epoch,
                health.epoch,
            )
            return
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
