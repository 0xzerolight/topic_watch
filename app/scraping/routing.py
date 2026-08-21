"""Provider routing with health-based cascade.

Tracks per-provider health in-memory and selects the first healthy
provider for each check cycle. Separate from the per-URL ``feed_health``
table (which tracks individual feed URLs for the UI dashboard).
"""

import logging
import time
from dataclasses import dataclass, field

from app.models import FeedMode, Topic
from app.scraping.providers import BingNewsProvider, GoogleNewsProvider, NewsProvider

logger = logging.getLogger(__name__)

# Default provider priority: Bing first (no redirect resolution),
# Google second (best coverage but fragile).
DEFAULT_PROVIDERS: list[NewsProvider] = [BingNewsProvider(), GoogleNewsProvider()]

_UNHEALTHY_COOLDOWN_SECONDS = 30 * 60.0
_FAILURE_THRESHOLD = 3
_PROBE_LEASE_SECONDS = 120.0
"""How long one half-open probe holds the right to test a recovering provider.

The lease exists only so a check that is cancelled, hangs or dies between taking
the probe and reporting its outcome cannot wedge recovery forever. The normal
release is the outcome itself.
"""


@dataclass
class _ProviderHealth:
    """One provider's local health record. Created on first failure, never deleted.

    Deleting it on success also destroyed ``epoch``, so the next failure recreated
    the counter at zero and an older success carrying that recycled value passed
    the equality check and wiped a newer cooldown (TW-AUD-021). The record now
    outlives any number of recoveries and ``epoch`` only ever counts up.
    """

    consecutive_failures: int = 0
    failed_at: float | None = None
    """Monotonic reading of the last failure. Liveness is measured monotonically —
    a wall-clock step must not expire or extend a cooldown (wave-A clock policy)."""
    epoch: int = 0
    probe_until: float | None = None
    """Monotonic deadline of an outstanding half-open probe, if one is out."""


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
        """The provider a topic would currently be served by — a display answer.

        Always names one, because "which provider is this AUTO topic on" has an
        answer even mid-outage. Callers about to make a request want
        :meth:`admit_provider` instead, which can refuse.
        """
        for provider in self.providers:
            if self._is_healthy(provider.name):
                return provider
        return self.providers[0]

    def admit_provider(self) -> NewsProvider | None:
        """The provider a fetch may use, or ``None`` when none may be used.

        ``None`` is the cooldown actually taking effect. Handing back an unhealthy
        provider "as best effort" meant every due topic attempted both providers
        during the shared 30-minute window and every failure pushed both deadlines
        forward again, so the cooldown suppressed no requests at all and kept a
        throttling upstream throttled (AUG-306).

        Once a cooldown has elapsed the provider is not simply reopened either:
        one caller takes a half-open probe and the rest still get ``None``, so a
        recovering provider meets one request rather than every due topic at once.
        """
        for provider in self.providers:
            if self._is_healthy(provider.name):
                return provider
        return self._take_probe()

    def get_next_provider(self, after: NewsProvider) -> NewsProvider | None:
        """Return the next healthy provider after the given one, or None.

        Only a healthy successor is offered. Falling back to a provider that is
        itself in cooldown was the other half of AUG-306: the within-cycle retry
        then made a second throttled request and refreshed that provider's
        deadline too.
        """
        found = False
        for provider in self.providers:
            if found and self._is_healthy(provider.name):
                return provider
            if provider.name == after.name:
                found = True
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

    def mark_unhealthy(self, provider_name: str) -> bool:
        """Record a failure for a provider.

        Returns ``True`` if this failure is the one that *crossed* the failure
        threshold (the provider just became unhealthy / entered cooldown), so the
        caller can log that transition at a louder level (OVH-133).
        """
        health = self._health.setdefault(provider_name, _ProviderHealth())
        was_below_threshold = health.consecutive_failures < _FAILURE_THRESHOLD
        health.consecutive_failures += 1
        health.epoch += 1
        health.failed_at = time.monotonic()
        health.probe_until = None  # the probe, if this was one, has reported
        logger.debug(
            "Provider %s: failure %d/%d",
            provider_name,
            health.consecutive_failures,
            _FAILURE_THRESHOLD,
        )
        return was_below_threshold and health.consecutive_failures >= _FAILURE_THRESHOLD

    def mark_healthy(self, provider_name: str, observed_epoch: int | None = None) -> bool:
        """Reset failure count for a provider on success.

        Monotonic accounting (OVH-127): if ``observed_epoch`` is given and the
        provider's epoch has advanced since (a concurrent check failed during
        this fetch's await), the success is stale and the reset is skipped so a
        freshly-tripped cooldown is not wiped. ``observed_epoch=None`` keeps the
        unconditional legacy reset for non-overlapping callers.

        Returns ``True`` if the provider had accumulated failures and was just
        reset (a recovery), so the caller can log the recovery (OVH-133).
        """
        health = self._health.get(provider_name)
        if health is None:
            return False
        if observed_epoch is not None and health.epoch != observed_epoch:
            logger.debug(
                "Provider %s: stale success (epoch %d != %d); cooldown kept",
                provider_name,
                observed_epoch,
                health.epoch,
            )
            return False
        recovered = health.consecutive_failures > 0
        health.consecutive_failures = 0
        health.failed_at = None
        health.probe_until = None
        return recovered

    def _take_probe(self) -> NewsProvider | None:
        """Claim the single half-open probe among the recovering providers.

        The candidate is the one whose cooldown elapsed earliest — the provider
        that has been waiting longest is the one worth re-testing first.
        """
        now = time.monotonic()
        candidates = [p for p in self.providers if self._is_recovering(p.name, now)]
        if not candidates:
            return None
        candidate = min(candidates, key=lambda p: self._health[p.name].failed_at or 0.0)
        health = self._health[candidate.name]
        if health.probe_until is not None and health.probe_until > now:
            return None  # a probe is already out; this check waits for its verdict
        health.probe_until = now + _PROBE_LEASE_SECONDS
        logger.info("Provider %s: cooldown elapsed, taking a half-open probe", candidate.name)
        return candidate

    def _is_healthy(self, provider_name: str) -> bool:
        """True while the provider has not crossed the consecutive-failure threshold."""
        health = self._health.get(provider_name)
        return not health or health.consecutive_failures < _FAILURE_THRESHOLD

    def _is_recovering(self, provider_name: str, now: float) -> bool:
        """True for an unhealthy provider whose cooldown has run out."""
        health = self._health.get(provider_name)
        if not health or health.consecutive_failures < _FAILURE_THRESHOLD:
            return False
        return health.failed_at is not None and (now - health.failed_at) > _UNHEALTHY_COOLDOWN_SECONDS


# Module-level singleton — all callers import this.
# The scheduler, CLI, and web layer share the same instance.
router = ProviderRouter()


def topic_owned_feed_urls(topic: Topic, exa_endpoint: str) -> list[str]:
    """The feed-health keys a topic currently answers for.

    ``feed_health`` is keyed by URL and has no topic column, so a feed a topic
    stops using — a removed manual URL, a renamed AUTO query, a deleted topic —
    leaves a row behind that diagnostics kept reporting as a live failing source
    (AUG-148). Ownership is therefore derived from current topic state wherever
    those rows are read.

    An AUTO topic owns every provider's URL, not just the one currently serving
    it, so a standby provider's health still traces back to its topic.
    """
    if topic.feed_mode == FeedMode.AUTO:
        return [provider.build_feed_url(topic) for provider in router.providers]
    if topic.feed_mode == FeedMode.EXA:
        return [exa_endpoint]
    return list(topic.feed_urls)
