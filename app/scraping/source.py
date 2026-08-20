"""Neutral source layer: the types every article source shares, and the registry.

A "source" is anything that can answer "what is new about this topic": the RSS
providers (AUTO), a topic's own feed URLs (MANUAL), or the Exa search API (EXA).
They differ only in how they reach the network, so everything they exchange with
the pipeline lives here rather than in any one source's module (TW-AUD-022):

- ``FeedEntry`` / ``FeedResponse`` — the entries a source returns and the
  identity plus capabilities of whatever produced them. ``provider_name`` and
  ``needs_url_resolution`` are response-level because AUTO can cascade between
  providers mid-fetch, so only the response knows which one actually answered.
- ``FeedHealthCallback`` / ``FeedStateLoader`` — the health side-channel.
- ``SourceRequest`` — the per-attempt inputs a fetcher needs beyond the topic.
- ``register_source`` / ``fetch_feeds_for_topic`` — mode-to-fetcher dispatch.

Sources register themselves at import, so adding one means writing its module
and registering it, not editing a dispatcher branch in an unrelated source's
module. This module imports no source, which is what keeps that possible.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from pydantic import BaseModel

from app.feed_backoff import BACKOFF_BASE_MINUTES, BACKOFF_CAP_HOURS
from app.models import FeedMode, Topic

if TYPE_CHECKING:
    from app.config import ExaSettings
    from app.models import FeedHealth
    from app.scraping.routing import ProviderRouter

logger = logging.getLogger(__name__)

FeedHealthCallback = Callable[
    [str, bool, str | None, str | None, str | None], None
]  # (feed_url, success, error_msg, etag, last_modified)

FeedStateLoader = Callable[[str], "FeedHealth | None"]
"""Returns the stored health row for a feed URL, or None when untracked."""


def url_hostname(url: str) -> str:
    """The lowercased hostname of a URL, or ``""`` when it has none.

    ``urlparse`` already strips userinfo and port and lowercases the host, so a
    crafted ``https://news.google.com@evil.example/`` yields ``evil.example``.
    A trailing root dot is dropped so ``example.com.`` and ``example.com`` are
    the same host.
    """
    try:
        host = urlparse(url).hostname or ""
    except ValueError:  # malformed netloc (bad port, unbalanced IPv6 brackets)
        return ""
    return host.lower().rstrip(".")


def host_matches(host: str, domain: str) -> bool:
    """True when ``host`` is ``domain`` itself or a subdomain of it.

    Source identity is a host fact, so it is decided on the parsed hostname and
    never on a substring of the raw URL (TW-AUD-031): ``news.google.com`` in a
    path or query belongs to whoever owns the host, and ``news.google.com.evil``
    is a different host that happens to start with the same labels.
    """
    return host == domain or host.endswith(f".{domain}")


class FeedEntry(BaseModel):
    """A single entry parsed from an RSS/Atom feed."""

    title: str
    url: str
    published: datetime | None = None
    summary: str = ""
    source_feed: str
    content: str | None = None
    """Pre-extracted full text, when the source already provides it (e.g. Exa search).
    ``None`` for RSS entries, whose text is fetched during content extraction. When set
    and non-empty, it short-circuits the network fetch in ``extract_article_content``."""


@dataclass(frozen=True)
class SourceIdentity:
    """Which source answered, and what the pipeline must do with its entries.

    ``needs_url_resolution`` is the one capability the pipeline branches on: only
    Google News hands back opaque redirects that need an async resolution pass.
    """

    name: str
    needs_url_resolution: bool = False


@dataclass
class FeedResponse:
    """Result of fetching feeds for a topic.

    Wraps the parsed entries with metadata about which provider was
    used, so downstream code can make provider-specific decisions
    (e.g. Google News URL resolution) without importing provider classes.

    ``feeds_total`` / ``feeds_failed`` expose per-fetch health so the check
    pipeline can distinguish a healthy partial yield from a degraded check where
    some sources silently dropped out (OVH-130). For AUTO mode a single provider
    is fetched (with at most one cascade), so the counts reflect that attempt;
    for MANUAL mode they count the topic's explicit feed URLs.
    """

    entries: list[FeedEntry] = field(default_factory=list)
    provider_name: str | None = None
    needs_url_resolution: bool = False
    feeds_total: int = 0
    feeds_failed: int = 0
    feeds_skipped: int = 0
    """MANUAL mode: feeds skipped this cycle because they are in a backoff window
    (persistently failing). For MANUAL mode ``feeds_total`` counts feeds ATTEMPTED
    (skipped feeds are excluded and surface here), so a backed-off feed is never
    miscounted as a partial failure."""

    @classmethod
    def from_source(
        cls,
        identity: SourceIdentity,
        *,
        entries: list[FeedEntry] | None = None,
        feeds_total: int = 0,
        feeds_failed: int = 0,
        feeds_skipped: int = 0,
        needs_url_resolution: bool | None = None,
    ) -> FeedResponse:
        """Build a response stamped with the identity and capabilities of its source.

        ``needs_url_resolution`` defaults to the source's own capability; a
        response with no entries overrides it to False, since there is nothing
        left to resolve.
        """
        return cls(
            entries=entries if entries is not None else [],
            provider_name=identity.name,
            needs_url_resolution=(
                identity.needs_url_resolution if needs_url_resolution is None else needs_url_resolution
            ),
            feeds_total=feeds_total,
            feeds_failed=feeds_failed,
            feeds_skipped=feeds_skipped,
        )


@dataclass(frozen=True)
class SourceRequest:
    """Everything a source fetcher needs for one attempt, besides the topic.

    One request object per logical fetch keeps every source's signature identical,
    which is what lets the registry dispatch without knowing who it is calling.
    Fields a given source does not use are simply ignored by it.
    """

    timeout: float
    max_attempts: int = 2
    max_results: int = 10
    health_callback: FeedHealthCallback | None = None
    feed_state_loader: FeedStateLoader | None = None
    backoff_base_minutes: int = BACKOFF_BASE_MINUTES
    backoff_cap_hours: int = BACKOFF_CAP_HOURS
    exa_settings: ExaSettings | None = None
    router: ProviderRouter | None = None


SourceFetcher = Callable[[Topic, SourceRequest], Awaitable[FeedResponse]]

_SOURCES: dict[FeedMode, SourceFetcher] = {}


def register_source(mode: FeedMode, fetcher: SourceFetcher) -> None:
    """Bind a feed mode to the coroutine that fetches it (called at import)."""
    _SOURCES[mode] = fetcher


async def fetch_feeds_for_topic(
    topic: Topic,
    timeout: float = 15.0,
    max_attempts: int = 2,
    health_callback: FeedHealthCallback | None = None,
    router: ProviderRouter | None = None,
    feed_state_loader: FeedStateLoader | None = None,
    backoff_base_minutes: int = BACKOFF_BASE_MINUTES,
    backoff_cap_hours: int = BACKOFF_CAP_HOURS,
    exa_settings: ExaSettings | None = None,
    max_results: int = 10,
) -> FeedResponse:
    """Fetch all entries for a topic from whichever source its feed mode names.

    For AUTO mode: uses the router to select a provider, with within-cycle
    fallback (max 1 retry with the next provider). For MANUAL mode: fetches all
    explicit feed URLs concurrently, skipping any in a backoff window. For EXA
    mode: queries the Exa search API (``exa_settings`` required; ``max_results``
    bounds the paid result count).

    ``feed_state_loader`` supplies the stored ``FeedHealth`` per URL — used to
    send conditional-GET validators (both RSS modes) and to skip backed-off feeds
    (MANUAL only; AUTO provider backoff is owned by ``ProviderRouter``).
    """
    request = SourceRequest(
        timeout=timeout,
        max_attempts=max_attempts,
        max_results=max_results,
        health_callback=health_callback,
        feed_state_loader=feed_state_loader,
        backoff_base_minutes=backoff_base_minutes,
        backoff_cap_hours=backoff_cap_hours,
        exa_settings=exa_settings,
        router=router,
    )
    return await _SOURCES[topic.feed_mode](topic, request)
