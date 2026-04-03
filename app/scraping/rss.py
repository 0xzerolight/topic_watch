"""RSS/Atom feed fetching and parsing.

Fetches feeds via httpx, parses with feedparser, and converts entries
to FeedEntry models ready for dedup and storage.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from calendar import timegm
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import struct_time
from typing import TYPE_CHECKING

import feedparser
import httpx
from pydantic import BaseModel

from app.models import FeedMode, Topic
from app.url_validation import is_private_url

if TYPE_CHECKING:
    from app.scraping.routing import ProviderRouter

logger = logging.getLogger(__name__)

FeedHealthCallback = Callable[[str, bool, str | None], None]  # (feed_url, success, error_msg)

_USER_AGENT = "TopicWatch/1.0.0 (RSS reader)"
_FEED_FETCH_TIMEOUT = 15.0


class FeedEntry(BaseModel):
    """A single entry parsed from an RSS/Atom feed."""

    title: str
    url: str
    published: datetime | None = None
    summary: str = ""
    source_feed: str


@dataclass
class FeedResponse:
    """Result of fetching feeds for a topic.

    Wraps the parsed entries with metadata about which provider was
    used, so downstream code can make provider-specific decisions
    (e.g. Google News URL resolution) without importing provider classes.
    """

    entries: list[FeedEntry] = field(default_factory=list)
    provider_name: str | None = None
    needs_url_resolution: bool = False


def compute_article_hash(url: str, title: str) -> str:
    """Compute a deterministic, case-insensitive content hash."""
    raw = f"{url}|{title}".lower()
    return hashlib.sha256(raw.encode()).hexdigest()


def _parse_feed_date(entry: dict) -> datetime | None:
    """Extract a datetime from a feedparser entry's date fields."""
    for date_field in ("published_parsed", "updated_parsed"):
        val = entry.get(date_field)
        if isinstance(val, struct_time):
            try:
                return datetime.fromtimestamp(timegm(val), tz=UTC)
            except (ValueError, OverflowError):
                continue
    return None


_GOOGLE_NEWS_HREF_RE = re.compile(r'<a[^>]+href=["\']([^"\']+)["\']', re.IGNORECASE)


def _resolve_google_news_url(link: str, description: str) -> str:
    """Extract the real article URL from a Google News RSS entry (fast path).

    Google News RSS entries use redirect URLs (news.google.com/rss/articles/...)
    as their <link>. Some entries embed the actual article URL as an <a href>
    in the description HTML. This is a zero-cost regex check that avoids HTTP
    requests. When it fails (e.g. Google embeds the same redirect URL in the
    description), the async resolver in google_news.py handles it later in
    the pipeline.
    """
    if "news.google.com/" not in link:
        return link
    match = _GOOGLE_NEWS_HREF_RE.search(description)
    if match:
        real_url = match.group(1)
        if real_url and not real_url.startswith("https://news.google.com/"):
            return real_url
    return link


def _parse_entry(raw_entry: dict, source_feed: str) -> FeedEntry | None:
    """Convert a feedparser entry dict to a FeedEntry, or None if invalid."""
    title = raw_entry.get("title", "").strip()
    url = raw_entry.get("link", "").strip()
    if not title or not url:
        return None

    # Atom/Reddit feeds store content in 'content' field
    summary = raw_entry.get("summary", "")
    if not summary:
        content_list = raw_entry.get("content", [])
        if content_list and isinstance(content_list, list):
            summary = content_list[0].get("value", "")

    # Google News RSS uses redirect URLs — resolve to actual article URLs
    url = _resolve_google_news_url(url, summary)

    return FeedEntry(
        title=title,
        url=url,
        published=_parse_feed_date(raw_entry),
        summary=summary,
        source_feed=source_feed,
    )


async def fetch_feed(
    feed_url: str,
    client: httpx.AsyncClient | None = None,
    timeout: float = _FEED_FETCH_TIMEOUT,
    max_attempts: int = 2,
    health_callback: FeedHealthCallback | None = None,
) -> list[FeedEntry]:
    """Fetch and parse a single RSS/Atom feed. Returns [] on any error."""
    if is_private_url(feed_url):
        logger.warning("Blocked fetch to private URL: %s", feed_url)
        return []
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
        )
    assert client is not None
    try:
        for attempt in range(max_attempts):
            try:
                response = await client.get(feed_url)
                response.raise_for_status()
                parsed = feedparser.parse(response.text)
                entries = []
                for raw in parsed.entries:
                    entry = _parse_entry(raw, feed_url)
                    if entry:
                        entries.append(entry)
                if health_callback:
                    health_callback(feed_url, True, None)
                return entries
            except httpx.TimeoutException as exc:
                if attempt < max_attempts - 1:
                    logger.debug("Timeout fetching feed (attempt %d): %s", attempt + 1, feed_url)
                    await asyncio.sleep(2)
                    continue
                logger.warning("Timeout fetching feed after %d attempts: %s", max_attempts, feed_url)
                if health_callback:
                    health_callback(feed_url, False, f"Timeout after {max_attempts} attempts: {exc}")
                return []
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code >= 500 and attempt < max_attempts - 1:
                    logger.debug(
                        "HTTP %d fetching feed (attempt %d): %s", exc.response.status_code, attempt + 1, feed_url
                    )
                    await asyncio.sleep(2)
                    continue
                logger.warning("HTTP %d fetching feed: %s", exc.response.status_code, feed_url)
                if health_callback:
                    health_callback(feed_url, False, f"HTTP {exc.response.status_code}")
                return []
            except httpx.NetworkError as exc:
                if attempt < max_attempts - 1:
                    logger.debug(
                        "Network error fetching feed (attempt %d): %s — %s",
                        attempt + 1,
                        feed_url,
                        type(exc).__name__,
                    )
                    await asyncio.sleep(2)
                    continue
                logger.warning(
                    "Network error fetching feed after %d attempts: %s — %s",
                    max_attempts,
                    feed_url,
                    type(exc).__name__,
                )
                if health_callback:
                    health_callback(feed_url, False, f"Network error: {type(exc).__name__}: {exc}")
                return []
            except Exception as exc:
                logger.warning("Error fetching feed: %s", feed_url, exc_info=True)
                if health_callback:
                    health_callback(feed_url, False, f"{type(exc).__name__}: {exc}")
                return []
        return []  # pragma: no cover
    finally:
        if owns_client:
            await client.aclose()


async def fetch_feeds_for_topic(
    topic: Topic,
    timeout: float = _FEED_FETCH_TIMEOUT,
    max_attempts: int = 2,
    health_callback: FeedHealthCallback | None = None,
    router: ProviderRouter | None = None,
) -> FeedResponse:
    """Fetch all feeds for a topic, deduplicated by URL.

    For AUTO mode: uses the router to select a provider, with
    within-cycle fallback (max 1 retry with the next provider).
    For MANUAL mode: fetches all explicit feed URLs concurrently.
    """
    if topic.feed_mode == FeedMode.AUTO:
        return await _fetch_auto(topic, timeout, max_attempts, health_callback, router)
    return await _fetch_manual(topic, timeout, max_attempts, health_callback)


async def _fetch_auto(
    topic: Topic,
    timeout: float,
    max_attempts: int,
    health_callback: FeedHealthCallback | None,
    router: ProviderRouter | None,
) -> FeedResponse:
    """AUTO mode: try provider, fallback to next on empty/error."""
    if router is None:
        from app.scraping.routing import router as default_router

        router = default_router

    provider = router.get_provider()

    async with httpx.AsyncClient(
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout,
        follow_redirects=True,
    ) as client:
        feed_url = provider.build_feed_url(topic)
        entries = await fetch_feed(
            feed_url, client, timeout=timeout, max_attempts=max_attempts, health_callback=health_callback
        )

        if entries:
            router.mark_healthy(provider.name)
            return FeedResponse(
                entries=entries,
                provider_name=provider.name,
                needs_url_resolution=provider.needs_url_resolution(),
            )

        # First provider failed or empty — try fallback
        router.mark_unhealthy(provider.name)
        next_provider = router.get_next_provider(provider)
        if next_provider is None:
            return FeedResponse(entries=[], provider_name=provider.name, needs_url_resolution=False)

        logger.info("Provider %s returned no entries, falling back to %s", provider.name, next_provider.name)
        feed_url = next_provider.build_feed_url(topic)
        entries = await fetch_feed(
            feed_url, client, timeout=timeout, max_attempts=max_attempts, health_callback=health_callback
        )

        if entries:
            router.mark_healthy(next_provider.name)
            return FeedResponse(
                entries=entries,
                provider_name=next_provider.name,
                needs_url_resolution=next_provider.needs_url_resolution(),
            )

        router.mark_unhealthy(next_provider.name)
        return FeedResponse(entries=[], provider_name=next_provider.name, needs_url_resolution=False)


async def _fetch_manual(
    topic: Topic,
    timeout: float,
    max_attempts: int,
    health_callback: FeedHealthCallback | None,
) -> FeedResponse:
    """MANUAL mode: fetch all explicit feed URLs concurrently."""
    if not topic.feed_urls:
        return FeedResponse()

    async with httpx.AsyncClient(
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout,
        follow_redirects=True,
    ) as client:
        tasks = [
            fetch_feed(url, client, timeout=timeout, max_attempts=max_attempts, health_callback=health_callback)
            for url in topic.feed_urls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    seen_urls: set[str] = set()
    entries: list[FeedEntry] = []
    for result in results:
        if isinstance(result, BaseException):
            logger.warning("Feed fetch failed: %s", result)
            continue
        for entry in result:
            if entry.url not in seen_urls:
                seen_urls.add(entry.url)
                entries.append(entry)

    return FeedResponse(entries=entries)
